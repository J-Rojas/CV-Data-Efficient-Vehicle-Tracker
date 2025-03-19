import os
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel, SegformerForSemanticSegmentation, SegformerFeatureExtractor, SegformerConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import JaccardIndex
from .losses import dice_loss, total_variation_loss, boundary_aware_loss, focal_loss, iou_loss

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SegmentationLightning(pl.LightningModule):
    def __init__(self, backbone_name="google/vit-base-patch16-224", num_classes=2, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        # Load the feature extractor and model
        model_name = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
        config = SegformerConfig(num_labels = num_classes, classifier_dropout_prob=0.5)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        self.model.decode_head.classifier.apply(init_weights)
        self.iou_metric = JaccardIndex(task="multiclass", num_classes=2).to(self.model.device)
        
        self.lr = lr        
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.decode_head.parameters():
            param.requires_grad = True

        print(self.model)


    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        logits = outputs.logits  # Shape: (batch_size, num_labels, H, W)

        # Upsample logits to the original image size
        # The feature extractor's size may differ from the original image size.
        logits = torch.nn.functional.interpolate(
            logits,
            size=pixel_values.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        return logits
    
    def combined_loss(self, logits, targets, alpha=0.3):        
        #ce = nn.functional.cross_entropy(logits, targets_long)
        target_two_channel = torch.stack([1 - targets, targets], dim=1)
        # Convert targets to one-hot for dice loss; assume num_classes=2
        pred_probs = torch.softmax(logits, dim=1)
        # focal loss is based on the cross entropy
        floss = focal_loss(logits, targets, use_bce=True) # use bce when using soft-targets
        dice = dice_loss(pred_probs, target_two_channel)
        # Compute boundary-aware loss, this already uses the CE loss
        b_loss = 0.0 #boundary_aware_loss(logits, targets, use_bce=True)
        iou = 0 # iou_loss(pred_probs[:,1,:,:], targets)
        return alpha * (iou + floss + b_loss) + (1 - alpha) * dice

    def criterion(self, logits, targets):
        use_bce = False
        seg_loss = self.combined_loss(logits, targets) 

        # Compute TV loss on the predicted probabilities (or on logits)
        # Here, we apply softmax to obtain probability maps
        prob_maps = torch.softmax(logits, dim=1)
        # Extract the probability for the positive class (assume index 1)
        pred_positive = prob_maps[:, 1, :, :]   # shape: (B, H, W)

        # Compute MSE loss between the predicted vehicle probability and fuzzy label   
        if use_bce:             
            loss = nn.functional.binary_cross_entropy_with_logits(pred_positive, targets)
        else:
            loss = nn.functional.mse_loss(pred_positive, targets)

        tv_loss = total_variation_loss(prob_maps)
        
        # Define a hyperparameter to weight the TV loss
        lambda_tv = 0.1
        lambda_ce = 0.9
        total_loss = lambda_ce * loss + (1 - lambda_ce) * seg_loss + lambda_tv * tv_loss

        return total_loss

    def criterion_mse(self, logits, targets):
        prob_maps = torch.softmax(logits, dim=1)
        pred_positive = prob_maps[:, 1, :, :]   # shape: (B, H, W)
        mse_loss = nn.functional.mse_loss(pred_positive, targets)
        prob_maps = torch.softmax(logits, dim=1)
        pred_positive = prob_maps[:, 1, :, :]   # shape: (B, H, W)

        return mse_loss # + iou_loss(pred_positive, targets)

    def validation_metric(self, logits, labels):
        preds = torch.argmax(logits, dim=1)  # (B, H, W)        
        self.iou_metric.update(preds, labels)

    def training_step(self, batch, batch_idx):
        # ensure dropout is enabled
        self.model.decode_head.train(True)
        if type(batch["pixel_values"]) == list:
            pixel_values = torch.stack(batch["pixel_values"], dim=0) 
            labels = torch.stack(batch["labels"], dim=0)     # (B, S, 3, H, W)

            # roll the sequence into the batch dimension
            pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
            labels = labels.view(-1, *labels.shape[2:])
        else:
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]

        # Forward pass: get the logits
        logits = self.forward(**{"pixel_values": pixel_values})
        
        labels = labels.squeeze() # (B, H, W) with class indices
        
        total_loss = self.criterion(logits, labels)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]      # (B, S, 3, H, W)

        # roll the sequence into the batch dimension
        if len(pixel_values.shape) == 5:
            pixel_values = torch.stack(batch["pixel_values"], dim=0)
            pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
            labels = torch.stack(batch["labels"], dim=0)
            labels = labels.view(-1, *labels.shape[2:])

        # Forward pass: get the logits
        logits = self.forward(**{"pixel_values": pixel_values})
        
        labels = batch["labels"].squeeze() # (B, H, W) with class indices
    
        labels_long = labels.long()

        # look at bottom half of the images    
        _, _, height, width = logits.shape 
        height_half = int(height / 2)            
        logits = logits[:,:,height_half:,:]
        labels = labels[:,height_half:,:]
        labels_long = labels_long[:,height_half:,:]
        pixel_values = pixel_values[:,:,height_half:,:]
        
        total_loss = self.criterion(logits, labels)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        
        self.validation_metric(logits, labels_long)

        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def on_validation_epoch_start(self) -> None:
        self.iou_metric.reset()
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        iou = self.iou_metric.compute().item()
        self.log("IoU", iou, on_step=False, on_epoch=True, prog_bar=True)
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold_mode="abs")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # this should be the metric you wish to monitor
            },
        }
        
    def train_dataloader(self):
        from .loader import train_loader
        return train_loader

    def val_dataloader(self):
        from .loader import val_loader
        return val_loader

# Example usage:
if __name__ == '__main__':
    
    if os.path.exists("./checkpoints/last.ckpt"):
        os.remove("./checkpoints/last.ckpt")

    # checkpoint saver
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",          # metric to monitor
        filename="checkpoint-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,                # save the best 3 models
        mode="min",                  # lower val_loss is better
        save_last=True,               # also save the last epoch
        dirpath="checkpoints/"
    )

    best_model = None
    best_avg_loss = float('inf')

    # Evaluate each seed by training for a few steps
    for seed in range(5):
        
        # Reinitialize model and trainer
        model = SegmentationLightning()
        trainer = Trainer(
            max_steps=1,                 # Train for 50 steps (adjust as needed)
            logger=False,                 # Disable logging for quick evaluation
            enable_checkpointing=False,
            enable_progress_bar=False            
        )
        
        trainer.fit(model)
        
        # Retrieve the training loss from the logged metrics
        loss_tensor = trainer.callback_metrics.get("train_loss")
        avg_loss = loss_tensor.item() if loss_tensor is not None else float('inf')
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_model = model
        

    trainer = Trainer(max_epochs=40, callbacks=[checkpoint_callback])
    trainer.fit(best_model)
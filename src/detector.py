import os
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel, SegformerForSemanticSegmentation, SegformerFeatureExtractor
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
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
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.config.num_labels = num_classes
        in_channels = self.model.decode_head.classifier.in_channels
        self.model.decode_head.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.model.decode_head.classifier.apply(init_weights)
        #self.model = SegmentationModel(backbone_model_name=backbone_name, num_classes=num_classes)
        # Use per-pixel CrossEntropyLoss (expects logits of shape [B, C, H, W] and labels of shape [B, H, W])
        #self.criterion = nn.CrossEntropyLoss()
        self.lr = lr        

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.decode_head.parameters():
            param.requires_grad = True

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
        return alpha * (floss + b_loss) + (1 - alpha) * dice

    def criterion(self, logits, targets):
        seg_loss = self.combined_loss(logits, targets)

        # Compute TV loss on the predicted probabilities (or on logits)
        # Here, we apply softmax to obtain probability maps
        prob_maps = torch.softmax(logits, dim=1)
        # Extract the probability for the positive class (assume index 1)
        pred_positive = prob_maps[:, 1, :, :]   # shape: (B, H, W)

        # Compute MSE loss between the predicted vehicle probability and fuzzy label
        mse_loss = nn.functional.mse_loss(pred_positive, targets)

        tv_loss = total_variation_loss(prob_maps)
        
        # Define a hyperparameter to weight the TV loss
        lambda_tv = 0.1
        lambda_mse = 0.7
        total_loss = lambda_mse * mse_loss + (1 - lambda_mse) * seg_loss + lambda_tv * tv_loss

        return total_loss

    def criterion_mse(self, logits, targets):
        prob_maps = torch.softmax(logits, dim=1)
        pred_positive = prob_maps[:, 1, :, :]   # shape: (B, H, W)
        mse_loss = nn.functional.mse_loss(pred_positive, targets)
        prob_maps = torch.softmax(logits, dim=1)
        pred_positive = prob_maps[:, 1, :, :]   # shape: (B, H, W)

        return mse_loss # + iou_loss(pred_positive, targets)


    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]      # (B, 3, H, W)
        
        # Forward pass: get the logits
        logits = self.forward(**{"pixel_values": pixel_values})
        
        labels = batch["labels"].squeeze() # (B, H, W) with class indices
        
        total_loss = self.criterion(logits, labels)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"].squeeze()
        logits = self(pixel_values)
        loss = self.focal_loss(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
        
    def train_dataloader(self):
        from .loader import train_loader
        return train_loader

# Example usage:
if __name__ == '__main__':
    # Load feature extractor to preprocess images
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    # Initialize our custom segmentation model using Swin-Tiny as backbone
    num_classes = 2  # For example, background vs. vehicle
    model = SegmentationLightning()

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

    trainer = Trainer(max_epochs=10, callbacks=[checkpoint_callback])
    trainer.fit(model)
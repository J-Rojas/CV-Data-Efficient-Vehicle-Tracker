import os
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from .losses import dice_loss, total_variation_loss, boundary_aware_loss, focal_loss

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_prob=0.1):
        super().__init__()
        # A simple segmentation head: a convolution, non-linearity, and a final conv layer
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        # A simple segmentation head: a convolution, non-linearity, and a final conv layer
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_prob)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x, output_size):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)        
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        # Upsample to match the output_size (e.g., original image resolution)
        x = nn.functional.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        return x

class SegmentationModel(nn.Module):
    def __init__(self, backbone_model_name, num_classes, patch_resolution=(14, 14)):
        super().__init__()
        # Load the pretrained Swin model from HuggingFace
        self.backbone = AutoModel.from_pretrained(backbone_model_name)
        # Assume the backbone configuration contains the hidden size
        hidden_dim = self.backbone.config.hidden_size
        # Assume a patch resolution of 7x7 for a 224x224 image.
        # This might be different depending on the model.
        self.patch_resolution = patch_resolution
        self.seg_head = SegmentationHead(in_channels=hidden_dim, num_classes=num_classes)

        # freeze the pre-trained backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values):
        # pixel_values: (batch, 3, height, width)
        outputs = self.backbone(pixel_values)
        # outputs.last_hidden_state has shape (batch, num_patches, hidden_dim)
        hidden_states = outputs.last_hidden_state
        batch_size = hidden_states.shape[0]
        H, W = self.patch_resolution
        # Reshape hidden_states to (batch, hidden_dim, H, W)
        tokens_without_cls = hidden_states[:, 1:, :]
        feature_map = tokens_without_cls.transpose(1, 2).reshape(batch_size, -1, H, W)
        # Use the segmentation head to produce per-pixel predictions at the original resolution
        seg_logits = self.seg_head(feature_map, output_size=pixel_values.shape[2:])
        return seg_logits


class SegmentationLightning(pl.LightningModule):
    def __init__(self, backbone_name="google/vit-base-patch16-224", num_classes=2, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = SegmentationModel(backbone_model_name=backbone_name, num_classes=num_classes)
        # Use per-pixel CrossEntropyLoss (expects logits of shape [B, C, H, W] and labels of shape [B, H, W])
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr        

    def forward(self, pixel_values):
        return self.model(pixel_values)

    def combined_loss(self, logits, targets, alpha=0.3):
        targets_long = targets.long()
        ce = nn.functional.cross_entropy(logits, targets_long)
        target_two_channel = torch.stack([1 - targets, targets], dim=1)
        # Convert targets to one-hot for dice loss; assume num_classes=2
        pred_probs = torch.softmax(logits, dim=1)
        # focal loss is based on the cross entropy
        floss = focal_loss(logits, targets_long)
        dice = dice_loss(pred_probs, target_two_channel)
        # Compute boundary-aware loss, this already uses the CE loss
        b_loss = 0.0 #boundary_aware_loss(logits, targets_long)
        return alpha * (floss + b_loss) + (1 - alpha) * dice

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]      # (B, 3, H, W)
        labels = batch["labels"].squeeze() # (B, H, W) with class indices
        logits = self(pixel_values)               # (B, num_classes, H, W)
        seg_loss = self.combined_loss(logits, labels)

        # Compute TV loss on the predicted probabilities (or on logits)
        # Here, we apply softmax to obtain probability maps
        prob_maps = torch.softmax(logits, dim=1)
        # Extract the probability for the positive class (assume index 1)
        pred_positive = prob_maps[:, 1, :, :]   # shape: (B, H, W)

        # Compute MSE loss between the predicted vehicle probability and fuzzy label
        mse_loss = nn.functional.mse_loss(pred_positive, labels)

        tv_loss = total_variation_loss(prob_maps)
        
        # Define a hyperparameter to weight the TV loss
        lambda_tv = 0.1
        lambda_mse = 0.5
        total_loss = lambda_mse * mse_loss + (1 - lambda_mse) * seg_loss + lambda_tv * tv_loss

        self.log("train_seg_loss", seg_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_tv_loss", tv_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
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
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
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

    # checkpoint saver
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",          # metric to monitor
        filename="checkpoint-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,                # save the best 3 models
        mode="min",                  # lower val_loss is better
        save_last=True,               # also save the last epoch
        dirpath="checkpoints/"
    )

    trainer = Trainer(max_epochs=5, callbacks=[checkpoint_callback])
    trainer.fit(model)
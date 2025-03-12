
import os
import torch
import pytorch_lightning as pl
import cv2
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from .loader import val_loader  # your data module from previous code
from .detector import SegmentationLightning  # your LightningModule
from .tools import get_bounding_box_from_mask, torch_to_cv2_image


def evaluate_model(checkpoint_path):
    # Load the trained model from a checkpoint.
    model = SegmentationLightning.load_from_checkpoint(checkpoint_path)
    model.freeze()  # freeze model and set to evaluation mode
    model.eval()

    # Create the validation DataLoader from the datamodule.
    #val_loader = val_loader

    # Prepare a metric, e.g., JaccardIndex for IoU (for binary segmentation, num_classes=2)
    iou_metric = JaccardIndex(task="multiclass", num_classes=2).to(model.device)
    
    total_loss = 0.0
    total_batches = 0

    criterion = model.criterion

    video_out = None
    with torch.no_grad():
        for batch in val_loader:

            pixel_values = batch["pixel_values"].to(model.device)  # (B, 3, H, W)
            labels = batch["labels"].to(model.device).long()              # (B, H, W) expected

            if video_out is None:
                layers, height, width = pixel_values.shape[1:]
                size = (width, height)
                video_out = cv2.VideoWriter("output_video.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      15, size)
            
            # Ensure labels are in the correct shape (B, H, W)
            if labels.dim() == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            
            logits = model(pixel_values)  # (B, num_classes, H, W)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_batches += 1
            
            # For IoU, get predicted class for each pixel
            preds = torch.argmax(logits, dim=1)  # (B, H, W)
            iou_metric.update(preds, labels)

            print((preds != 0).sum())

            for i in range(pixel_values.shape[0]):
                bbox = get_bounding_box_from_mask(preds[i].squeeze().detach().numpy())
                print(bbox)

                im = torch_to_cv2_image(pixel_values[i].detach())
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

                video_out.write(im)

    mean_loss = total_loss / total_batches
    mean_iou = iou_metric.compute().item()

    print(f"Validation Loss: {mean_loss:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")

    if video_out:
        video_out.release()

if __name__ == "__main__":
    # Set paths for your data and checkpoint
    checkpoint_path = "./checkpoints/last.ckpt"

    # Evaluate the model
    evaluate_model(checkpoint_path)
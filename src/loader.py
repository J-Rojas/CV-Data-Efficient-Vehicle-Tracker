import os
import glob
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from .tools import create_fuzzy_label_rect, create_label_rect
import torchvision.transforms as transforms
import pandas as pd
from transformers import SegformerFeatureExtractor

class VehicleSegmentationDataset(Dataset):
    def __init__(self, data_dir, labels_dir, feature_extractor, ignore=True):
        """
        Args:
            data_dir (str): Path to the folder containing input images.            
            transform (callable, optional): Transformation to apply to the input images.
            mask_transform (callable, optional): Transformation to apply to the masks.
        """
        self.feature_extractor = feature_extractor
        self.data_dir = data_dir 
        self.labels_dir = labels_dir
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))        
        self.labels_paths = sorted(glob.glob(os.path.join(labels_dir, "*_foreground_label.png")))  
        #print(self.labels_df)   

        frames_to_ignore = pd.read_csv(labels_dir + "/ignore.csv", header=None)
        self.ignored_frames = set((frames_to_ignore.to_numpy() - 1).flatten())

        # filter out ignored frames
        if ignore:
            self.image_paths = list(map(lambda x: x[1], filter(lambda x: x[0] not in self.ignored_frames, enumerate(self.image_paths))))
            self.labels_paths = list(map(lambda x: x[1], filter(lambda x: x[0] not in self.ignored_frames, enumerate(self.labels_paths))))
        
        self.transform = None   
        # For the masks, we need to use nearest neighbor interpolation to avoid label mixing.
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Read the CSV row corresponding to this sample
        #row = self.labels_df.iloc[idx]
        filename = self.image_paths[idx]
        filename_label = self.labels_paths[idx]
        #bbox = (int(row[2]), int(row[3]), int(row[6]), int(row[7]))

        # Load the image
        image_path = filename
        image = Image.open(image_path).convert("RGB")
        pixels = feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # Load the label image
        image_path = filename_label
        image = Image.open(image_path).convert("RGBA")        
        pixels_label = self.mask_transform(image)

        # only use the alpha channel to determine the mask
        mask = pixels_label[3,:,:]
                
        #print("Non zero mask values: ", (mask > 0.0).sum())
        #print("Zero mask values: ", (mask == 0).sum())

        return {"pixel_values": pixels, "labels": mask}


# Paths to your data directories
validation_images_dir = "./data/"
labels_dir = "./data_augment/"

model_name = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)

# Create datasets
train_dataset = VehicleSegmentationDataset(
    data_dir=validation_images_dir,
    labels_dir=labels_dir,
    feature_extractor=feature_extractor
)
val_dataset = VehicleSegmentationDataset(
    data_dir=validation_images_dir,
    labels_dir=labels_dir,
    feature_extractor=feature_extractor
)
eval_dataset = VehicleSegmentationDataset(
    data_dir=validation_images_dir,
    labels_dir=labels_dir,
    feature_extractor=feature_extractor,
    ignore=False
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=4)


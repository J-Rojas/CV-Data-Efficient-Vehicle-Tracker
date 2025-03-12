import os
import glob
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from .tools import create_fuzzy_label_rect
import torchvision.transforms as transforms
import pandas as pd

class VehicleSegmentationDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path to the folder containing input images.            
            transform (callable, optional): Transformation to apply to the input images.
            mask_transform (callable, optional): Transformation to apply to the masks.
        """
        self.data_dir = data_dir        
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.labels_df = pd.read_csv(data_dir + "/groundtruth.txt", header=None)  
        print(self.labels_df)      
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # For the masks, we need to use nearest neighbor interpolation to avoid label mixing.
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])



    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Read the CSV row corresponding to this sample
        row = self.labels_df.iloc[idx]
        filename = self.image_paths[idx]
        bbox = (int(row[2]), int(row[3]), int(row[6]), int(row[7]))

        # Load the image
        image_path = filename
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Create a blank mask and draw the bounding box on it
        mask = create_fuzzy_label_rect((height, width), bbox, factor=1.0)

        # Convert the NumPy array to a PIL Image
        mask = Image.fromarray(mask)
        
        # Apply image transformations
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Apply mask transformations; use NEAREST interpolation to preserve label values
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)
            # Remove channel dimension and convert to long type for loss functions like CrossEntropyLoss
            mask = mask.squeeze()

        #print("Non zero mask values: ", (mask != 0).sum())
        #print("Zero mask values: ", (mask == 0).sum())

        return {"pixel_values": image, "labels": mask}


# Paths to your data directories
train_images_dir = "./data/"

# Create datasets
train_dataset = VehicleSegmentationDataset(
    data_dir=train_images_dir
)
val_dataset = VehicleSegmentationDataset(
    data_dir=train_images_dir
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

import os
import glob
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from .tools import create_fuzzy_label_rect, create_label_rect
from .augment import generate_random_sequence
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

class VehicleSegmentationAugmentedDataset(Dataset):

    def __init__(self, data_dir, feature_extractor, ignore=True, num_items=100):
        """
        Args:
            data_dir (str): Path to the folder containing input images.            
            transform (callable, optional): Transformation to apply to the input images.
            mask_transform (callable, optional): Transformation to apply to the masks.
        """
        self.feature_extractor = feature_extractor
        self.data_dir = data_dir 
        self.labels_dir = labels_dir
        self.bg_image_paths = sorted(glob.glob(os.path.join(data_dir, "*_background.jpg")))        
        self.fg_image_paths = sorted(glob.glob(os.path.join(data_dir, "*_foreground.png")))        
        self.labels_paths = sorted(glob.glob(os.path.join(labels_dir, "*_foreground_label.png")))  
        #print(self.labels_df)   

        frames_to_ignore = pd.read_csv(labels_dir + "/ignore.csv", header=None)
        self.ignored_frames = set((frames_to_ignore.to_numpy() - 1).flatten())

        # filter out ignored frames
        if ignore:
            self.fg_image_paths = list(map(lambda x: x[1], filter(lambda x: x[0] not in self.ignored_frames, enumerate(self.fg_image_paths))))
            self.bg_image_paths = list(map(lambda x: x[1], filter(lambda x: x[0] not in self.ignored_frames, enumerate(self.bg_image_paths))))
            self.labels_paths = list(map(lambda x: x[1], filter(lambda x: x[0] not in self.ignored_frames, enumerate(self.labels_paths))))
        
        self.transform = None   
        # For the masks, we need to use nearest neighbor interpolation to avoid label mixing.
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        self.num_items = num_items
        self.sequence_range = [16, 16]        
    
    def __len__(self):
        return self.num_items
    
    def __getitem__(self, idx):

        images, image_labels = generate_random_sequence(self.fg_image_paths, self.bg_image_paths, length=[self.sequence_range, self.sequence_range])

        pixels_values_seq = []
        mask_seq = []

        for im, label in zip(images, image_labels):                
            pixels = feature_extractor(images=im, return_tensors="pt")["pixel_values"].squeeze(0)                  
            arr_uint8 = (np.clip(label, 0, 1) * 255).astype(np.uint8)
            pixels_label = self.mask_transform(Image.fromarray(arr_uint8))

            # only use the alpha channel to determine the mask
            mask = pixels_label[3,:,:]

            pixels_values_seq.append(pixels)
            mask_seq.append(mask)

        return {"pixel_values": pixels_values_seq, "labels": mask_seq}


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

train_dataset = VehicleSegmentationAugmentedDataset(
    data_dir=labels_dir,
    feature_extractor=feature_extractor,
    num_items=250
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

#eval_dataset = VehicleSegmentationAugmentedDataset(
#    data_dir=labels_dir,
#    feature_extractor=feature_extractor,
#    ignore=True,
#    num_items=250
#)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=4)


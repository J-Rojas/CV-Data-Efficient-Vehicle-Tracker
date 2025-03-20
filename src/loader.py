from numpy._typing._array_like import NDArray
import os
import glob
from typing import Any
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from .tools import create_fuzzy_label_rect, create_label_rect, calculate_optical_flow
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
        
        # Merge the previous and current frames into one image vertically for input into the network

        images = []
        labels = []
        for i in range(idx - 1, idx + 2):
            if 0 <= i < len(self.image_paths):
                filename = self.image_paths[i]
                filename_label = self.labels_paths[i]
                            
                # Load the image                
                image = Image.open(filename).convert("RGB")

                images.append(np.array(image))

                # Load the label image                
                image = Image.open(filename_label).convert("RGBA")                
                image = np.array(image)
                labels.append(image)
            else:
                images.append(None)
                labels.append(None)

        for i in range(len(images)):
            if images[i] is None:
                # generate an image with pure noise
                images[i] = images[1].copy()
                labels[i] = labels[1].copy()            

        pad_width = ((0, 0), (0, 0), (0, 1))        
        
        # merge 2nd and 1st images        
        image = np.concatenate([images[0], images[1]], axis=0)        
        pixels = feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # replace top
        height_half = pixels.shape[1] // 2
        upper_half = pixels[:,:height_half,:].numpy().astype(np.uint8).transpose(1, 2, 0)
        lower_half = pixels[:,height_half:,:].numpy().astype(np.uint8).transpose(1, 2, 0)
        flow = torch.Tensor(np.pad(calculate_optical_flow(upper_half, lower_half), pad_width=pad_width, constant_values=0)).float().permute(2, 0, 1)
        pixels[:,:height_half,:] = flow

        # Load the label image
        pixels_label = np.concatenate([labels[2], labels[1]], axis=0)    
        pixels_label = self.mask_transform(Image.fromarray(pixels_label))
        
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

        images, image_labels, bboxes = generate_random_sequence(self.fg_image_paths, self.bg_image_paths, length=[self.sequence_range, self.sequence_range])

        # stack 2nd, 1st images
        image_input: NDArray[Any] = np.concatenate([images[0], images[1]], axis=0)
        # stack 2nd and 3rd labels
        label_input = np.concatenate([image_labels[2], image_labels[1]], axis=0)
        # reposition bboxes for merge
        bboxes[0] = np.array([bboxes[0][0] + images[0].shape[0], bboxes[0][1], bboxes[0][2] + images[0].shape[0], bboxes[0][3]])
        
        pixels = feature_extractor(images=image_input, return_tensors="pt")["pixel_values"].squeeze(0)                  

        # replace top
        pad_width = ((0, 0), (0, 0), (0, 1))  
        height_half = pixels.shape[1] // 2
        upper_half = pixels[:,:height_half,:].numpy().astype(np.uint8).transpose(1, 2, 0)
        lower_half = pixels[:,height_half:,:].numpy().astype(np.uint8).transpose(1, 2, 0)
        flow = torch.Tensor(np.pad(calculate_optical_flow(upper_half, lower_half), pad_width=pad_width, constant_values=0)).float().permute(2, 0, 1)
        pixels[:,:height_half,:] = flow

        arr_uint8 = (np.clip(label_input, 0, 1) * 255).astype(np.uint8)
        pixels_label = self.mask_transform(Image.fromarray(arr_uint8))

        #print(pixels.shape, image_input.shape)

        # image scaling factor
        scale_y = float(pixels.shape[1]) / image_input.shape[0] # pixels is in C, H, W, image_input is in H, W, C
        scale_x = float(pixels.shape[2]) / image_input.shape[1]

        # recalculate bboxes after image rescaling
        scale_matrix = np.array([[scale_y, 0.0], [0.0, scale_x]])
        

        #print("scale", scale_matrix)
        #print("bboxes before", bboxes)
        bboxes_trans = np.array([np.matmul(scale_matrix, bbox.reshape(2, 2).T).T.reshape(4) for bbox in bboxes]).astype(int)
        #print("bboxes", bboxes_trans)

        
        #print(bboxes_trans)

        # only use the alpha channel to determine the mask
        mask = pixels_label[3,:,:]

        return {"pixel_values": pixels, "labels": mask, "boxes": bboxes_trans}


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
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4)


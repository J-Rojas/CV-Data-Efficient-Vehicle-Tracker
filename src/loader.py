from numpy._typing._array_like import NDArray
import os
import glob
from typing import Any
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from .tools import create_fuzzy_label_rect, create_label_rect, calculate_optical_flow
from .augment import generate_random_sequence, merge_background_and_foreground_sequences, get_fixed_bg_params, get_fixed_fg_params, get_fixed_bg_params_none
import torchvision.transforms as transforms
import pandas as pd
from transformers import SegformerFeatureExtractor

def get_image_sequences(idx, image_paths, labels_paths):
    images = []
    labels = []
    for i in range(idx - 1, idx + 2):
        if 0 <= i < len(image_paths):
            filename = image_paths[i]
            filename_label = labels_paths[i]
                        
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

    return images, labels

class VehicleDatasetBase(Dataset):

    def __init__(self):
        self.feature_extractor = feature_extractor
        # For the masks, we need to use nearest neighbor interpolation to avoid label mixing.
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def _prepare_output_data(self, images, image_labels=None, bboxes=None, shift_labels=True):
        # stack 2nd, 1st images
        image_input: NDArray[Any] = np.concatenate([images[0], images[1]], axis=0)
        mask = None
        # stack 2nd and 3rd labels
        if image_labels is not None:
            label_input = np.concatenate([image_labels[2], image_labels[1]], axis=0)
            arr_uint8 = (np.clip(label_input, 0, 1) * 255).astype(np.uint8)
            pixels_label = self.mask_transform(Image.fromarray(arr_uint8))

            # only use the alpha channel to determine the mask
            mask = pixels_label[3,:,:]
            
        pixels = self.feature_extractor(images=image_input, return_tensors="pt")["pixel_values"].squeeze(0)

        # replace top
        pad_width = ((0, 0), (0, 0), (0, 1))  
        height_half = pixels.shape[1] // 2
        upper_half = pixels[:,:height_half,:].numpy().astype(np.uint8).transpose(1, 2, 0)
        lower_half = pixels[:,height_half:,:].numpy().astype(np.uint8).transpose(1, 2, 0)
        #flow = torch.Tensor(np.pad(calculate_optical_flow(upper_half, lower_half), pad_width=pad_width, constant_values=0)).float().permute(2, 0, 1) 
            
        # replace with the image difference       
        pixels[0,:height_half,:] = pixels[:,height_half:,:].mean(dim=0) - pixels[:,:height_half,:].mean(dim=0)
        pixels[1,:height_half,:] = pixels[:,:height_half,:].mean(dim=0)

        #print(pixels.shape, image_input.shape)

        # image scaling factor
        scale_y = float(pixels.shape[1]) / image_input.shape[0] # pixels is in C, H, W, image_input is in H, W, C
        scale_x = float(pixels.shape[2]) / image_input.shape[1]

        # recalculate bboxes after image rescaling
        scale_matrix = np.array([[scale_y, 0.0], [0.0, scale_x]])

        # reposition bboxes for merge
        bboxes_trans = None
        if bboxes is not None:
            if shift_labels:
                bboxes[0] = np.array([bboxes[0][0] + images[0].shape[0], bboxes[0][1], bboxes[0][2] + images[0].shape[0], bboxes[0][3]])
            #print("scale", scale_matrix)
            #print("bboxes before", bboxes)
            bboxes_trans = torch.tensor(np.array([np.matmul(scale_matrix, bbox.reshape(2, 2).T).T.reshape(4) for bbox in bboxes]).astype(int))
        #print("bboxes", bboxes_trans)
        #print(bboxes_trans)
        
        return pixels, mask, bboxes_trans


class VehicleSegmentationDataset(VehicleDatasetBase):
    def __init__(self, data_dir, labels_dir, feature_extractor, ignore=True):
        """
        Args:
            data_dir (str): Path to the folder containing input images.            
            transform (callable, optional): Transformation to apply to the input images.
            mask_transform (callable, optional): Transformation to apply to the masks.
        """
        super().__init__()
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


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        # Merge the previous and current frames into one image vertically for input into the network

        images, labels = get_image_sequences(idx, self.image_paths, self.labels_paths)

        pad_width = ((0, 0), (0, 0), (0, 1))        
        
        # merge 2nd and 1st images    
        # replace with the image difference       
        #diff_0 = images[1].mean(axis=2) - images[0].mean(axis=2)
        #diff_1 = images[2].mean(axis=2) - images[1].mean(axis=2)        
        #diff_image = np.stack([images[1].mean(axis=2), diff_0, diff_1]).transpose(1, 2, 0)
        
        image = np.concatenate([images[0], images[1]], axis=0)        
        pixels = feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # replace top
        height_half = pixels.shape[1] // 2
        #upper_half = pixels[:,:height_half,:].numpy().astype(np.uint8).transpose(1, 2, 0)
        #lower_half = pixels[:,height_half:,:].numpy().astype(np.uint8).transpose(1, 2, 0)
        #flow = torch.Tensor(np.pad(calculate_optical_flow(upper_half, lower_half), pad_width=pad_width, constant_values=0)).float().permute(2, 0, 1)
        #pixels[:,:height_half,:] = flow
        pixels[0,:height_half,:] = pixels[:,height_half:,:].mean(dim=0) - pixels[:,:height_half,:].mean(dim=0)
        pixels[1,:height_half,:] = pixels[:,:height_half,:].mean(dim=0)
        
        # Load the label image
        pixels_label = np.concatenate([labels[2], labels[1]], axis=0)    
        pixels_label = self.mask_transform(Image.fromarray(pixels_label))
        
        # only use the alpha channel to determine the mask
        mask = pixels_label[3,:,:]
        
        #print("Non zero mask values: ", (mask > 0.0).sum())
        #print("Zero mask values: ", (mask == 0).sum())

        return {"pixel_values": pixels, "labels": mask}

class VehicleSegmentationAugmentedDataset(VehicleDatasetBase):

    def __init__(self, data_dir, labels_dir, feature_extractor, ignore=True, num_items=100):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.data_dir = data_dir 
        self.labels_dir = labels_dir
        self.bg_image_paths = sorted(glob.glob(os.path.join(labels_dir, "*_background.jpg")))        
        self.fg_image_paths = sorted(glob.glob(os.path.join(labels_dir, "*_foreground.png")))        
        self.labels_paths = sorted(glob.glob(os.path.join(labels_dir, "*_foreground_label.png")))  
        #print(self.labels_df)   

        frames_to_ignore = pd.read_csv(labels_dir + "/ignore.csv", header=None)
        self.ignored_frames = set((frames_to_ignore.to_numpy() - 1).flatten())

        # filter out ignored frames
        if ignore:
            self.fg_image_paths = list(map(lambda x: x[1], filter(lambda x: x[0] not in self.ignored_frames, enumerate(self.fg_image_paths))))
            self.bg_image_paths = list(map(lambda x: x[1], filter(lambda x: x[0] not in self.ignored_frames, enumerate(self.bg_image_paths))))
            self.labels_paths = list(map(lambda x: x[1], filter(lambda x: x[0] not in self.ignored_frames, enumerate(self.labels_paths))))
        
        self.num_items = num_items
        self.sequence_range = [16, 16]        

        self.labels_df = pd.read_csv(f"{data_dir}/groundtruth.txt", header=None)
        self.boxes = [(row[3], row[2], row[7], row[6]) for idx, row in self.labels_df.iterrows()]
    
    def __len__(self):
        return self.num_items
    
    def __getitem__(self, idx):

        fg_trans = get_fixed_fg_params()
        bg_trans = get_fixed_bg_params()

        selected_bg, selected_fg, start_positions, theta = generate_random_sequence(self.fg_image_paths, self.bg_image_paths, self.boxes, length=3, position_range=[[0, 1.0], [0, 0.25]])
        images, image_labels, bboxes = merge_background_and_foreground_sequences(selected_bg, selected_fg, start_positions, theta, fg_trans, bg_trans)
        pixels, mask, bboxes_trans = self._prepare_output_data(images, image_labels, bboxes)
        
        return {"pixel_values": pixels, "labels": mask, "boxes": bboxes_trans }

class VehicleSegmentationAugmentedEvaluationDataset(VehicleSegmentationAugmentedDataset):

    def __init__(self, validation_dir, augmentation_data_dir, feature_extractor, num_vehicles=1):
        super().__init__(validation_dir, augmentation_data_dir, feature_extractor, ignore=False)
        self.bg_image_paths = sorted(glob.glob(os.path.join(validation_dir, "*.jpg")))        
        self.num_vehicles = num_vehicles        
        self.vehicle_data = [{
           "sequence_data": generate_random_sequence(
                self.fg_image_paths, 
                self.bg_image_paths,
                self.boxes,
                length=len(self.bg_image_paths),
                position_range=[[0, 1.0], [0, 0.75]],
                velocity_range=[0.0,0.1],
                velocity_noise=[0.005,0.01],
                direction_range=[45, 120]                              
            ),
            "fg_trans": get_fixed_fg_params()
        } for i in range(1, num_vehicles)]         
    

    def __len__(self):
        return len(self.bg_image_paths)

    def __getitem__(self, idx):

        max_bg = len(self.bg_image_paths)

        images, image_labels = get_image_sequences(idx, self.bg_image_paths, self.labels_paths)
                
        row = self.labels_df.iloc[idx]
        label_bboxes = [[(row[3], row[2], row[7], row[6])]]
        
        if self.num_vehicles > 1:

            for data in self.vehicle_data:
                fg_trans_params = data["fg_trans"]
                seq_data = data["sequence_data"]

                # get the slices of data for this portion of the sequence, but we use the fixed background images that were already generated
                selected_bg, selected_fg, positions, theta = seq_data
                selected_bg, selected_fg_partial, positions_partial = images, selected_fg[max(0, idx-1):min(max_bg, idx + 2)], positions[max(0, idx-1):min(max_bg, idx + 2)]

                # pad the sequence when we are at the ends
                if idx == 0:
                    selected_fg_partial.insert(0, selected_fg[0])
                    positions_partial.insert(0, positions[0])
                elif idx == max_bg - 1:
                    selected_fg_partial.append(selected_fg[-1])
                    positions_partial.append(positions[-1])

                # generate new vehicles into the scene, using the preloaded images with the original vehicle as backgrounds
                images, image_labels, bboxes = merge_background_and_foreground_sequences(selected_bg, selected_fg_partial, positions_partial, theta, fg_trans_params=fg_trans_params, bg_trans_params=get_fixed_bg_params_none())

                # append the boxes to the list that contains the boxes for each vehicle (current frame only)
                label_bboxes.append([bboxes[1]])
                
        all_vehicle_bboxes = np.array(label_bboxes)

        pixels, mask, trans_bboxes = self._prepare_output_data(images, image_labels, all_vehicle_bboxes.reshape(-1, 4), shift_labels=False )
        
        return {"pixel_values": pixels, "labels": mask, "boxes": trans_bboxes.reshape(self.num_vehicles, -1, 4) }


class VehicleSegmentationEvaluationDataset(VehicleDatasetBase):

    def __init__(self, file_or_dir):
        """
        Args:
            file_or_dir (str): input directory (to load directory with images) or a path to a video file
        """
        super().__init__()
        
        self.image_paths = []
        self.video_loader = None
        self.labels_df = None
        if os.path.isdir(file_or_dir):
            self.image_paths = sorted(glob.glob(f"{file_or_dir}/*.jpg"))
        else:
            self.video_loader = cv2.VideoCapture(file_or_dir, cv2.CAP_FFMPEG)
            if not self.video_loader.isOpened():
                raise Exception("could not open video file")
        if os.path.exists(f"{file_or_dir}/groundtruth.txt"):
            self.labels_df = pd.read_csv(f"{file_or_dir}/groundtruth.txt", header=None)
    
    def __len__(self):
        return int(self.video_loader.get(cv2.CAP_PROP_FRAME_COUNT)) if self.video_loader else len(self.image_paths)

    def __getitem__(self, idx):

        frame_count = self.__len__()

        label_bboxes = None
        if self.labels_df is not None:
            row = self.labels_df.iloc[idx]
            label_bboxes = np.array([[(row[3], row[2], row[7], row[6])]])

        # get frames
        frames = []
        for i in range(idx-1, idx+2):
            if i >= 0 and i < frame_count:
                if self.video_loader:
                    self.video_loader.set(cv2.CAP_PROP_POS_FRAMES, i)
                    im = self.video_loader.read()[1]                    
                else:
                    im = cv2.imread(self.image_paths[idx])
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                frames.append(im)
            else:
                frames.append(None)

        frames[0] = frames[1].copy()
        frames[-1] = frames[-2].copy()

        pixels, _, bboxes_trans = self._prepare_output_data(frames, None, label_bboxes, shift_labels=False )

        return {"pixel_values": pixels, "boxes": bboxes_trans.reshape(1, -1, 4) }


# Paths to your data directories
validation_images_dir = "./data/"
labels_dir = "./data_augment/"

model_name = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)

# Create datasets
#train_dataset = VehicleSegmentationDataset(
#    data_dir=validation_images_dir,
#    labels_dir=labels_dir,    
#    feature_extractor=feature_extractor 
#)

train_dataset = VehicleSegmentationAugmentedDataset(
    data_dir=validation_images_dir,
    labels_dir=labels_dir,
    feature_extractor=feature_extractor,
    num_items=250
)

val_dataset = VehicleSegmentationDataset(
    data_dir=validation_images_dir,
    labels_dir=labels_dir,
    feature_extractor=feature_extractor
)

eval_dataset = VehicleSegmentationAugmentedEvaluationDataset(
    validation_dir=validation_images_dir,
    augmentation_data_dir=labels_dir,
    feature_extractor=feature_extractor,
    num_vehicles=1
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


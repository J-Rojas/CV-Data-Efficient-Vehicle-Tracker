
import torch
import numpy as np
import random
import cv2
import albumentations as A
import torchvision.transforms as T
import math
from .tools import size_from_bbox

def get_fixed_fg_params():
    return {
        "rotate": 0, 
        "scale": random.uniform(0.8, 1.2),
        "shear": (random.uniform(-10, 10), random.uniform(-10, 10)),
        "brightness": random.uniform(0.7, 1.3),  
        "contrast": random.uniform(0.7, 1.3),
        "saturation":random.uniform(0.7, 1.3),
        "hue": random.uniform(-0.1, 0.1)
    }

def get_fixed_bg_params():
    return {
        "rotate": random.uniform(-15, 15), 
        "scale": random.uniform(0.9, 1.1),       
        "shear": (random.uniform(-10, 10), random.uniform(-10, 10)),        
        "brightness": random.uniform(0.7, 1.3),
        "contrast": random.uniform(0.7, 1.3),
        "saturation": random.uniform(0.7, 1.3),
        "hue": random.uniform(-0.1, 0.1)
    }

def get_fixed_bg_params_none():
    return {
        "rotate": 0, 
        "scale": 1.0,       
        "shear": 0,        
        "brightness": 1.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "hue": 0
    }


def apply_deterministic_fg_transform(image, params):
    transform = A.Compose([
        A.Affine(
            rotate=params["rotate"],
            scale=params["scale"],
            shear=params["shear"],
            translate_percent=0,
            p=1.0
        ),
        A.ColorJitter(
            brightness=(params["brightness"], params["brightness"]),
            contrast=(params["contrast"], params["contrast"]),
            saturation=(params["saturation"], params["saturation"]),
            hue=(params["hue"], params["hue"]),
            p=1.0
        )
    ])
    result = transform(image=image)
    return result["image"]

def apply_deterministic_mask_transform(mask, params):
    transform = A.Compose([
        A.Affine(
            rotate=params["rotate"],
            scale=params["scale"],
            shear=params["shear"],
            translate_percent=0,
            interpolation=cv2.INTER_NEAREST,
            p=1.0
        )
    ])
    result = transform(image=mask)
    return result["image"]

def apply_deterministic_bg_transform(image, params):
    transform = A.Compose([
        A.ColorJitter(
            brightness=(params["brightness"], params["brightness"]),
            contrast=(params["contrast"], params["contrast"]),
            saturation=(params["saturation"], params["saturation"]),
            hue=(params["hue"], params["hue"]),
            p=1.0
        )
    ])
    result = transform(image=image)
    return result["image"]


def rotate_foreground_with_padding(fg, angle):
    h, w = fg.shape[:2]
    center = (w // 2, h // 2)

    # Compute the new bounding box size after rotation
    abs_cos = abs(np.cos(np.radians(angle)))
    abs_sin = abs(np.sin(np.radians(angle)))
    new_w = int(np.ceil(w * abs_cos + h * abs_sin))  # Ensure rounding up
    new_h = int(np.ceil(h * abs_cos + w * abs_sin))

    # Ensure the new dimensions are at least the same as the original
    new_w = max(new_w, w)
    new_h = max(new_h, h)

    # Create a fully transparent canvas large enough to fit rotated content
    expanded_fg = np.zeros((new_h, new_w, 4), dtype=np.uint8)

    # Compute new offsets to center the original image inside the larger canvas
    x_offset = (new_w - w) // 2
    y_offset = (new_h - h) // 2

    # Place the original image at the center of the larger canvas
    expanded_fg[y_offset:y_offset + h, x_offset:x_offset + w] = fg

    # Compute the new center for rotation
    new_center = (new_w // 2, new_h // 2)

    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(new_center, angle, 1.0)

    # Rotate the entire padded foreground
    rotated_fg = cv2.warpAffine(expanded_fg, rotation_matrix, (new_w, new_h), 
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return rotated_fg

def generate_random_sequence(files_fg, files_bg, bboxes,
                             start=[[0, 237], [0, 237]], 
                             length=3, 
                             scale_size = [272, 640],
                             direction_range=[-40, 40], 
                             position_range=[[0.0, 1.0], [0.0, 1.0]], 
                             velocity_range=[0.1, 0.5],  
                             velocity_noise=[-0.05, 0.05],
                             scale_range=[0.75, 1.25]):
    
    start[0][0] = min(start[0][0], len(files_fg) - length)
    start[0][1] = min(start[0][1], len(files_fg) - length + 1)
    start[1][0] = min(start[1][0], len(files_bg) - length)
    start[1][1] = min(start[1][1], len(files_bg) - length + 1)
    
    # Select a sequential set of images ensuring continuity in selection
    img_ranges = torch.tensor([torch.randint(*start[0], (1,)), torch.randint(*start[1], (1,))])
    
    img_ranges = torch.vstack((img_ranges, img_ranges + length)).squeeze().T

    # adjust ranges
    # Select images within the dataset limits
    selected_fg = [files_fg[i] for i in range(*torch.clip(img_ranges[0], 0, len(files_fg)))]
    selected_bg = [files_bg[i] for i in range(*torch.clip(img_ranges[1], 0, len(files_bg)))]
    
    # randomize the scale
    scale = np.array([
        random.uniform(scale_range[0], scale_range[1]),
        random.uniform(scale_range[0], scale_range[1])
    ])
    fg_sizes = [scale * size_from_bbox(bboxes[i]) for i in range(*torch.clip(img_ranges[0], 0, len(files_fg)))]
    
    # Initial placement (normalized positions)
    start_position = np.array([
        random.uniform(position_range[0][0], position_range[0][1]),
        random.uniform(position_range[1][0], position_range[1][1])
    ])

    # Random movement direction
    theta = random.uniform(direction_range[0], direction_range[1])
    theta_rad = np.radians(theta)
    direction_vector = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    # constant velocity 
    velocity = random.uniform(velocity_range[0], velocity_range[1])

    # Update the foreground's normalized position.
    start_positions = []
    for i in range(len(selected_fg)):
        velocity_magnitude = velocity * fg_sizes[i][1] / scale_size
        velocity_vector = direction_vector * velocity_magnitude
        velocity_sample = velocity_vector + direction_vector * np.random.uniform(velocity_noise[0], velocity_noise[1], size=2) * fg_sizes[i][1] / scale_size
        start_position += velocity_sample
        #start_position = np.clip(start_position, 0.0, 1.0)
        start_positions.append(start_position.copy())

    return selected_bg, selected_fg, start_positions, theta


def merge_background_and_foreground_sequences(selected_bg, selected_fg, start_positions, theta, fg_trans_params=get_fixed_fg_params(), bg_trans_params=get_fixed_bg_params()):
    
    sequence = []
    sequence_labels = []
    sequence_bbox = []
    
    for i, (fg_path, bg_path) in enumerate(zip(selected_fg, selected_bg)):
        if type(fg_path) == str:
            fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        else:
            fg = fg_path # image pixels
        if type(bg_path) == str:
            bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        else:
            bg = bg_path # image pixels
        if fg is None or bg is None:
            continue

        # Apply foreground rotation
        fg_rgb = rotate_foreground_with_padding(fg, theta)

        # Separate foreground alpha if available.
        if fg.shape[-1] == 4:
            fg_rgb = fg[:, :, :3]
            orig_alpha = fg[:, :, 3]
        else:
            fg_rgb = fg
            orig_alpha = np.ones(fg_rgb.shape[:2], dtype=np.uint8) * 255

        # Apply the fixed foreground transformation.
        fg_trans = apply_deterministic_fg_transform(fg_rgb, fg_trans_params)
        # Apply the fixed transformation to the alpha mask.
        alpha_trans = apply_deterministic_mask_transform(orig_alpha, fg_trans_params)
        # Apply the fixed background transformation.
        bg_trans = apply_deterministic_bg_transform(bg, bg_trans_params)
        
        # Normalize the transformed alpha to [0,1] and replicate to 3 channels.
        alpha_norm = alpha_trans.astype(np.float32) / 255.0
        alpha_norm_3c = np.repeat(alpha_norm[:, :, np.newaxis], 3, axis=2)

        # Compute placement of the foreground on the background.
        pos_y = int(start_positions[i][0] * bg_trans.shape[0])
        pos_x = int(start_positions[i][1] * bg_trans.shape[1])
        x_end = min(pos_x + fg_trans.shape[1], bg_trans.shape[1])
        y_end = min(pos_y + fg_trans.shape[0], bg_trans.shape[0])

        clipped_h, clipped_w = max(0, y_end - max(0, pos_y)), max(0, x_end - max(0, pos_x))

        blend_fg = fg_trans[:clipped_h, :clipped_w]
        alpha_norm = alpha_norm[:clipped_h, :clipped_w]
        blend_alpha = alpha_norm_3c[:clipped_h, :clipped_w]
        bg_crop = bg_trans[max(0, pos_y):max(0, max(0,pos_y) +blend_fg.shape[0]), max(0, pos_x):max(0, max(0, pos_x)+blend_fg.shape[1])]

        blended = (blend_fg.astype(np.float32) * blend_alpha +
                   bg_crop.astype(np.float32) * (1 - blend_alpha)).astype(np.uint8)

        bg_trans[max(0, pos_y):max(0, y_end), max(0, pos_x):max(0, x_end)] = blended

        sequence.append(bg_trans)

        fg_label = np.zeros((bg_trans.shape[0], bg_trans.shape[1], 4), dtype=np.uint8)
        fg_label[max(0, pos_y):max(0, y_end), max(0, pos_x):max(0, x_end)] = np.concatenate([blend_fg, alpha_norm[:, :, np.newaxis] * 255], axis=2)
        sequence_labels.append(fg_label)
                
        sequence_bbox.append(np.array([pos_y, pos_x, y_end, x_end]))

    return sequence, sequence_labels, sequence_bbox


if __name__ == '__main__':
    
    import glob
    import pathlib
    import pandas as pd
    import cv2

    foreground = sorted(glob.glob("data_augment/*_foreground.png"))
    background = sorted(glob.glob("data_augment/*_background.jpg"))

    labels_boxes = pd.read_csv("data/groundtruth.txt", header=None)
    ignore = pd.read_csv("data_augment/ignore.csv", header=None)        
    ignore = set((ignore.to_numpy() - 1).flatten())

    # remove the ignored values from the list
    boxes = [(row[3], row[2], row[7], row[6]) for idx, row in labels_boxes.iterrows()]
    foreground = list(map(lambda x: x[1], filter(lambda x: x[0] not in ignore, enumerate(foreground))))
    background = list(map(lambda x: x[1], filter(lambda x: x[0] not in ignore, enumerate(background))))
    
    fg_trans = get_fixed_fg_params()
    bg_trans = get_fixed_bg_params()
    selected_bg, selected_fg, start_positions, theta = generate_random_sequence(foreground, background, boxes, position_range=[0, 0.5], length=len(foreground))
    merged, labels, _ = merge_background_and_foreground_sequences(selected_bg, selected_fg, start_positions, theta, fg_trans, bg_trans)

    for idx, (im, label) in enumerate(zip(merged, labels)):
        cv2.imwrite(f"data_sequence/{idx}.jpg", im)
        cv2.imwrite(f"data_sequence/{idx}_label.png", label)

    

import torch
import numpy as np
import random
import cv2
import albumentations as A
import torchvision.transforms as T

import numpy as np
import math
import cv2

import torch
import numpy as np
import random
import cv2
import math


def get_fixed_fg_params():
    return {
        "rotate": 0, 
        "scale": random.uniform(0.5, 1.5),
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

def generate_random_sequence(files_fg, files_bg, 
                             start=[[0, 237], [0, 237]], 
                             length=[[10, 15], [10, 15]], 
                             direction_range=[-40, 40], 
                             position_range=[0.0, 1.0], 
                             velocity_range=[0.1, 0.5],  
                             velocity_noise=[0.01, 0.05]):
    length = 3
    start[0][0] = min(start[0][0], len(files_fg) - length)
    start[0][1] = min(start[0][1], len(files_fg) - length)
    start[1][0] = min(start[1][0], len(files_bg) - length)
    start[1][1] = min(start[1][1], len(files_bg) - length)
    
    # Select a sequential set of images ensuring continuity in selection
    img_ranges = torch.tensor([torch.randint(*start[0], (1,)), torch.randint(*start[1], (1,))])
    
    img_ranges = torch.vstack((img_ranges, img_ranges + length)).squeeze().T

    # adjust ranges

    # Select images within the dataset limits
    selected_fg = [files_fg[i] for i in range(*torch.clip(img_ranges[0], 0, len(files_fg)-1))]
    selected_bg = [files_bg[i] for i in range(*torch.clip(img_ranges[1], 0, len(files_bg)-1))]
    
    # Random movement direction
    theta = random.uniform(direction_range[0], direction_range[1])
    theta_rad = np.radians(theta)
    direction_vector = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    # Initial placement (normalized positions)
    start_position = np.array([
        random.uniform(position_range[0], position_range[1]),
        random.uniform(position_range[0], position_range[1])
    ])

    # Lock in fixed parameters for each stream.
    fg_params = get_fixed_fg_params()
    bg_params = get_fixed_bg_params()

    sequence = []
    sequence_labels = []
    
    for fg_path, bg_path in zip(selected_fg, selected_bg):
        fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
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
        fg_trans = apply_deterministic_fg_transform(fg_rgb, fg_params)
        # Apply the fixed transformation to the alpha mask.
        alpha_trans = apply_deterministic_mask_transform(orig_alpha, fg_params)
        # Apply the fixed background transformation.
        bg_trans = apply_deterministic_bg_transform(bg, bg_params)
        
        # Normalize the transformed alpha to [0,1] and replicate to 3 channels.
        alpha_norm = alpha_trans.astype(np.float32) / 255.0
        alpha_norm_3c = np.repeat(alpha_norm[:, :, np.newaxis], 3, axis=2)

        # Compute placement of the foreground on the background.
        pos_x = int(start_position[0] * (bg_trans.shape[1] - fg_trans.shape[1]))
        pos_y = int(start_position[1] * (bg_trans.shape[0] - fg_trans.shape[0]))
        x_end = min(pos_x + fg_trans.shape[1], bg_trans.shape[1])
        y_end = min(pos_y + fg_trans.shape[0], bg_trans.shape[0])

        blend_fg = fg_trans[:y_end - pos_y, :x_end - pos_x]
        blend_alpha = alpha_norm_3c[:y_end - pos_y, :x_end - pos_x]
        bg_crop = bg_trans[pos_y:y_end, pos_x:x_end]

        blended = (blend_fg.astype(np.float32) * blend_alpha +
                   bg_crop.astype(np.float32) * (1 - blend_alpha)).astype(np.uint8)
        bg_trans[pos_y:y_end, pos_x:x_end] = blended

        sequence.append(bg_trans)

        fg_label = np.zeros((bg_trans.shape[0], bg_trans.shape[1], 4))
        fg_label[pos_y:y_end, pos_x:x_end] = np.concatenate([blend_fg, alpha_norm[:, :, np.newaxis]], axis=2)
        sequence_labels.append(fg_label)

        # Update the foreground's normalized position.
        velocity_magnitude = random.uniform(velocity_range[0], velocity_range[1]) * fg_trans.shape[1]
        velocity = direction_vector * velocity_magnitude
        velocity += np.random.uniform(velocity_noise[0], velocity_noise[1], size=2) * fg_trans.shape[1]
        start_position += velocity / np.array([bg_trans.shape[1], bg_trans.shape[0]])
        start_position = np.clip(start_position, 0.0, 1.0)

    return sequence, sequence_labels


if __name__ == '__main__':
    
    import glob
    import pathlib
    import pandas as pd
    import cv2

    foreground = sorted(glob.glob("data_augment/*_foreground.png"))
    background = sorted(glob.glob("data_augment/*_background.jpg"))

    ignore = pd.read_csv("data_augment/ignore.csv", header=None)        
    ignore = set((ignore.to_numpy() - 1).flatten())

    # remove the ignored values from the list
    foreground = list(map(lambda x: x[1], filter(lambda x: x[0] not in ignore, enumerate(foreground))))
    background = list(map(lambda x: x[1], filter(lambda x: x[0] not in ignore, enumerate(background))))
    
    merged, labels = generate_random_sequence(foreground, background, position_range=[0, 0.5])

    for idx, im in enumerate(merged):
        cv2.imwrite(f"data_sequence/{idx}.jpg", im)

    
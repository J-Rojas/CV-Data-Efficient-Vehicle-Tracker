import cv2
import torch
import numpy as np
from transformers import SamModel, SamProcessor
from .tools import extract_masked_pixels, inpaint_pixels_with_mask, extend_image_with_edge, crop_to_nonzero, save_with_transparency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def custom_inpaint(im, bbox):
    # take the mean of the pixels to the right and the left of the bbox to inpaint
    x1, y1, x2, y2 = bbox
    height, width = im.shape[:2]
    b_width = int(max(abs(x1 - x2) * 1.1, 40))
    b_height = int(abs(y1 - y2))
    
    right_crop = im[y1:y2, min(x1 + b_width, width):min(x2 + b_width, width), :]
    right_crop = extend_image_with_edge(right_crop, (b_height, b_width, 3), side="start")

    left_crop = im[y1:y2, max(x1 - b_width, 0):max(x2 - b_width, 0), :]
    left_crop = extend_image_with_edge(left_crop, (b_height, b_width, 3), side="end")

    new_im = im.copy()
    averaged = ((right_crop.astype(np.float32) + left_crop.astype(np.float32)) / 2.0)
    new_im[y1:y2, x1:x2] = np.clip(averaged[:abs(y1-y2), :abs(x1-x2), :], 0, 255).astype(np.uint8)

    return new_im
    
def generate_segment_mask(im, bbox):
    
    # Load the SAM model (adjust model_type and checkpoint path as needed)
    model_type = "vit_b"  # or another variant
    checkpoint = "path/to/sam_vit_b.pth"
    input_points = [[[bbox[0], bbox[1]], [bbox[2], bbox[3]]]]

    inputs = sam_processor(im, input_points=input_points, return_tensors="pt")

    # Move each tensor in the input dictionary to the CUDA device.
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = sam_model(**inputs)
    masks = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores

    assert(len(masks) == 1)

    return masks[0].squeeze().transpose(0, 2).transpose(0, 1).numpy().astype(np.uint8)

def extract_vehicle_from_frame(im, bbox, border_perc=0.8):
    
    height, width = im.shape[:2]

    x1, y1, x2, y2 = bbox
    bbox_orig = bbox

    # grow the box by the border_perc factor
    mask_w_ = abs(x2 - x1)
    mask_h_ = abs(y2 - y1)
    mask_w = mask_w_ * border_perc
    mask_h = mask_h_ * border_perc

    x1 -= (mask_w - mask_w_) // 2
    y1 -= (mask_h - mask_h_) // 2
    x2 += (mask_w - mask_w_) // 2
    y2 += (mask_h - mask_h_) // 2
    
    bbox = (int(x1),int(y1), int(x2), int(y2))

    mask = generate_segment_mask(im, bbox)
    
    print(mask.shape, im.shape)
    print("Mask is uint8", mask.dtype == np.uint8)
    if mask.ndim == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # clip out the masked region
    foreground_full = extract_masked_pixels(im, mask)
    
    # reduce to only non-zero pixels
    foreground = crop_to_nonzero(foreground_full)

    # inpaint the masked region
    background = custom_inpaint(im, bbox_orig)

    return foreground, foreground_full, background

if __name__ == '__main__':
    
    import glob
    import pathlib
    import pandas as pd
    labels = pd.read_csv("./data/groundtruth.txt", header=None)
    for img_file, (idx, row) in zip(glob.glob("./data/*.jpg"), labels.iterrows()):
        im = cv2.imread(img_file)
        bbox = (int(row[2]), int(row[3]), int(row[6]), int(row[7]))

        foreground, foreground_full, background = extract_vehicle_from_frame(im, bbox)

        basename = pathlib.Path(img_file).name.replace(".jpg", "")

        save_with_transparency(foreground, f"./data_augment/{basename}_foreground.png")
        save_with_transparency(foreground_full, f"./data_augment/{basename}_foreground_label.png")
        cv2.imwrite(f"./data_augment/{basename}_background.jpg", background)

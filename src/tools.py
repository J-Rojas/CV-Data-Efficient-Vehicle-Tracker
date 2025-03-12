import cv2
import numpy as np

def get_bounding_box_from_mask(mask) -> tuple | None:
    """
    Given a binary segmentation mask (numpy array with shape (H, W))
    where vehicle pixels are 1 and background is 0, this function computes
    the smallest axis-aligned bounding box around the detected vehicle region.
    
    Returns:
        A tuple (x, y, w, h) representing the bounding box, or None if no contour is found.
    """
    # Convert mask to uint8 (if not already) and scale to 0-255
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours of the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None  # No region detected
    
    # Optionally, select the largest contour if there are multiple
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Compute the bounding rectangle (x, y, width, height)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, x + w, y + h

def create_fuzzy_label_rect(image_shape, bbox, factor=0.1):
    """
    Create a fuzzy label for a rectangular region defined by a bounding box.
    
    Args:
        image_shape (tuple): Shape of the full image mask (height, width).
        bbox (tuple): Bounding box in the form (x1, y1, x2, y2).
        factor (float): Factor to scale the kernel size and sigma relative to the bbox dimensions.
    
    Returns:
        numpy.ndarray: A fuzzy mask with values between 0 and 1, where only the bbox region is blurred.
    """
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    
    # Create a blank binary mask of the full image
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    # Fill the rectangle
    cv2.rectangle(binary_mask, (x1, y1), (x2, y2), 255)
    
    # Crop the rectangular region from the binary mask
    crop = binary_mask[y1:y2, x1:x2].astype(np.float32)
    
    # Compute kernel sizes based on the dimensions of the bbox
    bbox_height = crop.shape[0]
    bbox_width = crop.shape[1]
    
    kernel_height = max(3, int(bbox_height * factor))
    kernel_width = max(3, int(bbox_width * factor))
    # Ensure kernel sizes are odd
    if kernel_height % 2 == 0:
        kernel_height += 1
    if kernel_width % 2 == 0:
        kernel_width += 1
        
    # Compute sigma values proportionally
    sigma_y = bbox_height * factor
    sigma_x = bbox_width * factor
    
    # Apply Gaussian blur to the cropped region
    blurred_crop = cv2.GaussianBlur(crop, (kernel_width, kernel_height), sigmaX=sigma_x, sigmaY=sigma_y)
    # Normalize the blurred crop to the range [0, 1]
    blurred_crop = blurred_crop / (blurred_crop.max() + 1e-6)
    
    # Create an output mask and place the blurred crop back into its original location
    fuzzy_mask = np.zeros((height, width), dtype=np.float32)
    fuzzy_mask[y1:y2, x1:x2] = blurred_crop
    
    return fuzzy_mask

def torch_to_cv2_image(tensor, denormalize=True):
    """
    Converts a PyTorch image tensor to a format suitable for OpenCV.
    
    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W) with values typically in [0, 1].
        denormalize (bool): If True, reverse the ImageNet normalization.
    
    Returns:
        numpy.ndarray: An image in BGR format with dtype uint8.
    """
    # Ensure tensor is on CPU and detach from graph
    img = tensor.cpu().detach()
    
    # Convert from tensor (C, H, W) to numpy array (H, W, C)
    img = img.numpy().transpose(1, 2, 0)
    
    if denormalize:
        # Assuming ImageNet normalization was applied: mean and std per channel
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean  # Reverse normalization
    
    # Clip to [0, 1] and scale to 0-255
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    # Convert from RGB (common in PyTorch) to BGR (used by OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

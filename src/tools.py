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

def create_label_rect(image_shape, bbox):
    """
    Create a label for a rectangular region defined by a bounding box.
    
    Args:
        image_shape (tuple): Shape of the full image mask (height, width).
        bbox (tuple): Bounding box in the form (x1, y1, x2, y2).
            
    Returns:
        numpy.ndarray: A mask with values between 0 and 255, where only the bbox region is blurred.
    """
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    
    # Create a blank binary mask of the full image
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    # Fill the rectangle
    cv2.rectangle(binary_mask, (x1, y1), (x2, y2), 1)
    
    return binary_mask.astype(np.float32)

def create_fuzzy_label_rect(image_shape, bbox, factor=0.1, sigma_factor=0.3):
    """
    Create a fuzzy elliptical mask using an analytic 2D Gaussian function.
    The Gaussian is computed over the bbox such that the peak is at the center,
    and the values decay outward.

    Args:
        image_shape (tuple): Shape of the full image mask (height, width).
        bbox (tuple): Bounding box in the form (x1, y1, x2, y2).
        sigma_factor (float): Fraction of the bbox dimensions to use as sigma.

    Returns:
        numpy.ndarray: A fuzzy mask with values in [0, 1] where the Gaussian 
                       is computed within the bbox and zeros elsewhere.
    """
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Coordinates for the bbox region
    xs = np.arange(bbox_width)
    ys = np.arange(bbox_height)
    xv, yv = np.meshgrid(xs, ys)

    # Center of the bbox (in local coordinates)
    center_x = bbox_width / 2.0
    center_y = bbox_height / 2.0

    # Standard deviations in x and y (controls the spread of the Gaussian)
    sigma_x = bbox_width * sigma_factor
    sigma_y = bbox_height * sigma_factor

    # Compute the 2D Gaussian function:
    # G(x,y) = exp( -[(x - center_x)^2 / (2*sigma_x^2) + (y - center_y)^2 / (2*sigma_y^2)] )
    gaussian = np.exp(-(((xv - center_x) ** 2) / (2 * sigma_x ** 2) +
                        ((yv - center_y) ** 2) / (2 * sigma_y ** 2)))
    gaussian = gaussian.astype(np.float32)
    
    # Optionally, you can normalize the gaussian to ensure the peak is 1:
    gaussian /= gaussian.max() + 1e-6

    # Create a full image mask and place the gaussian in the bbox region
    full_mask = np.zeros((height, width), dtype=np.float32)
    full_mask[y1:y2, x1:x2] = gaussian

    return full_mask

def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlay a mask on an image, where the mask values are in the range [0, 1].
    The overlay is computed per pixel by multiplying the mask by alpha.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        mask (numpy.ndarray): Mask with values in [0, 1] (or in [0,255], which will be normalized).
        color (tuple): BGR color for the overlay.
        alpha (float): Maximum transparency factor for the overlay.
        
    Returns:
        numpy.ndarray: Image with the overlay.
    """
    # Convert mask to float32 in the range [0,1]
    if mask.dtype == np.uint8:
        # If the mask is [0,255], convert it to [0,1]
        if mask.max() > 1:
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = mask.astype(np.float32)
    else:
        mask = mask.astype(np.float32)
    
    # Create a colored overlay image
    colored_overlay = np.full(image.shape, color, dtype=np.float32)
    
    # Convert the original image to float32
    image_float = image.astype(np.float32)
    
    # The overlay factor is (alpha * mask) which varies per pixel.
    # For each pixel, the final color is:
    #   blended = image * (1 - alpha*mask) + colored_overlay * (alpha*mask)
    # mask[..., np.newaxis] expands the mask to have the same number of channels as the image.
    blended = image_float * (1 - alpha * mask[..., np.newaxis]) + colored_overlay * (alpha * mask[..., np.newaxis])
    
    # Clip the output to [0,255] and convert back to uint8
    output = np.clip(blended, 0, 255).astype(np.uint8)
    return output

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

import cv2
import numpy as np
import math
from numpy.lib.stride_tricks import sliding_window_view

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

def extract_masked_pixels(im, mask):
    return cv2.bitwise_and(im, im, mask=mask)

def inpaint_pixels_with_mask(im, mask, radius=5):
    return cv2.inpaint(im, mask, radius, cv2.INPAINT_NS)


def tile_to_size(matrix, target_shape):
    """
    Tiles the input matrix until the resulting matrix is at least as large as the target shape,
    and then crops it to exactly the target shape.
    
    Args:
        matrix (np.ndarray): The input array (2D or 3D).
        target_shape (tuple): The desired shape, e.g. (M, N) for a 2D array or (M, N, C) for a 3D array.
        
    Returns:
        np.ndarray: The tiled (and cropped) array of shape target_shape.
    """
    # Get current shape (assuming at least 2 dimensions)
    m, n = matrix.shape[:2]
    target_m, target_n = target_shape[:2]
    
    # Compute number of repetitions needed along each dimension
    rep_m = math.ceil(target_m / m)
    rep_n = math.ceil(target_n / n)
    
    # Tile the matrix. If matrix has more than 2 dimensions, tile only along the first two.
    if matrix.ndim == 2:
        tiled = np.tile(matrix, (rep_m, rep_n))
    else:
        # For a 3D array, tile along height and width only
        tiled = np.tile(matrix, (rep_m, rep_n, 1))
    
    # Crop the tiled array to the target size
    if matrix.ndim == 2:
        return tiled[:target_m, :target_n]
    else:
        return tiled[:target_m, :target_n, :]

def extend_image_with_edge(image, target_shape, side="end"):
    """
    Extend a 3D image (height, width, channels) to the target shape by
    filling extra space with the edge values from either the start or end.
    
    Args:
        image (np.ndarray): Input image with shape (H, W, C).
        target_shape (tuple): Desired shape (target_H, target_W, target_C).
                              target_C must equal the number of channels in the image.
        side (str): "end" pads at the bottom/right; "start" pads at the top/left.
        
    Returns:
        np.ndarray: Extended image with shape target_shape.
    """
    H, W, C = image.shape
    target_H, target_W, target_C = target_shape
    if target_C != C:
        raise ValueError("The target channel count must match the input image channels.")
    
    pad_H = max(0, target_H - H)
    pad_W = max(0, target_W - W)
    
    if side == "end":
        pad_top, pad_bottom = 0, pad_H
        pad_left, pad_right = 0, pad_W
    elif side == "start":
        pad_top, pad_bottom = pad_H, 0
        pad_left, pad_right = pad_W, 0
    else:
        raise ValueError("side must be either 'start' or 'end'")
    
    # Pad only the first two dimensions (height and width); no padding for channels.
    extended_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
    return extended_image

def crop_to_nonzero(image):
    """
    Crops the input image to the smallest rectangle that contains all non-zero pixels.
    
    Args:
        image (numpy.ndarray): Input image (grayscale or color).
    
    Returns:
        numpy.ndarray: Cropped image.
    """
    # If the image is color, convert to grayscale for the purpose of finding non-zero areas.
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Find coordinates of non-zero pixels.
    nonzero = cv2.findNonZero(gray)
    if nonzero is None:
        # If no non-zero pixels found, return the original image.
        return image

    # Get the bounding rectangle for these non-zero pixels.
    x, y, w, h = cv2.boundingRect(nonzero)

    # Crop the image to the bounding rectangle.
    cropped = image[y:y+h, x:x+w]
    return cropped

def save_with_transparency(image, output_path):
    """
    Converts a 3-channel BGR image to a 4-channel BGRA image where
    pixels that are completely zero are set to transparent and then saves the image as PNG.

    Args:
        image (np.ndarray): Input image in BGR format (H, W, 3) with dtype uint8.
        output_path (str): Path to save the PNG image.
    """
    # Create an alpha channel:
    # Where all three channels are 0, set alpha to 0 (transparent), otherwise 255 (opaque)
    alpha = np.where(np.all(image == 0, axis=2), 0, 255).astype(np.uint8)
    
    # Stack the alpha channel to create a BGRA image
    bgra = np.dstack((image, alpha))
    
    # Save the image as PNG
    cv2.imwrite(output_path, bgra)

def iou(box1, box2):
    # Determine the coordinates of the intersection rectangle
    x_left   = max(box1[0], box2[0])
    y_top    = max(box1[1], box2[1])
    x_right  = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # If there is no overlap, the intersection area is zero.
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Compute the area of the intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute the union area by using the formula: union = A + B - intersection
    union_area = box1_area + box2_area - intersection_area
    
    # Compute the IoU
    iou_value = intersection_area / union_area
    return iou_value

def slice_from_bbox(bbox):
    return slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])

def normalized_cross_correlation(patch1, patch2):
    # Ensure patches are floating point
    patch1 = patch1.astype(np.float32)
    patch2 = patch2.astype(np.float32)
    
    # Subtract mean
    patch1_mean = patch1 - np.mean(patch1)
    patch2_mean = patch2 - np.mean(patch2)
    
    # Compute numerator and denominator
    numerator = np.sum(patch1_mean * patch2_mean)
    denominator = np.sqrt(np.sum(patch1_mean**2) * np.sum(patch2_mean**2))
    
    if denominator == 0:
        return 0  # Avoid division by zero
    return numerator / denominator

def patch_matching_cross_correlation(image, template):
    
    # determine the larger image
    image, template = (image, template) if image.size > template.size else (template, image)

    H, W = image.shape
    h, w = template.shape

    # clip it down
    h, w = min(H, h), min(W, w)
    template = template[:h, :w]

    # Extract all windows of the size of template from the image
    windows = sliding_window_view(image, (h, w))  # shape: (H-h+1, W-w+1, h, w)

    # Compute means and standard deviations for normalization
    template_mean = template.mean()
    template_std = template.std()

    # Compute window means and standard deviations along the last two axes (the window dimensions)
    windows_mean = windows.mean(axis=(-1, -2))
    windows_std = windows.std(axis=(-1, -2))

    # Normalize windows and template
    norm_windows = (windows - windows_mean[..., None, None])
    norm_template = template - template_mean

    # Compute cross correlation (summing the product over window dimensions)
    numerator = np.sum(norm_windows * norm_template, axis=(-1, -2))
    denominator = windows_std * template_std * h * w

    # Avoid division by zero
    corr_map = np.where(denominator == 0, 0, numerator / denominator)
    return corr_map

def calculate_optical_flow(frame1, frame2):
    # Convert to grayscale, as optical flow methods usually work on single channel images
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow using Farneback method.
    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,     
        flow=None,   
        pyr_scale=0.5,      # scaling factor between pyramid layers
        levels=3,           # number of pyramid layers
        winsize=15,         # averaging window size; larger values can handle larger motion
        iterations=3,       # number of iterations at each pyramid level
        poly_n=5,           # size of pixel neighborhood for polynomial expansion
        poly_sigma=1.2,     # standard deviation of the Gaussian that is used to smooth derivatives
        flags=0
    )
    return flow

def rescale_bboxes(src_shape, target_shape, bboxes):
    # image scaling factor
    scale_y = float(target_shape[0]) / src_shape[0] # pixels is in C, H, W, image_input is in H, W, C
    scale_x = float(target_shape[1]) / src_shape[1]

    # recalculate bboxes after image rescaling
    scale_matrix = np.array([[scale_y, 0.0], [0.0, scale_x]])
    
    #print("scale", scale_matrix)
    #print("bboxes before", bboxes)
    bboxes_trans = np.array([np.matmul(scale_matrix, bbox.reshape(2, 2).T).T.reshape(4) for bbox in bboxes]).astype(int)
    
    return bboxes_trans
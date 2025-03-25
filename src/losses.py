# Copyright (c) 2025, Jose Rojas (https://github.com/J-Rojas)
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_distance_map(mask_np):
    """
    Compute the distance transform of the inverse of the binary mask.
    Args:
        mask_np (numpy.ndarray): 2D binary mask (values 0 or 1).
    Returns:
        distance_map (numpy.ndarray): 2D distance map.
    """
    # The distance transform is computed on the background (1 - mask)
    distance_map = distance_transform_edt(1 - mask_np)
    return distance_map

def boundary_aware_loss(pred_logits, target_mask, alpha=1.0, cap_value=0.5, use_bce=False):
    """
    Compute a boundary-aware loss by weighting the per-pixel CrossEntropy loss 
    with a distance map computed from the ground truth mask.
    
    Args:
        pred_logits (torch.Tensor): Predicted logits, shape (B, num_classes, H, W).
        target_mask (torch.Tensor): Ground truth mask, shape (B, H, W) with values {0,1}.
        
    Returns:
        torch.Tensor: Scalar loss.
    """
    # Standard per-pixel CrossEntropy loss (no reduction)
    if use_bce:
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits[:, 1, :, :], target_mask, reduction='none')        
    else:
        ce_loss = F.cross_entropy(pred_logits, target_mask, reduction='none')
    
    # For each sample in the batch, compute a distance map
    B, H, W = target_mask.shape
    weight_maps = []
    for i in range(B):
        # Convert to numpy (ensure type is uint8 for distance transform)
        mask_np = target_mask[i].cpu().numpy().astype(np.uint8)
        # Compute the distance map; note that lower values mean closer to the boundary.
        dist_map = compute_distance_map(mask_np)
        # Optionally, you could invert this map or compute a function of the distance to give higher weight near boundaries.
        # For this example, we'll use an inverse formulation: weight = 1 / (distance + epsilon)
        epsilon = 1e-6
        weight_np = alpha / (dist_map + epsilon)
        weight_np = np.clip(weight_np, 0, cap_value)
        # Normalize weight map so that its mean is 1
        weight_np = weight_np / (np.mean(weight_np) + epsilon)
        weight_tensor = torch.tensor(weight_np, dtype=pred_logits.dtype, device=pred_logits.device)
        weight_maps.append(weight_tensor)
    
    weight_maps = torch.stack(weight_maps, dim=0)  # shape (B, H, W)
    
    # Multiply the CE loss with the weight map and take the mean
    weighted_loss = ce_loss * weight_maps
    return weighted_loss.mean()

def total_variation_loss(x):
        """
        Compute the Total Variation (TV) loss.
        Args:
            x (torch.Tensor): Tensor of shape (B, C, H, W).
        Returns:
            tv_loss (torch.Tensor): A scalar loss value.
        """
        # Compute absolute differences along height and width dimensions
        h_diff = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        w_diff = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return h_diff + w_diff

def consistency_loss(x, bbox1, bbox2):
    """
    This loss compares two different regions for similarity, the more dissimilar the higher the loss
    """
    
    total = 0
    for b, bb1, bb2 in zip(x, bbox1, bbox2):
        # ensure the boxes are same in sizes        
        region1 = b[bb1[0]:bb1[2], bb1[1]:bb1[3]]
        region2 = b[bb2[0]:bb2[2], bb2[1]:bb2[3]]
        region2 = region2[:region1.shape[0], :region1.shape[1]]
        region1 = region1[:region2.shape[0], :region2.shape[1]]
        total = total + ((region1 - region2) ** 2).mean()

    return total

def iou_consistency_loss(x, bbox1, bbox2):

    """
    This loss compares two different regions using the IoU loss, the more dissimilar the higher the loss
    """
    
    total = 0
    for b, bb1, bb2 in zip(x, bbox1, bbox2):
        # ensure the boxes are same in sizes        
        region1 = b[bb1[0]:bb1[2], bb1[1]:bb1[3]]
        region2 = b[bb2[0]:bb2[2], bb2[1]:bb2[3]]
        region2 = region2[:region1.shape[0], :region1.shape[1]]
        region1 = region1[:region2.shape[0], :region2.shape[1]]
        total += iou_loss(region1.unsqueeze(0), region2.unsqueeze(0))

    return total

# A simple focal loss implementation, deal with class imbalances due to pixels being biased towards background class
def focal_loss(logits, targets, alpha=0.8, gamma=2.0, use_bce=False):
    # logits: (B, num_classes, H, W)
    # targets: (B, H, W)
    if use_bce:
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits[:, 1, :, :], targets, reduction='none')
    else:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # probability of correct classification
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

def dice_loss(pred, target, smooth=1.0):
    # pred: probabilities after softmax of shape (B, num_classes, H, W)
    # target: one-hot encoded of shape (B, num_classes, H, W)
    intersection = (pred * target).sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)
    return 1 - dice.mean()

def iou_loss(pred, target, eps=1e-6):
    """
    Compute the Jaccard index (IoU) loss for binary segmentation.
    
    Args:
        pred (torch.Tensor): Predicted probabilities of shape (B, H, W) or (B, 1, H, W).
        target (torch.Tensor): Ground truth binary mask of shape (B, H, W) or (B, 1, H, W).
        eps (float): Small value to avoid division by zero.
        
    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Ensure both pred and target are of shape (B, H, W)
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
    
    jaccard = (intersection + eps) / (union + eps)
    loss = 1 - jaccard  # Loss is 1 - Jaccard index
    return loss.mean()
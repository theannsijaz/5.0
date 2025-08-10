import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MaskedSemanticModeling(nn.Module):
    """
    Implements masked semantic modeling as described in the SOLIDER paper.
    This module randomly masks out semantic parts and predicts their semantic labels.
    """
    def __init__(self, mask_ratio=0.3, num_semantic_parts=3):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.num_semantic_parts = num_semantic_parts
        
    def generate_mask(self, feature_maps):
        """
        Generate random masks for semantic parts.
        Each semantic part has a probability of mask_ratio to be masked.
        """
        B, C, H, W = feature_maps.shape
        device = feature_maps.device
        
        # Create spatial coordinate grids
        y_coords = torch.linspace(0, 1, H, device=device).view(H, 1).expand(H, W)
        
        # Initialize mask (1 means keep, 0 means mask)
        mask = torch.ones((B, H, W), device=device)
        
        # Define semantic regions based on y-coordinates
        upper_mask = y_coords < 0.45  # Upper body
        middle_mask = (y_coords >= 0.45) & (y_coords < 0.80)  # Lower body
        lower_mask = y_coords >= 0.80  # Shoes
        
        # For each batch
        for b in range(B):
            # Randomly decide which parts to mask
            if random.random() < self.mask_ratio:
                mask[b][upper_mask] = 0  # Mask upper body
            if random.random() < self.mask_ratio:
                mask[b][middle_mask] = 0  # Mask lower body
            if random.random() < self.mask_ratio:
                mask[b][lower_mask] = 0  # Mask shoes
                
        return mask.unsqueeze(1)  # Add channel dimension
        
    def forward(self, feature_maps):
        """
        Apply masking to feature maps and return both masked features and mask
        """
        # Generate mask
        mask = self.generate_mask(feature_maps)
        
        # Apply mask to features
        masked_features = feature_maps * mask
        
        return {
            'masked_features': masked_features,
            'mask': mask,
            'original_features': feature_maps
        }

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader, Sampler
import itertools
import os
from PIL import Image
from torchvision import transforms
import random
from collections import defaultdict
DEVICE_IDS = [0, 1]

# Memory optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free = total - cached
            print(f"GPU {i}: Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Free: {free:.2f}GB, Total: {total:.2f}GB")

def clear_gpu_memory():
    """Clear GPU memory cache and garbage collect."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("✓ GPU memory cleared")

def check_gpu_distribution(model):
    """Check if model is properly distributed across GPUs."""
    if hasattr(model, 'module'):  # DataParallel
        print(f"✓ Model is wrapped with DataParallel")
        print(f"  - Device IDs: {model.device_ids}")
        print(f"  - Output device: {model.output_device}")
        
        # Check parameter distribution
        for i, device_id in enumerate(model.device_ids):
            device_params = sum(p.numel() for p in model.module.parameters() if p.device.index == device_id)
            total_params = sum(p.numel() for p in model.module.parameters())
            print(f"  - GPU {device_id}: {device_params:,} parameters ({device_params/total_params*100:.1f}%)")
    else:
        print(f"✓ Model is on single device: {next(model.parameters()).device}")

def verify_gpu_utilization():
    """Verify that both GPUs are being utilized during training."""
    print("✓ GPU Utilization Check:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  - GPU {i}: {allocated:.3f}GB allocated, {reserved:.3f}GB reserved")
    
    # Check if both GPUs have significant memory usage
    gpu0_mem = torch.cuda.memory_allocated(0) / 1024**3
    gpu1_mem = torch.cuda.memory_allocated(1) / 1024**3
    
    if gpu1_mem > 0.1:  # More than 100MB on GPU1
        print(f"✓ Both GPUs are being utilized (GPU1: {gpu1_mem:.3f}GB)")
    else:
        print(f"⚠️  GPU1 appears underutilized (only {gpu1_mem:.3f}GB allocated)")
        print(f"   This indicates DataParallel is not working properly!")

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader, Sampler
import itertools
import os
from PIL import Image
from torchvision import transforms
import random
from collections import defaultdict

class PKSampler(Sampler):
    """
    Fixed PK Sampler for Person Re-ID: P persons × K images per person
    """
    def __init__(self, data_source, P=16, K=4):
        self.data_source = data_source
        self.P = P  # Number of persons per batch
        self.K = K  # Number of images per person
        
        # Group samples by person ID
        self.pid_to_indices = defaultdict(list)
        for idx, (_, pid) in enumerate(data_source.samples):
            self.pid_to_indices[pid].append(idx)
        
        # Filter out persons with less than K images
        self.valid_pids = [pid for pid, indices in self.pid_to_indices.items() 
                          if len(indices) >= self.K]
        
        if len(self.valid_pids) < self.P:
            raise ValueError(f"Not enough persons with at least {self.K} images. "
                           f"Found {len(self.valid_pids)}, need {self.P}")
        
        # Calculate total number of samples we'll generate
        self.num_batches = len(self.valid_pids) // self.P
        self.total_size = self.num_batches * self.P * self.K
    
    def __iter__(self):
        """Fixed iterator that yields individual indices, not batches"""
        # Shuffle valid PIDs for each epoch
        shuffled_pids = self.valid_pids.copy()
        random.shuffle(shuffled_pids)
        
        # Generate all indices for this epoch
        all_indices = []
        
        for batch_start in range(0, len(shuffled_pids) - self.P + 1, self.P):
            # Select P persons for this batch
            batch_pids = shuffled_pids[batch_start:batch_start + self.P]
            
            batch_indices = []
            for pid in batch_pids:
                # Randomly select K images for this person
                available_indices = self.pid_to_indices[pid]
                if len(available_indices) >= self.K:
                    selected_indices = random.sample(available_indices, self.K)
                    batch_indices.extend(selected_indices)
            
            # Shuffle within batch to avoid ordering bias
            random.shuffle(batch_indices)
            all_indices.extend(batch_indices)
        
        # Yield individual indices
        for idx in all_indices:
            yield idx
    
    def __len__(self):
        return self.total_size

class PersonReIDTrainDataset(torch.utils.data.Dataset):
    """
    Dataset for training set: expects structure Dataset/train/<pid>/*.jpg
    Returns (image, label)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List of (img_path, label)
        self.label_map = {}  # pid (str) -> label (int)
        self._prepare()

    def _prepare(self):
        pids = sorted(os.listdir(self.root_dir))
        self.label_map = {pid: idx for idx, pid in enumerate(pids)}
        for pid in pids:
            pid_dir = os.path.join(self.root_dir, pid)
            if not os.path.isdir(pid_dir):
                continue
            for fname in os.listdir(pid_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(pid_dir, fname), self.label_map[pid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
        

class PersonReIDTestDataset(torch.utils.data.Dataset):
    """
    Dataset for query/gallery set: expects structure Dataset/query/*.jpg or Dataset/gallery/*.jpg
    Returns (image, label, cam_id)
    """
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        self.samples = []  # List of (img_path, label, cam_id)
        self._prepare()

    def _prepare(self):
        for fname in os.listdir(self.dir_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Example: 0001_c1s1_001051_00.jpg
                parts = fname.split('_')
                if len(parts) < 2:
                    continue
                label = int(parts[0])
                cam_id = int(parts[1][1])  # e.g., c1 -> 1
                self.samples.append((os.path.join(self.dir_path, fname), label, cam_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, cam_id = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, cam_id


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class SpatialSemanticClustering(nn.Module):
    """
    Improved spatial-level semantic clustering for CNN feature maps.
    """
    def __init__(self, feature_dim, num_semantic_parts=3, momentum=0.99):
        super(SpatialSemanticClustering, self).__init__()
        self.feature_dim = feature_dim
        self.num_semantic_parts = num_semantic_parts
        self.momentum = momentum
        
        # Semantic head with dropout for robustness
        self.semantic_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 4, num_semantic_parts + 1)
        )
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def spatial_semantic_labeling(self, feature_maps):
        """Generate spatial semantic labels based on human priors."""
        B, C, H, W = feature_maps.shape
        device = feature_maps.device
        
        # Create spatial coordinate grids
        y_coords = torch.linspace(0, 1, H, device=device).view(H, 1).expand(H, W)
        
        # Human prior: spatial semantic assignment
        semantic_labels = torch.zeros(H, W, dtype=torch.long, device=device)
        
        # Upper body: top 40%
        upper_mask = y_coords < 0.4
        semantic_labels[upper_mask] = 0
        
        # Lower body: middle 40%
        middle_mask = (y_coords >= 0.4) & (y_coords < 0.8)
        semantic_labels[middle_mask] = 1
        
        # Shoes: bottom 20%
        lower_mask = y_coords >= 0.8
        semantic_labels[lower_mask] = 2
        
        return semantic_labels
    
    def foreground_background_clustering(self, feature_maps):
        """Separate foreground and background with improved stability."""
        B, C, H, W = feature_maps.shape
        
        # Calculate feature magnitude
        feature_magnitude = torch.norm(feature_maps, dim=1, p=2)  # [B, H, W]
        
        # Adaptive threshold with clamping for stability
        batch_mean = feature_magnitude.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        batch_std = feature_magnitude.std(dim=(1, 2), keepdim=True)     # [B, 1, 1]
        
        # Clamp std to avoid division by zero
        batch_std = torch.clamp(batch_std, min=1e-6)
        fg_threshold = batch_mean + 0.5 * batch_std  # [B, 1, 1]
        
        # Create foreground mask and ensure it's [B, H, W]
        fg_mask = feature_magnitude > fg_threshold  # Broadcasting to [B, H, W]
        
        return fg_mask
    
    def forward(self, student_features, teacher_features=None):
        """Forward pass with improved error handling."""
        B, C, H, W = student_features.shape
        device = student_features.device
        
        # Use teacher features if available
        clustering_features = teacher_features if teacher_features is not None else student_features
        
        # Generate pseudo semantic labels
        fg_mask = self.foreground_background_clustering(clustering_features)
        spatial_labels = self.spatial_semantic_labeling(clustering_features)
        
        # Combine foreground mask with spatial labels - TorchScript compatible
        pseudo_labels = torch.full((B, H, W), self.num_semantic_parts, 
                                 dtype=torch.long, device=device)
        
        # Vectorized operation instead of loop
        # fg_mask is [B, H, W], spatial_labels is [H, W]
        # Ensure fg_mask is [B, H, W] and spatial_labels is expanded to [B, H, W]
        if fg_mask.dim() == 4:  # If fg_mask is [B, 1, H, W]
            fg_mask = fg_mask.squeeze(1)  # Remove the extra dimension
        
        spatial_labels_expanded = spatial_labels.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        pseudo_labels = torch.where(fg_mask, spatial_labels_expanded, pseudo_labels)
        
        # Flatten for classification
        student_flat = student_features.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = pseudo_labels.reshape(-1)
        
        # Semantic classification
        semantic_logits = self.semantic_head(student_flat)
        # Use label smoothing for better training stability
        semantic_loss = F.cross_entropy(semantic_logits, labels_flat, 
                                       reduction='mean', label_smoothing=0.1)
        
        return {
            'semantic_loss': semantic_loss,
            'pseudo_labels': pseudo_labels,
            'foreground_mask': fg_mask,
            'semantic_logits': semantic_logits.view(B, H, W, -1)
        }

class SemanticController(nn.Module):
    """Improved semantic controller with better parameter handling."""
    def __init__(self, feature_dim):
        super(SemanticController, self).__init__()
        self.feature_dim = feature_dim
        
        # Lambda encoding networks
        self.weight_encoder = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Softplus()
        )
        
        self.bias_encoder = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, feature_maps, lambda_val=0.5):
        """Apply semantic control with robust parameter handling."""
        B, C, H, W = feature_maps.shape
        device = feature_maps.device
        
        if isinstance(lambda_val, (int, float)):
            lambda_tensor = torch.tensor([[float(lambda_val)]], device=device, dtype=torch.float32)
        elif isinstance(lambda_val, torch.Tensor):
            if lambda_val.dim() == 0:
                lambda_tensor = lambda_val.unsqueeze(0).unsqueeze(0).to(device).float()
            elif lambda_val.dim() == 1:
                lambda_tensor = lambda_val.unsqueeze(1).to(device).float()
            else:
                lambda_tensor = lambda_val.to(device).float()
        else:
            lambda_tensor = torch.tensor([[0.5]], device=device, dtype=torch.float32)
        
        # Ensure we have the right shape [1, 1]
        if lambda_tensor.numel() == 0:
            lambda_tensor = torch.tensor([[0.5]], device=device, dtype=torch.float32)
        elif lambda_tensor.shape != (1, 1):
            lambda_tensor = lambda_tensor.view(1, 1)
        
        # Encode lambda into weights and biases
        weights = self.weight_encoder(lambda_tensor)  # [1, C]
        biases = self.bias_encoder(lambda_tensor)     # [1, C]
        
        # Expand for broadcasting
        weights = weights.view(1, C, 1, 1).expand(B, C, H, W)
        biases = biases.view(1, C, 1, 1).expand(B, C, H, W)
        
        # Apply semantic control
        controlled_features = weights * feature_maps + biases
        
        return controlled_features

class SOLIDERCNNBlock(nn.Module):
    """
    ResNet block integrated with SOLIDER semantic control.
    Maintains ResNet structure while adding semantic controllability.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(SOLIDERCNNBlock, self).__init__()
        
        # Standard ResNet block components
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
        # SOLIDER semantic controller
        self.semantic_controller = SemanticController(out_channels)
        
    def forward(self, x, lambda_val=0.5):
        identity = x
        
        # Standard ResNet forward pass
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsampling if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Apply semantic control before adding residual
        out = self.semantic_controller(out, lambda_val)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out

class MultiScaleFeatureFusion(nn.Module):
    """
    Fixed multi-scale feature fusion with proper dimension handling.
    """
    def __init__(self, feature_dims=None, output_dim=2048):
        super(MultiScaleFeatureFusion, self).__init__()
        
        # Default ResNet50 dimensions if not provided
        if feature_dims is None:
            feature_dims = [256, 512, 1024, 2048]
        
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Projection layers to align dimensions
        self.projections = nn.ModuleList()
        for dim in feature_dims:
            if dim != output_dim:
                self.projections.append(nn.Sequential(
                    nn.Conv2d(dim, output_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(output_dim),
                    nn.ReLU(inplace=True)
                ))
            else:
                # Identity projection for same dimension
                self.projections.append(nn.Identity())
        
        # Attention mechanism for scale weighting
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_dim, output_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim // 4, len(feature_dims), 1),
            nn.Sigmoid()
        )
        
    def forward(self, multi_scale_features):
        """Fuse multi-scale features with proper error handling."""
        # Fixed implementation for TorchScript compatibility
        # We know we have exactly 4 features: [x1, x2, x3, x4]
        if len(multi_scale_features) != 4:
            # Fallback for unexpected number of features
            if len(multi_scale_features) == 0:
                device = torch.device('cpu')
                return torch.zeros(1, self.output_dim, 8, 4, device=device)
            # Use the last feature as fallback
            return multi_scale_features[-1]
        
        # Extract individual features
        feat1, feat2, feat3, feat4 = multi_scale_features
        
        # Get target size from the highest resolution feature (last one)
        target_size = feat4.shape[2:]
        
        # Project and resize features individually
        proj1 = self.projections[0](feat1)
        proj2 = self.projections[1](feat2)
        proj3 = self.projections[2](feat3)
        proj4 = self.projections[3](feat4)
        
        # Resize if needed
        if proj1.shape[2:] != target_size:
            proj1 = F.interpolate(proj1, size=target_size, mode='bilinear', align_corners=False)
        if proj2.shape[2:] != target_size:
            proj2 = F.interpolate(proj2, size=target_size, mode='bilinear', align_corners=False)
        if proj3.shape[2:] != target_size:
            proj3 = F.interpolate(proj3, size=target_size, mode='bilinear', align_corners=False)
        if proj4.shape[2:] != target_size:
            proj4 = F.interpolate(proj4, size=target_size, mode='bilinear', align_corners=False)
        
        # Stack features for attention computation
        stacked_features = torch.stack([proj1, proj2, proj3, proj4], dim=1)  # [B, 4, C, H, W]
        B, num_scales, C, H, W = stacked_features.shape
        
        # Compute attention weights using mean feature
        mean_feature = torch.mean(stacked_features, dim=1)  # [B, C, H, W]
        attention_weights = self.scale_attention(mean_feature)  # [B, 4, 1, 1]
        
        # Apply attention and fuse
        attention_weights = attention_weights.unsqueeze(2)  # [B, 4, 1, 1, 1]
        weighted_features = stacked_features * attention_weights
        fused_features = torch.sum(weighted_features, dim=1)  # [B, C, H, W]
        
        return fused_features

class SOLIDERPersonReIDModel(nn.Module):
    """
    Fixed SOLIDER Person Re-ID model with improved stability.
    """
    def __init__(self, num_classes, feature_dim=2048, num_semantic_parts=3):
        super(SOLIDERPersonReIDModel, self).__init__()
        
        # Load ResNet50 backbone
        resnet = models.resnet50(pretrained=True)
        
        # Extract stages
        self.stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stage1 = resnet.layer1  # 256 channels
        self.stage2 = resnet.layer2  # 512 channels  
        self.stage3 = resnet.layer3  # 1024 channels
        self.stage4 = resnet.layer4  # 2048 channels
        
        # Multi-scale fusion with correct dimensions
        self.multi_scale_fusion = MultiScaleFeatureFusion(
            feature_dims=[256, 512, 1024, 2048],
            output_dim=feature_dim
        )
        
        # Semantic clustering
        self.semantic_clustering = SpatialSemanticClustering(
            feature_dim=feature_dim,
            num_semantic_parts=num_semantic_parts
        )
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_neck = nn.BatchNorm1d(feature_dim)
        self.bn_neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        
        self._init_params()
    
    def _init_params(self):
        """Initialize parameters."""
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out')
        nn.init.constant_(self.bn_neck.weight, 1)
        nn.init.constant_(self.bn_neck.bias, 0)
    
    def forward(self, x, lambda_val=0.5, return_semantic_loss=False, teacher_features=None):
        """
        Forward pass with consistent output handling.
        """
        # Extract multi-scale features
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)  
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        # Multi-scale fusion - TorchScript compatible
        fused_features = self.multi_scale_fusion([x1, x2, x3, x4])
        
        # Global pooling and classification
        pooled_features = self.global_pool(fused_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        features = self.bn_neck(pooled_features)
        logits = self.classifier(features)
        
        # Return based on request
        if return_semantic_loss:
            # Semantic clustering for training
            semantic_output = self.semantic_clustering(fused_features, teacher_features)
            return features, logits, semantic_output
        else:
            return features, logits

def create_solider_model(num_classes):
    """Factory function to create SOLIDER model."""
    return SOLIDERPersonReIDModel(num_classes=num_classes)


class SOLIDERFIDITrainer:
    """
    Fixed SOLIDER-enhanced FIDI trainer with better error handling.
    """
    def __init__(self, model, num_classes, device='cuda', 
                 alpha=1.05, beta=0.5, lr=3.5e-4, weight_decay=5e-4,
                 loss_strategy='adaptive', semantic_weight=0.5):
        
        # Device setup with multi-GPU support
        if isinstance(device, (list, tuple)):
            assert torch.cuda.is_available(), "CUDA must be available for multi-GPU."
            assert len(device) >= 2, "Multi-GPU requires at least 2 devices."
            
            # Clear GPU memory first
            torch.cuda.empty_cache()
            
            # CRITICAL FIX: Move model to first GPU BEFORE wrapping with DataParallel
            self.device = torch.device(f"cuda:{device[0]}")
            model = model.to(self.device)  # Move to first GPU FIRST
            
            # Use DataParallel with specified device IDs - this will distribute properly
            self.model = nn.DataParallel(model, device_ids=device, output_device=device[0])
            self.is_parallel = True
            
            print(f"DataParallel initialized with devices: {device}")
        else:
            self.device = torch.device(device)
            self.model = model.to(self.device)
            self.is_parallel = False
        
        self.num_classes = num_classes
        self.fidi_loss = FIDILoss(alpha=alpha, beta=beta)
        self.ce_loss = nn.CrossEntropyLoss()
        self.loss_strategy = loss_strategy
        self.semantic_weight = semantic_weight
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=40, gamma=0.1
        )
        
        # Training state
        self.loss_history = {'fidi': [], 'ce': [], 'semantic': []}
        self.best_mAP = 0.0
        self.stage_switch_epoch = 100  # Switch to SOLIDER stage after this epoch
        

    
    def get_model(self):
        """Get the actual model (handle DataParallel wrapper)"""
        return self.model.module if self.is_parallel else self.model
    
    def get_loss_weights(self, epoch, total_epochs, strategy=None):
        """Get dynamic loss weights based on training progress."""
        if strategy is None:
            strategy = self.loss_strategy
            
        progress = epoch / total_epochs
        
        if strategy == 'conservative':
            fidi_weight = min(0.8, progress * 1.5)
            cls_weight = max(0.8, 1.2 - progress)
            
        elif strategy == 'progressive':
            import math
            fidi_weight = 0.5 * (1 + math.tanh(4 * (progress - 0.5)))
            cls_weight = 1.0 - 0.3 * progress
            
        elif strategy == 'adaptive':
            if len(self.loss_history['fidi']) > 5:
                recent_fidi = sum(self.loss_history['fidi'][-5:]) / 5
                recent_ce = sum(self.loss_history['ce'][-5:]) / 5
                
                if recent_fidi > recent_ce * 2:
                    fidi_weight = max(0.3, min(0.7, 0.5 - 0.2 * (recent_fidi / recent_ce - 2)))
                    cls_weight = 1.0
                elif recent_ce > recent_fidi * 2:
                    fidi_weight = min(1.0, 0.5 + 0.3 * (recent_ce / recent_fidi - 2))
                    cls_weight = max(0.7, 1.0 - 0.2 * (recent_ce / recent_fidi - 2))
                else:
                    fidi_weight = 0.5 + 0.3 * progress
                    cls_weight = 1.0 - 0.2 * progress
            else:
                fidi_weight = 0.3 + 0.3 * progress
                cls_weight = 1.0
                
        elif strategy == 'fixed':
            fidi_weight = 0.7
            cls_weight = 1.0
            
        else:  # 'original'
            fidi_weight = min(1.0, epoch / (total_epochs * 0.3))
            cls_weight = max(0.5, 1.0 - epoch / (total_epochs * 0.8))
        
        return fidi_weight, cls_weight
    
    def train_epoch(self, dataloader, epoch=0, total_epochs=120):
        """
        Fixed train_epoch method that handles both FIDI and SOLIDER stages.
        """
        # Determine training stage
        if epoch < self.stage_switch_epoch:
            return self._train_epoch_stage1(dataloader, epoch, total_epochs)
        else:
            if epoch == self.stage_switch_epoch:
                print("=" * 50)
                print("SWITCHING TO SOLIDER STAGE")
                print("=" * 50)
                # Reduce learning rate for SOLIDER stage
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
            
            return self._train_epoch_stage2(dataloader, epoch, total_epochs)
    
    def _train_epoch_stage1(self, dataloader, epoch, total_epochs):
        """Stage 1: FIDI training only."""
        self.model.train()
        total_loss = 0.0
        total_fidi_loss = 0.0
        total_ce_loss = 0.0
        total_semantic_loss = 0.0  # Always 0 in stage 1
        
        batch_losses = []
        batch_fidi_losses = []
        batch_ce_losses = []
        batch_semantic_losses = []
        
        fidi_weight, cls_weight = self.get_loss_weights(epoch, total_epochs)
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Standard forward pass (no semantic loss)
            features, logits = self.model(images, return_semantic_loss=False)
            
            fidi_loss = self.fidi_loss(features, labels)
            ce_loss = self.ce_loss(logits, labels)
            loss = fidi_weight * fidi_loss + cls_weight * ce_loss
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Clear cache to prevent memory buildup
            if batch_idx % 5 == 0:  # Clear every 5 batches
                torch.cuda.empty_cache()
            
            # Track losses
            batch_loss = loss.item()
            batch_fidi = fidi_loss.item()
            batch_ce = ce_loss.item()
            batch_semantic = 0.0
            
            total_loss += batch_loss
            total_fidi_loss += batch_fidi
            total_ce_loss += batch_ce
            total_semantic_loss += batch_semantic
            
            batch_losses.append(batch_loss)
            batch_fidi_losses.append(batch_fidi)
            batch_ce_losses.append(batch_ce)
            batch_semantic_losses.append(batch_semantic)
            
            if batch_idx % 10 == 0:  # Changed from 50 to 10 for more frequent updates
                print(f'FIDI Stage - Batch {batch_idx}: Loss={batch_loss:.6f}, '
                      f'FIDI={batch_fidi:.6f}×{fidi_weight:.2f}, '
                      f'CE={batch_ce:.6f}×{cls_weight:.2f}')
                if batch_idx % 50 == 0:  # Print memory every 50 batches
                    print_gpu_memory()
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_fidi = total_fidi_loss / num_batches
        avg_ce = total_ce_loss / num_batches
        avg_semantic = total_semantic_loss / num_batches
        
        # Update history
        self.loss_history['fidi'].append(avg_fidi)
        self.loss_history['ce'].append(avg_ce)
        self.loss_history['semantic'].append(avg_semantic)
        
        # Final memory cleanup for this epoch
        clear_gpu_memory()
        
        return avg_loss, avg_fidi, avg_ce, avg_semantic, batch_losses, batch_fidi_losses, batch_ce_losses, batch_semantic_losses
    
    def _train_epoch_stage2(self, dataloader, epoch, total_epochs):
        """Stage 2: SOLIDER training with semantic supervision."""
        self.model.train()
        total_loss = 0.0
        total_fidi_loss = 0.0
        total_ce_loss = 0.0
        total_semantic_loss = 0.0
        
        batch_losses = []
        batch_fidi_losses = []
        batch_ce_losses = []
        batch_semantic_losses = []
        
        # Generate lambda values for semantic control
        num_batches = len(dataloader)
        lambda_vals = torch.bernoulli(torch.full((num_batches,), 0.5))
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Get lambda value for this batch
            current_lambda = float(lambda_vals[batch_idx % len(lambda_vals)].item())
            
            # Forward pass with semantic loss
            # Remove try/except for better stability
            actual_model = self.get_model()
            features, logits, semantic_output = actual_model(
                images, lambda_val=current_lambda, return_semantic_loss=True
            )
            
            # Extract semantic loss safely
            if isinstance(semantic_output, dict) and 'semantic_loss' in semantic_output:
                semantic_loss = semantic_output['semantic_loss']
            else:
                semantic_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Compute losses
            fidi_loss = self.fidi_loss(features, labels)
            ce_loss = self.ce_loss(logits, labels)
            
            # Get weights and combine losses
            fidi_weight, cls_weight = self.get_loss_weights(epoch, total_epochs)
            loss = (fidi_weight * fidi_loss + 
                   cls_weight * ce_loss + 
                   self.semantic_weight * semantic_loss)
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Clear cache to prevent memory buildup
            if batch_idx % 3 == 0:  # Clear every 3 batches for better memory management
                torch.cuda.empty_cache()
                if batch_idx % 10 == 0:  # More aggressive cleanup every 10 batches
                    import gc
                    gc.collect()
            
            # Track losses safely
            batch_loss = loss.item()
            batch_fidi = fidi_loss.item()
            batch_ce = ce_loss.item()
            batch_semantic = semantic_loss.item() if hasattr(semantic_loss, 'item') else float(semantic_loss)
            
            total_loss += batch_loss
            total_fidi_loss += batch_fidi
            total_ce_loss += batch_ce
            total_semantic_loss += batch_semantic
            
            batch_losses.append(batch_loss)
            batch_fidi_losses.append(batch_fidi)
            batch_ce_losses.append(batch_ce)
            batch_semantic_losses.append(batch_semantic)
            
            if batch_idx % 10 == 0:  # Changed from 50 to 10 for more frequent updates
                print(f'SOLIDER Stage - Batch {batch_idx}: Loss={batch_loss:.6f}, '
                      f'FIDI={batch_fidi:.6f}×{fidi_weight:.2f}, '
                      f'CE={batch_ce:.6f}×{cls_weight:.2f}, '
                      f'Semantic={batch_semantic:.6f}×{self.semantic_weight:.2f}, '
                      f'Lambda={current_lambda:.1f}')
            if batch_idx % 20 == 0:  # Print memory every 20 batches
                print_gpu_memory()
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_fidi = total_fidi_loss / num_batches
        avg_ce = total_ce_loss / num_batches
        avg_semantic = total_semantic_loss / num_batches
        
        # Update history
        self.loss_history['fidi'].append(avg_fidi)
        self.loss_history['ce'].append(avg_ce)
        self.loss_history['semantic'].append(avg_semantic)
        
        # Final memory cleanup for this epoch
        clear_gpu_memory()
        
        return avg_loss, avg_fidi, avg_ce, avg_semantic, batch_losses, batch_fidi_losses, batch_ce_losses, batch_semantic_losses
    
    def evaluate(self, query_dataloader, gallery_dataloader):
        """Evaluation with proper SOLIDER model handling."""
        self.model.eval()
        query_features = []
        query_labels = []
        query_cam_ids = []
        
        # Optimal lambda for evaluation (appearance-focused)
        eval_lambda = 0.15
        
        with torch.no_grad():
            for images, labels, cam_ids in query_dataloader:
                images = images.to(self.device, non_blocking=True)
                
                actual_model = self.get_model()
                features, logits = actual_model(
                    images, lambda_val=eval_lambda, return_semantic_loss=False
                )
                
                query_features.append(features.cpu())
                query_labels.extend(labels.numpy())
                query_cam_ids.extend(cam_ids.numpy())
        
        query_features = torch.cat(query_features, dim=0)
        query_features = F.normalize(query_features, p=2, dim=1)
        
        gallery_features = []
        gallery_labels = []
        gallery_cam_ids = []
        
        with torch.no_grad():
            for images, labels, cam_ids in gallery_dataloader:
                images = images.to(self.device, non_blocking=True)

                actual_model = self.get_model()
                features, logits = actual_model(
                    images, lambda_val=eval_lambda, return_semantic_loss=False
                )
                
                gallery_features.append(features.cpu())
                gallery_labels.extend(labels.numpy())
                gallery_cam_ids.extend(cam_ids.numpy())
        
        gallery_features = torch.cat(gallery_features, dim=0)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)
        
        dist_matrix = torch.cdist(query_features, gallery_features, p=2)
        cmc, mAP = self.compute_cmc_map(
            dist_matrix, query_labels, gallery_labels, 
            query_cam_ids, gallery_cam_ids
        )
        
        return cmc, mAP
    
    def compute_cmc_map(self, dist_matrix, query_labels, gallery_labels, 
                       query_cam_ids, gallery_cam_ids, max_rank=50):
        """CMC and mAP computation (unchanged from your original)."""
        num_q, num_g = dist_matrix.shape
        if num_g < max_rank:
            max_rank = num_g
            print(f"Note: number of gallery samples is quite small, got {num_g}")
        
        indices = torch.argsort(dist_matrix, dim=1)
        matches = (torch.tensor(gallery_labels)[indices] == 
                  torch.tensor(query_labels).view(-1, 1))
        
        all_cmc = []
        all_AP = []
        num_valid_q = 0
        
        for q_idx in range(num_q):
            q_pid = query_labels[q_idx]
            q_camid = query_cam_ids[q_idx]
            order = indices[q_idx]
            
            remove = torch.tensor([(gallery_labels[i] == q_pid) & 
                                 (gallery_cam_ids[i] == q_camid) 
                                 for i in order])
            keep = ~remove
            orig_cmc = matches[q_idx][keep]
            
            if not torch.any(orig_cmc):
                continue
            
            cmc = orig_cmc.cumsum(0)
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1
            
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum(0)
            tmp_cmc = tmp_cmc / (torch.arange(len(tmp_cmc)) + 1.0)
            tmp_cmc = tmp_cmc * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)
        
        if num_valid_q == 0:
            raise RuntimeError("No valid query")
        
        all_cmc = torch.stack(all_cmc, dim=0).float()
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = sum(all_AP) / len(all_AP)
        
        return all_cmc, mAP


# In[ ]:


class FIDILoss(nn.Module):
    """
    Fine-grained Difference-aware (FIDI) Pairwise Loss
    Corrected implementation following the paper exactly
    """
    def __init__(self, alpha=1.05, beta=0.5):
        super(FIDILoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-8
    
    def forward(self, features, labels):
        """
        Args:
            features: tensor of shape (batch_size, feature_dim)
            labels: tensor of shape (batch_size,)
        """
        # Compute pairwise distances
        distances = self.compute_pairwise_distances(features)
        
        # Compute ground truth binary relationship matrix K
        labels = labels.view(-1, 1)
        k_matrix = (labels == labels.T).float()  # 1 if same identity, 0 otherwise
        
        # Compute learned probability distribution U using exponential function
        u_matrix = torch.exp(-self.beta * distances)
        
        # Compute D(U||K) + D(K||U)
        d_u_k = self.compute_kl_divergence(u_matrix, k_matrix)
        d_k_u = self.compute_kl_divergence(k_matrix, u_matrix)
        
        total_loss = d_u_k + d_k_u
        
        return total_loss
    
    def compute_pairwise_distances(self, features):
        """Compute Euclidean distances between all pairs of features"""
        n = features.size(0)
        # Expand features to compute all pairwise distances
        features_1 = features.unsqueeze(1).expand(n, n, -1)
        features_2 = features.unsqueeze(0).expand(n, n, -1)
        
        # Compute Euclidean distance
        distances = torch.sqrt(torch.sum((features_1 - features_2) ** 2, dim=2) + self.eps)
        
        return distances
    
    def compute_kl_divergence(self, p_matrix, q_matrix):
        """
        Compute KL divergence D(P||Q) following Equation (5) from the paper:
        D(P||Q) = Σ p_ij * log(α * p_ij / ((α-1) * p_ij + q_ij))
        """
        # Clamp to avoid numerical issues
        p_matrix = torch.clamp(p_matrix, min=self.eps, max=1-self.eps)
        q_matrix = torch.clamp(q_matrix, min=self.eps, max=1-self.eps)
        
        # Compute the denominator: (α-1) * p_ij + q_ij
        denominator = (self.alpha - 1) * p_matrix + q_matrix
        denominator = torch.clamp(denominator, min=self.eps)
        
        # Compute the fraction: α * p_ij / denominator
        numerator = self.alpha * p_matrix
        fraction = numerator / denominator
        fraction = torch.clamp(fraction, min=self.eps)
        
        # Compute KL divergence: p_ij * log(fraction)
        kl_div = p_matrix * torch.log(fraction)
        
        # Exclude diagonal elements (self-comparisons) and compute mean
        mask = ~torch.eye(p_matrix.size(0), dtype=torch.bool, device=p_matrix.device)
        kl_div = kl_div[mask].mean()
        
        return kl_div


# In[ ]:


# =========================
# 5. Trainer Class
# =========================
class FIDITrainer:
    """
    Improved Training framework for Person Re-ID with FIDI loss
    """
    def __init__(self, model, num_classes, device='cuda', 
                 alpha=1.05, beta=0.5, lr=3.5e-4, weight_decay=5e-4,
                 loss_strategy='adaptive'):
        # Multi-GPU support
        if isinstance(device, (list, tuple)):
            assert torch.cuda.is_available(), "CUDA must be available for multi-GPU."
            assert len(device) >= 2, "Multi-GPU requires at least 2 devices."
            
            # Place model on the first GPU, then use DataParallel
            self.device = torch.device(f"cuda:{device[0]}")
            model = model.to(self.device)  # Move to first GPU first
            
            # Use DataParallel with specified device IDs
            self.model = nn.DataParallel(model, device_ids=device, output_device=device[0])
            self.is_parallel = True
            
            print(f"✓ DataParallel initialized with devices: {device}")
            print(f"✓ Model distributed across {len(device)} GPUs")
        else:
            self.device = torch.device(device)
            self.model = model.to(self.device)
            self.is_parallel = False
        
        self.num_classes = num_classes
        self.fidi_loss = FIDILoss(alpha=alpha, beta=beta)
        self.ce_loss = nn.CrossEntropyLoss()
        self.loss_strategy = loss_strategy
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=40, gamma=0.1
        )
        
        # For adaptive strategy
        self.loss_history = {'fidi': [], 'ce': []}
        self.best_mAP = 0.0
        
        # Verify model is properly initialized
        if hasattr(self, 'is_parallel') and self.is_parallel:
            try:
                test_input = torch.randn(1, 3, 256, 128, device=self.device)
                with torch.no_grad():
                    _ = self.model(test_input)
                print(f"✓ FIDITrainer model verification successful")
            except Exception as e:
                print(f"⚠️  FIDITrainer model verification failed: {e}")
    
    def get_model(self):
        """Get the actual model (handle DataParallel wrapper)"""
        return self.model.module if hasattr(self, 'is_parallel') and self.is_parallel else self.model
    
    def get_loss_weights(self, epoch, total_epochs, strategy=None):
        """
        Multiple loss weighting strategies based on training progress and loss magnitudes
        """
        if strategy is None:
            strategy = self.loss_strategy
            
        progress = epoch / total_epochs
        
        if strategy == 'conservative':
            # More conservative approach - slower FIDI ramp-up, maintain CE importance
            fidi_weight = min(0.8, progress * 1.5)  # Max 0.8, reaches it at 53% of training
            cls_weight = max(0.8, 1.2 - progress)   # Min 0.8, gradual decrease
            
        elif strategy == 'progressive':
            # Gradual transition with smooth curves
            import math
            fidi_weight = 0.5 * (1 + math.tanh(4 * (progress - 0.5)))  # Sigmoid-like curve
            cls_weight = 1.0 - 0.3 * progress  # Linear decrease to 0.7
            
        elif strategy == 'adaptive':
            # Adaptive based on loss magnitudes (requires loss history)
            if len(self.loss_history['fidi']) > 5:
                # Calculate recent loss ratios
                recent_fidi = sum(self.loss_history['fidi'][-5:]) / 5
                recent_ce = sum(self.loss_history['ce'][-5:]) / 5
                
                # Balance weights based on loss magnitudes
                if recent_fidi > recent_ce * 2:  # FIDI much larger
                    fidi_weight = max(0.3, min(0.7, 0.5 - 0.2 * (recent_fidi / recent_ce - 2)))
                    cls_weight = 1.0
                elif recent_ce > recent_fidi * 2:  # CE much larger
                    fidi_weight = min(1.0, 0.5 + 0.3 * (recent_ce / recent_fidi - 2))
                    cls_weight = max(0.7, 1.0 - 0.2 * (recent_ce / recent_fidi - 2))
                else:  # Balanced
                    fidi_weight = 0.5 + 0.3 * progress
                    cls_weight = 1.0 - 0.2 * progress
            else:
                # Early training fallback
                fidi_weight = 0.3 + 0.3 * progress
                cls_weight = 1.0
                
        elif strategy == 'fixed':
            # Simple fixed weights
            fidi_weight = 0.7
            cls_weight = 1.0
            
        else:  # 'original' - your current strategy
            fidi_weight = min(1.0, epoch / (total_epochs * 0.3))
            cls_weight = max(0.5, 1.0 - epoch / (total_epochs * 0.8))
        
        return fidi_weight, cls_weight
    
    def train_epoch(self, dataloader, epoch=0, total_epochs=120):
        self.model.train()
        total_loss = 0.0
        total_fidi_loss = 0.0
        total_ce_loss = 0.0
        batch_losses = []
        batch_fidi_losses = []
        batch_ce_losses = []
        
        # Get dynamic weights for this epoch
        fidi_weight, cls_weight = self.get_loss_weights(epoch, total_epochs)
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            features, logits = self.model(images)
            fidi_loss = self.fidi_loss(features, labels)
            ce_loss = self.ce_loss(logits, labels)
            
            # Apply dynamic weighting
            loss = fidi_weight * fidi_loss + cls_weight * ce_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Clear cache to prevent memory buildup
            if batch_idx % 20 == 0:  # Clear every 5 batches
                torch.cuda.empty_cache()
            
            # Store all batch values
            batch_loss = loss.item()
            batch_fidi = fidi_loss.item()
            batch_ce = ce_loss.item()
            
            total_loss += batch_loss
            total_fidi_loss += batch_fidi
            total_ce_loss += batch_ce
            
            batch_losses.append(batch_loss)
            batch_fidi_losses.append(batch_fidi)
            batch_ce_losses.append(batch_ce)
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}: Loss={batch_loss:.6f}, '
                      f'FIDI={batch_fidi:.6f}×{fidi_weight:.2f}, '
                      f'CE={batch_ce:.6f}×{cls_weight:.2f}')
                print_gpu_memory()
        
        avg_loss = total_loss / len(dataloader)
        avg_fidi_loss = total_fidi_loss / len(dataloader)
        avg_ce_loss = total_ce_loss / len(dataloader)
        
        # Calculate additional statistics
        min_loss = min(batch_losses)
        max_loss = max(batch_losses)
        std_loss = np.std(batch_losses)
        
        print(f'Epoch Summary: Avg Loss={avg_loss:.6f}, Min={min_loss:.6f}, Max={max_loss:.6f}, Std={std_loss:.6f}')
        print(f'FIDI: Avg={avg_fidi_loss:.6f}, Min={min(batch_fidi_losses):.6f}, Max={max(batch_fidi_losses):.6f}')
        print(f'CE: Avg={avg_ce_loss:.6f}, Min={min(batch_ce_losses):.6f}, Max={max(batch_ce_losses):.6f}')
        
        # Store loss history for adaptive strategy
        self.loss_history['fidi'].append(avg_fidi_loss)
        self.loss_history['ce'].append(avg_ce_loss)
        if len(self.loss_history['fidi']) > 20:  # Keep only recent history
            self.loss_history['fidi'].pop(0)
            self.loss_history['ce'].pop(0)
        
        return avg_loss, avg_fidi_loss, avg_ce_loss, batch_losses, batch_fidi_losses, batch_ce_losses
    
    def evaluate(self, query_dataloader, gallery_dataloader):
        self.model.eval()
        query_features = []
        query_labels = []
        query_cam_ids = []
        
        with torch.no_grad():
            for images, labels, cam_ids in query_dataloader:
                images = images.to(self.device)
                features, _ = self.model(images)
                query_features.append(features.cpu())
                query_labels.extend(labels.numpy())
                query_cam_ids.extend(cam_ids.numpy())
        
        query_features = torch.cat(query_features, dim=0)
        query_features = F.normalize(query_features, p=2, dim=1)
        
        gallery_features = []
        gallery_labels = []
        gallery_cam_ids = []
        
        with torch.no_grad():
            for images, labels, cam_ids in gallery_dataloader:
                images = images.to(self.device)
                features, _ = self.model(images)
                gallery_features.append(features.cpu())
                gallery_labels.extend(labels.numpy())
                gallery_cam_ids.extend(cam_ids.numpy())
        
        gallery_features = torch.cat(gallery_features, dim=0)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)
        
        dist_matrix = torch.cdist(query_features, gallery_features, p=2)
        cmc, mAP = self.compute_cmc_map(
            dist_matrix, query_labels, gallery_labels, 
            query_cam_ids, gallery_cam_ids
        )
        
        return cmc, mAP
    
    def compute_cmc_map(self, dist_matrix, query_labels, gallery_labels, 
                       query_cam_ids, gallery_cam_ids, max_rank=50):
        num_q, num_g = dist_matrix.shape
        if num_g < max_rank:
            max_rank = num_g
            print(f"Note: number of gallery samples is quite small, got {num_g}")
        
        indices = torch.argsort(dist_matrix, dim=1)
        matches = (torch.tensor(gallery_labels)[indices] == 
                  torch.tensor(query_labels).view(-1, 1))
        
        all_cmc = []
        all_AP = []
        num_valid_q = 0
        
        for q_idx in range(num_q):
            q_pid = query_labels[q_idx]
            q_camid = query_cam_ids[q_idx]
            order = indices[q_idx]
            
            remove = torch.tensor([(gallery_labels[i] == q_pid) & 
                                 (gallery_cam_ids[i] == q_camid) 
                                 for i in order])
            keep = ~remove
            orig_cmc = matches[q_idx][keep]
            
            if not torch.any(orig_cmc):
                continue
            
            cmc = orig_cmc.cumsum(0)
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1
            
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum(0)
            tmp_cmc = tmp_cmc / (torch.arange(len(tmp_cmc)) + 1.0)
            tmp_cmc = tmp_cmc * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)
        
        if num_valid_q == 0:
            raise RuntimeError("No valid query")
        
        all_cmc = torch.stack(all_cmc, dim=0).float()
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = sum(all_AP) / len(all_AP)
        
        return all_cmc, mAP
    
    def train(self, train_dataloader, query_dataloader, gallery_dataloader, 
              num_epochs=120, eval_freq=10):
        print(f"Starting training with '{self.loss_strategy}' loss weighting strategy...")
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            # Get current weights for logging
            fidi_weight, cls_weight = self.get_loss_weights(epoch, num_epochs)
            print(f'Loss weights - FIDI: {fidi_weight:.3f}, CE: {cls_weight:.3f}')
            
            avg_loss, avg_fidi_loss, avg_ce_loss, batch_losses, batch_fidi_losses, batch_ce_losses = self.train_epoch(
                train_dataloader, epoch, num_epochs
            )
            print(f'Train Loss: {avg_loss:.4f}, FIDI Loss: {avg_fidi_loss:.4f}, '
                  f'CE Loss: {avg_ce_loss:.4f}')
            
            self.scheduler.step()
            
            if (epoch + 1) % eval_freq == 0:
                print("Evaluating...")
                cmc, mAP = self.evaluate(query_dataloader, gallery_dataloader)
                print(f'Rank-1: {cmc[0]:.4f}, Rank-5: {cmc[4]:.4f}, '
                      f'Rank-10: {cmc[9]:.4f}, mAP: {mAP:.4f}')
                
                if mAP > self.best_mAP:
                    self.best_mAP = mAP
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'mAP': mAP,
                        'cmc': cmc,
                        'loss_strategy': self.loss_strategy,
                        'fidi_weight': fidi_weight,
                        'cls_weight': cls_weight,
                    }, 'best_model.pth')
                    print(f'New best mAP: {self.best_mAP:.4f}')
        
        print(f'\nTraining completed. Best mAP: {self.best_mAP:.4f}')
        return self.best_mAP


# In[ ]:


# =========================
# 6. Tune-able Parameters / Config
# =========================

# Set PyTorch memory allocation to prevent fragmentation
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# PK Sampling parameters - Optimized for dual GPU
P = 16   # Number of persons per batch (reduced from 16)
K = 8   # Number of images per person
batch_size = P * K  # This will be 64 for optimal PK sampling with dual GPU

num_epochs = 200
# Optimize for dual GPU setup
if torch.cuda.device_count() >= 2:
    device = [0, 1]
    num_workers = 4  # Reduced workers for stability
    prefetch_factor = 2  # Reduced prefetch for stability
else:
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4
    prefetch_factor = 2

alpha = 1.05
beta = 0.5
lr = 3.5e-4
weight_decay = 5e-4
image_height = 256
image_width = 128
train_dir = os.path.join('Dataset', 'train')
query_dir = os.path.join('Dataset', 'query')
gallery_dir = os.path.join('Dataset', 'gallery')


# In[ ]:


# 7. Data Transforms & DataLoaders – SOLIDER-only (no fallbacks)

train_transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.Pad(10, padding_mode='edge'),
    transforms.RandomCrop((image_height, image_width)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.15, 0.15, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02,0.4), ratio=(0.3,3.3), value='random'),
])

test_transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Datasets
train_dataset = PersonReIDTrainDataset(train_dir, transform=train_transform)
query_dataset = PersonReIDTestDataset(query_dir, transform=test_transform)
gallery_dataset = PersonReIDTestDataset(gallery_dir, transform=test_transform)

num_classes = len(train_dataset.label_map)

# PKSampler (error if not enough PIDs/images)
pk_sampler = PKSampler(train_dataset, P=P, K=K)

# Optimize batch sizes for dual GPU
if isinstance(device, (list, tuple)) and len(device) >= 2:
    effective_batch_size = P * K * len(device)
else:
    effective_batch_size = P * K

# Always use PK sampling
train_loader = DataLoader(
    train_dataset,
    sampler=pk_sampler,
    batch_size=P*K,
    num_workers=num_workers,
    pin_memory=False,  # Disabled to prevent CUDA illegal memory access
    prefetch_factor=prefetch_factor,
    drop_last=True
)



query_loader = DataLoader(
    query_dataset,
    batch_size=P*K,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,  # Disabled to prevent CUDA illegal memory access
    prefetch_factor=prefetch_factor
)

gallery_loader = DataLoader(
    gallery_dataset,
    batch_size=P*K,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,  # Disabled to prevent CUDA illegal memory access
    prefetch_factor=prefetch_factor
)

print(f"Train samples: {len(train_dataset)}, PIDs: {num_classes}")
print(f"DataLoaders ready: train {len(train_loader)} batches, "
      f"query {len(query_loader)}, gallery {len(gallery_loader)}")


# In[ ]:


# 8. SOLIDER Model & Trainer Initialization – no fallbacks

# Model
model = SOLIDERPersonReIDModel(num_classes=num_classes)

# Trainer
trainer = SOLIDERFIDITrainer(
    model=model,
    num_classes=num_classes,
    device=device,
    alpha=alpha,
    beta=beta,
    lr=lr,
    weight_decay=weight_decay,
    loss_strategy='progressive',
    semantic_weight=0.5
)

# Enforce SOLIDER stage from epoch 0
trainer.stage_switch_epoch = 0

print(f"SOLIDER model with {num_classes} classes")
print(f"Trainer initialized – SOLIDER stage from epoch 0 onwards")
print(f"Using device(s): {device}")


# In[9]:


# # =========================
# # ONNX Export for SOLIDER Model (Netron Visualization) - SIMPLIFIED CORE
# # =========================
# import torch
# import torch.nn as nn
# import os

# def export_solider_model_to_onnx():
#     """
#     Export the SOLIDER CNN model to ONNX format for Netron visualization
#     """
#     print("Exporting SOLIDER CNN model to ONNX...")
    
#     # Create SOLIDER model instance
#     solider_model = SOLIDERPersonReIDModel(num_classes=num_classes)
    
#     # Determine device for dummy input
#     if isinstance(device, (list, tuple)):
#         dummy_device = f"cuda:{device[0]}" if torch.cuda.is_available() else "cpu"
#     else:
#         dummy_device = device if torch.cuda.is_available() else "cpu"
    
#     # Move model to device
#     solider_model = solider_model.to(dummy_device)
#     solider_model.eval()
    
#     # Create sample input tensor
#     batch_size = 1
#     sample_input = torch.randn(batch_size, 3, image_height, image_width, device=dummy_device)
    
#     # Define ONNX export paths
#     onnx_dir = "onnx_models"
#     os.makedirs(onnx_dir, exist_ok=True)
    
#     # Create a wrapper that exports just the core backbone without the final layers
#     class SOLIDERCoreWrapper(nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model
            
#             # Extract the core backbone (stages 0-4)
#             self.stage0 = model.stage0
#             self.stage1 = model.stage1
#             self.stage2 = model.stage2
#             self.stage3 = model.stage3
#             self.stage4 = model.stage4
            
#             # Extract multi-scale fusion
#             self.multi_scale_fusion = model.multi_scale_fusion
            
#             # Extract semantic clustering (without final layers)
#             self.semantic_clustering = model.semantic_clustering
        
#         def forward(self, x):
#             # Forward through stages
#             stage0_out = self.stage0(x)
#             stage1_out = self.stage1(stage0_out)
#             stage2_out = self.stage2(stage1_out)
#             stage3_out = self.stage3(stage2_out)
#             stage4_out = self.stage4(stage3_out)
            
#             # Multi-scale fusion
#             fused_features = self.multi_scale_fusion([stage1_out, stage2_out, stage3_out, stage4_out])
            
#             # Global pooling
#             pooled_features = torch.nn.functional.adaptive_avg_pool2d(fused_features, (1, 1))
#             pooled_features = pooled_features.view(pooled_features.size(0), -1)
            
#             # Return intermediate features for visualization
#             return pooled_features, stage4_out
    
#     # Export core architecture
#     core_onnx_path = os.path.join(onnx_dir, "solider_core_architecture.onnx")
#     core_wrapper = SOLIDERCoreWrapper(solider_model)
    
#     try:
#         print("Exporting core SOLIDER architecture...")
#         torch.onnx.export(
#             core_wrapper,
#             sample_input,
#             core_onnx_path,
#             export_params=True,
#             opset_version=11,
#             do_constant_folding=True,
#             input_names=['input_image'],
#             output_names=['pooled_features', 'stage4_features'],
#             dynamic_axes={
#                 'input_image': {0: 'batch_size'}, 
#                 'pooled_features': {0: 'batch_size'}, 
#                 'stage4_features': {0: 'batch_size'}
#             },
#             verbose=False
#         )
#         print(f"✓ SOLIDER core architecture exported to: {core_onnx_path}")
        
#     except Exception as e:
#         print(f"❌ Failed to export core architecture: {str(e)}")
        
#         # Try an even simpler approach - just the backbone stages
#         print("Trying simplified backbone export...")
        
#         class SOLIDERBackboneWrapper(nn.Module):
#             def __init__(self, model):
#                 super().__init__()
#                 self.stage0 = model.stage0
#                 self.stage1 = model.stage1
#                 self.stage2 = model.stage2
#                 self.stage3 = model.stage3
#                 self.stage4 = model.stage4
            
#             def forward(self, x):
#                 x = self.stage0(x)
#                 x = self.stage1(x)
#                 x = self.stage2(x)
#                 x = self.stage3(x)
#                 x = self.stage4(x)
#                 return x
        
#         backbone_onnx_path = os.path.join(onnx_dir, "solider_backbone_stages.onnx")
#         backbone_wrapper = SOLIDERBackboneWrapper(solider_model)
        
#         try:
#             torch.onnx.export(
#                 backbone_wrapper,
#                 sample_input,
#                 backbone_onnx_path,
#                 export_params=True,
#                 opset_version=11,
#                 do_constant_folding=True,
#                 input_names=['input_image'],
#                 output_names=['backbone_output'],
#                 dynamic_axes={
#                     'input_image': {0: 'batch_size'}, 
#                     'backbone_output': {0: 'batch_size'}
#                 },
#                 verbose=False
#             )
#             print(f"✓ SOLIDER backbone stages exported to: {backbone_onnx_path}")
#             core_onnx_path = backbone_onnx_path
            
#         except Exception as e2:
#             print(f"❌ Failed to export backbone stages: {str(e2)}")
#             return None
    
#     # Print model statistics
#     total_params = sum(p.numel() for p in solider_model.parameters())
#     trainable_params = sum(p.numel() for p in solider_model.parameters() if p.requires_grad)
    
#     print(f"\n📊 SOLIDER Model Statistics:")
#     print(f"   • Total parameters: {total_params:,}")
#     print(f"   • Trainable parameters: {trainable_params:,}")
#     print(f"   • Input shape: {tuple(sample_input.shape)}")
#     print(f"   • Number of classes: {num_classes}")
    
#     print(f"\n🔍 Netron Visualization:")
#     print(f"   1. Open Netron (https://netron.app/)")
#     print(f"   2. Load the exported ONNX file: {core_onnx_path}")
    
#     print(f"\n💡 Key SOLIDER Components to Look For:")
#     print(f"   • Multi-scale feature fusion (stages 1-4)")
#     print(f"   • Spatial semantic clustering")
#     print(f"   • Semantic controller modules")
#     print(f"   • ResNet backbone with SOLIDER blocks")
    
#     return {'core': core_onnx_path}

# # Execute the export
# try:
#     exported_models = export_solider_model_to_onnx()
#     if exported_models:
#         print("\n✅ SOLIDER model successfully exported to ONNX!")
#         print("   You can now visualize it in Netron!")
#     else:
#         print("\n❌ Failed to export SOLIDER model")
        
# except Exception as e:
#     print(f"❌ Error during export: {str(e)}")
#     print("Make sure the SOLIDERPersonReIDModel class has been defined and all dependencies are imported.")


# In[ ]:


# =========================
# 9. Training Setup and Configuration
# =========================
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import sys
from datetime import datetime
import os
import json

# Global log file variable
log_file = None

# Create a custom print function that writes to both console and file
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    global log_file
    if log_file is not None:
        print(*args, **kwargs, file=log_file)
        log_file.flush()  # Ensure immediate writing

# Note: Log file will be created in the main execution block
print("SOLIDER model and trainer initialized successfully!")
print("Training will start when script is run directly.")

eval_freq = 10  # Evaluation frequency

def update_training_plots(epochs, train_losses, fidi_losses, ce_losses, semantic_losses, 
                         eval_epochs, rank1s, maps, save_dir="training_plots"):
    """Update the same training plot files every epoch."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Total Loss', linewidth=2)
    plt.plot(epochs, fidi_losses, label='FIDI Loss', linewidth=2)
    plt.plot(epochs, ce_losses, label='CE Loss', linewidth=2)
    plt.plot(epochs, semantic_losses, label='Semantic Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Losses - Current Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to fixed filename (overwrites previous version)
    loss_filename = f"{save_dir}/training_losses.png"
    plt.savefig(loss_filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create accuracy plot (only if we have evaluation data)
    if eval_epochs and rank1s and maps:
        plt.figure(figsize=(10, 6))
        plt.plot(eval_epochs, rank1s, label='Rank-1 Accuracy', linewidth=2, marker='o')
        plt.plot(eval_epochs, maps, label='mAP', linewidth=2, marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title(f'Validation Performance - Current Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to fixed filename (overwrites previous version)
        acc_filename = f"{save_dir}/validation_accuracy.png"
        plt.savefig(acc_filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return loss_filename, acc_filename
    else:
        return loss_filename, None




# =========================
# Main Execution Block
# =========================
if __name__ == "__main__":
    print("=" * 80)
    print("SOLIDER Person Re-ID Training Script")
    print("=" * 80)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA devices: {torch.cuda.device_count()}")
        
        # Show GPU information for dual setup
        if torch.cuda.device_count() >= 2:
            print(f"✓ GPU 0: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU 1: {torch.cuda.get_device_name(1)}")
            print(f"✓ Memory GPU 0: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"✓ Memory GPU 1: {torch.cuda.get_device_properties(1).total_memory / 1e9:.1f} GB")
            print(f"✓ Using DataParallel with devices: {device}")
        else:
            print(f"✓ Single GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available, using CPU")
    
    # Check dataset paths
    print(f"\n📁 Dataset paths:")
    print(f"   Train: {train_dir}")
    print(f"   Query: {query_dir}")
    print(f"   Gallery: {gallery_dir}")
    
    # Verify datasets exist
    if not os.path.exists(train_dir):
        print(f"❌ Train directory not found: {train_dir}")
        exit(1)
    if not os.path.exists(query_dir):
        print(f"❌ Query directory not found: {query_dir}")
        exit(1)
    if not os.path.exists(gallery_dir):
        print(f"❌ Gallery directory not found: {gallery_dir}")
        exit(1)
    print("✓ All dataset directories found")
    
    # Print configuration
    print(f"\n⚙️  Configuration:")
    if isinstance(device, (list, tuple)) and len(device) >= 2:
        print(f"   Effective batch size: {effective_batch_size} (P={P}, K={K}, GPUs={len(device)})")
    else:
        print(f"   Batch size: {P*K} (P={P}, K={K})")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   Device: {device}")
    print(f"   Workers: {num_workers}, Prefetch: {prefetch_factor}")
    print(f"   FIDI params: α={alpha}, β={beta}")
    print(f"   Image size: {image_height}x{image_width}")
    print(f"   Number of classes: {num_classes}")
    
    # Create necessary directories
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("training_plots", exist_ok=True)
    
    # Create a more descriptive log filename with training parameters
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"logs/solider_training_P{P}K{K}_epochs{num_epochs}_{timestamp}.log"
    log_file = open(log_filename, 'w')
    
    # Log training start
    log_print(f"Training started at: {datetime.now()}")
    log_print(f"Using device: {device}")
    log_print(f"FIDI parameters: alpha={alpha}, beta={beta}")
    log_print(f"PK sampling: P={P}, K={K}, batch_size={P*K}")
    log_print(f"Number of classes: {num_classes}")
    log_print("="*80)
    
    print(f"\n🚀 Starting training...")
    print("=" * 80)
    
    try:
        # Start training automatically
        train_losses = []
        fidi_losses = []
        ce_losses = []
        semantic_losses = []
        epochs = []
        rank1s = []
        maps = []
        eval_epochs = []

        for epoch in range(num_epochs):
            log_print(f'\nEpoch {epoch+1}/{num_epochs}')
            log_print('-' * 50)
            
            # Log epoch start
            log_print(f"Starting epoch {epoch+1}/{num_epochs}")
            
            avg_loss, avg_fidi_loss, avg_ce_loss, avg_semantic_loss, batch_losses, batch_fidi_losses, batch_ce_losses, batch_semantic_losses = trainer.train_epoch(train_loader, epoch, num_epochs)
            
            # Get current loss weights for this epoch
            fidi_weight, cls_weight = trainer.get_loss_weights(epoch, num_epochs)
            
            # Log training results with weights
            log_print(f"Epoch {epoch+1} Training Results:")
            log_print(f"  - Total Loss: {avg_loss:.6f}")
            log_print(f"  - FIDI Loss: {avg_fidi_loss:.6f} × {fidi_weight:.2f}")
            log_print(f"  - CE Loss: {avg_ce_loss:.6f} × {cls_weight:.2f}")
            log_print(f"  - Semantic Loss: {avg_semantic_loss:.6f} × {trainer.semantic_weight:.2f}")
            
            train_losses.append(avg_loss)
            fidi_losses.append(avg_fidi_loss)
            ce_losses.append(avg_ce_loss)
            semantic_losses.append(avg_semantic_loss)
            epochs.append(epoch + 1)

            # Evaluate and collect accuracy/mAP
            if (epoch + 1) % eval_freq == 0 or (epoch + 1) == num_epochs:
                log_print("Evaluating...")
                cmc, mAP = trainer.evaluate(query_loader, gallery_loader)
                rank1 = float(cmc[0].item())
                rank1s.append(rank1)
                maps.append(float(mAP))
                eval_epochs.append(epoch + 1)
                log_print(f'Rank-1: {rank1:.4f}, mAP: {mAP:.4f}')
                
                # Log evaluation results
                log_print(f"Epoch {epoch+1} Evaluation Results:")
                log_print(f"  - Rank-1 Accuracy: {rank1:.4f}")
                log_print(f"  - mAP: {mAP:.4f}")

            # Update training plots (overwrites the same files)
            loss_file, acc_file = update_training_plots(
                epochs, train_losses, fidi_losses, ce_losses, semantic_losses,
                eval_epochs, rank1s, maps
            )
            log_print(f"Training plots updated: {loss_file}")
            if acc_file:
                log_print(f"Accuracy plot updated: {acc_file}")
            
            # Log learning rate
            current_lr = trainer.optimizer.param_groups[0]['lr']
            log_print(f"Current Learning Rate: {current_lr:.6f}")

            trainer.scheduler.step()

            # Save models every 5 epochs
            if (epoch + 1) % 5 == 0:
                log_print(f"\nSaving models at epoch {epoch+1}...")
                
                # Create weights directory if it doesn't exist
                os.makedirs("weights", exist_ok=True)
                
                # Save PyTorch state dict (.pth file)
                pth_path = f"weights/checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict()
                }, pth_path)
                log_print(f"PyTorch model saved: {pth_path}")
                
                # Save TorchScript model (.pt file) - also in weights folder
                model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                model_to_save.eval()
                
                # Use tracing instead of scripting for better compatibility
                try:
                    # Create dummy input for tracing
                    dummy_input = torch.randn(1, 3, image_height, image_width, device=next(model_to_save.parameters()).device)
                    traced = torch.jit.trace(model_to_save, dummy_input)
                    script_path = f"weights/checkpoint_epoch_{epoch+1}.pt"
                    traced.save(script_path)
                    log_print(f"TorchScript model saved: {script_path}")
                except Exception as e:
                    log_print(f"Warning: Failed to save TorchScript model: {e}")
                    # Continue without TorchScript export
                    script_path = None
                    
                log_print(f"Model saving completed for epoch {epoch+1}")

        # Close the log file at the end
        log_file.close()

        log_print("Training completed successfully!")
        
        # Log final summary
        log_print(f"\n" + "="*80)
        log_print("FINAL TRAINING SUMMARY")
        log_print("="*80)
        log_print(f"Total epochs completed: {num_epochs}")
        log_print(f"Final Rank-1 Accuracy: {rank1s[-1] if rank1s else 0.0:.4f}")
        log_print(f"Final mAP: {maps[-1] if maps else 0.0:.4f}")
        log_print(f"Best Rank-1 Accuracy: {max(rank1s) if rank1s else 0.0:.4f}")
        log_print(f"Best mAP: {max(maps) if maps else 0.0:.4f}")
        log_print(f"Training completed at: {datetime.now()}")
        log_print("="*80)
        
        # Create final summary plots
        plt.figure(figsize=(15, 10))
        
        # Loss subplot
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, label='Total Loss', linewidth=2)
        plt.plot(epochs, fidi_losses, label='FIDI Loss', linewidth=2)
        plt.plot(epochs, ce_losses, label='CE Loss', linewidth=2)
        plt.plot(epochs, semantic_losses, label='Semantic Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses - Complete Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy subplot
        if eval_epochs and rank1s and maps:
            plt.subplot(2, 2, 2)
            plt.plot(eval_epochs, rank1s, label='Rank-1 Accuracy', linewidth=2, marker='o')
            plt.plot(eval_epochs, maps, label='mAP', linewidth=2, marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Validation Performance - Complete Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Loss comparison subplot
        plt.subplot(2, 2, 3)
        plt.plot(epochs, fidi_losses, label='FIDI Loss', linewidth=2)
        plt.plot(epochs, ce_losses, label='CE Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('FIDI vs CE Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Final metrics subplot
        if eval_epochs and rank1s and maps:
            plt.subplot(2, 2, 4)
            plt.bar(['Rank-1', 'mAP'], [rank1s[-1], maps[-1]], color=['blue', 'orange'])
            plt.ylabel('Score')
            plt.title('Final Performance Metrics')
            plt.ylim(0, 1)
            for i, v in enumerate([rank1s[-1], maps[-1]]):
                plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        final_summary_plot = "training_plots/final_training_summary.png"
        plt.savefig(final_summary_plot, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save final results
        final_results = {
            'final_rank1': rank1s[-1] if rank1s else 0.0,
            'final_map': maps[-1] if maps else 0.0,
            'best_rank1': max(rank1s) if rank1s else 0.0,
            'best_map': max(maps) if maps else 0.0,
            'total_epochs': num_epochs,
            'training_completed': True,
            'final_summary_plot': final_summary_plot
        }
        
        with open('final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Final Results:")
        print(f"   Final Rank-1: {final_results['final_rank1']:.4f}")
        print(f"   Final mAP: {final_results['final_map']:.4f}")
        print(f"   Best Rank-1: {final_results['best_rank1']:.4f}")
        print(f"   Best mAP: {final_results['best_map']:.4f}")
        print(f"📁 Models saved in: weights/")
        print(f"📝 Log saved as: {log_filename}")
        print(f"📊 Results saved as: final_results.json")
        print(f"📈 Training plots saved in: training_plots/")
        print(f"📊 Final summary plot: {final_summary_plot}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        log_print("Training interrupted by user")
        log_file.close()
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        log_print(f"Training failed with error: {str(e)}")
        log_file.close()
        raise e



#!/usr/bin/env python
# coding: utf-8

"""
SOLIDER: Semantic-Oriented Learning with Identity-aware Discriminative Embeddings for Re-identification.

Key properties:
- ResNet-50 backbone (torchvision) up to layer4, global average pooling → 2048-d embedding
- Three losses and separate heads:
  1) FIDI loss head (metric head with BN + L2 normalize)
  2) CE/ID head (BNNeck + linear classifier)
  3) Semantic head (1x1 conv head over spatial feature map)
- Dynamic loss weighting using DWA (Dynamic Weight Averaging). Weights sum to 1.0
- No external project imports; everything is defined here

This file intentionally avoids complex pipelines and stages; it is designed to be
easy to read, adapt, and reuse.
"""

import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FIDILoss(nn.Module):
    """
    Fine-grained Difference-aware (FIDI) Pairwise Loss

    Implements the α-divergence formulation from the paper:
    L_fidi = D(U||K) + D(K||U)

    Where:
    - U = exp(-β * d(zi, zj)) : learned probability distribution
    - K = binary ground truth matrix (1 if same identity, 0 otherwise)
    - D(P||Q) = Σ p_ij * log(α * p_ij / ((α-1) * p_ij + q_ij)) : α-divergence
    """

    def __init__(self, alpha: float = 1.05, beta: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-8

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        distances = self._compute_pairwise_distances(features)
        labels = labels.view(-1, 1)
        k_matrix = (labels == labels.T).float()
        u_matrix = torch.exp(-self.beta * distances)
        d_u_k = self._compute_alpha_divergence(u_matrix, k_matrix)
        d_k_u = self._compute_alpha_divergence(k_matrix, u_matrix)
        return d_u_k + d_k_u

    def _compute_pairwise_distances(self, features: torch.Tensor) -> torch.Tensor:
        # Euclidean distance with numerical stability
        squared = torch.sum(features ** 2, dim=1, keepdim=True)
        distances = squared + squared.T - 2.0 * (features @ features.T)
        distances = torch.clamp(distances, min=0.0)
        return torch.sqrt(distances + self.eps)

    def _compute_alpha_divergence(self, p_matrix: torch.Tensor, q_matrix: torch.Tensor) -> torch.Tensor:
        p_matrix = torch.clamp(p_matrix, min=self.eps, max=1.0 - self.eps)
        q_matrix = torch.clamp(q_matrix, min=self.eps, max=1.0 - self.eps)
        denominator = (self.alpha - 1.0) * p_matrix + q_matrix
        denominator = torch.clamp(denominator, min=self.eps)
        fraction = (self.alpha * p_matrix) / denominator
        fraction = torch.clamp(fraction, min=self.eps)
        alpha_div = p_matrix * torch.log(fraction)
        # Exclude diagonal
        mask = ~torch.eye(p_matrix.size(0), dtype=torch.bool, device=p_matrix.device)
        return alpha_div[mask].mean()


class DynamicWeightAveraging:
    """
    Dynamic Weight Averaging (DWA) for multi-task learning.
    - Uses relative rate of loss change to assign weights.
    - Here adapted to 3 tasks (FIDI, CE, Semantic) with weights that sum to 1.0.
    - Reference: End-to-End Multi-Task Learning with Attention (Liu et al., CVPR 2019)
    """

    def __init__(self, num_tasks: int = 3, temperature: float = 2.0) -> None:
        assert num_tasks == 3, "This implementation expects exactly 3 tasks (FIDI, CE, Semantic)."
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.history: List[List[float]] = [[] for _ in range(num_tasks)]

    def update(self, task_losses: List[float]) -> None:
        for idx, loss_value in enumerate(task_losses):
            self.history[idx].append(float(loss_value))
            # Keep short history for stability
            if len(self.history[idx]) > 50:
                self.history[idx].pop(0)

    def get_weights(self) -> Tuple[float, float, float]:
        # If not enough history, use equal weighting
        if any(len(h) < 2 for h in self.history):
            return (1.0 / self.num_tasks, 1.0 / self.num_tasks, 1.0 / self.num_tasks)

        ratios = []
        for h in self.history:
            ratios.append(h[-1] / (h[-2] + 1e-8))

        # Softmax over ratios/temperature → weights
        exp_values = [math.exp(r / self.temperature) for r in ratios]
        denom = sum(exp_values) + 1e-8
        weights = [v / denom for v in exp_values]
        # Normalize again for numerical safety
        total = sum(weights) + 1e-8
        weights = [w / total for w in weights]
        return float(weights[0]), float(weights[1]), float(weights[2])


class SOLIDERPersonReIDModel(nn.Module):
    """
    SOLIDER Person Re-ID model with improved stability.
    
    Key components:
    1. ResNet50 backbone with multi-scale features
    2. Multi-scale feature fusion with attention
    3. Spatial semantic clustering for body parts
    4. FIDI loss for fine-grained metric learning
    """
    def __init__(self, num_classes, feature_dim=2048, num_semantic_parts=3):
        super(SOLIDERPersonReIDModel, self).__init__()
        
        # Model parameters
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_semantic_parts = num_semantic_parts
        self.background_label = num_semantic_parts  # Last index is background
        
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
        
        # Initialize parameters
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out')
        nn.init.constant_(self.bn_neck.weight, 1)
        nn.init.constant_(self.bn_neck.bias, 0)

    @staticmethod
    def _build_resnet50(use_imagenet_weights: bool) -> models.ResNet:
        # Torchvision API changed; support both styles
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if use_imagenet_weights else None
            backbone = models.resnet50(weights=weights)
        except Exception:
            backbone = models.resnet50(pretrained=use_imagenet_weights)
        return backbone

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.classifier_id.weight, mode='fan_out')
        # BN layers are already initialized by torchvision; keep defaults

    def _foreground_mask(self, features: torch.Tensor) -> torch.Tensor:
        # features: B x C x H x W
        magnitude = torch.norm(features, p=2, dim=1)  # B x H x W
        median = magnitude.view(magnitude.size(0), -1).median(dim=1).values.view(-1, 1, 1)
        mean = magnitude.mean(dim=(1, 2), keepdim=True)
        threshold = 0.5 * (median + mean)
        return magnitude >= threshold  # bool mask

    def _spatial_semantic_labels(self, features: torch.Tensor) -> torch.Tensor:
        # Create vertical stripes: upper(0), middle(1), lower(2)
        _, _, H, W = features.shape
        device = features.device
        y_coords = torch.linspace(0, 1, steps=H, device=device).view(H, 1).expand(H, W)
        labels = torch.full((H, W), self.background_label, dtype=torch.long, device=device)
        # thirds
        upper_mask = (y_coords < 1.0 / 3.0)
        middle_mask = (y_coords >= 1.0 / 3.0) & (y_coords < 2.0 / 3.0)
        lower_mask = (y_coords >= 2.0 / 3.0)
        labels[upper_mask] = 0
        labels[middle_mask] = 1
        labels[lower_mask] = 2 if self.num_semantic_parts >= 3 else 1
        return labels

    def _generate_pseudo_labels(self, semantic_features: torch.Tensor) -> torch.Tensor:
        # Combine foreground mask with spatial stripes
        # semantic_features: B x 1024 x H x W
        B, _, H, W = semantic_features.shape
        fg_mask = self._foreground_mask(semantic_features)  # B x H x W (bool)
        spatial = self._spatial_semantic_labels(semantic_features)  # H x W
        pseudo = torch.full((B, H, W), self.background_label, dtype=torch.long, device=semantic_features.device)
        spatial_expanded = spatial.unsqueeze(0).expand(B, -1, -1)
        pseudo = torch.where(fg_mask, spatial_expanded, pseudo)
        return pseudo

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
        
        # Multi-scale fusion
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

    @staticmethod
    def semantic_loss_from_logits(
        semantic_logits: torch.Tensor,
        semantic_pseudo_labels: torch.Tensor,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        # Cross-entropy over all pixels
        B, C, H, W = semantic_logits.shape
        logits_flat = semantic_logits.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = semantic_pseudo_labels.reshape(-1)
        return F.cross_entropy(logits_flat, labels_flat, label_smoothing=label_smoothing)


class SpatialSemanticClustering(nn.Module):
    """
    Improved spatial-level semantic clustering for CNN feature maps.
    """
    def __init__(self, feature_dim, num_semantic_parts=3):
        super(SpatialSemanticClustering, self).__init__()
        self.feature_dim = feature_dim
        self.num_semantic_parts = num_semantic_parts
        
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

class SimpleTrainer:
    """
    Minimal trainer for SOLIDER with three losses:
    - FIDI loss (metric)
    - CE loss (ID classification)
    - Semantic loss (spatial classification with pseudo-labels)

    Uses Dynamic Weight Averaging (DWA) so the three weights sum to 1.0.
    """

    def __init__(
        self,
        model: SOLIDERPersonReIDModel,
        num_classes: int,
        device: str | torch.device | List[int] = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 3e-4,
        weight_decay: float = 5e-4,
        fidi_alpha: float = 1.05,
        fidi_beta: float = 0.5,
    ) -> None:
        # Device and optional DataParallel
        if isinstance(device, (list, tuple)):
            assert torch.cuda.is_available(), "CUDA must be available for multi-GPU."
            self.device = torch.device("cuda")
            model = model.cuda()
            self.model = nn.DataParallel(model, device_ids=list(device))
        else:
            self.device = torch.device(device)
            self.model = model.to(self.device)
        self.num_classes = num_classes
        self.fidi_loss_fn = FIDILoss(alpha=fidi_alpha, beta=fidi_beta)
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 80], gamma=0.1)
        self.dwa = DynamicWeightAveraging(num_tasks=3, temperature=2.0)

    def train_one_epoch(self, dataloader: torch.utils.data.DataLoader, epoch_index: int) -> Dict[str, float]:
        self.model.train()
        total_loss_value = 0.0
        total_fidi_value = 0.0
        total_ce_value = 0.0
        total_sem_value = 0.0
        num_batches = 0

        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            features, logits, semantic_output = self.model(images, return_semantic_loss=True)

            # Losses
            fidi_value = self.fidi_loss_fn(features, labels)
            ce_value = self.ce_loss_fn(logits, labels)
            sem_value = semantic_output['semantic_loss']

            # Update DWA and fetch weights that sum to 1.0
            self.dwa.update([fidi_value.item(), ce_value.item(), sem_value.item()])
            w_fidi, w_ce, w_sem = self.dwa.get_weights()

            loss = w_fidi * fidi_value + w_ce * ce_value + w_sem * sem_value

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss_value += float(loss.item())
            total_fidi_value += float(fidi_value.item())
            total_ce_value += float(ce_value.item())
            total_sem_value += float(sem_value.item())
            num_batches += 1

        self.scheduler.step()

        if num_batches == 0:
            num_batches = 1

        return {
            "loss": total_loss_value / num_batches,
            "fidi": total_fidi_value / num_batches,
            "ce": total_ce_value / num_batches,
            "semantic": total_sem_value / num_batches,
        }

    @torch.no_grad()
    def extract_features(self, dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        all_features: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images = images.to(self.device, non_blocking=True)
            features, _ = self.model(images)
            all_features.append(features.cpu())
            all_labels.append(labels)
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    @torch.no_grad()
    def evaluate(
        self,
        query_dataloader: torch.utils.data.DataLoader,
        gallery_dataloader: torch.utils.data.DataLoader,
        max_rank: int = 50,
    ) -> Tuple[torch.Tensor, float]:
        self.model.eval()

        # Extract and normalize features
        q_feat, q_labels = self.extract_features(query_dataloader)
        g_feat, g_labels = self.extract_features(gallery_dataloader)
        q_feat = F.normalize(q_feat, p=2, dim=1)
        g_feat = F.normalize(g_feat, p=2, dim=1)

        # Compute distance matrix
        dist_matrix = torch.cdist(q_feat, g_feat, p=2)

        # For CMC/mAP, we also need camera ids; reuse label tensors as placeholders if cams are not provided
        # Here we expect test dataloaders to yield (image, label, cam_id). If not, set cam_id to zeros.
        # Try to pull cam_ids from underlying dataset if attribute exists
        def get_cam_ids(loader):
            if hasattr(loader.dataset, 'samples') and len(loader.dataset.samples) > 0 and len(loader.dataset.samples[0]) >= 3:
                # PersonReIDTestDataset stores (path,label,cam_id)
                return torch.tensor([s[2] for s in loader.dataset.samples], dtype=torch.long)
            # Fallback: zeros
            return torch.zeros(len(loader.dataset), dtype=torch.long)

        q_cam = get_cam_ids(query_dataloader)
        g_cam = get_cam_ids(gallery_dataloader)

        return self.compute_cmc_map(dist_matrix, q_labels.numpy(), g_labels.numpy(), q_cam.numpy(), g_cam.numpy(), max_rank=max_rank)

    @staticmethod
    def compute_cmc_map(
        dist_matrix: torch.Tensor,
        query_labels,
        gallery_labels,
        query_cam_ids,
        gallery_cam_ids,
        max_rank: int = 50,
    ) -> Tuple[torch.Tensor, float]:
        num_q, num_g = dist_matrix.shape
        if num_g < max_rank:
            max_rank = num_g

        indices = torch.argsort(dist_matrix, dim=1)
        matches = (torch.tensor(gallery_labels)[indices] == torch.tensor(query_labels).view(-1, 1))

        all_cmc = []
        all_AP = []
        num_valid_q = 0

        for q_idx in range(num_q):
            q_pid = query_labels[q_idx]
            q_camid = query_cam_ids[q_idx]
            order = indices[q_idx]

            remove = torch.tensor([(gallery_labels[i] == q_pid) & (gallery_cam_ids[i] == q_camid) for i in order])
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
        mAP = float(sum(all_AP) / len(all_AP))
        return all_cmc, mAP


# Optional quick usage example (no datasets are created here):
# if __name__ == "__main__":
#     num_classes = 100
#     model = VisNet(num_classes=num_classes)
#     trainer = SimpleTrainer(model=model, num_classes=num_classes)
#     # Prepare your DataLoader that yields (images, labels)
#     # stats = trainer.train_one_epoch(train_loader, epoch_index=0)
#     # print(stats)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# CRITICAL FIX: Set matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

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
# Removed external project imports; VisNet is self-contained in this file


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
    def __init__(self, data_source, P=4, K=8):
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

# Removed large SOLIDER-specific clustering block

# Removed SemanticController (not needed for VisNet)

# Removed SOLIDER-specific CNN block (not used in VisNet)

# Removed MultiScaleFeatureFusion (VisNet uses plain ResNet-50 backbone)

# Removed SOLIDER model (replaced by VisNet)

def create_solider_model(num_classes: int) -> SOLIDERPersonReIDModel:
    """Factory function to create VisNet model."""
    return SOLIDERPersonReIDModel(num_classes=num_classes)


"""Removed residual SOLIDER trainer code."""


# In[ ]:


class FIDILoss(nn.Module):
    """
    Fine-grained Difference-aware (FIDI) Pairwise Loss
    
    Implements the exact formulation from the paper:
    L_fidi = D(U||K) + D(K||U)
    
    Where:
    - U = exp(-β * d(zi, zj)) : learned probability distribution  
    - K = binary ground truth matrix (1 if same identity, 0 otherwise)
    - D(P||Q) = Σ p_ij * log(α * p_ij / ((α-1) * p_ij + q_ij)) : α-divergence
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
        
        # Compute D(U||K) + D(K||U) using α-divergence as per paper
        d_u_k = self.compute_alpha_divergence(u_matrix, k_matrix)
        d_k_u = self.compute_alpha_divergence(k_matrix, u_matrix)
        
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
    
    def compute_alpha_divergence(self, p_matrix, q_matrix):
        """
        Compute α-divergence D(P||Q) following Equation (5) from the paper:
        D(P||Q) = Σ p_ij * log(α * p_ij / ((α-1) * p_ij + q_ij))
        
        This is the exact α-divergence formulation from the FIDI paper.
        For D(U||K): p=u_matrix, q=k_matrix  
        For D(K||U): p=k_matrix, q=u_matrix
        """
        # Clamp p_matrix to avoid numerical issues, but be careful with k_matrix
        p_matrix = torch.clamp(p_matrix, min=self.eps, max=1-self.eps)
        
        # k_matrix is binary {0,1}, but we need to handle q_matrix generically
        # since this function is used for both D(U||K) and D(K||U)
        q_matrix = torch.clamp(q_matrix, min=self.eps, max=1-self.eps)
        
        # Compute the denominator: (α-1) * p_ij + q_ij
        denominator = (self.alpha - 1) * p_matrix + q_matrix
        denominator = torch.clamp(denominator, min=self.eps)
        
        # Compute the fraction: α * p_ij / denominator
        numerator = self.alpha * p_matrix
        fraction = numerator / denominator
        fraction = torch.clamp(fraction, min=self.eps)
        
        # Compute α-divergence: p_ij * log(fraction)
        alpha_div = p_matrix * torch.log(fraction)
        
        # Exclude diagonal elements (self-comparisons) and compute mean
        mask = ~torch.eye(p_matrix.size(0), dtype=torch.bool, device=p_matrix.device)
        alpha_div = alpha_div[mask].mean()
        
        return alpha_div


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
            # Use generic cuda device for DataParallel
            self.device = torch.device("cuda")
            # Move model to cuda first, then wrap in DataParallel
            model = model.cuda()
            self.model = nn.DataParallel(model, device_ids=device)
        else:
            self.device = torch.device(device)
            self.model = model.to(self.device)
        
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
            
        else:  # 'original'
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
              num_epochs=None, eval_freq=10):
        """
        Train the model with two stages:
        1. FIDI stage (epochs 0-49): Only FIDI and CE loss
        2. SOLIDER stage (epochs 50-99): FIDI, CE, and semantic losses
        """
        # Use class-defined total epochs if num_epochs is not provided
        if num_epochs is None:
            num_epochs = self.total_epochs
        
        print(f"Starting training with single-stage FIDI+CE+Semantic...")
        
        for epoch in range(num_epochs):
            # Stop training if we've reached total epochs
            if epoch >= self.total_epochs:
                print(f"Reached total epochs limit ({self.total_epochs}). Stopping training.")
                break
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
# PK Sampling parameters
P = 8  # Number of persons per batch
K = 12   # Number of images per person
batch_size = P * K  # This will be 64 for optimal PK sampling

num_epochs = 100  # Total epochs
device = [0, 1] if torch.cuda.device_count() > 1 else ('cuda' if torch.cuda.is_available() else 'cpu')
alpha = 1.05
beta = 0.5  # Increased for better FIDI sensitivity
lr = 3.5e-4  # Reduced for better convergence
weight_decay = 5e-4
num_workers = 8
prefetch_factor = 4
image_height = 256
image_width = 128
train_dir = os.path.join('Dataset', 'train')
query_dir = os.path.join('Dataset', 'query')
gallery_dir = os.path.join('Dataset', 'gallery')

# train_dir = '/home/anns/Downloads/dataSet/train'
# query_dir = '/home/anns/Downloads/dataSet/query'
# gallery_dir = '/home/anns/Downloads/dataSet/gallery'


# In[ ]:


# 7. Data Transforms & DataLoaders

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

# Always use PK sampling
train_loader = DataLoader(
    train_dataset,
    sampler=pk_sampler,
    batch_size=P*K,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=prefetch_factor,
    drop_last=True
)

query_loader = DataLoader(
    query_dataset,
    batch_size=P*K,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=prefetch_factor
)

gallery_loader = DataLoader(
    gallery_dataset,
    batch_size=P*K,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=prefetch_factor
)

print(f"✓ Train samples: {len(train_dataset)}, PIDs: {num_classes}")
print(f"✓ DataLoaders ready: train {len(train_loader)} batches, "
      f"query {len(query_loader)}, gallery {len(gallery_loader)}")


# In[ ]:


# 8. SOLIDER Model & Trainer Initialization

# Model
solider_model = SOLIDERPersonReIDModel(num_classes=num_classes)

# Export SOLIDER core to ONNX for Netron visualization
def export_solider_to_onnx(model: SOLIDERPersonReIDModel, export_path: str) -> None:
    model.eval()
    export_device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, image_height, image_width, device=export_device)

    class SOLIDERCore(nn.Module):
        def __init__(self, model_: SOLIDERPersonReIDModel):
            super().__init__()
            # Keep full model for visualization
            self.stage0 = model_.stage0
            self.stage1 = model_.stage1
            self.stage2 = model_.stage2
            self.stage3 = model_.stage3
            self.stage4 = model_.stage4
            self.multi_scale_fusion = model_.multi_scale_fusion
            self.semantic_clustering = model_.semantic_clustering
            self.global_pool = model_.global_pool
            self.bn_neck = model_.bn_neck
            self.classifier = model_.classifier
        
        def forward(self, x):
            # Full forward pass for visualization
            x0 = self.stage0(x)
            x1 = self.stage1(x0)
            x2 = self.stage2(x1)
            x3 = self.stage3(x2)
            x4 = self.stage4(x3)
            
            # Multi-scale fusion
            fused_features = self.multi_scale_fusion([x1, x2, x3, x4])
            
            # Global pooling and feature extraction
            pooled_features = self.global_pool(fused_features)
            pooled_features = pooled_features.view(pooled_features.size(0), -1)
            features = self.bn_neck(pooled_features)
            logits = self.classifier(features)
            
            # For visualization only - no loss computation
            semantic_features = self.semantic_clustering(fused_features, None)
            
            return features, logits, semantic_features['semantic_logits']

    core = SOLIDERCore(model).to(export_device)
    torch.onnx.export(
        core,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=12,  # Using opset 12 for cross_entropy_loss support
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['features', 'logits', 'semantic_features'],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'features': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'semantic_features': {0: 'batch_size'}
        },
        verbose=False,
    )

os.makedirs("onnx_models", exist_ok=True)
solider_model = solider_model.to(device if isinstance(device, str) else (f"cuda:{device[0]}" if torch.cuda.is_available() and isinstance(device, (list, tuple)) else 'cpu'))
export_path = os.path.join("onnx_models", "restored.onnx")
try:
    export_solider_to_onnx(solider_model, export_path)
    print(f"✓ SOLIDER core exported to ONNX: {export_path}")
except Exception as ex:
    print(f"Warning: Failed to export SOLIDER ONNX: {ex}")

# Trainer
trainer = SimpleTrainer(
    model=solider_model,
    num_classes=num_classes,
    device=device,
    learning_rate=lr,
    weight_decay=weight_decay,
    fidi_alpha=alpha,
    fidi_beta=beta,
)

print(f"✓ SOLIDER model with {num_classes} classes")
print(f"Using device(s): {device}")


# In[9]:


# =========================
# (Removed) ONNX export and SOLIDER-specific utilities
# =========================
import torch
import torch.nn as nn
import os

# (All removed)



# In[ ]:


# =========================
# 9. Training Setup and Configuration
# =========================
# matplotlib already imported with Agg backend at top of file
import sys
from datetime import datetime
import os
import json
import time

eval_freq = 10


# =========================
# Main Execution Block
# =========================
if __name__ == "__main__":
    print("=" * 80)
    print("SOLIDER Person Re-ID Training Script")
    print("=" * 80)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA devices: {torch.cuda.device_count()}")
    else:
        print("⚠ CUDA not available, using CPU")
    
    print(f"\n📁 Dataset paths:")
    print(f"   Train: {train_dir}")
    print(f"   Query: {query_dir}")
    print(f"   Gallery: {gallery_dir}")
    
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
    
    print(f"\n⚙️  Configuration:")
    print(f"   Batch size: {P*K} (P={P}, K={K})")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   Device: {device}")
    print(f"   FIDI params: α={alpha}, β={beta}")
    print(f"   Image size: {image_height}x{image_width}")
    print(f"   Number of classes: {num_classes}")
    
    os.makedirs("weights", exist_ok=True)
    print(f"\n🚀 Starting training...")
    print("=" * 80)
    
    # Track histories for plotting
    train_losses, fidi_losses, ce_losses, semantic_losses = [], [], [], []
    acc_epochs, rank1s, rank3s, rank5s, maps = [], [], [], [], []

    for epoch in range(num_epochs):
        stats = trainer.train_one_epoch(train_loader, epoch_index=epoch)
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {stats['loss']:.4f} | FIDI: {stats['fidi']:.4f} | CE: {stats['ce']:.4f} | Semantic: {stats['semantic']:.4f}"
        )

        # Update histories and loss plot every epoch
        train_losses.append(stats['loss'])
        fidi_losses.append(stats['fidi'])
        ce_losses.append(stats['ce'])
        semantic_losses.append(stats['semantic'])
        os.makedirs("training_plots", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Total Loss', linewidth=2)
        plt.plot(range(1, len(fidi_losses) + 1), fidi_losses, label='FIDI Loss', linewidth=2)
        plt.plot(range(1, len(ce_losses) + 1), ce_losses, label='CE Loss', linewidth=2)
        plt.plot(range(1, len(semantic_losses) + 1), semantic_losses, label='Semantic Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_plots/training_losses.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        if ((epoch + 1) % 25 == 0) or ((epoch + 1) == num_epochs):
            ckpt_path = os.path.join("weights", f"visnet_epoch_{epoch+1}.pth")
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

        # Evaluate every 10 epochs, update accuracy plot, and save JIT model
        if (epoch + 1) % 10 == 0:
            cmc, mAP = trainer.evaluate(query_loader, gallery_loader)
            r1 = float(cmc[0].item()) if cmc.numel() > 0 else 0.0
            r3 = float(cmc[2].item()) if cmc.numel() > 2 else 0.0
            r5 = float(cmc[4].item()) if cmc.numel() > 4 else 0.0
            print(f"Eval @ epoch {epoch+1}: Rank-1={r1:.4f}, Rank-3={r3:.4f}, Rank-5={r5:.4f}, mAP={mAP:.4f}")

            # histories for plotting
            acc_epochs.append(epoch + 1)
            rank1s.append(r1)
            rank3s.append(r3)
            rank5s.append(r5)
            maps.append(mAP)

            os.makedirs("training_plots", exist_ok=True)
            plt.figure(figsize=(12, 6))
            plt.plot(acc_epochs, rank1s, label='Rank-1', marker='o')
            plt.plot(acc_epochs, rank3s, label='Rank-3', marker='^')
            plt.plot(acc_epochs, rank5s, label='Rank-5', marker='v')
            plt.plot(acc_epochs, maps, label='mAP', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Validation Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_plots/validation_accuracy.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
            # Save state dict (.pth) and TorchScript (.pt) at evaluation time
        os.makedirs("weights", exist_ok=True)
        pth_eval_path = os.path.join("weights", f"visnet_epoch_{epoch+1}.pth")
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
            },
            pth_eval_path,
        )
        print(f"Saved checkpoint: {pth_eval_path}")

        # TorchScript export (inference-ready .pt)
        model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        model_to_save.eval()
        example_input = torch.randn(1, 3, image_height, image_width, device=next(model_to_save.parameters()).device)
        try:
            traced = torch.jit.trace(model_to_save, example_input)
            ts_path = os.path.join("weights", f"visnet_epoch_{epoch+1}.pt")
            traced.save(ts_path)
            print(f"Saved TorchScript model: {ts_path}")
        except Exception as ex:
            print(f"Warning: TorchScript export failed at epoch {epoch+1}: {ex}")
        
        print("=" * 80)
    print("Training finished.")



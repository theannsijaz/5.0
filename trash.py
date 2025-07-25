#!/usr/bin/env python
# coding: utf-8

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
import math

# =========================
# 1. Random Erasing Transform (Paper's Recommended Data Augmentation)
# =========================
class RandomErasing:
    """
    Random Erasing Data Augmentation as mentioned in the paper
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[1]
                return img

        return img

# =========================
# 2. PK Batch Sampler (Critical for FIDI Loss Performance)
# =========================
class PKSampler(Sampler):
    """
    PK Sampler: Sample P identities and K images per identity in each batch
    This is crucial for the FIDI loss to work optimally as mentioned in the paper
    """
    def __init__(self, dataset, batch_size, num_instances=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        # Group samples by identity
        self.index_pid = defaultdict(list)
        for index, (_, pid) in enumerate(dataset.samples):
            self.index_pid[pid].append(index)
        
        self.pids = list(self.index_pid.keys())
        
        # Estimate number of batches per epoch
        self.length = len(self.pids) // self.num_pids_per_batch * self.num_pids_per_batch

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        
        for pid in self.pids:
            idxs = self.index_pid[pid]
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            else:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=False)
            batch_idxs_dict[pid] = idxs

        avai_pids = list(batch_idxs_dict.keys())
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid]
                final_idxs.extend(batch_idxs)
                avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

# =========================
# 3. Enhanced Dataset Classes
# =========================
class PersonReIDTrainDataset(torch.utils.data.Dataset):
    """
    Enhanced dataset for training with better sample tracking
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
                if fname.lower().endswith('.jpg'):
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
    Dataset for query/gallery set
    """
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        self.samples = []  # List of (img_path, label, cam_id)
        self._prepare()

    def _prepare(self):
        for fname in os.listdir(self.dir_path):
            if fname.lower().endswith('.jpg'):
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

# =========================
# 4. Enhanced FIDI Loss (GPU Optimized)
# =========================
class FIDILoss(nn.Module):
    """
    GPU-optimized Fine-grained Difference-aware (FIDI) Pairwise Loss
    """
    def __init__(self, alpha=1.05, beta=0.5):
        super(FIDILoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-8

    def forward(self, features, labels):
        # Normalize features for better stability
        features = F.normalize(features, p=2, dim=1)
        
        # Compute pairwise distances efficiently
        dist_mat = torch.cdist(features, features, p=2)
        
        # Create label mask
        labels = labels.view(-1, 1)
        label_mask = (labels == labels.T).float()
        
        # Convert distances to probabilities using exponential function
        u_probs = torch.exp(-self.beta * dist_mat)
        
        # Compute symmetric KL divergence
        loss = self._compute_kl_divergence(u_probs, label_mask) + \
               self._compute_kl_divergence(label_mask, u_probs)
        
        return loss

    def _compute_kl_divergence(self, p, q):
        # Clamp probabilities to avoid numerical issues
        p = torch.clamp(p, min=self.eps, max=1-self.eps)
        q = torch.clamp(q, min=self.eps, max=1-self.eps)
        
        # Compute KL divergence according to paper's formula
        denominator = (self.alpha - 1) * p + q
        denominator = torch.clamp(denominator, min=self.eps)
        
        fraction = (self.alpha * p) / denominator
        fraction = torch.clamp(fraction, min=self.eps)
        
        kl_div = p * torch.log(fraction)
        
        # Exclude diagonal elements (self-comparison)
        mask = ~torch.eye(p.size(0), dtype=torch.bool, device=p.device)
        kl_div = kl_div[mask].mean()
        
        return kl_div

# =========================
# 5. Enhanced PersonReID Model
# =========================
class PersonReIDModel(nn.Module):
    """
    Person Re-identification model following paper's architecture
    """
    def __init__(self, num_classes, feature_dim=2048):
        super(PersonReIDModel, self).__init__()
        # ResNet50 backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Batch Normalization neck (critical component from paper)
        self.bn_neck = nn.BatchNorm1d(feature_dim)
        self.bn_neck.bias.requires_grad_(False)
        
        # Classification layer
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        
        self._init_params()

    def _init_params(self):
        # Initialize parameters as in the paper
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out')
        nn.init.constant_(self.bn_neck.weight, 1)
        nn.init.constant_(self.bn_neck.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Features for FIDI loss (before BN)
        global_feat = x
        
        # Features after BN neck
        feat = self.bn_neck(global_feat)
        
        # Classification logits
        logits = self.classifier(feat)
        
        if self.training:
            return global_feat, logits  # Return global features for FIDI loss
        else:
            return feat  # Return BN features for inference

# =========================
# 6. Enhanced Trainer with Paper's Exact Configuration
# =========================
class FIDITrainer:
    """
    Training framework following the paper's best practices exactly
    """
    def __init__(self, model, num_classes, device='cuda', 
                 alpha=1.05, beta=0.5, lr=3.5e-4, weight_decay=5e-4):
        # Multi-GPU support
        if isinstance(device, (list, tuple)):
            assert torch.cuda.is_available(), "CUDA must be available for multi-GPU."
            self.device = torch.device(f"cuda:{device[0]}")
            model = model.to(self.device)
            self.model = nn.DataParallel(model, device_ids=device)
        else:
            self.device = torch.device(device)
            self.model = model.to(self.device)
        
        self.num_classes = num_classes
        
        # Loss functions
        self.fidi_loss = FIDILoss(alpha=alpha, beta=beta).to(self.device)
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
        
        # Optimizer and scheduler (paper's configuration)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=40, gamma=0.1
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_fidi_loss = 0.0
        total_ce_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            global_feat, logits = self.model(images)
            
            # Compute losses
            fidi_loss = self.fidi_loss(global_feat, labels)
            ce_loss = self.ce_loss(logits, labels)
            
            # Total loss (paper uses equal weighting)
            loss = fidi_loss + ce_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_fidi_loss += fidi_loss.item()
            total_ce_loss += ce_loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}: Loss={loss.item():.4f}, '
                      f'FIDI={fidi_loss.item():.4f}, CE={ce_loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        avg_fidi_loss = total_fidi_loss / len(dataloader)
        avg_ce_loss = total_ce_loss / len(dataloader)
        
        return avg_loss, avg_fidi_loss, avg_ce_loss

    def evaluate(self, query_dataloader, gallery_dataloader):
        self.model.eval()
        
        # Extract query features
        query_features = []
        query_labels = []
        query_cam_ids = []
        
        with torch.no_grad():
            for images, labels, cam_ids in query_dataloader:
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)  # Get BN features for inference
                query_features.append(features.cpu())
                query_labels.extend(labels.numpy())
                query_cam_ids.extend(cam_ids.numpy())
        
        query_features = torch.cat(query_features, dim=0)
        query_features = F.normalize(query_features, p=2, dim=1)
        
        # Extract gallery features
        gallery_features = []
        gallery_labels = []
        gallery_cam_ids = []
        
        with torch.no_grad():
            for images, labels, cam_ids in gallery_dataloader:
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)  # Get BN features for inference
                gallery_features.append(features.cpu())
                gallery_labels.extend(labels.numpy())
                gallery_cam_ids.extend(cam_ids.numpy())
        
        gallery_features = torch.cat(gallery_features, dim=0)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)
        
        # Compute distance matrix
        dist_matrix = torch.cdist(query_features, gallery_features, p=2)
        
        # Compute CMC and mAP
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
            
            # Compute Average Precision
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
        print("Starting training with FIDI loss and PK sampling...")
        best_mAP = 0.0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            avg_loss, avg_fidi_loss, avg_ce_loss = self.train_epoch(train_dataloader)
            print(f'Train Loss: {avg_loss:.4f}, FIDI Loss: {avg_fidi_loss:.4f}, '
                  f'CE Loss: {avg_ce_loss:.4f}')
            
            self.scheduler.step()
            
            if (epoch + 1) % eval_freq == 0:
                print("Evaluating...")
                cmc, mAP = self.evaluate(query_dataloader, gallery_dataloader)
                print(f'Rank-1: {cmc[0]:.4f}, Rank-5: {cmc[4]:.4f}, '
                      f'Rank-10: {cmc[9]:.4f}, mAP: {mAP:.4f}')
                
                if mAP > best_mAP:
                    best_mAP = mAP
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'mAP': mAP,
                        'cmc': cmc,
                    }, 'best_fidi_model.pth')
                    print(f'New best mAP: {best_mAP:.4f}')
        
        print(f'\nTraining completed. Best mAP: {best_mAP:.4f}')

# =========================
# 7. Complete Training Setup with Paper's Best Configuration
# =========================
def setup_training():
    # Paper's exact configuration
    batch_size = 128
    num_instances = 4  # K in PK sampling (4 images per identity)
    num_epochs = 120
    device = [0, 1] if torch.cuda.device_count() > 1 else ('cuda' if torch.cuda.is_available() else 'cpu')
    alpha = 1.05
    beta = 0.5
    lr = 3.5e-4
    weight_decay = 5e-4
    num_workers = 8
    prefetch_factor = 4
    image_height = 256
    image_width = 128
    
    # Directory paths
    train_dir = os.path.join('Dataset', 'train')
    query_dir = os.path.join('Dataset', 'query')
    gallery_dir = os.path.join('Dataset', 'gallery')
    
    # Enhanced data transforms with Random Erasing
    train_transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5)  # Paper's recommended data augmentation
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Create datasets
    train_dataset = PersonReIDTrainDataset(train_dir, transform=train_transform)
    query_dataset = PersonReIDTestDataset(query_dir, transform=test_transform)
    gallery_dataset = PersonReIDTestDataset(gallery_dir, transform=test_transform)
    
    num_classes = len(train_dataset.label_map)
    print(f"Number of training identities: {num_classes}")
    print(f"Number of training images: {len(train_dataset)}")
    
    # Create PK sampler for optimal batch composition
    pk_sampler = PKSampler(train_dataset, batch_size, num_instances)
    
    # Create data loaders with PK sampling
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=pk_sampler,  # Use PK sampler instead of random sampling
        num_workers=num_workers, 
        pin_memory=True, 
        prefetch_factor=prefetch_factor
    )
    
    query_loader = DataLoader(
        query_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        prefetch_factor=prefetch_factor
    )
    
    gallery_loader = DataLoader(
        gallery_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        prefetch_factor=prefetch_factor
    )
    
    # Create model and trainer
    model = PersonReIDModel(num_classes=num_classes)
    trainer = FIDITrainer(
        model=model,
        num_classes=num_classes,
        device=device,
        alpha=alpha,
        beta=beta,
        lr=lr,
        weight_decay=weight_decay
    )
    
    print(f"Using device: {device}")
    print(f"FIDI parameters: alpha={alpha}, beta={beta}")
    print(f"Batch composition: {batch_size//num_instances} identities × {num_instances} images")
    
    return trainer, train_loader, query_loader, gallery_loader

# Run the complete training
if __name__ == "__main__":
    trainer, train_loader, query_loader, gallery_loader = setup_training()
    trainer.train(train_loader, query_loader, gallery_loader, num_epochs=120, eval_freq=10)
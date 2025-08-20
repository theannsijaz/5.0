"""
This script compares your exported JIT model (.pt) with several popular person re-id models
using the Torchreid library. It evaluates model size, FLOPs, parameter count, and performance
metrics (Rank-1, Rank-3, mAP) on the Market1501 dataset.

Instructions:
- Place your exported model as 'checkpoint_epoch_50.pt' in the same directory as this script.
- Install torchreid: pip install torchreid
"""

import os
import re
import torch
import torchreid
from torchreid.utils import compute_model_complexity
from torchreid import metrics
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.amp as amp  # Updated import for autocast
import torch.backends.cudnn as cudnn

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class Market1501Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # List all jpg files
        for filename in os.listdir(root_dir):
            if filename.endswith('.jpg'):
                # Parse pid and camid from filename (e.g., 0001_c1s1_001051_00.jpg or -1_c1s1_000401_03.jpg)
                pid, camid = self._parse_filename(filename)
                self.samples.append((os.path.join(root_dir, filename), pid, camid))
    
    def _parse_filename(self, filename):
        pattern = r'([\-\d]+)_c(\d+)s\d+_\d+'
        match = re.match(pattern, filename)
        if match:
            pid = int(match.group(1))
            camid = int(match.group(2))
            return pid, camid
        return -1, -1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, pid, camid = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(pid, dtype=torch.long), torch.tensor(camid, dtype=torch.long)

def get_model_size(model_path):
    try:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return f"{size_mb:.2f}"
    except Exception as e:
        return f"Error\n  Error: {str(e)}"

def _is_script_submodule_present(module: torch.nn.Module) -> bool:
    try:
        import torch.jit as jit
        if isinstance(module, (jit.ScriptModule, jit.RecursiveScriptModule)):
            return True
        for child in module.children():
            if _is_script_submodule_present(child):
                return True
        return False
    except Exception:
        return False

def get_flops_params(model, input_size=(1, 3, 256, 128)):
    try:
        if _is_script_submodule_present(model):
            try:
                param_count = sum(p.numel() for p in model.parameters())
                return "Unsupported for JIT", f"{param_count/1e6:.2f}M"
            except Exception as inner:
                return f"Error\n  Error: {str(inner)}", f"Error\n  Error: {str(inner)}"
        flops, params = compute_model_complexity(model, input_size=input_size)
        return f"{flops/1e9:.2f}B", f"{params/1e6:.2f}M"
    except Exception as e:
        return f"Error\n  Error: {str(e)}", f"Error\n  Error: {str(e)}"

def _l2_normalize(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features, p=2, dim=1)

@torch.no_grad()
def extract_features(model, data_loader, device, use_amp=True):
    model.eval()
    features = []
    pids = []
    camids = []
    
    for batch_idx, (imgs, batch_pids, batch_camids) in enumerate(data_loader):
        imgs = imgs.to(device)
        with amp.autocast(device_type='cuda', enabled=use_amp):  # Updated autocast
            outputs = model(imgs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            if isinstance(outputs, dict):
                outputs = outputs.get('features', outputs.get('embeddings', outputs.get('feat', outputs)))
            if outputs.dim() > 2:
                outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, (1, 1)).flatten(1)
            outputs = _l2_normalize(outputs)
        
        features.append(outputs.cpu())
        pids.extend(batch_pids.tolist())
        camids.extend(batch_camids.tolist())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
    
    features = torch.cat(features, dim=0).numpy()
    return features, pids, camids

def evaluate_model(model, query_loader, gallery_loader, device='cuda'):
    try:
        model = model.to(device)
        model.eval()
        
        print("\nExtracting query features...")
        query_features, query_pids, query_camids = extract_features(model, query_loader, device)
        
        print("\nExtracting gallery features...")
        gallery_features, gallery_pids, gallery_camids = extract_features(model, gallery_loader, device)
        
        print("\nComputing metrics...")
        # Using GPU for distance matrix computation if it fits in memory
        if query_features.shape[0] * gallery_features.shape[0] * 4 < 8e9:  # 8GB threshold
            q_tensor = torch.from_numpy(query_features).to(device)
            g_tensor = torch.from_numpy(gallery_features).to(device)
            distmat = -torch.mm(q_tensor, g_tensor.t()).cpu().numpy()
        else:
            distmat = -np.matmul(query_features, gallery_features.T)
        
        # Evaluate - now correctly handling the return values
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            np.array(query_pids),
            np.array(gallery_pids),
            np.array(query_camids),
            np.array(gallery_camids),
            use_metric_cuhk03=False
        )
        
        rank1 = cmc[0] * 100
        rank3 = cmc[2] * 100
        mAP = mAP * 100
        
        return f"{rank1:.2f}", f"{rank3:.2f}", f"{mAP:.2f}"
    except Exception as e:
        import traceback
        print(f"Error during evaluation: {str(e)}")
        print(traceback.format_exc())
        return f"Error\n  Error: {str(e)}", f"Error\n  Error: {str(e)}", f"Error\n  Error: {str(e)}"

def load_jit_model(model_path, device='cuda'):
    try:
        model = torch.jit.load(model_path, map_location=device)
        model = model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Error\n  Error: {str(e)}"

def compare_models():
    print("Starting model comparison...")
    # Use CUDA with optimizations
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
    print(f"Using CUDA for GPU acceleration")
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets using your local data
    query_dataset = Market1501Dataset(root_dir='Dataset/query', transform=transform)
    gallery_dataset = Market1501Dataset(root_dir='Dataset/gallery', transform=transform)
    
    # Create data loaders with larger batch size for CUDA
    query_loader = DataLoader(
        query_dataset, 
        batch_size=256,  # Increased batch size
        shuffle=False,
        num_workers=8,  # Increased workers
        pin_memory=True  # Enable pin memory for faster data transfer
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=256,  # Increased batch size
        shuffle=False,
        num_workers=8,  # Increased workers
        pin_memory=True  # Enable pin memory for faster data transfer
    )
    
    print(f"Loaded {len(query_dataset)} query images and {len(gallery_dataset)} gallery images")
    
    # List of models to compare
    models = [
        {
            'name': 'YourModel',
            'type': 'jit',
            'path': 'checkpoint_epoch_50.pt'  # Updated to use relative path
        },
        {
            'name': 'osnet_x1_0',
            'type': 'torchreid',
            'model_name': 'osnet_x1_0'
        },
        {
            'name': 'resnet50_fc512',
            'type': 'torchreid',
            'model_name': 'resnet50_fc512'
        },
        {
            'name': 'mobilenetv2_x1_0',
            'type': 'torchreid',
            'model_name': 'mobilenetv2_x1_0'
        }
    ]

    results = []

    for m in models:
        print(f"\nEvaluating {m['name']}...")
        row = {'Model': m['name']}
        if m['type'] == 'jit':
            # Your exported model
            if not os.path.exists(m['path']):
                row.update({
                    'Size(MB)': f"Error\n  Error: File not found: {os.path.basename(m['path'])}",
                    'FLOPs': "Error",
                    'Params': "Error",
                    'Rank-1': "Error",
                    'Rank-3': "Error",
                    'mAP': "Error"
                })
                results.append(row)
                continue

            row['Size(MB)'] = get_model_size(m['path'])
            model, err = load_jit_model(m['path'], device=device)
            if model is None:
                row.update({
                    'FLOPs': err,
                    'Params': err,
                    'Rank-1': err,
                    'Rank-3': err,
                    'mAP': err
                })
                results.append(row)
                continue

            # Wrap the JIT model
            class JitWrapper(torch.nn.Module):
                def __init__(self, jit_model):
                    super().__init__()
                    self.jit_model = jit_model
                def forward(self, x):
                    y = self.jit_model(x)
                    if isinstance(y, (tuple, list)):
                        y = y[0]
                    if isinstance(y, dict):
                        y = y.get('features', y.get('embeddings', y.get('feat', y)))
                    if y.dim() > 2:
                        y = torch.nn.functional.adaptive_avg_pool2d(y, (1, 1)).flatten(1)
                    return _l2_normalize(y)

            wrapper = JitWrapper(model)
            flops, params = get_flops_params(wrapper)
            row['FLOPs'] = flops
            row['Params'] = params

            # Evaluate on Market1501
            rank1, rank3, mAP = evaluate_model(wrapper, query_loader, gallery_loader, device=device)
            row['Rank-1'] = rank1
            row['Rank-3'] = rank3
            row['mAP'] = mAP

        else:
            # Torchreid built-in models
            try:
                model = torchreid.models.build_model(
                    name=m['model_name'],
                    num_classes=751,  # Market1501
                    pretrained=True
                ).to(device)
                model.eval()
                
                row['Size(MB)'] = "N/A"
                flops, params = get_flops_params(model)
                row['FLOPs'] = flops
                row['Params'] = params

                # Evaluate on Market1501
                rank1, rank3, mAP = evaluate_model(model, query_loader, gallery_loader, device=device)
                row['Rank-1'] = rank1
                row['Rank-3'] = rank3
                row['mAP'] = mAP
            except Exception as e:
                err = f"Error\n  Error: {str(e)}"
                row.update({
                    'Size(MB)': "Error",
                    'FLOPs': err,
                    'Params': err,
                    'Rank-1': err,
                    'Rank-3': err,
                    'mAP': err
                })
        results.append(row)

    # Write results to comparison.txt
    print("\nWriting results to comparison.txt...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Number of results: {len(results)}")
    
    try:
        with open('comparison.txt', 'w') as f:
            header = "{:<20} {:<10} {:<15} {:<10} {:<10} {:<10} {:<10}\n".format(
                "Model", "Size(MB)", "FLOPs", "Params", "Rank-1", "Rank-3", "mAP"
            )
            f.write(header)
            f.write("="*90 + "\n")
            for row in results:
                f.write("{:<20} {:<10} {:<15} {:<10} {:<10} {:<10} {:<10}\n".format(
                    row.get('Model', 'N/A'),
                    row.get('Size(MB)', 'N/A'),
                    row.get('FLOPs', 'N/A'),
                    row.get('Params', 'N/A'),
                    row.get('Rank-1', 'N/A'),
                    row.get('Rank-3', 'N/A'),
                    row.get('mAP', 'N/A')
                ))
        print("Results successfully written to comparison.txt")
    except Exception as e:
        print(f"Error writing results file: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: create a basic results file
        try:
            print("Creating fallback results file...")
            with open('comparison_fallback.txt', 'w') as f:
                f.write("Model comparison results (fallback)\n")
                f.write("="*50 + "\n")
                f.write(f"Script completed with {len(results)} models\n")
                f.write(f"Working directory: {os.getcwd()}\n")
                for row in results:
                    f.write(f"{row.get('Model', 'N/A')}: {row}\n")
            print("Fallback file created: comparison_fallback.txt")
        except Exception as fallback_error:
            print(f"Failed to create fallback file: {fallback_error}")

if __name__ == "__main__":
    compare_models()
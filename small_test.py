# =========================
# 0. Instructions
# =========================
#This file provides a small test for the person re-identification model.
#It loads a query image and a gallery of images, extracts features, and computes similarities.
#It then visualizes the results and saves the result as a PNG image.
#It uses a cosine similarity threshold for ID assignment.
#It uses a batch size for feature extraction.
# =========================
import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

# ========== CONFIG ===========
# Path to TorchScript checkpoint
CHECKPOINT_PATH = 'checkpoint_epoch_15.pt'  # Change as needed
QUERY_DIR = 'Dataset/query'
GALLERY_DIR = 'Dataset/gallery'
RESULT_PNG = 'query_top10_result.png'
IMAGE_SIZE = (256, 128)

# ========== DEVICE SELECTION ===========
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
print(f"Using device: {DEVICE}")

# ========== UTILS ===========
def get_pid_from_filename(filename):
    # Market-1501: 0001_c1s1_001051_00.jpg -> 0001
    return int(filename.split('_')[0])

def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

# ========== LOAD MODEL ===========
print("Loading model...")
model = torch.jit.load(CHECKPOINT_PATH, map_location=DEVICE)
model.eval()

# ========== PICK QUERY IMAGE ===========
query_files = [f for f in os.listdir(QUERY_DIR) if f.lower().endswith('.jpg')]
query_file = query_files[0]  # Pick the first query image (or change index)
query_path = os.path.join(QUERY_DIR, query_file)
query_img = load_image(query_path)
query_pid = get_pid_from_filename(query_file)
query_tensor = preprocess(query_img).to(DEVICE)

# ========== LOAD GALLERY IMAGES ===========
print("Loading gallery images and extracting features...")
gallery_files = [f for f in os.listdir(GALLERY_DIR) if f.lower().endswith('.jpg')]
gallery_imgs = []
gallery_pids = []
gallery_tensors = []
for fname in tqdm(gallery_files, desc="Gallery images"):
    img = load_image(os.path.join(GALLERY_DIR, fname))
    gallery_imgs.append(img)
    gallery_pids.append(get_pid_from_filename(fname))
    gallery_tensors.append(preprocess(img))
gallery_tensors = torch.cat(gallery_tensors, dim=0).to(DEVICE)

# ========== FEATURE EXTRACTION ===========
print("Extracting features...")

# Extract query feature
with torch.no_grad():
    query_feat, _ = model(query_tensor)
    query_feat = torch.nn.functional.normalize(query_feat, p=2, dim=1)

# Batch-wise gallery feature extraction
BATCH_SIZE = 32
gallery_feats = []
with torch.no_grad():
    for i in tqdm(range(0, len(gallery_imgs), BATCH_SIZE), desc="Gallery feature extraction"):
        batch_imgs = gallery_imgs[i:i+BATCH_SIZE]
        batch_tensors = torch.cat([preprocess(img) for img in batch_imgs], dim=0).to(DEVICE)
        feats, _ = model(batch_tensors)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        gallery_feats.append(feats.cpu())
gallery_feats = torch.cat(gallery_feats, dim=0)

# ========== SIMILARITY & RANKING ===========
print("Computing similarities and ranking...")
sims = torch.mm(query_feat, gallery_feats.t()).cpu().numpy().flatten()
topk_idx = np.argsort(-sims)[:10]  # Top 10 most similar

# ========== VISUALIZATION ===========
print("Visualizing and saving result...")
thumb_w, thumb_h = IMAGE_SIZE
margin = 10
font = ImageFont.load_default()

result_w = thumb_w * 11 + margin * 12
result_h = thumb_h + margin * 2 + 20
result_img = Image.new('RGB', (result_w, result_h), (255, 255, 255))
draw = ImageDraw.Draw(result_img)

# Paste query image
result_img.paste(query_img.resize(IMAGE_SIZE), (margin, margin))
draw.text((margin, margin + thumb_h + 2), 'Query', fill=(0,0,0), font=font)

# Paste top 10 gallery images
for i, idx in enumerate(tqdm(topk_idx, desc="Visualizing top 10")):
    gal_img = gallery_imgs[idx].resize(IMAGE_SIZE)
    x = (i+1) * (thumb_w + margin) + margin
    y = margin
    result_img.paste(gal_img, (x, y))
    # Draw rectangle: green if correct, red otherwise
    color = (0,255,0) if gallery_pids[idx] == query_pid else (255,0,0)
    draw.rectangle([x, y, x+thumb_w-1, y+thumb_h-1], outline=color, width=4)
    # Similarity score
    sim_score = sims[idx]
    draw.text((x, y + thumb_h + 2), f'{sim_score:.2f}', fill=color, font=font)

# Save result
result_img.save(RESULT_PNG)
print(f'Result saved to {RESULT_PNG}')

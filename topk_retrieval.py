import os
import random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================
# 1. Tuneable Parameters
# =========================
MODEL_PATH = 'checkpoint_epoch_50.pt'
QUERY_DIR = '/home/anns/Downloads/dataSet/query'
GALLERY_DIR = '/home/anns/Downloads/dataSet/gallery'
IMAGE_SIZE = (256, 128)
TOP_K = [1, 5, 10, 20]
NUM_VIS = 40  # Number of random queries to visualize
RESULTS_DIR = 'Results Old 2'
VIS_DIR = os.path.join(RESULTS_DIR, 'topk_vis')
FONT_PATH = None
CMC_PLOT_PATH = os.path.join(RESULTS_DIR, 'cmc_curve.png')
STATS_TXT_PATH = os.path.join(RESULTS_DIR, 'results.txt')
VIS_TOPK = 5  # Only show top 5 in visualization
VIS_IMAGE_SIZE = (128, 256)  # (width, height) for better readability

# Device and batch size selection
device = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)
if device == 'cuda' and torch.cuda.device_count() > 1:
    BATCH_SIZE = 256
elif device == 'mps':
    BATCH_SIZE = 48
else:
    BATCH_SIZE = 48
print(f"Using device: {device}, batch size: {BATCH_SIZE}")

# =========================
# 2. Load Model
# =========================
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()

# =========================
# 3. Preprocessing
# =========================
def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img)

# =========================
# 4. Load Images & Extract Features
# =========================
def load_image_paths(dir_path):
    return sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith('.jpg')])

def get_pid_cam_from_filename(filename):
    # Market-1501: 0001_c1s1_001051_00.jpg
    parts = os.path.basename(filename).split('_')
    pid = int(parts[0])
    camid = int(parts[1][1])  # c1 -> 1
    return pid, camid

def extract_features(image_paths):
    features = []
    pids = []
    camids = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Extracting features"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        imgs = [preprocess(Image.open(p).convert('RGB')) for p in batch_paths]
        imgs = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats, _ = model(imgs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        features.append(feats.cpu())
        for p in batch_paths:
            pid, camid = get_pid_cam_from_filename(p)
            pids.append(pid)
            camids.append(camid)
    features = torch.cat(features, dim=0)
    return features, np.array(pids), np.array(camids)

# =========================
# 5. Compute Metrics
# =========================
def compute_cmc_map(query_feats, query_pids, query_camids, gallery_feats, gallery_pids, gallery_camids, topk=TOP_K):
    num_q = query_feats.size(0)
    num_g = gallery_feats.size(0)
    sim_matrix = torch.mm(query_feats, gallery_feats.t()).cpu().numpy()
    indices = np.argsort(-sim_matrix, axis=1)  # Descending order
    matches = (gallery_pids[indices] == query_pids[:, None])
    all_cmc = []
    all_AP = []
    for q_idx in range(num_q):
        q_pid = query_pids[q_idx]
        q_camid = query_camids[q_idx]
        order = indices[q_idx]
        remove = (gallery_pids[order] == q_pid) & (gallery_camids[order] == q_camid)
        keep = ~remove
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max(TOP_K)])
        # AP
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        tmp_cmc = tmp_cmc * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    if len(all_cmc) == 0:
        raise RuntimeError("No valid query")
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc_scores = [all_cmc[:, k-1].mean() for k in TOP_K]
    mAP = np.mean(all_AP)
    # For full CMC curve
    full_cmc = all_cmc.mean(axis=0)
    return cmc_scores, mAP, indices, full_cmc

# =========================
# 6. Visualization
# =========================
def visualize_topk(vis_query_indices, query_paths, gallery_paths, indices, matches, vis_dir, k=VIS_TOPK, font_path=None):
    os.makedirs(vis_dir, exist_ok=True)
    font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
    for idx, q_idx in enumerate(vis_query_indices):
        q_path = query_paths[q_idx]
        vis_img = Image.new('RGB', (VIS_IMAGE_SIZE[0]*(k+1), VIS_IMAGE_SIZE[1]), (255,255,255))
        q_img = Image.open(q_path).convert('RGB').resize(VIS_IMAGE_SIZE)
        vis_img.paste(q_img, (0,0))
        for j in range(k):
            g_idx = indices[q_idx, j]
            g_img = Image.open(gallery_paths[g_idx]).convert('RGB').resize(VIS_IMAGE_SIZE)
            vis_img.paste(g_img, ((j+1)*VIS_IMAGE_SIZE[0], 0))
            draw = ImageDraw.Draw(vis_img)
            color = (0,255,0) if matches[q_idx, j] else (255,0,0)
            draw.rectangle([(j+1)*VIS_IMAGE_SIZE[0], 0, (j+2)*VIS_IMAGE_SIZE[0]-1, VIS_IMAGE_SIZE[1]-1], outline=color, width=4)
        vis_img.save(os.path.join(vis_dir, f'query_{idx}_top{k}.png'))

# =========================
# 7. Save CMC Curve
# =========================
def save_cmc_curve(full_cmc, save_path, topk=TOP_K):
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(1, len(full_cmc)+1), full_cmc, label='CMC Curve')
    for k in topk:
        plt.scatter([k], [full_cmc[k-1]], label=f'Rank-{k}: {full_cmc[k-1]*100:.2f}%')
    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    plt.title('CMC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =========================
# 8. Save Results TXT
# =========================
def save_stats_txt(stats_path, cmc_scores, mAP, topk=TOP_K):
    with open(stats_path, 'w') as f:
        f.write('==== Person Re-ID Results (cosine similarity, single-shot) ====\n')
        for k, score in zip(topk, cmc_scores):
            f.write(f'Rank-{k}: {score*100:.2f}%\n')
        f.write(f'mAP: {mAP*100:.2f}%\n')

# =========================
# 9. Main Evaluation
# =========================
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading query and gallery images...")
    query_paths = load_image_paths(QUERY_DIR)
    gallery_paths = load_image_paths(GALLERY_DIR)
    print(f"# Queries: {len(query_paths)}, # Gallery: {len(gallery_paths)}")
    print("Extracting query features...")
    query_feats, query_pids, query_camids = extract_features(query_paths)
    print("Extracting gallery features...")
    gallery_feats, gallery_pids, gallery_camids = extract_features(gallery_paths)
    print("Computing CMC and mAP...")
    cmc_scores, mAP, indices, full_cmc = compute_cmc_map(query_feats, query_pids, query_camids, gallery_feats, gallery_pids, gallery_camids)
    print("\n==== Results (cosine similarity, single-shot) ====")
    for k, score in zip(TOP_K, cmc_scores):
        print(f"Rank-{k}: {score*100:.2f}%")
    print(f"mAP: {mAP*100:.2f}%")
    # Save stats
    save_stats_txt(STATS_TXT_PATH, cmc_scores, mAP, topk=TOP_K)
    print(f"Results saved to {STATS_TXT_PATH}")
    # Save CMC curve
    save_cmc_curve(full_cmc, CMC_PLOT_PATH, topk=TOP_K)
    print(f"CMC curve saved to {CMC_PLOT_PATH}")
    # Visualization
    if NUM_VIS > 0:
        print(f"Saving top-{VIS_TOPK} retrieval visualizations for {NUM_VIS} queries...")
        matches = (gallery_pids[indices] == query_pids[:, None])
        # Find queries with correct prediction in top 3
        correct_in_top3 = [i for i in range(len(query_paths)) if matches[i, indices[i, :3]].any()]
        num_correct = min(len(correct_in_top3), NUM_VIS // 2)
        num_random = NUM_VIS - num_correct
        # Sample from correct_in_top3
        vis_indices_correct = random.sample(correct_in_top3, num_correct) if num_correct > 0 else []
        # Sample the rest randomly, excluding already selected
        remaining_indices = list(set(range(len(query_paths))) - set(vis_indices_correct))
        if num_random > len(remaining_indices):
            num_random = len(remaining_indices)
        vis_indices_random = random.sample(remaining_indices, num_random) if num_random > 0 else []
        vis_indices = vis_indices_correct + vis_indices_random
        random.shuffle(vis_indices)  # Shuffle for variety
        visualize_topk(vis_indices, query_paths, gallery_paths, indices, matches, VIS_DIR, k=VIS_TOPK, font_path=FONT_PATH)
        print(f"Visualizations saved in {VIS_DIR}/")
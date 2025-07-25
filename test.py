# =========================
# 0. Instructions
# =========================
#This file provides the complete pipeline for person re-identification to support video.
#It uses YOLOv8m for person detection and a re-ID model for feature extraction.
#It then assigns IDs to the detected persons and draws the results on the video.
#It also saves the output video.
#It uses a dynamic gallery to store the features of the detected persons.
#It uses a cosine similarity threshold for ID assignment.
#It uses a batch size for feature extraction.
# =========================
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# =========================
# 1. Tuneable Parameters
# =========================
VIDEO_PATH = '/Users/beta/Downloads/Video Footages/1_IPC_20250723@6/output2.mp4'  # Path to input video
YOLOV8_WEIGHTS = '/Users/Shared/Person Re-ID Test/yolov8m.pt'   # Path to YOLOv8m weights
REID_MODEL_PATH = 'checkpoint_epoch_50.pt'  # Path to TorchScript re-ID model
SAVE_OUTPUT_VIDEO = True  # Set to False to not save output video
OUTPUT_VIDEO = '/Users/Shared/4.0/1.mp4'  # Output video file name (used only if SAVE_OUTPUT_VIDEO is True)
IMAGE_SIZE = (256, 128)           # Re-ID model input size
BATCH_SIZE = 32                   # Batch size for feature extraction
SIM_THRESHOLD = 0.4               # Cosine similarity threshold for ID assignment
DEVICE = (
    'mps' if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available() else
    'cpu'
)
CONFIDENCE_THRESHOLD = 0.4        # YOLO person detection confidence
NMS_IOU_THRESHOLD = 0.5           # YOLO NMS IoU threshold
SHOW_VIDEO = True                 # Set to False to only save output
FONT_PATH = None                  # Path to a .ttf font file for drawing (optional)

# =========================
# 2. Load Models
# =========================
print(f"Using device: {DEVICE}")
print("Loading YOLOv8m detector...")
from ultralytics import YOLO
person_detector = YOLO(YOLOV8_WEIGHTS)

print("Loading re-ID model...")
reid_model = torch.jit.load(REID_MODEL_PATH, map_location=DEVICE)
reid_model.eval()

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
# 4. Dynamic Gallery
# =========================
gallery_features = []  # List of torch tensors (features)
gallery_ids = []       # List of assigned IDs
next_id = 0

# =========================
# 5. Video Processing
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Output video will be saved to: {OUTPUT_VIDEO}")
if SAVE_OUTPUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    OUTPUT_VIDEO = '/Users/Shared/4.0/1.avi'  # Change extension to .avi
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"[ERROR] Failed to open VideoWriter for {OUTPUT_VIDEO}")
        print(f"Check if the directory exists and you have write permissions.")
        import sys
        sys.exit(1)
else:
    out = None

font = ImageFont.truetype(FONT_PATH, 18) if FONT_PATH else ImageFont.load_default()

frame_idx = 0
pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    orig_frame = frame.copy()
    # 1. Detect persons
    results = person_detector(frame, conf=CONFIDENCE_THRESHOLD, iou=NMS_IOU_THRESHOLD)
    boxes = []
    crops = []
    for det in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # class 0 is 'person' in COCO
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            boxes.append((x1, y1, x2, y2))
            crop = frame[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crops.append(preprocess(crop_pil))
    if not crops:
        if SAVE_OUTPUT_VIDEO and out is not None:
            out.write(frame)
        pbar.update(1)
        if SHOW_VIDEO:
            cv2.imshow('Person Re-ID', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        continue
    # 2. Extract features in batch
    crops_tensor = torch.stack(crops).to(DEVICE)
    with torch.no_grad():
        features, _ = reid_model(crops_tensor)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    # 3. Assign IDs using dynamic gallery
    assigned_ids = []
    for feat in features:
        if gallery_features:
            sims = torch.stack([torch.dot(feat, gfeat) for gfeat in gallery_features])
            max_sim, idx = torch.max(sims, dim=0)
            if max_sim.item() > SIM_THRESHOLD:
                assigned_ids.append(gallery_ids[idx.item()])
            else:
                assigned_ids.append(next_id)
                gallery_features.append(feat.detach().to(DEVICE))
                gallery_ids.append(next_id)
                next_id += 1
        else:
            assigned_ids.append(next_id)
            gallery_features.append(feat.detach().to(DEVICE))
            gallery_ids.append(next_id)
            next_id += 1
    # 4. Draw results
    pil_frame = Image.fromarray(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_frame)
    for (x1, y1, x2, y2), pid in zip(boxes, assigned_ids):
        color = (0, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 18), f'ID: {pid}', fill=color, font=font)
    frame_out = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
    if SAVE_OUTPUT_VIDEO and out is not None:
        out.write(frame_out)
    if SHOW_VIDEO:
        cv2.imshow('Person Re-ID', frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pbar.update(1)

cap.release()
if SAVE_OUTPUT_VIDEO and out is not None:
    out.release()
cv2.destroyAllWindows()
pbar.close()
if SAVE_OUTPUT_VIDEO:
    print(f"Output saved to {OUTPUT_VIDEO}")

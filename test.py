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
VIDEO_PATH = '/Users/beta/Downloads/2954065-hd_1920_1080_30fps.mp4' # Path to input video
YOLOV8_WEIGHTS = '/Users/Shared/Person Re-ID Test/yolov8m.pt'   # Path to YOLOv8m weights
REID_MODEL_PATH = '/Users/Shared/5.0/Market_1501_checkpoint_epoch_50_82.pt'  # Path to TorchScript re-ID model

# Target Mode Settings
TARGET_MODE = False  # Set to True to track only a specific target person
DUAL_TARGET_MODE = False  # Set to True to track two target persons simultaneously
TARGET_IMAGE_PATH = '/Users/beta/Downloads/hehe.png'  # Path to target person's image
TARGET_NAME = 'John'  # Name of the target person to display

# Dual Target Settings (used when DUAL_TARGET_MODE = True)
TARGET1_IMAGE_PATH = '/Users/Shared/Golden Cave/xa.png'  # Path to first target person's image
TARGET1_NAME = 'Talha'  # Name of the first target person
TARGET1_SIM_THRESHOLD = 0.50  # Similarity threshold for first target (can be different from main threshold)
TARGET2_IMAGE_PATH = '/Users/Shared/Golden Cave/sample.png'  # Path to second target person's image
TARGET2_NAME = 'Anns'  # Name of the second target person
TARGET2_SIM_THRESHOLD = 0.55  # Similarity threshold for second target (can be different from main threshold)

# Multiple Images per Target Settings
USE_MULTIPLE_IMAGES_PER_TARGET = True  # Set to True to use multiple images per target
TARGET1_IMAGE_PATH_2 = '/Users/Shared/Golden Cave/x.png'  # Second image for first target (optional)
TARGET1_IMAGE_PATH_3 = '/Users/Shared/Golden Cave/talha.png'  # Third image for first target (optional)
TARGET2_IMAGE_PATH_2 = '/Users/Shared/Golden Cave/ya.png'  # Second image for second target (optional)
TARGET2_IMAGE_PATH_3 = '/Users/Shared/Golden Cave/anns.png'  # Third image for second target (optional)

# Dynamic Threshold Adjustment Settings
ENABLE_DYNAMIC_THRESHOLDS = False  # Set to True to enable threshold changes during processing
THRESHOLD_TIME_RANGES = [
    # Format: [start_time, end_time, target1_threshold, target2_threshold]
    # Times in format: "MM:SS" or "HH:MM:SS"
    ["0:00", "0:17", 0.41, 0.35],    # Default thresholds before your ranges
    ["0:18", "0:21", 0.30, 0.35],    # Your specific range
    ["0:22", "0:36", 0.41, 0.35],    # Back to default
    ["0:37", "0:38", 0.70, 0.35],    # Your specific range
    ["0:39", "0:42", 0.41, 0.35],    # Back to default
    ["0:43", "0:44", 0.41, 0.40],    # Your specific range
    ["0:45", "1:00", 0.41, 0.35],    # Back to default
    ["1:01", "1:02", 0.41, 0.20],    # Your specific range
    ["1:03", "99:59", 0.41, 0.35],   # Default for rest of video
]

SHOW_SIMILARITY_AND_CONFIDENCE = True  # Show similarity score and YOLO confid ence on bounding box

SAVE_OUTPUT_VIDEO = False  # Set to False to not save output video
OUTPUT_VIDEO = '/Users/Shared/5.0/track_anns&talha_front_leftFINAL.mp4'  # Output video file name (used only if SAVE_OUTPUT_VIDEO is True)
VIDEO_QUALITY = 'high'  # Options: 'high', 'medium', 'low' - affects codec selection
IMAGE_SIZE = (256, 128)           # Re-ID model input size
BATCH_SIZE = 64                   # Increased batch size for better GPU utilization
SIM_THRESHOLD = 0.3               # Cosine similarity threshold for ID assignment
DEVICE = (
    'mps' if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available() else
    'cpu'
)

# GPU Optimization Settings
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed

# Memory optimization for Mac GPU
if DEVICE == 'mps':
    torch.mps.empty_cache()  # Clear MPS cache
CONFIDENCE_THRESHOLD = 0.3        # YOLO person detection confidence
NMS_IOU_THRESHOLD = 0.5           # YOLO NMS IoU threshold
SHOW_VIDEO = True                 # Set to False to only save output
FAST_PROCESSING = False           # Set to True to skip live display for faster processing
FONT_PATH = None                  # Path to a .ttf font file for drawing (optional)

# =========================
# 2. Load Models
# =========================
print(f"Using device: {DEVICE}")
print("Loading YOLOv8m detector...")
from ultralytics import YOLO
person_detector = YOLO(YOLOV8_WEIGHTS)
person_detector.to(DEVICE)  # Move YOLO model to GPU

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

def time_to_frames(time_str, fps):
    """Convert time string (MM:SS or HH:MM:SS) to frame number"""
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS format
        minutes, seconds = parts
        total_seconds = int(minutes) * 60 + int(seconds)
    elif len(parts) == 3:  # HH:MM:SS format
        hours, minutes, seconds = parts
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use MM:SS or HH:MM:SS")
    
    return int(total_seconds * fps)

def create_continuous_frame_ranges(time_ranges, fps):
    """Create continuous frame ranges with no gaps"""
    frame_ranges = []
    for i, time_range in enumerate(time_ranges):
        start_time, end_time, target1_thresh, target2_thresh = time_range
        start_frame = time_to_frames(start_time, fps)
        end_frame = time_to_frames(end_time, fps)
        
        # Ensure continuous ranges
        if i > 0:
            # Start from the frame after the previous range ended
            start_frame = frame_ranges[-1][1] + 1
        
        frame_ranges.append([start_frame, end_frame, target1_thresh, target2_thresh])
    
    return frame_ranges

# =========================
# 4. Target Feature Extraction (for Target Mode)
# =========================
target_feature = None
target1_feature = None
target2_feature = None

if DUAL_TARGET_MODE:
    print("Loading dual target images...")
    try:
        # Load and preprocess first target image(s)
        target1_img = Image.open(TARGET1_IMAGE_PATH).convert('RGB')
        target1_img = target1_img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        target1_tensor = preprocess(target1_img).unsqueeze(0).to(DEVICE)
        
        # Load second image for target1 if enabled and file exists
        target1_feature_2 = None
        if USE_MULTIPLE_IMAGES_PER_TARGET and os.path.exists(TARGET1_IMAGE_PATH_2):
            try:
                target1_img_2 = Image.open(TARGET1_IMAGE_PATH_2).convert('RGB')
                target1_img_2 = target1_img_2.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                target1_tensor_2 = preprocess(target1_img_2).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    target1_feature_2, _ = reid_model(target1_tensor_2)
                    target1_feature_2 = torch.nn.functional.normalize(target1_feature_2, p=2, dim=1)
                    target1_feature_2 = target1_feature_2.squeeze(0)
                print(f"Loaded second image for {TARGET1_NAME}")
            except Exception as e:
                print(f"Warning: Could not load second image for {TARGET1_NAME}: {e}")
        
        # Load third image for target1 if enabled and file exists
        target1_feature_3 = None
        if USE_MULTIPLE_IMAGES_PER_TARGET and os.path.exists(TARGET1_IMAGE_PATH_3):
            try:
                target1_img_3 = Image.open(TARGET1_IMAGE_PATH_3).convert('RGB')
                target1_img_3 = target1_img_3.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                target1_tensor_3 = preprocess(target1_img_3).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    target1_feature_3, _ = reid_model(target1_tensor_3)
                    target1_feature_3 = torch.nn.functional.normalize(target1_feature_3, p=2, dim=1)
                    target1_feature_3 = target1_feature_3.squeeze(0)
                print(f"Loaded third image for {TARGET1_NAME}")
            except Exception as e:
                print(f"Warning: Could not load third image for {TARGET1_NAME}: {e}")
        
        # Load and preprocess second target image(s)
        target2_img = Image.open(TARGET2_IMAGE_PATH).convert('RGB')
        target2_img = target2_img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        target2_tensor = preprocess(target2_img).unsqueeze(0).to(DEVICE)
        
        # Load second image for target2 if enabled and file exists
        target2_feature_2 = None
        if USE_MULTIPLE_IMAGES_PER_TARGET and os.path.exists(TARGET2_IMAGE_PATH_2):
            try:
                target2_img_2 = Image.open(TARGET2_IMAGE_PATH_2).convert('RGB')
                target2_img_2 = target2_img_2.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                target2_tensor_2 = preprocess(target2_img_2).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    target2_feature_2, _ = reid_model(target2_tensor_2)
                    target2_feature_2 = torch.nn.functional.normalize(target2_feature_2, p=2, dim=1)
                    target2_feature_2 = target2_feature_2.squeeze(0)
                print(f"Loaded second image for {TARGET2_NAME}")
            except Exception as e:
                print(f"Warning: Could not load second image for {TARGET2_NAME}: {e}")
        
        # Load third image for target2 if enabled and file exists
        target2_feature_3 = None
        if USE_MULTIPLE_IMAGES_PER_TARGET and os.path.exists(TARGET2_IMAGE_PATH_3):
            try:
                target2_img_3 = Image.open(TARGET2_IMAGE_PATH_3).convert('RGB')
                target2_img_3 = target2_img_3.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                target2_tensor_3 = preprocess(target2_img_3).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    target2_feature_3, _ = reid_model(target2_tensor_3)
                    target2_feature_3 = torch.nn.functional.normalize(target2_feature_3, p=2, dim=1)
                    target2_feature_3 = target2_feature_3.squeeze(0)
                print(f"Loaded third image for {TARGET2_NAME}")
            except Exception as e:
                print(f"Warning: Could not load third image for {TARGET2_NAME}: {e}")
        
        # Extract target features
        with torch.no_grad():
            target1_feature, _ = reid_model(target1_tensor)
            target1_feature = torch.nn.functional.normalize(target1_feature, p=2, dim=1)
            target1_feature = target1_feature.squeeze(0)
            
            target2_feature, _ = reid_model(target2_tensor)
            target2_feature = torch.nn.functional.normalize(target2_feature, p=2, dim=1)
            target2_feature = target2_feature.squeeze(0)
        
        print(f"Dual target features extracted successfully for: {TARGET1_NAME} and {TARGET2_NAME}")
        print(f"Target images resized to: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
        print(f"Using individual thresholds: {TARGET1_NAME} ({TARGET1_SIM_THRESHOLD:.3f}), {TARGET2_NAME} ({TARGET2_SIM_THRESHOLD:.3f})")
        if USE_MULTIPLE_IMAGES_PER_TARGET:
            target1_count = 1 + (1 if target1_feature_2 is not None else 0) + (1 if target1_feature_3 is not None else 0)
            target2_count = 1 + (1 if target2_feature_2 is not None else 0) + (1 if target2_feature_3 is not None else 0)
            print(f"Multiple images mode: {TARGET1_NAME} has {target1_count} image(s), {TARGET2_NAME} has {target2_count} image(s)")
    except Exception as e:
        print(f"Error loading dual target images: {e}")
        print("Falling back to normal mode...")
        DUAL_TARGET_MODE = False
        TARGET_MODE = False

elif TARGET_MODE:
    print(f"Loading target image from: {TARGET_IMAGE_PATH}")
    try:
        # Load and preprocess target image
        target_img = Image.open(TARGET_IMAGE_PATH).convert('RGB')
        # Resize to model's expected input size (256x128)
        target_img = target_img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        target_tensor = preprocess(target_img).unsqueeze(0).to(DEVICE)
        
        # Extract target features
        with torch.no_grad():
            target_feature, _ = reid_model(target_tensor)
            target_feature = torch.nn.functional.normalize(target_feature, p=2, dim=1)
            target_feature = target_feature.squeeze(0)
        
        print(f"Target feature extracted successfully for: {TARGET_NAME}")
        print(f"Target image resized to: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    except Exception as e:
        print(f"Error loading target image: {e}")
        print("Falling back to normal mode...")
        TARGET_MODE = False

# =========================
# 5. Dynamic Gallery (for Normal Mode)
# =========================
gallery_features = []  # List of torch tensors (features)
gallery_ids = []       # List of assigned IDs
next_id = 0

# =========================
# 6. Video Processing
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Convert time ranges to frame ranges for dynamic thresholds
threshold_frame_ranges = []
if ENABLE_DYNAMIC_THRESHOLDS and DUAL_TARGET_MODE:
    print("Converting time ranges to frame ranges...")
    threshold_frame_ranges = create_continuous_frame_ranges(THRESHOLD_TIME_RANGES, fps)
    for frame_range in threshold_frame_ranges:
        start_frame, end_frame, target1_thresh, target2_thresh = frame_range
        start_time = f"{int(start_frame // (fps * 60)):02d}:{int((start_frame // fps) % 60):02d}"
        end_time = f"{int(end_frame // (fps * 60)):02d}:{int((end_frame // fps) % 60):02d}"
        print(f"Frames {start_frame}-{end_frame} ({start_time}-{end_time}) → Thresh: {target1_thresh:.3f}, {target2_thresh:.3f}")

print(f"Output video will be saved to: {OUTPUT_VIDEO}")

# Display processing mode
if FAST_PROCESSING:
    print("FAST PROCESSING MODE: Skipping live display for faster processing")
    print("Video will be saved without live preview")
else:
    print("NORMAL MODE: Showing live preview during processing")

# Check if output directory exists
import os
output_dir = os.path.dirname(OUTPUT_VIDEO)
if not os.path.exists(output_dir):
    print(f"[ERROR] Output directory does not exist: {output_dir}")
    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

if SAVE_OUTPUT_VIDEO:
    # Use appropriate codec based on quality setting and file format
    if VIDEO_QUALITY == 'high':
        if OUTPUT_VIDEO.endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # High quality MP4
        elif OUTPUT_VIDEO.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # High quality AVI
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif VIDEO_QUALITY == 'medium':
        if OUTPUT_VIDEO.endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif OUTPUT_VIDEO.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:  # low quality
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Lower quality but faster
    
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"[ERROR] Failed to open VideoWriter for {OUTPUT_VIDEO}")
        print(f"Trying alternative codec...")
        
        # Try different codecs for the same file format
        if OUTPUT_VIDEO.endswith('.mp4'):
            # Try H.264 codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
            if not out.isOpened():
                # Try XVID as last resort
                OUTPUT_VIDEO = OUTPUT_VIDEO.replace('.mp4', '.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
                print(f"Switched to AVI format: {OUTPUT_VIDEO}")
        
        if not out.isOpened():
            print(f"[ERROR] Failed to open VideoWriter for {OUTPUT_VIDEO}")
            print(f"Check if the directory exists and you have write permissions.")
            print(f"Trying to create directory: {output_dir}")
            try:
                os.makedirs(output_dir, exist_ok=True)
                # Try one more time with MJPG
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
                if not out.isOpened():
                    import sys
                    sys.exit(1)
                else:
                    print(f"Successfully created video writer with MJPG codec")
            except Exception as e:
                print(f"Final error: {e}")
                import sys
                sys.exit(1)
else:
    out = None

# Create different font sizes for different display modes
if SHOW_SIMILARITY_AND_CONFIDENCE:
    font = ImageFont.truetype(FONT_PATH, 18) if FONT_PATH else ImageFont.load_default()
else:
    # Much bigger font for names only
    font = ImageFont.truetype(FONT_PATH, 64) if FONT_PATH else ImageFont.load_default()

frame_idx = 0
pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    # Dynamic threshold adjustment
    if ENABLE_DYNAMIC_THRESHOLDS and DUAL_TARGET_MODE:
        # Check if current frame falls within any threshold range
        threshold_found = False
        for frame_range in threshold_frame_ranges:
            start_frame, end_frame, target1_thresh, target2_thresh = frame_range
            if start_frame <= frame_idx <= end_frame:
                threshold_found = True
                # Update thresholds if they've changed
                if TARGET1_SIM_THRESHOLD != target1_thresh or TARGET2_SIM_THRESHOLD != target2_thresh:
                    TARGET1_SIM_THRESHOLD = target1_thresh
                    TARGET2_SIM_THRESHOLD = target2_thresh
                    current_time = frame_idx / fps
                    minutes = int(current_time // 60)
                    seconds = int(current_time % 60)
                    print(f"Frame {frame_idx} ({minutes:02d}:{seconds:02d}): Thresholds changed to {TARGET1_NAME}({TARGET1_SIM_THRESHOLD:.3f}), {TARGET2_NAME}({TARGET2_SIM_THRESHOLD:.3f})")
                break
        
        # Debug: If no threshold range found, this shouldn't happen now
        if not threshold_found and frame_idx % 30 == 0:  # Print every 30 frames for debugging
            current_time = frame_idx / fps
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            print(f"WARNING: Frame {frame_idx} ({minutes:02d}:{seconds:02d}) not in any threshold range!")
        
        # Debug: Print current thresholds every 30 frames around the problematic time
        if frame_idx % 30 == 0:
            current_time = frame_idx / fps
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            if 35 <= seconds <= 40:  # Around 0:37
                print(f"DEBUG: Frame {frame_idx} ({minutes:02d}:{seconds:02d}) - Current thresholds: {TARGET1_NAME}({TARGET1_SIM_THRESHOLD:.3f}), {TARGET2_NAME}({TARGET2_SIM_THRESHOLD:.3f})")
    
    orig_frame = frame.copy()
    # 1. Detect persons
    results = person_detector(frame, conf=CONFIDENCE_THRESHOLD, iou=NMS_IOU_THRESHOLD, verbose=False)
    boxes = []
    crops = []
    confidences = []
    
    # Process detections more efficiently
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        detections = results[0].boxes.data
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            if int(cls) == 0:  # class 0 is 'person' in COCO
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                boxes.append((x1, y1, x2, y2))
                confidences.append(conf)
                crop = frame[y1:y2, x1:x2]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                crops.append(preprocess(crop_pil))
    if not crops:
        if SAVE_OUTPUT_VIDEO and out is not None:
            out.write(frame)
        pbar.update(1)
        if SHOW_VIDEO and not FAST_PROCESSING:
            cv2.imshow('Person Re-ID', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        continue
    # 2. Extract features in batch (optimized for GPU)
    if len(crops) > 0:
        crops_tensor = torch.stack(crops).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            features, _ = reid_model(crops_tensor)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
    else:
        features = torch.empty(0, device=DEVICE)
    
    # 3. Assign IDs based on mode
    assigned_ids = []
    similarity_scores = []
    
    if DUAL_TARGET_MODE:
        # Dual Target Mode: Track two target persons (GPU optimized)
        #print(f"DEBUG: DUAL_TARGET_MODE is True, processing {len(features)} features")
        if len(features) > 0:
            # Calculate all similarities at once on GPU
            sim1_all = torch.mm(features, target1_feature.unsqueeze(1)).squeeze(1)
            sim2_all = torch.mm(features, target2_feature.unsqueeze(1)).squeeze(1)
            
            # Calculate similarities with second and third images if available
            if USE_MULTIPLE_IMAGES_PER_TARGET and target1_feature_2 is not None:
                sim1_all_2 = torch.mm(features, target1_feature_2.unsqueeze(1)).squeeze(1)
                # Take the best similarity from first two images for target1
                sim1_all = torch.max(sim1_all, sim1_all_2)
            
            if USE_MULTIPLE_IMAGES_PER_TARGET and target1_feature_3 is not None:
                sim1_all_3 = torch.mm(features, target1_feature_3.unsqueeze(1)).squeeze(1)
                # Take the best similarity from all three images for target1
                sim1_all = torch.max(sim1_all, sim1_all_3)
            
            if USE_MULTIPLE_IMAGES_PER_TARGET and target2_feature_2 is not None:
                sim2_all_2 = torch.mm(features, target2_feature_2.unsqueeze(1)).squeeze(1)
                # Take the best similarity from first two images for target2
                sim2_all = torch.max(sim2_all, sim2_all_2)
            
            if USE_MULTIPLE_IMAGES_PER_TARGET and target2_feature_3 is not None:
                sim2_all_3 = torch.mm(features, target2_feature_3.unsqueeze(1)).squeeze(1)
                # Take the best similarity from all three images for target2
                sim2_all = torch.max(sim2_all, sim2_all_3)
            
            # Determine best matches using individual thresholds
            for i in range(len(features)):
                sim1 = sim1_all[i]
                sim2 = sim2_all[i]
                
                # Check if each target meets its individual threshold
                target1_valid = sim1.item() > TARGET1_SIM_THRESHOLD
                target2_valid = sim2.item() > TARGET2_SIM_THRESHOLD
                
                if target1_valid and target2_valid:
                    # Both targets are valid, choose the one with higher similarity
                    if sim1.item() > sim2.item():
                        assigned_ids.append(TARGET1_NAME)
                        similarity_scores.append(sim1.item())
                    else:
                        assigned_ids.append(TARGET2_NAME)
                        similarity_scores.append(sim2.item())
                elif target1_valid:
                    # Only target1 is valid
                    assigned_ids.append(TARGET1_NAME)
                    similarity_scores.append(sim1.item())
                elif target2_valid:
                    # Only target2 is valid
                    assigned_ids.append(TARGET2_NAME)
                    similarity_scores.append(sim2.item())
                else:
                    # Neither target meets threshold
                    assigned_ids.append(None)
                    similarity_scores.append(max(sim1.item(), sim2.item()))
        
        # Handle multiple matches for each target type
        target1_indices = [i for i, pid in enumerate(assigned_ids) if pid == TARGET1_NAME]
        target2_indices = [i for i, pid in enumerate(assigned_ids) if pid == TARGET2_NAME]
        
        # Debug: Print assigned IDs
        if assigned_ids:
            print(f"DEBUG: Assigned IDs: {assigned_ids}")
            print(f"DEBUG: Target1 indices: {target1_indices}, Target2 indices: {target2_indices}")
        
        # Keep only the best match for each target
        if len(target1_indices) > 1:
            best_idx = max(target1_indices, key=lambda i: similarity_scores[i])
            for i in target1_indices:
                if i != best_idx:
                    assigned_ids[i] = None
            print(f"Multiple {TARGET1_NAME} found. Keeping best match with similarity: {similarity_scores[best_idx]:.3f}")
        
        if len(target2_indices) > 1:
            best_idx = max(target2_indices, key=lambda i: similarity_scores[i])
            for i in target2_indices:
                if i != best_idx:
                    assigned_ids[i] = None
            print(f"Multiple {TARGET2_NAME} found. Keeping best match with similarity: {similarity_scores[best_idx]:.3f}")
    
    elif TARGET_MODE:
        # Target Mode: Only track the target person (GPU optimized)
        if len(features) > 0:
            # Calculate all similarities at once on GPU
            similarities = torch.mm(features, target_feature.unsqueeze(1)).squeeze(1)
            
            for sim in similarities:
                similarity_scores.append(sim.item())
                if sim.item() > SIM_THRESHOLD:
                    assigned_ids.append(TARGET_NAME)  # Use target name instead of ID
                else:
                    assigned_ids.append(None)  # Not the target person
        
        # If multiple targets found, keep only the one with highest similarity
        target_indices = [i for i, pid in enumerate(assigned_ids) if pid is not None]
        if len(target_indices) > 1:
            # Find the index with highest similarity
            best_idx = max(target_indices, key=lambda i: similarity_scores[i])
            # Set all other targets to None
            for i in target_indices:
                if i != best_idx:
                    assigned_ids[i] = None
            print(f"Multiple targets found. Keeping best match with similarity: {similarity_scores[best_idx]:.3f}")
    else:
        # Normal Mode: Use dynamic gallery (GPU optimized)
        print(f"DEBUG: Normal mode triggered, DUAL_TARGET_MODE={DUAL_TARGET_MODE}, TARGET_MODE={TARGET_MODE}")
        if len(features) > 0:
            for feat in features:
                if gallery_features:
                    # Calculate similarities with all gallery features at once
                    gallery_tensor = torch.stack(gallery_features)
                    sims = torch.mm(feat.unsqueeze(0), gallery_tensor.t()).squeeze(0)
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
    
    for i, ((x1, y1, x2, y2), pid) in enumerate(zip(boxes, assigned_ids)):
        if DUAL_TARGET_MODE:
            if pid is not None:  # This is one of the target persons
                if pid == TARGET1_NAME:
                    color = (0, 255, 0)  # Green for first target
                else:  # TARGET2_NAME
                    color = (255, 0, 0)  # Red for second target
                
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Prepare text to display
                if SHOW_SIMILARITY_AND_CONFIDENCE:
                    sim_score = similarity_scores[i]
                    conf_score = confidences[i]
                    if ENABLE_DYNAMIC_THRESHOLDS:
                        # Show current threshold for this target
                        current_threshold = TARGET1_SIM_THRESHOLD if pid == TARGET1_NAME else TARGET2_SIM_THRESHOLD
                        text = f'{pid} | Sim: {sim_score:.3f} | Conf: {conf_score:.3f} | Thresh: {current_threshold:.3f}'
                        
                        # Debug: Print when Talha is detected around 0:37
                        current_time = frame_idx / fps
                        minutes = int(current_time // 60)
                        seconds = int(current_time % 60)
                        if pid == TARGET1_NAME and 35 <= seconds <= 40:
                            print(f"DISPLAY DEBUG: Frame {frame_idx} ({minutes:02d}:{seconds:02d}) - {pid} threshold: {current_threshold:.3f}")
                    else:
                        text = f'{pid} | Sim: {sim_score:.3f} | Conf: {conf_score:.3f}'
                    draw.text((x1, y1 - 18), text, fill=color, font=font)
                else:
                    text = f'{pid}'
                    # Position text at the very top left corner of the bounding box
                    text_x = x1 + 5  # Small offset from the left edge
                    text_y = y1 - 15  # Small offset above the top edge
                    draw.text((text_x, text_y), text, fill=color, font=font)
            # Don't draw anything for non-target persons
        
        elif TARGET_MODE:
            if pid is not None:  # This is the target person
                color = (0, 255, 0)  # Green for target
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Prepare text to display
                if SHOW_SIMILARITY_AND_CONFIDENCE:
                    sim_score = similarity_scores[i]
                    conf_score = confidences[i]
                    if ENABLE_DYNAMIC_THRESHOLDS:
                        # Show current threshold for this target
                        text = f'{pid} | Sim: {sim_score:.3f} | Conf: {conf_score:.3f} | Thresh: {SIM_THRESHOLD:.3f}'
                    else:
                        text = f'{pid} | Sim: {sim_score:.3f} | Conf: {conf_score:.3f}'
                    draw.text((x1, y1 - 18), text, fill=color, font=font)
                else:
                    text = f'{pid}'
                    # Position text at the very top left corner of the bounding box
                    text_x = x1 + 5  # Small offset from the left edge
                    text_y = y1 - 15  # Small offset above the top edge
                    draw.text((text_x, text_y), text, fill=color, font=font)
            # Don't draw anything for non-target persons
        else:
            # Normal mode: draw all detected persons
            color = (0, 255, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 - 18), f'ID: {pid}', fill=color, font=font)
    frame_out = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
    if SAVE_OUTPUT_VIDEO and out is not None:
        out.write(frame_out)
    if SHOW_VIDEO and not FAST_PROCESSING:
        cv2.imshow('Person Re-ID', frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pbar.update(1)
    
    # Periodic memory cleanup for Mac GPU
    if frame_idx % 100 == 0 and DEVICE == 'mps':
        torch.mps.empty_cache()

cap.release()
if SAVE_OUTPUT_VIDEO and out is not None:
    out.release()
pbar.close()

# Clean up video display
if SHOW_VIDEO and not FAST_PROCESSING:
    cv2.destroyAllWindows()
    # Add a small delay to ensure windows are properly closed
    cv2.waitKey(1)

if SAVE_OUTPUT_VIDEO:
    print(f"Output saved to {OUTPUT_VIDEO}")

print("Video processing completed.")

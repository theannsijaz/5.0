# Project Change Log

## Session Started: $(date)

### Project Overview
- Working directory: /Users/Shared/5.0
- Project appears to be a computer vision/deep learning project with image datasets
- Contains Dataset folder with gallery, query, and train subdirectories
- Has checkpoint files (.pt) suggesting a PyTorch-based model
- Contains analysis files and test scripts

### File Structure Analysis
- **Dataset/gallery/**: 19,730 files (19,729 *.jpg, 1 *.db)
- **Dataset/query/**: 3,366 files (3,365 *.jpg, 1 *.db)  
- **Dataset/train/**: Multiple subdirectories with varying numbers of *.jpg files
- **Root files**: iust_analysis.json, IUST_checkpoint_epoch_50.pt, Market_1501_checkpoint_epoch_50_82.pt, small_test.py, and other Python/notebook files

### Changes Made
- **Created project_log.md** - Initial log file to track all project changes
- **Analyzed train.py** - Comprehensive training script with advanced features

### Project Analysis
Based on examining the code files, this is a **Person Re-Identification (Re-ID) project** with the following components:

1. **small_test.py** - Simple test script that:
   - Loads a query image and gallery images
   - Extracts features using a TorchScript model
   - Computes similarities and ranks results
   - Visualizes top-10 matches with color coding (green=correct, red=incorrect)

2. **test.py** - Complete video processing pipeline that:
   - Uses YOLOv8m for person detection
   - Applies re-ID model for feature extraction
   - Implements dynamic gallery for ID assignment
   - Processes video frames and saves output

3. **topk_retrieval.py** - Comprehensive evaluation script that:
   - Computes CMC (Cumulative Matching Characteristic) curves
   - Calculates mAP (mean Average Precision)
   - Generates visualizations of top-k retrievals
   - Saves detailed statistics and plots

4. **Dataset Structure**:
   - **gallery/**: 19,730 images for matching against
   - **query/**: 3,366 query images to find matches for
   - **train/**: Training data organized by person IDs (0002, 0007, etc.)

5. **Model Architecture**:
   - **Backbone**: ResNet50 (pretrained on ImageNet)
   - **Architecture Details**:
     - Uses `list(resnet.children())[:-1]` which means **ALL layers except the final classification layer**
     - This includes: Conv1, BatchNorm1, ReLU, MaxPool, Layer1, Layer2, Layer3, Layer4
     - **Total layers used**: ~48 layers (not 50, as the final fc layer is removed)
   - **Additional Components**:
     - Global Average Pooling: `nn.AdaptiveAvgPool2d((1, 1))`
     - Batch Normalization Neck: `nn.BatchNorm1d(2048)` with frozen bias
     - Classifier: `nn.Linear(2048, num_classes)` without bias
   - **Feature Dimension**: 2048 (ResNet50's final feature dimension)
   - **Output**: Both features (for similarity) and logits (for classification)

6. **Model Checkpoints**:
   - IUST_checkpoint_epoch_50.pt (99MB)
   - Market_1501_checkpoint_epoch_50_82.pt (96MB)

7. **train.py - Advanced Training Script**:
   - **PK Sampler**: Implements P×K sampling strategy (P=16 persons, K=4 images per person)
   - **Batch Size**: 64 (16×4) for optimal Person Re-ID training
   - **Advanced Features**:
     - Custom `PKSampler` class for balanced person sampling
     - FIDI (Fine-grained Difference-aware) loss function
     - Combined loss: FIDI + CrossEntropy
     - Multi-GPU support with DataParallel
     - Live plotting with matplotlib during training
     - Automatic TorchScript model export every 5 epochs
     - ONNX export for model visualization
   - **Training Configuration**:
     - Learning rate: 3.5e-4 with StepLR scheduler (step_size=40, gamma=0.1)
     - Weight decay: 5e-4
     - FIDI parameters: alpha=1.05, beta=0.5
     - Image size: 256×128
     - Evaluation frequency: every 10 epochs
   - **Data Augmentation**:
     - Random horizontal flip for training
     - ImageNet normalization
     - Resize to 256×128
   - **Evaluation Metrics**:
     - CMC (Cumulative Matching Characteristic) curves
     - mAP (mean Average Precision)
     - Rank-1, Rank-5, Rank-10 accuracy
   - **Model Export**:
     - ONNX format for Netron visualization
     - TorchScript format for deployment
     - Best model checkpoint based on mAP

---
*This log will be updated with each file modification during this session.* 
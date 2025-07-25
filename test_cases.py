# =========================
# 0. Instructions
# =========================
#This file provides a test for the person re-identification model.
#It loads a query image and a gallery of images, extracts features, and computes similarities.
#It then visualizes the results and saves the result as a PNG image.
#It uses a cosine similarity threshold for ID assignment.
#It uses a batch size for feature extraction.
# =========================

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the JIT compiled model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the JIT model from .pt file
model = torch.jit.load("/Users/Shared/4.0/checkpoint_epoch_5.pt", map_location=device)
model.eval()

print(f"JIT model loaded successfully on {device}")

# Define preprocessing transformation
def preprocess(img):
    IMAGE_SIZE = (224, 224)  # Define image size if not already defined
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def extract_features_jit(image_path, model):
    """Extract features using JIT compiled model with preprocessing"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).to(device)
    
    with torch.no_grad():
        features = model(image_tensor)
        # Handle tuple outputs from model
        if isinstance(features, tuple):
            features = features[0]  # Take the first element if it's a tuple
    return features

def rank_all_gallery_images(query_image_path, gallery_path, model):
    """
    Rank all images in the gallery folder against the query image
    """
    query_features = extract_features_jit(query_image_path, model)
    
    similarities = []
    image_paths = []
    
    # Collect all images in gallery
    for root, _, files in os.walk(gallery_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                # Skip if it's the query image itself
                if os.path.abspath(image_path) != os.path.abspath(query_image_path):
                    image_paths.append(image_path)
                    # Extract features and compute similarity
                    features = extract_features_jit(image_path, model)
                    
                    # Ensure both tensors are properly shaped for cosine similarity
                    query_flat = query_features.view(query_features.size(0), -1)
                    features_flat = features.view(features.size(0), -1)
                    
                    similarity = F.cosine_similarity(query_flat, features_flat, dim=1).item()
                    similarities.append(similarity)
    
    # Sort all images by similarity (highest to lowest)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    
    print(f"Ranking {len(image_paths)} gallery images against query image:")
    print(f"Query: {query_image_path}")
    print("\nTop ranked matches:")
    
    # Display ranking results
    for rank, idx in enumerate(ranked_indices[:20], 1):  # Show top 20
        image_name = os.path.basename(image_paths[idx])
        print(f"Rank {rank:2d}: {image_name} (Similarity: {similarities[idx]:.4f})")
    
    # Create comprehensive visualization showing query and all ranked images
    total_images = min(len(ranked_indices) + 1, 25)  # Limit to 25 total images for better visualization
    
    # Calculate grid dimensions for better layout
    if total_images <= 16:
        cols = 4
        rows = int(np.ceil(total_images / cols))
    elif total_images <= 25:
        cols = 5
        rows = int(np.ceil(total_images / cols))
    else:
        cols = 6
        rows = int(np.ceil(total_images / cols))
    
    # Create the main figure with high DPI for better quality
    fig = plt.figure(figsize=(cols * 4, rows * 4), dpi=150)
    
    # Show query image first
    plt.subplot(rows, cols, 1)
    query_img = Image.open(query_image_path)
    plt.imshow(query_img)
    plt.title(f"QUERY IMAGE\n{os.path.basename(query_image_path)}", 
              fontweight='bold', color='red', fontsize=14)
    plt.axis('off')
    
    # Show ranked gallery images with their similarity scores
    images_to_show = min(len(ranked_indices), (rows * cols) - 1)  # -1 for query image
    
    for i, idx in enumerate(ranked_indices[:images_to_show], 2):
        plt.subplot(rows, cols, i)
        try:
            image = Image.open(image_paths[idx])
            plt.imshow(image)
            
            # Color code based on similarity score
            sim_score = similarities[idx]
            if sim_score >= 0.8:
                color = 'green'
            elif sim_score >= 0.6:
                color = 'orange'
            else:
                color = 'red'
                
            plt.title(f"Rank {i-1}\nSim: {sim_score:.3f}\n{os.path.basename(image_paths[idx])}", 
                     color=color, fontsize=12)
            plt.axis('off')
        except Exception as e:
            print(f"Error loading image {image_paths[idx]}: {e}")
            continue
    
    plt.tight_layout(pad=2.0)
    
    # Save the main ranking figure
    query_name = os.path.splitext(os.path.basename(query_image_path))[0]
    main_output_path = f'ranking_results_15{query_name}.png'
    plt.savefig(main_output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nMain ranking visualization saved as: {main_output_path}")
    plt.show()
    
    # Create a detailed similarity analysis figure
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    
    # Plot 1: Top 15 similarities as horizontal bar chart
    y_pos = np.arange(min(15, len(similarities)))
    top_similarities = [similarities[ranked_indices[i]] for i in range(min(15, len(similarities)))]
    top_names = [os.path.basename(image_paths[ranked_indices[i]]) for i in range(min(15, len(similarities)))]
    
    bars = ax1.barh(y_pos, top_similarities)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"R{i+1}: {name[:20]}..." if len(name) > 20 else f"R{i+1}: {name}" 
                        for i, name in enumerate(top_names)])
    ax1.set_xlabel('Cosine Similarity Score')
    ax1.set_title('Top 15 Image Similarities')
    ax1.invert_yaxis()
    
    # Color bars based on similarity
    for i, bar in enumerate(bars):
        sim = top_similarities[i]
        if sim >= 0.8:
            bar.set_color('green')
        elif sim >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Plot 2: Distribution of similarities
    ax2.hist(similarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Number of Images')
    ax2.set_title('Distribution of Similarity Scores')
    ax2.axvline(np.mean(similarities), color='red', linestyle='--', 
                label=f'Mean: {np.mean(similarities):.3f}')
    ax2.legend()
    
    # Plot 3: Top 5 matches as thumbnails
    ax3.set_title('Top 5 Matches', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    for i in range(min(5, len(ranked_indices))):
        idx = ranked_indices[i]
        # Create sub-subplot for each top match
        sub_ax = fig2.add_subplot(2, 2, 3, frameon=False)
        sub_ax.axis('off')
        
        # Calculate position for thumbnail
        x_offset = 0.02 + (i * 0.18)
        y_offset = 0.02
        
        # Add thumbnail
        thumbnail_ax = fig2.add_axes([x_offset, y_offset, 0.15, 0.15])
        try:
            thumb_img = Image.open(image_paths[idx])
            thumbnail_ax.imshow(thumb_img)
            thumbnail_ax.set_title(f"R{i+1}: {similarities[idx]:.3f}", fontsize=10)
            thumbnail_ax.axis('off')
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
    
    # Plot 4: Statistics summary
    ax4.axis('off')
    stats_text = f"""
    RANKING STATISTICS
    
    Query Image: {os.path.basename(query_image_path)}
    Gallery Path: {gallery_path}
    
    Total Gallery Images: {len(similarities)}
    
    Similarity Statistics:
    • Average: {np.mean(similarities):.4f}
    • Maximum: {np.max(similarities):.4f}
    • Minimum: {np.min(similarities):.4f}
    • Std Dev: {np.std(similarities):.4f}
    
    Performance Thresholds:
    • High Similarity (>0.8): {sum(1 for s in similarities if s > 0.8)} images
    • Medium Similarity (0.6-0.8): {sum(1 for s in similarities if 0.6 < s <= 0.8)} images
    • Low Similarity (<0.6): {sum(1 for s in similarities if s <= 0.6)} images
    
    Top 3 Matches:
    """
    
    for i in range(min(3, len(ranked_indices))):
        idx = ranked_indices[i]
        stats_text += f"  {i+1}. {os.path.basename(image_paths[idx])} ({similarities[idx]:.4f})\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the analysis figure
    analysis_output_path = f'similarity_analysis_{query_name}.png'
    plt.savefig(analysis_output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Detailed analysis saved as: {analysis_output_path}")
    plt.show()
    
    return ranked_indices, similarities, image_paths

# Test with a query image against all gallery images
query_image_path = 'Test Cases/Test Case 1/0 (3).jpg'  # Update with your query image path
gallery_path = 'Test Cases/Test Case 1'  # Update with your gallery folder path

# Rank all gallery images against the query
ranked_indices, similarities, image_paths = rank_all_gallery_images(query_image_path, gallery_path, model)

print(f"\nSummary Statistics:")
print(f"Total gallery images: {len(similarities)}")
print(f"Average similarity: {np.mean(similarities):.4f}")
print(f"Max similarity: {np.max(similarities):.4f}")
print(f"Min similarity: {np.min(similarities):.4f}")
print(f"Images with similarity > 0.8: {sum(1 for s in similarities if s > 0.8)}")
print(f"Images with similarity > 0.6: {sum(1 for s in similarities if s > 0.6)}")

print(f"\nOutput files saved:")
query_name = os.path.splitext(os.path.basename(query_image_path))[0]
print(f"1. ranking_results_{query_name}.png - Main ranking visualization")
print(f"2. similarity_analysis_{query_name}.png - Detailed analysis")

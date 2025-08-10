#!/usr/bin/env python3
"""
Multi-GPU Memory Test - Matches actual training setup with DataParallel
This properly tests the DataParallel configuration that causes OOM
"""

import os
import torch
import torch.nn as nn
import gc
from train_updated_model_definition import SOLIDERPersonReIDModel, FIDILoss

def print_gpu_memory(stage_name):
    """Print memory for all GPUs"""
    if not torch.cuda.is_available():
        print(f"{stage_name:20} | CPU mode")
        return [0]
    
    gpu_count = torch.cuda.device_count()
    memories = []
    
    print(f"{stage_name:20}", end="")
    for i in range(gpu_count):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f" | GPU{i}: {allocated:.2f}GB/{reserved:.2f}GB", end="")
        memories.append(allocated)
    print()
    return memories

def test_multi_gpu_memory():
    """Test memory with actual DataParallel setup"""
    
    print("=" * 80)
    print("🔬 MULTI-GPU MEMORY TEST (DataParallel)")
    print("=" * 80)
    
    # Exact configuration from your training
    P, K = 6, 12
    batch_size = P * K  # 72
    num_classes = 751
    device = [0, 1]  # Multi-GPU like your training
    
    print(f"Configuration: batch_size={batch_size}, Multi-GPU={device}")
    print("-" * 80)
    
    # Clear all GPU memory
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    gc.collect()
    
    print_gpu_memory("Initial")
    
    # Create model EXACTLY like your trainer does
    print("Creating model with DataParallel (like your training)...")
    model = SOLIDERPersonReIDModel(num_classes=num_classes)
    
    # Move to first GPU and wrap with DataParallel (exact copy of your trainer code)
    model = model.to(torch.device(f"cuda:{device[0]}"))
    model = nn.DataParallel(model, device_ids=device)
    
    print_gpu_memory("Model + DataParallel")
    
    # Create test data on first GPU (like your training)
    test_images = torch.randn(batch_size, 3, 256, 128, device=f"cuda:{device[0]}")
    test_labels = torch.randint(0, num_classes, (batch_size,), device=f"cuda:{device[0]}")
    
    print_gpu_memory("Test data loaded")
    
    # Create losses (like your training)
    fidi_loss_fn = FIDILoss(alpha=1.05, beta=0.5)
    ce_loss_fn = nn.CrossEntropyLoss()
    
    print("\n🧪 Testing FIDI Stage (Training Mode)...")
    try:
        model.train()  # Training mode (not inference!)
        
        # Forward pass
        features, logits = model(test_images, return_semantic_loss=False)
        print_gpu_memory("FIDI forward")
        
        # Compute losses
        fidi_loss = fidi_loss_fn(features, test_labels)
        ce_loss = ce_loss_fn(logits, test_labels)
        total_loss = 0.7 * fidi_loss + 1.0 * ce_loss
        print_gpu_memory("FIDI losses")
        
        # Backward pass (this uses more memory)
        total_loss.backward()
        print_gpu_memory("FIDI backward")
        
        print("✅ FIDI stage successful!")
        
        # Cleanup
        model.zero_grad()
        del features, logits, fidi_loss, ce_loss, total_loss
        torch.cuda.empty_cache()
        print_gpu_memory("FIDI cleaned")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ FIDI stage OOM: {e}")
        return False
    
    print("\n🧪 Testing SOLIDER Stage (Training Mode)...")
    try:
        model.train()
        
        # Forward pass with semantic loss (the problematic part)
        features, logits, semantic_output = model(
            test_images, 
            lambda_val=0.5, 
            return_semantic_loss=True
        )
        print_gpu_memory("SOLIDER forward")
        
        # Extract semantic loss and ensure it's scalar
        semantic_loss = semantic_output['semantic_loss']
        if semantic_loss.dim() > 0:
            semantic_loss = semantic_loss.mean()
        print_gpu_memory("Semantic extracted")
        
        # Compute all losses (exactly like your training)
        fidi_loss = fidi_loss_fn(features, test_labels)
        ce_loss = ce_loss_fn(logits, test_labels)
        total_loss = 0.32 * fidi_loss + 0.88 * ce_loss + 0.5 * semantic_loss
        print_gpu_memory("All losses computed")
        
        # Ensure total_loss is scalar before backward pass
        if total_loss.dim() > 0:
            total_loss = total_loss.mean()
        
        # The critical backward pass
        total_loss.backward()
        print_gpu_memory("SOLIDER backward")
        
        print("✅ SOLIDER stage successful!")
        
        # Cleanup
        model.zero_grad()
        del features, logits, semantic_output, semantic_loss, fidi_loss, ce_loss, total_loss
        torch.cuda.empty_cache()
        print_gpu_memory("SOLIDER cleaned")
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ SOLIDER stage OOM: {e}")
        print(f"💡 This is the exact issue you're hitting at epoch 101!")
        return False

def test_batch_size_with_dataparallel():
    """Test different batch sizes with DataParallel"""
    print(f"\n" + "=" * 80)
    print("🔍 TESTING BATCH SIZES WITH DATAPARALLEL")
    print("=" * 80)
    
    device = [0, 1]
    num_classes = 751
    
    # Test configurations
    test_configs = [
        (4, 8),   # 32
        (4, 10),  # 40
        (5, 8),   # 40
        (5, 10),  # 50
        (6, 8),   # 48
        (6, 10),  # 60
        (6, 12),  # 72 (your current)
    ]
    
    working_configs = []
    
    for P, K in test_configs:
        batch_size = P * K
        print(f"\nTesting P={P}, K={K} (batch_size={batch_size}) with DataParallel...")
        
        try:
            # Clear memory
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
            gc.collect()
            
            # Create model with DataParallel
            model = SOLIDERPersonReIDModel(num_classes=num_classes)
            model = model.to(torch.device(f"cuda:{device[0]}"))
            model = nn.DataParallel(model, device_ids=device)
            
            # Create test data
            test_images = torch.randn(batch_size, 3, 256, 128, device=f"cuda:{device[0]}")
            test_labels = torch.randint(0, num_classes, (batch_size,), device=f"cuda:{device[0]}")
            
            # Test SOLIDER stage with training mode
            model.train()
            features, logits, semantic_output = model(
                test_images, 
                lambda_val=0.5, 
                return_semantic_loss=True
            )
            
            # Test backward pass with proper scalar handling
            semantic_loss = semantic_output['semantic_loss']
            if semantic_loss.dim() > 0:
                semantic_loss = semantic_loss.mean()
            ce_loss = nn.CrossEntropyLoss()(logits, test_labels)
            total_loss = ce_loss + 0.5 * semantic_loss
            
            # Ensure scalar before backward
            if total_loss.dim() > 0:
                total_loss = total_loss.mean()
            total_loss.backward()
            
            print(f"✅ P={P}, K={K} works with DataParallel!")
            working_configs.append((P, K, batch_size))
            
            # Cleanup
            del model, test_images, test_labels, features, logits, semantic_output
            del semantic_loss, ce_loss, total_loss
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ P={P}, K={K} OOM with DataParallel")
        except Exception as e:
            print(f"❌ P={P}, K={K} error: {e}")
    
    print(f"\n📊 DataParallel Results:")
    if working_configs:
        print(f"✅ Working configurations:")
        for P, K, batch_size in working_configs:
            print(f"   P={P}, K={K} (batch_size={batch_size})")
        
        max_working = max(working_configs, key=lambda x: x[2])
        print(f"🏆 Maximum working batch size: {max_working[2]} (P={max_working[0]}, K={max_working[1]})")
        
        if max_working[2] < 72:
            print(f"⚠️  Your current config P=6, K=12 (batch_size=72) is too large!")
            print(f"💡 Recommended: P={max_working[0]}, K={max_working[1]} (batch_size={max_working[2]})")
    else:
        print(f"❌ No configurations worked with DataParallel - need single GPU")

def suggest_optimizations():
    """Suggest specific optimizations based on test results"""
    print(f"\n" + "=" * 80)
    print("💡 OPTIMIZATION SUGGESTIONS")
    print("=" * 80)
    
    print("1. 🔧 Set environment variable for memory fragmentation:")
    print("   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print("   (Add this to your .bashrc or run before training)")
    
    print("\n2. 🎯 Reduce batch size in train_updated_model_definition.py:")
    print("   Change: P = 6, K = 12  # 72 batch")
    print("   To:     P = 6, K = 10  # 60 batch")
    print("   Or:     P = 5, K = 10  # 50 batch")
    
    print("\n3. 🖼️  Reduce image resolution temporarily:")
    print("   Change: image_height = 256, image_width = 128")
    print("   To:     image_height = 224, image_width = 112")
    
    print("\n4. 📱 Try single GPU if multi-GPU still fails:")
    print("   Change: device = [0, 1]")
    print("   To:     device = 'cuda'  # or device = 0")
    
    print("\n5. 🧠 Enable gradient checkpointing more aggressively:")
    print("   (Already implemented in your memory optimizations)")

def main():
    print("🚀 Multi-GPU Memory Test - Exact Training Setup")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test exact multi-GPU setup
    success = test_multi_gpu_memory()
    
    # Test batch size limits with DataParallel
    test_batch_size_with_dataparallel()
    
    # Provide optimization suggestions
    suggest_optimizations()
    
    print(f"\n" + "=" * 80)
    if success:
        print("🎉 SUCCESS! Your current config should work at epoch 101!")
    else:
        print("💥 CONFIRMED: This reproduces your epoch 101 OOM issue")
        print("🔧 Follow the optimization suggestions above")
    print("=" * 80)

if __name__ == "__main__":
    main()

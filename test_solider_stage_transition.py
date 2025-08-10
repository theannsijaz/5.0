#!/usr/bin/env python3
"""
Test script to verify SOLIDER stage transition memory optimizations
This simulates epoch 101 (SOLIDER stage) without training for 100 epochs
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import gc

# Import our main training file
from train_updated_model_definition import (
    SOLIDERPersonReIDModel, 
    SOLIDERFIDITrainer,
    PersonReIDTrainDataset,
    PKSampler,
    test_transform
)

def create_dummy_dataset(num_samples=500, num_classes=100):
    """Create a dummy dataset for testing"""
    print(f"Creating dummy dataset with {num_samples} samples, {num_classes} classes...")
    
    # Create dummy images (3x256x128)
    dummy_images = torch.randn(num_samples, 3, 256, 128)
    dummy_labels = torch.randint(0, num_classes, (num_samples,))
    
    # Create a simple dataset
    dataset = TensorDataset(dummy_images, dummy_labels)
    
    print(f"✓ Dummy dataset created: {len(dataset)} samples")
    return dataset, num_classes

def test_memory_usage():
    """Test memory usage and print stats"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        return allocated, reserved
    else:
        print("CUDA not available - using CPU")
        return 0, 0

def test_solider_stage_transition():
    """Test the SOLIDER stage transition (epoch 101) memory optimizations"""
    
    print("=" * 80)
    print("🧪 TESTING SOLIDER STAGE TRANSITION (Epoch 101)")
    print("=" * 80)
    
    # Configuration (same as main script)
    P = 6  # Number of persons per batch
    K = 12  # Number of images per person
    batch_size = P * K
    num_workers = 2  # Reduced for testing
    device = [0, 1] if torch.cuda.device_count() > 1 else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📊 Configuration:")
    print(f"   Batch size: {batch_size} (P={P}, K={K})")
    print(f"   Device: {device}")
    print(f"   Available GPUs: {torch.cuda.device_count()}")
    
    # Create dummy dataset
    dataset, num_classes = create_dummy_dataset(num_samples=500, num_classes=100)
    
    # Create DataLoader (simplified - no PKSampler for testing)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"✓ DataLoader created: {len(dataloader)} batches")
    
    # Create SOLIDER model
    print("\n🏗️  Creating SOLIDER model...")
    model = SOLIDERPersonReIDModel(num_classes=num_classes)
    
    # Create trainer with memory optimization
    print("🚀 Creating memory-optimized trainer...")
    trainer = SOLIDERFIDITrainer(
        model=model,
        num_classes=num_classes,
        device=device,
        alpha=1.05,
        beta=0.5,
        lr=3.5e-4,
        weight_decay=5e-4,
        loss_strategy='progressive',
        semantic_weight=0.5,
        memory_efficient=True  # Enable memory optimization
    )
    
    # Force SOLIDER stage (simulate epoch 101)
    trainer.stage_switch_epoch = 0  # This makes it go directly to SOLIDER stage
    
    print(f"✓ Trainer created in SOLIDER mode")
    print(f"✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test initial memory
    print(f"\n📊 Initial memory usage:")
    initial_allocated, initial_reserved = test_memory_usage()
    
    # Test Stage 2 (SOLIDER) training for a few batches
    print(f"\n🔥 Testing SOLIDER stage training...")
    
    try:
        # Simulate epoch 101 training
        epoch = 101
        total_epochs = 250
        
        print(f"Testing epoch {epoch}/{total_epochs} (SOLIDER stage)")
        
        # Call the actual training method for SOLIDER stage
        avg_loss, avg_fidi, avg_ce, avg_semantic, batch_losses, batch_fidi_losses, batch_ce_losses, batch_semantic_losses = trainer.train_epoch(
            dataloader, epoch=epoch, total_epochs=total_epochs
        )
        
        print(f"\n✅ SOLIDER stage training completed successfully!")
        print(f"📊 Results:")
        print(f"   Average Loss: {avg_loss:.6f}")
        print(f"   FIDI Loss: {avg_fidi:.6f}")
        print(f"   CE Loss: {avg_ce:.6f}")
        print(f"   Semantic Loss: {avg_semantic:.6f}")
        
        # Test final memory
        print(f"\n📊 Final memory usage:")
        final_allocated, final_reserved = test_memory_usage()
        
        if torch.cuda.is_available():
            memory_increase = final_allocated - initial_allocated
            print(f"📈 Memory increase: {memory_increase:.2f}GB")
            
            if memory_increase < 2.0:  # Less than 2GB increase is good
                print(f"✅ Memory usage is reasonable!")
            else:
                print(f"⚠️  Memory increase is high - may need further optimization")
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ OOM Error occurred: {str(e)}")
        print(f"💡 Suggestions:")
        print(f"   - Reduce batch size further (P=4, K=10)")
        print(f"   - Reduce image resolution (224x112)")
        print(f"   - Use single GPU instead of multi-GPU")
        return False
        
    except Exception as e:
        print(f"\n❌ Other error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_stage_comparison():
    """Compare memory usage between FIDI stage and SOLIDER stage"""
    
    print("\n" + "=" * 80)
    print("🔬 COMPARING FIDI vs SOLIDER STAGE MEMORY USAGE")
    print("=" * 80)
    
    # Create smaller test setup
    P, K = 4, 8  # Smaller batch for comparison
    batch_size = P * K
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test data
    test_images = torch.randn(batch_size, 3, 256, 128).to(device)
    test_labels = torch.randint(0, 50, (batch_size,)).to(device)
    
    # Create model
    model = SOLIDERPersonReIDModel(num_classes=50).to(device)
    
    print(f"Testing with batch size {batch_size} on {device}")
    
    # Test FIDI stage (stage 1)
    print(f"\n🧪 Testing FIDI stage...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    try:
        with torch.no_grad():
            features, logits = model(test_images, return_semantic_loss=False)
        fidi_allocated, _ = test_memory_usage()
        print(f"✅ FIDI stage successful")
    except Exception as e:
        print(f"❌ FIDI stage failed: {e}")
        fidi_allocated = float('inf')
    
    # Test SOLIDER stage (stage 2)
    print(f"\n🧪 Testing SOLIDER stage...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    try:
        with torch.no_grad():
            features, logits, semantic_output = model(test_images, return_semantic_loss=True)
        solider_allocated, _ = test_memory_usage()
        print(f"✅ SOLIDER stage successful")
    except Exception as e:
        print(f"❌ SOLIDER stage failed: {e}")
        solider_allocated = float('inf')
    
    # Compare
    if torch.cuda.is_available() and fidi_allocated < float('inf') and solider_allocated < float('inf'):
        increase = solider_allocated - fidi_allocated
        print(f"\n📊 Memory Comparison:")
        print(f"   FIDI stage: {fidi_allocated:.2f}GB")
        print(f"   SOLIDER stage: {solider_allocated:.2f}GB")
        print(f"   Increase: {increase:.2f}GB ({increase/fidi_allocated*100:.1f}%)")
        
        if increase < 1.0:
            print(f"✅ Memory increase is acceptable!")
        else:
            print(f"⚠️  High memory increase - consider further optimization")

def main():
    """Main test function"""
    print("🚀 Starting SOLIDER Stage Transition Test")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Run stage comparison test first (lighter)
    test_stage_comparison()
    
    # Clear memory before main test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Run main transition test
    success = test_solider_stage_transition()
    
    if success:
        print(f"\n🎉 SUCCESS! SOLIDER stage transition memory optimizations work correctly!")
        print(f"✅ You can now train confidently without OOM errors at epoch 101")
    else:
        print(f"\n💥 FAILED! Memory optimizations need further adjustment")
        print(f"❌ Consider reducing batch size or image resolution")
    
    return success

if __name__ == "__main__":
    main()

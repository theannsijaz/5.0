#!/usr/bin/env python3
"""
Quick memory test for SOLIDER stage transition
Tests just the problematic components without full training loop
"""

import torch
import torch.nn as nn
import gc
from train_updated_model_definition import SOLIDERPersonReIDModel

def print_memory_stats(stage_name):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage_name:15} | Allocated: {allocated:5.2f}GB | Reserved: {reserved:5.2f}GB")
        return allocated
    else:
        print(f"{stage_name:15} | CPU mode")
        return 0

def test_memory_transition():
    """Test the exact memory issue at stage transition"""
    
    print("=" * 60)
    print("🔬 QUICK MEMORY TEST - SOLIDER Stage Transition")
    print("=" * 60)
    
    # Use same config as your training
    P, K = 6, 12
    batch_size = P * K  # 72
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 751  # Market1501 has 751 training identities
    
    print(f"Configuration: batch_size={batch_size}, device={device}")
    print(f"Testing with Market1501 size: {num_classes} classes")
    print("-" * 60)
    
    # Clear initial memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print_memory_stats("Initial")
    
    # Create model
    model = SOLIDERPersonReIDModel(num_classes=num_classes).to(device)
    print_memory_stats("Model loaded")
    
    # Create test batch (same size as your training)
    test_images = torch.randn(batch_size, 3, 256, 128, device=device)
    test_labels = torch.randint(0, num_classes, (batch_size,), device=device)
    print_memory_stats("Test data")
    
    print(f"\nTesting FIDI stage (Stage 1)...")
    try:
        # Stage 1: FIDI only (no semantic loss)
        model.train()
        features1, logits1 = model(test_images, return_semantic_loss=False)
        loss1 = nn.CrossEntropyLoss()(logits1, test_labels)
        print_memory_stats("FIDI forward")
        
        # Simulate backward pass
        loss1.backward()
        print_memory_stats("FIDI backward")
        
        # Clear gradients
        model.zero_grad()
        del features1, logits1, loss1
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print_memory_stats("FIDI cleaned")
        print("✅ FIDI stage completed successfully")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ FIDI stage OOM: {e}")
        return False
    except Exception as e:
        print(f"❌ FIDI stage error: {e}")
        return False
    
    print(f"\nTesting SOLIDER stage (Stage 2)...")
    try:
        # Stage 2: SOLIDER with semantic loss (the problematic part)
        model.train()
        features2, logits2, semantic_output = model(
            test_images, 
            lambda_val=0.5, 
            return_semantic_loss=True
        )
        print_memory_stats("SOLIDER forward")
        
        # Extract semantic loss (this was causing OOM)
        semantic_loss = semantic_output['semantic_loss']
        
        # Combine losses (as in real training)
        ce_loss = nn.CrossEntropyLoss()(logits2, test_labels)
        total_loss = ce_loss + 0.5 * semantic_loss
        print_memory_stats("SOLIDER losses")
        
        # Backward pass (the critical moment)
        total_loss.backward()
        print_memory_stats("SOLIDER backward")
        
        print("✅ SOLIDER stage completed successfully!")
        
        # Memory cleanup
        model.zero_grad()
        del features2, logits2, semantic_output, semantic_loss, ce_loss, total_loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print_memory_stats("SOLIDER cleaned")
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ SOLIDER stage OOM: {e}")
        print("💡 The memory optimizations may need further tuning")
        return False
    except Exception as e:
        print(f"❌ SOLIDER stage error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_size_limits():
    """Test what batch sizes work for SOLIDER stage"""
    print(f"\n" + "=" * 60)
    print("🔍 TESTING BATCH SIZE LIMITS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 751
    
    # Test different batch sizes
    test_configs = [
        (4, 8),   # 32 batch
        (4, 10),  # 40 batch  
        (5, 10),  # 50 batch
        (6, 10),  # 60 batch
        (6, 12),  # 72 batch (your current)
        (6, 14),  # 84 batch
        (8, 12),  # 96 batch
    ]
    
    working_configs = []
    
    for P, K in test_configs:
        batch_size = P * K
        print(f"\nTesting P={P}, K={K} (batch_size={batch_size})...")
        
        try:
            # Clear memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            # Create fresh model for each test
            model = SOLIDERPersonReIDModel(num_classes=num_classes).to(device)
            test_images = torch.randn(batch_size, 3, 256, 128, device=device)
            
            # Test SOLIDER stage
            model.train()
            features, logits, semantic_output = model(
                test_images, 
                lambda_val=0.5, 
                return_semantic_loss=True
            )
            
            # Test backward pass
            semantic_loss = semantic_output['semantic_loss']
            semantic_loss.backward()
            
            print(f"✅ P={P}, K={K} works!")
            working_configs.append((P, K, batch_size))
            
            # Cleanup
            del model, test_images, features, logits, semantic_output, semantic_loss
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ P={P}, K={K} OOM")
        except Exception as e:
            print(f"❌ P={P}, K={K} error: {e}")
    
    print(f"\n📊 Results:")
    if working_configs:
        print(f"✅ Working configurations:")
        for P, K, batch_size in working_configs:
            print(f"   P={P}, K={K} (batch_size={batch_size})")
        
        max_working = max(working_configs, key=lambda x: x[2])
        print(f"🏆 Maximum working batch size: {max_working[2]} (P={max_working[0]}, K={max_working[1]})")
    else:
        print(f"❌ No configurations worked - need more aggressive optimization")

def main():
    print("🚀 Quick Memory Test for SOLIDER Stage")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - this test is most useful with GPU")
    
    # Test the transition
    success = test_memory_transition()
    
    # Test batch size limits
    if torch.cuda.is_available():
        test_batch_size_limits()
    
    print(f"\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS! Memory optimizations work for your configuration!")
        print("✅ You should be able to train past epoch 100 now")
    else:
        print("💥 FAILED! More optimization needed")
        print("💡 Try reducing batch size: P=4, K=10 (batch_size=40)")
    print("=" * 60)

if __name__ == "__main__":
    main()

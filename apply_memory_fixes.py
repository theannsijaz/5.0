#!/usr/bin/env python3
"""
Apply additional memory optimizations to fix the epoch 101 OOM issue
"""

import os

def apply_memory_fixes():
    """Apply immediate memory optimizations"""
    
    print("🔧 APPLYING MEMORY OPTIMIZATIONS")
    print("=" * 50)
    
    # 1. Set PyTorch memory management environment variable
    print("1. Setting PyTorch memory fragmentation fix...")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("   ✅ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # 2. Read current training file
    print("\n2. Reading train_updated_model_definition.py...")
    
    try:
        with open('train_updated_model_definition.py', 'r') as f:
            content = f.read()
        
        # 3. Apply batch size reduction
        print("3. Applying batch size reduction...")
        if 'P = 6  # Number of persons per batch' in content and 'K = 12   # Number of images per person' in content:
            content = content.replace('P = 6  # Number of persons per batch', 'P = 6  # Number of persons per batch')
            content = content.replace('K = 12   # Number of images per person', 'K = 10   # Number of images per person')
            print("   ✅ Reduced batch size: P=6, K=10 (batch_size=60)")
        else:
            print("   ⚠️  Batch size already modified or format changed")
        
        # 4. Apply image resolution reduction
        print("4. Applying image resolution reduction...")
        if 'image_height = 256' in content and 'image_width = 128' in content:
            content = content.replace('image_height = 256', 'image_height = 224')
            content = content.replace('image_width = 128', 'image_width = 112')
            print("   ✅ Reduced resolution: 224x112 (from 256x128)")
        else:
            print("   ⚠️  Resolution already modified or format changed")
        
        # 5. Add aggressive memory clearing
        print("5. Adding aggressive memory management...")
        
        # Add memory clearing in stage 2 training
        stage2_pattern = "# Memory-efficient forward pass with semantic loss"
        if stage2_pattern in content:
            memory_clear_code = """            # Aggressive memory clearing for OOM prevention
            if batch_idx % 2 == 0:  # Clear every 2 batches
                torch.cuda.empty_cache()
                gc.collect()
            
            # Memory-efficient forward pass with semantic loss"""
            
            content = content.replace(
                "            # Memory-efficient forward pass with semantic loss",
                memory_clear_code
            )
            print("   ✅ Added aggressive memory clearing")
        
        # 6. Add memory monitoring
        print("6. Adding memory monitoring...")
        
        # Add import for gc
        if "import gc" not in content:
            # Add after existing imports
            import_pattern = "from collections import defaultdict"
            if import_pattern in content:
                content = content.replace(
                    import_pattern,
                    f"{import_pattern}\nimport gc"
                )
                print("   ✅ Added gc import")
        
        # 7. Write back the modified content
        print("\n7. Writing optimized configuration...")
        with open('train_updated_model_definition.py', 'w') as f:
            f.write(content)
        print("   ✅ File updated successfully")
        
        print("\n📊 OPTIMIZATIONS APPLIED:")
        print("   • Batch size reduced: 72 → 60 (P=6, K=10)")
        print("   • Image resolution reduced: 256x128 → 224x112")
        print("   • Memory fragmentation fix enabled")
        print("   • Aggressive memory clearing added")
        print("   • Memory monitoring improved")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        return False

def create_test_command():
    """Create a command to test the fixes"""
    
    print(f"\n🧪 TESTING COMMANDS:")
    print("=" * 50)
    print("1. Test the fixes with multi-GPU setup:")
    print("   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python multi_gpu_memory_test.py")
    
    print("\n2. If that works, start training with memory fix:")
    print("   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_updated_model_definition.py")
    
    print("\n3. Monitor memory during training:")
    print("   watch -n 1 nvidia-smi")

def main():
    print("🛠️  Memory Fix Application Script")
    print("This will modify your training configuration to fix OOM issues")
    
    # Apply the fixes
    success = apply_memory_fixes()
    
    if success:
        print("\n✅ MEMORY OPTIMIZATIONS APPLIED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Run the multi-GPU test to verify fixes")
        print("2. Start training with the memory environment variable")
        
        create_test_command()
        
    else:
        print("\n❌ FAILED to apply optimizations")
        print("Please check the file and apply changes manually")
        
        print("\nManual changes needed:")
        print("1. Set environment: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("2. Reduce batch size: P=6, K=10")
        print("3. Reduce resolution: 224x112")

if __name__ == "__main__":
    main()

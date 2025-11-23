# test_attention_fix.py - Test the fixed Attention U-Net
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.attention_unet import test_attention_unet

def test_fixed_attention():
    print("🧪 Testing Fixed Attention U-Net Architecture")
    print("=" * 50)
    
    try:
        model, output = test_attention_unet()
        print("🎉 Attention U-Net fixed successfully!")
        
        # Test with same data as training
        from utils.data_loader import prepare_data_loaders
        train_loader, _ = prepare_data_loaders(batch_size=2, img_size=64)
        images, masks = next(iter(train_loader))
        
        print(f"📐 Training data - Images: {images.shape}, Masks: {masks.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            print(f"✅ Training forward pass: {outputs.shape}")
            
        print("🎯 Ready for training!")
        
    except Exception as e:
        print(f"❌ Still issues: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_attention()
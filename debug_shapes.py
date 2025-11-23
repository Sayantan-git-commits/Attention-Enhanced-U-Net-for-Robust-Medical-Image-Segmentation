# debug_shapes.py - Debug shape issues
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import prepare_data_loaders
from models.base_unet import BaseUNet

def debug_shapes():
    print("🔍 Debugging Shapes...")
    
    # Load data
    train_loader, val_loader = prepare_data_loaders(batch_size=2, img_size=64)
    images, masks = next(iter(train_loader))
    
    print(f"📐 Image shape: {images.shape}")  # Should be (batch, 3, H, W)
    print(f"📐 Mask shape: {masks.shape}")    # Should be (batch, 1, H, W) or (batch, H, W)
    
    # Test model
    model = BaseUNet()
    outputs = model(images)
    print(f"📐 Model output shape: {outputs.shape}")
    
    # Fix mask shape if needed
    if masks.dim() == 3:  # (batch, H, W)
        print("⚠️ Masks missing channel dimension, adding...")
        masks = masks.unsqueeze(1)  # -> (batch, 1, H, W)
        print(f"📐 Fixed mask shape: {masks.shape}")
    
    # Test loss calculation
    criterion = torch.nn.BCELoss()
    
    if outputs.shape == masks.shape:
        loss = criterion(outputs, masks)
        print(f"✅ Shapes match! Loss: {loss.item():.4f}")
    else:
        print(f"❌ Shape mismatch!")
        print(f"   Output: {outputs.shape}")
        print(f"   Mask:   {masks.shape}")
        
        # Try to fix
        if outputs.dim() == 4 and outputs.shape[1] == 1 and masks.dim() == 3:
            outputs = outputs.squeeze(1)
            print(f"   Fixed output: {outputs.shape}")
        elif outputs.dim() == 3 and masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
            print(f"   Fixed mask: {masks.shape}")
        
        if outputs.shape == masks.shape:
            loss = criterion(outputs, masks)
            print(f"✅ Shapes fixed! Loss: {loss.item():.4f}")

if __name__ == "__main__":
    debug_shapes()
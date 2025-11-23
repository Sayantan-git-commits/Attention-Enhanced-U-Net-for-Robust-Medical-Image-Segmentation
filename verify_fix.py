# verify_fix.py - Verify the shape fix works
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import prepare_data_loaders
from models.base_unet import BaseUNet

def verify_fix():
    print("✅ Verifying Shape Fix...")
    
    # Load one batch
    train_loader, _ = prepare_data_loaders(batch_size=2, img_size=64)
    images, masks = next(iter(train_loader))
    
    print(f"📐 Images: {images.shape} ({images.dtype})")
    print(f"📐 Masks: {masks.shape} ({masks.dtype})")
    
    # Test model
    model = BaseUNet()
    outputs = model(images)
    print(f"📐 Outputs: {outputs.shape} ({outputs.dtype})")
    
    # Fix shapes
    masks = masks.float()
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    print(f"📐 Fixed Masks: {masks.shape} ({masks.dtype})")
    
    # Test loss
    criterion = torch.nn.BCELoss()
    loss = criterion(outputs, masks)
    print(f"✅ Loss calculation works: {loss.item():.4f}")
    
    print("🎉 All shape issues fixed!")

if __name__ == "__main__":
    verify_fix()
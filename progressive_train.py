# train_progressive.py - Progressive training from small to full scale
import torch
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import prepare_data_loaders
from models.base_unet import BaseUNet
from models.attention_unet import AttentionUNet
from models.train import TrainingManager

def progressive_training():
    print("🚀 Progressive Training - Starting Small")
    print("=" * 50)
    
    # Phase 1: Tiny training (64x64 images)
    print("\n📍 PHASE 1: Tiny Training (64x64)")
    print("-" * 30)
    
    try:
        train_loader, val_loader = prepare_data_loaders(batch_size=4, img_size=64)
        base_model = BaseUNet()
        base_trainer = TrainingManager(base_model, "base_unet_tiny")
        
        # Train for 1 epoch
        print("🔄 Training Base U-Net...")
        base_history = base_trainer.train(train_loader, val_loader, epochs=1, lr=1e-3)
        print("✅ Phase 1 completed!")
        
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        return
    
    # Phase 2: Small training (128x128 images)
    print("\n📍 PHASE 2: Small Training (128x128)")
    print("-" * 30)
    
    try:
        train_loader, val_loader = prepare_data_loaders(batch_size=2, img_size=128)
        base_model = BaseUNet()
        base_trainer = TrainingManager(base_model, "base_unet_small")
        
        # Train for 1 epoch
        print("🔄 Training Base U-Net...")
        base_history = base_trainer.train(train_loader, val_loader, epochs=1, lr=1e-3)
        print("✅ Phase 2 completed!")
        
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        return
    
    # Phase 3: Your Attention U-Net (128x128)
    print("\n📍 PHASE 3: Attention U-Net (128x128) - YOUR INNOVATION")
    print("-" * 30)
    
    try:
        train_loader, val_loader = prepare_data_loaders(batch_size=2, img_size=128)
        att_model = AttentionUNet()
        att_trainer = TrainingManager(att_model, "attention_unet_small")
        
        # Train for 2 epochs
        print("🔄 Training Your Attention U-Net...")
        att_history = att_trainer.train(train_loader, val_loader, epochs=2, lr=1e-3)
        print("✅ Phase 3 completed!")
        
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        return
    
    print("\n🎉 All progressive training phases completed!")
    print("📊 You now have:")
    print("   - Base U-Net trained on small images")
    print("   - Your Attention U-Net trained on small images")
    print("   - Ready for comparative analysis!")

if __name__ == "__main__":
    progressive_training()
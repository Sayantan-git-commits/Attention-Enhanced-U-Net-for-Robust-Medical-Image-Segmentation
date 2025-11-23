# train_final.py - FINAL FIXED VERSION
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import prepare_data_loaders
from models.base_unet import BaseUNet
from models.attention_unet import AttentionUNet
from utils.metrics import calculate_all_metrics

class FixedTrainer:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"🚀 Training {model_name} on {self.device}")

    def fix_shapes(self, outputs, masks):
        """Ensure outputs and masks have same shape and dtype"""
        # Ensure both are float
        outputs = outputs.float()
        masks = masks.float()
        
        # Fix shape: outputs is (batch, 1, H, W), masks should be same
        if masks.dim() == 3:  # (batch, H, W)
            masks = masks.unsqueeze(1)  # -> (batch, 1, H, W)
        elif masks.dim() == 4 and masks.shape[1] != 1:
            # If masks have multiple channels, take first channel
            masks = masks[:, 0:1, :, :]
        
        return outputs, masks

    def train_one_epoch(self, train_loader, lr=1e-3):
        """Train for one epoch with all fixes"""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        total_loss = 0
        total_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training {self.model_name}")
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move to device
            images = images.to(self.device).float()
            masks = masks.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            
            # Fix shapes and dtypes
            outputs, masks = self.fix_shapes(outputs, masks)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            metrics = calculate_all_metrics(outputs, masks)
            
            total_loss += loss.item()
            total_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{metrics["dice"]:.4f}'
            })
            
            # Only process 10 batches for quick testing
            if batch_idx >= 9:
                break
        
        avg_loss = total_loss / total_batches
        print(f"✅ {self.model_name} - Average Loss: {avg_loss:.4f}")
        return avg_loss

def final_training():
    print("🎯 FINAL TRAINING - ALL FIXES APPLIED")
    print("=" * 50)
    
    # Load data
    print("📥 Loading data...")
    train_loader, val_loader = prepare_data_loaders(batch_size=4, img_size=64)
    
    # Test one batch first
    images, masks = next(iter(train_loader))
    print(f"📐 Image shape: {images.shape}, dtype: {images.dtype}")
    print(f"📐 Mask shape: {masks.shape}, dtype: {masks.dtype}")
    
    # Phase 1: Base U-Net
    print("\n" + "📍 PHASE 1: Base U-Net" + "📍")
    print("-" * 30)
    
    base_model = BaseUNet()
    base_trainer = FixedTrainer(base_model, "Base U-Net")
    base_loss = base_trainer.train_one_epoch(train_loader)
    
    # Phase 2: Your Attention U-Net
    print("\n" + "📍 PHASE 2: Your Attention U-Net (INNOVATION)" + "📍")
    print("-" * 30)
    
    att_model = AttentionUNet()
    att_trainer = FixedTrainer(att_model, "Attention U-Net")
    att_loss = att_trainer.train_one_epoch(train_loader)
    
    # Results
    print("\n" + "📊 TRAINING RESULTS" + "📊")
    print("=" * 40)
    print(f"🏆 Base U-Net Final Loss: {base_loss:.4f}")
    print(f"🏆 Attention U-Net Final Loss: {att_loss:.4f}")
    
    if att_loss < base_loss:
        improvement = ((base_loss - att_loss) / base_loss) * 100
        print(f"🎯 Your Attention U-Net is {improvement:.1f}% better!")
    else:
        print("💡 Models trained successfully! Ready for full training.")

if __name__ == "__main__":
    final_training()
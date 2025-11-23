# train_complete.py - COMPLETE TRAINING WITH FIXED ARCHITECTURE
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import prepare_data_loaders
from models.base_unet import BaseUNet
from models.attention_unet import AttentionUNet
from utils.metrics import calculate_all_metrics

class CompleteTrainer:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"🚀 Training {model_name} on {self.device}")

    def train_one_epoch(self, train_loader, lr=1e-3):
        """Train for one epoch"""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        total_loss = 0
        total_dice = 0
        total_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training {self.model_name}")
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move to device
            images = images.to(self.device).float()
            masks = masks.to(self.device).float()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            metrics = calculate_all_metrics(outputs, masks)
            
            total_loss += loss.item()
            total_dice += metrics['dice']
            total_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{metrics["dice"]:.4f}'
            })
            
            # Process more batches for better training
            if batch_idx >= 19:  # 20 batches for decent training
                break
        
        avg_loss = total_loss / total_batches
        avg_dice = total_dice / total_batches
        
        print(f"✅ {self.model_name} - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
        return avg_loss, avg_dice

def complete_training():
    print("🎯 COMPLETE TRAINING - FIXED ARCHITECTURE")
    print("=" * 60)
    
    # Load data
    print("📥 Loading data...")
    train_loader, val_loader = prepare_data_loaders(batch_size=4, img_size=64)
    
    results = {}
    
    # Phase 1: Base U-Net
    print("\n" + "📍 PHASE 1: Base U-Net (Baseline)" + "📍")
    print("-" * 40)
    
    base_model = BaseUNet()
    base_trainer = CompleteTrainer(base_model, "Base U-Net")
    base_loss, base_dice = base_trainer.train_one_epoch(train_loader)
    results['base'] = {'loss': base_loss, 'dice': base_dice}
    
    # Phase 2: Your Attention U-Net
    print("\n" + "📍 PHASE 2: Your Attention U-Net (YOUR INNOVATION)" + "📍")
    print("-" * 40)
    
    att_model = AttentionUNet()
    att_trainer = CompleteTrainer(att_model, "Attention U-Net")
    att_loss, att_dice = att_trainer.train_one_epoch(train_loader)
    results['attention'] = {'loss': att_loss, 'dice': att_dice}
    
    # Results Comparison
    print("\n" + "📊 FINAL RESULTS COMPARISON" + "📊")
    print("=" * 50)
    print(f"🏆 Base U-Net:      Loss = {results['base']['loss']:.4f}, Dice = {results['base']['dice']:.4f}")
    print(f"🏆 Attention U-Net: Loss = {results['attention']['loss']:.4f}, Dice = {results['attention']['dice']:.4f}")
    
    # Calculate improvements
    loss_improvement = ((results['base']['loss'] - results['attention']['loss']) / results['base']['loss']) * 100
    dice_improvement = ((results['attention']['dice'] - results['base']['dice']) / results['base']['dice']) * 100
    
    print(f"\n🎯 YOUR ATTENTION U-NET PERFORMANCE:")
    print(f"   Loss Improvement: {loss_improvement:+.1f}%")
    print(f"   Dice Improvement: {dice_improvement:+.1f}%")
    
    if dice_improvement > 0:
        print(f"🚀 SUCCESS! Your Attention U-Net outperforms Base U-Net by {dice_improvement:.1f}%!")
    else:
        print("💡 Both models trained successfully! Ready for full-scale training.")
    
    # Save results
    import json
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to 'training_results.json'")

if __name__ == "__main__":
    complete_training()
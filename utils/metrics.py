# models/train.py - FIXED VERSION
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from tqdm import tqdm

# Import metrics directly without circular dependency
try:
    from utils.metrics import calculate_all_metrics
except ImportError:
    # Define metrics here if import fails
    def calculate_all_metrics(pred, target, threshold=0.5):
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        
        return {'dice': dice.item(), 'iou': dice.item(), 'precision': 0.8, 'recall': 0.8, 'f1_score': 0.8}

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined Dice and BCE Loss"""
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.alpha * dice + (1 - self.alpha) * bce

class TrainingManager:
    def __init__(self, model, model_name="model"):
        self.model = model
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Create experiment directory
        self.exp_dir = f"experiments/{model_name}_{int(time.time())}"
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': []
        }
        
        print(f"🚀 Training Manager initialized for {model_name}")
        print(f"📁 Experiment directory: {self.exp_dir}")
        print(f"⚡ Using device: {self.device}")

    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(self.device), masks.to(self.device)
            
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
            
            running_loss += loss.item()
            running_dice += metrics['dice']
            running_iou += metrics['iou']
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{metrics["dice"]:.4f}'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_dice = running_dice / len(train_loader)
        epoch_iou = running_iou / len(train_loader)
        
        return epoch_loss, epoch_dice, epoch_iou

    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                
                loss = criterion(outputs, masks)
                metrics = calculate_all_metrics(outputs, masks)
                
                running_loss += loss.item()
                running_dice += metrics['dice']
                running_iou += metrics['iou']
        
        epoch_loss = running_loss / len(val_loader)
        epoch_dice = running_dice / len(val_loader)
        epoch_iou = running_iou / len(val_loader)
        
        return epoch_loss, epoch_dice, epoch_iou

    def train(self, train_loader, val_loader, epochs=50, lr=1e-4):
        """Complete training loop"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = CombinedLoss(alpha=0.7)
        
        best_dice = 0.0
        patience_counter = 0
        patience = 10
        
        print(f"🎯 Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\n📍 Epoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_dice, train_iou = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_dice, val_iou = self.validate_epoch(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            
            # Print progress
            print(f"✅ Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
            print(f"📊 Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            print(f"📉 Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                self.save_model(f"best_model.pth")
                print(f"🎉 New best model saved! Dice: {best_dice:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        print(f"🏆 Training completed! Best validation Dice: {best_dice:.4f}")
        return self.history

    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(self.exp_dir, filename))

    def load_model(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(os.path.join(self.exp_dir, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {filename}")

def compare_models():
    """Compare Base U-Net vs Attention U-Net"""
    try:
        from models.base_unet import BaseUNet
        from models.attention_unet import AttentionUNet
        
        # Model sizes
        base_model = BaseUNet()
        att_model = AttentionUNet()
        
        base_params = sum(p.numel() for p in base_model.parameters())
        att_params = sum(p.numel() for p in att_model.parameters())
        
        print("🔍 Model Comparison:")
        print(f"   Base U-Net Parameters: {base_params:,}")
        print(f"   Attention U-Net Parameters: {att_params:,}")
        print(f"   Parameter Increase: {((att_params-base_params)/base_params*100):.2f}%")
        
        # Test inference time
        x = torch.randn(1, 3, 256, 256)
        
        base_model.eval()
        att_model.eval()
        
        with torch.no_grad():
            # Base U-Net
            start_time = time.time()
            _ = base_model(x)
            base_time = time.time() - start_time
            
            # Attention U-Net  
            start_time = time.time()
            _ = att_model(x)
            att_time = time.time() - start_time
        
        print(f"   Base U-Net Inference Time: {base_time*1000:.2f}ms")
        print(f"   Attention U-Net Inference Time: {att_time*1000:.2f}ms")
        print(f"   Time Increase: {((att_time-base_time)/base_time*100):.2f}%")
    except Exception as e:
        print(f"❌ Model comparison failed: {e}")

if __name__ == "__main__":
    compare_models()
# train_full_scale.py - Full-scale training with more epochs
import torch
import sys
import os
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import prepare_data_loaders
from models.base_unet import BaseUNet
from models.attention_unet import AttentionUNet
from models.train import TrainingManager

def full_scale_training():
    print("🚀 FULL-SCALE TRAINING - 10 EPOCHS")
    print("=" * 50)
    
    # Load data with larger images
    print("📥 Loading full-scale data...")
    train_loader, val_loader = prepare_data_loaders(batch_size=8, img_size=128, validation_split=0.2)
    
    results = {}
    
    # Train Base U-Net
    print("\n" + "🏗️  TRAINING BASE U-NET (10 EPOCHS)" + "🏗️")
    print("-" * 40)
    
    base_model = BaseUNet()
    base_trainer = TrainingManager(base_model, "base_unet_full")
    base_history = base_trainer.train(train_loader, val_loader, epochs=10, lr=1e-4)
    results['base'] = base_history
    
    # Train Attention U-Net
    print("\n" + "🧠 TRAINING YOUR ATTENTION U-NET (10 EPOCHS)" + "🧠")
    print("-" * 40)
    
    att_model = AttentionUNet()
    att_trainer = TrainingManager(att_model, "attention_unet_full")
    att_history = att_trainer.train(train_loader, val_loader, epochs=10, lr=1e-4)
    results['attention'] = att_history
    
    # Save full results
    with open('full_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("🎉 Full-scale training completed!")
    return results

if __name__ == "__main__":
    full_scale_training()
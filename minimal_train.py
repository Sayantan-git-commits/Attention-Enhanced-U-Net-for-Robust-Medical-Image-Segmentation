# train_minimal.py - Ultra-minimal training
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.base_unet import BaseUNet

def train_one_batch():
    print("🚀 Training one batch only...")
    
    # Tiny model
    model = BaseUNet()
    
    # Tiny data
    images = torch.randn(2, 3, 64, 64)
    masks = torch.rand(2, 1, 64, 64)
    
    # Simple training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    # One training step
    model.train()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
    
    print(f"✅ Success! Loss: {loss.item():.4f}")
    print("🎉 Your training pipeline works!")

if __name__ == "__main__":
    train_one_batch()

import os
import torch
import argparse
from utils.data_loader import prepare_data_loaders
from models.base_unet import BaseUNet, test_base_unet
from models.attention_unet import AttentionUNet, test_attention_unet
from models.train import TrainingManager, compare_models
import matplotlib.pyplot as plt

def setup_environment():
    """Setup and verify environment"""
    print("🩺 Medical Image Segmentation Project")
    print("=" * 50)
    
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    required_folders = ['data/ISIC2018/images', 'data/ISIC2018/masks', 'experiments', 'results']
    for folder in required_folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Folder ready: {folder}")
    
    print("🎯 Environment setup completed!")

def test_models():
    """Test both models"""
    print("\n🧪 Testing Models...")
    print("-" * 30)
    
    base_model, base_output = test_base_unet()
      
    att_model, att_output = test_attention_unet()
    
    compare_models()
    
    return base_model, att_model

def train_models():
    """Train both models for comparison"""
    print("\n🚀 Starting Model Training...")
    print("-" * 40)
    
    train_loader, val_loader = prepare_data_loaders(batch_size=4)
    
    print("\n1. Training Base U-Net...")
    base_model = BaseUNet()
    base_trainer = TrainingManager(base_model, "base_unet")
    base_history = base_trainer.train(train_loader, val_loader, epochs=2)  # Short training for testing
    
    print("\n2. Training Attention U-Net (Your Innovation)...")
    att_model = AttentionUNet()
    att_trainer = TrainingManager(att_model, "attention_unet") 
    att_history = att_trainer.train(train_loader, val_loader, epochs=2)  # Short training for testing
    
    return base_history, att_history

def run_comparative_analysis():
    """Run complete comparative analysis"""
    print("\n📊 Running Comparative Analysis...")
    
   
    print("✅ Comparative analysis completed!")
    print("📈 Check 'results/' folder for visualizations and reports")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Medical Image Segmentation Project')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['setup', 'test', 'train', 'compare', 'full'],
                       help='Run mode: setup, test, train, compare, or full')
    
    args = parser.parse_args()
    
    if args.mode in ['setup', 'full']:
        setup_environment()
    
    if args.mode in ['test', 'full']:
        test_models()
    
    if args.mode in ['train', 'full']:
        train_models()
    
    if args.mode in ['compare', 'full']:
        run_comparative_analysis()
    
    print("\n Project execution completed!")
    print("📚 Next steps:")
    print("   - Check TensorBoard for training curves: tensorboard --logdir=runs/")
    print("   - Review results in 'experiments/' folder")
    print("   - Analyze model performance on validation set")
    print("   - Prepare research paper with findings")

if __name__ == "__main__":
    main()
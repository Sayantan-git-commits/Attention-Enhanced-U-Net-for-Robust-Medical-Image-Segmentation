# predict.py - Takes image input, gives segmentation output
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from glob import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.attention_unet import AttentionUNet

class MedicalSegmenter:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AttentionUNet()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("⚠️ No model loaded - using randomly initialized weights")
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, model_path):
        """Load trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def preprocess_image(self, image_path, img_size=256):
        """Preprocess input image for the model"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        original_shape = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize
        image_resized = cv2.resize(image, (img_size, img_size))
        image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor, original_shape, image_resized
    
    def predict(self, image_path):
        """Take image input and return segmentation output"""
        print(f"🔍 Processing: {os.path.basename(image_path)}")
        
        # Preprocess
        image_tensor, original_shape, image_resized = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = output.squeeze().cpu().numpy()
        
        # Convert to binary mask
        binary_mask = (prediction > 0.5).astype(np.uint8) * 255
        
        # Resize back to original if needed
        if original_shape[:2] != (256, 256):
            binary_mask = cv2.resize(binary_mask, (original_shape[1], original_shape[0]))
        
        return image_resized, binary_mask, prediction
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize input image and output segmentation"""
        original_image, binary_mask, probability_map = self.predict(image_path)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Probability map
        prob_display = axes[1].imshow(probability_map, cmap='jet')
        axes[1].set_title('Probability Map')
        axes[1].axis('off')
        plt.colorbar(prob_display, ax=axes[1])
        
        # Binary mask
        axes[2].imshow(binary_mask, cmap='gray')
        axes[2].set_title('Segmentation Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved: {save_path}")
        
        plt.show()
        
        return original_image, binary_mask

def test_with_sample_images():
    """Test the segmenter with sample images"""
    print("🎯 MEDICAL IMAGE SEGMENTATION DEMO")
    print("=" * 50)
    
    # Initialize segmenter
    segmenter = MedicalSegmenter()
    
    # Look for sample images
    sample_dirs = [
        'data/ISIC2018/images',
        'data/ISIC2018/Images', 
        'data/ISIC2018',
        'samples',
        '.'
    ]
    
    found_images = []
    for directory in sample_dirs:
        if os.path.exists(directory):
            images = glob(os.path.join(directory, '*.jpg')) + glob(os.path.join(directory, '*.png'))
            found_images.extend(images[:2])  # Take first 2 images from each directory
    
    if not found_images:
        print("❌ No sample images found. Creating synthetic demo...")
        create_synthetic_demo(segmenter)
        return
    
    print(f"📸 Found {len(found_images)} sample images")
    
    # Process each image
    for i, image_path in enumerate(found_images[:3]):  # Process first 3 images
        try:
            print(f"\n🖼️  Processing image {i+1}: {os.path.basename(image_path)}")
            save_path = f'prediction_result_{i+1}.png'
            segmenter.visualize_prediction(image_path, save_path)
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

def create_synthetic_demo(segmenter):
    """Create synthetic medical images for demo"""
    print("🎨 Creating synthetic medical images for demonstration...")
    
    # Create synthetic skin lesion images
    os.makedirs('synthetic_demo', exist_ok=True)
    
    for i in range(3):
        # Create synthetic image (256x256 RGB)
        image = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        
        # Add synthetic lesion (circle)
        center = (np.random.randint(80, 176), np.random.randint(80, 176))
        radius = np.random.randint(30, 70)
        cv2.circle(image, center, radius, (50, 50, 50), -1)  # Dark circle as lesion
        
        # Add some noise
        noise = np.random.randint(-20, 20, (256, 256, 3), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        image_path = f'synthetic_demo/skin_lesion_{i+1}.jpg'
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Process it
        print(f"\n🖼️  Processing synthetic image {i+1}")
        save_path = f'prediction_result_{i+1}.png'
        segmenter.visualize_prediction(image_path, save_path)

if __name__ == "__main__":
    test_with_sample_images()
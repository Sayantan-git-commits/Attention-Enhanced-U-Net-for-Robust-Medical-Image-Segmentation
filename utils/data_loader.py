# utils/data_loader.py - COMPLETE UPDATED VERSION
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, img_size=256, is_training=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size
        self.is_training = is_training
        
        # Get all image files
        self.image_files = sorted(glob(os.path.join(image_dir, "*.jpg")))
        
        if not self.image_files:
            print("⚠️ No real images found. Using synthetic data for testing...")
            self.use_synthetic = True
            self.num_samples = 100
        else:
            self.use_synthetic = False
            print(f"✅ Found {len(self.image_files)} real images!")
            
            # Verify masks exist and create valid pairs
            self.valid_pairs = []
            for img_path in self.image_files:
                base_name = os.path.basename(img_path).replace('.jpg', '')
                
                # Try different mask naming conventions
                mask_paths = [
                    os.path.join(mask_dir, f"{base_name}_segmentation.png"),
                    os.path.join(mask_dir, f"{base_name}.png"),
                    os.path.join(mask_dir, f"{base_name}_mask.png"),
                ]
                
                mask_found = False
                for mask_path in mask_paths:
                    if os.path.exists(mask_path):
                        self.valid_pairs.append((img_path, mask_path))
                        mask_found = True
                        break
                
                if not mask_found:
                    print(f"⚠️ No mask found for: {base_name}")
            
            print(f"✅ Found {len(self.valid_pairs)} valid image-mask pairs")
            
            if len(self.valid_pairs) == 0:
                print("❌ No valid masks found. Using synthetic data.")
                self.use_synthetic = True
                self.num_samples = 100
    
    def __len__(self):
        if self.use_synthetic:
            return self.num_samples
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        if self.use_synthetic:
            return self._get_synthetic_sample()
        
        # Use real data
        img_path, mask_path = self.valid_pairs[idx]
        
        try:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not read image: {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Read mask
            mask = cv2.imread(mask_path, 0)  # Read as grayscale
            if mask is None:
                raise ValueError(f"Could not read mask: {mask_path}")
            
            # Apply transformations
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                # Default processing
                image = cv2.resize(image, (self.img_size, self.img_size))
                mask = cv2.resize(mask, (self.img_size, self.img_size))
                
                # Convert to tensor and normalize - FIXED FOR SHAPES
                image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
                mask = torch.from_numpy(mask).float() / 255.0  # Keep as (H, W) for now
            
            # Ensure mask is binary (threshold at 0.5)
            mask = (mask > 0.5).float()
            
            # CRITICAL FIX: Add channel dimension to mask
            if mask.dim() == 2:  # If (H, W)
                mask = mask.unsqueeze(0)  # -> (1, H, W)
            elif mask.dim() == 3 and mask.shape[0] == 1:  # If (1, H, W) from transform
                pass  # Already correct
            elif mask.dim() == 3 and mask.shape[2] == 1:  # If (H, W, 1)
                mask = mask.permute(2, 0, 1)  # -> (1, H, W)
            
            # Add noise for training (to test robustness)
            if self.is_training and random.random() > 0.7:
                image = self._add_medical_noise(image)
            
            return image, mask
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return self._get_synthetic_sample()
    
    def _get_synthetic_sample(self):
        """Generate synthetic skin lesion data with CORRECT SHAPES"""
        # Create synthetic image (skin-like texture)
        image = torch.rand(3, self.img_size, self.img_size) * 0.3 + 0.4  # Skin tone base
        
        # Create synthetic lesion (irregular shape)
        center_x, center_y = random.randint(50, self.img_size-50), random.randint(50, self.img_size-50)
        radius_x, radius_y = random.randint(20, 60), random.randint(20, 60)
        
        y, x = torch.meshgrid(torch.linspace(0, self.img_size-1, self.img_size), 
                             torch.linspace(0, self.img_size-1, self.img_size))
        
        # Elliptical lesion with noise
        lesion = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 < 1
        lesion = lesion.float()
        
        # Add texture to lesion
        lesion_texture = torch.randn_like(lesion) * 0.1
        lesion = torch.clamp(lesion + lesion_texture, 0, 1)
        
        # Apply lesion to image (darker region)
        image = image * (1 - lesion.unsqueeze(0)) + (image * 0.3) * lesion.unsqueeze(0)
        
        # Mask is the lesion area - CORRECT SHAPE: (1, H, W)
        mask = lesion.unsqueeze(0)
        
        return image, mask
    
    def _add_medical_noise(self, image):
        """Add realistic medical image noise"""
        # Gaussian noise
        noise = torch.randn_like(image) * 0.05
        image = image + noise
        
        # Simulate hair artifacts (random lines)
        if random.random() > 0.5:
            image = self._add_hair_artifacts(image)
        
        return torch.clamp(image, 0, 1)
    
    def _add_hair_artifacts(self, image):
        """Add synthetic hair artifacts"""
        h, w = image.shape[1], image.shape[2]
        for _ in range(random.randint(1, 3)):
            x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
            
            # Create line mask
            line_mask = np.zeros((h, w))
            cv2.line(line_mask, (x1, y1), (x2, y2), 1, thickness=random.randint(1, 2))
            
            line_mask = torch.from_numpy(line_mask).float()
            # Darken the hair areas
            image = image * (1 - line_mask.unsqueeze(0)) + image * 0.7 * line_mask.unsqueeze(0)
        
        return image

def get_transforms(img_size=256, is_training=True):
    """Data augmentation transforms with CORRECT SHAPE HANDLING"""
    if is_training:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

def prepare_data_loaders(batch_size=8, validation_split=0.2, img_size=256):
    """Prepare data loaders - automatically uses real data if available"""
    print("📥 Preparing data loaders...")
    
    train_transform = get_transforms(img_size, is_training=True)
    val_transform = get_transforms(img_size, is_training=False)
    
    # Create dataset
    dataset = ISICDataset(
        image_dir="data/ISIC2018/images",
        mask_dir="data/ISIC2018/masks",
        transform=train_transform,
        img_size=img_size,
        is_training=True
    )
    
    if dataset.use_synthetic:
        print("🔧 Using SYNTHETIC data for training")
        # Use synthetic dataset directly
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Apply validation transform to val dataset
        val_dataset.dataset.transform = val_transform
    else:
        print("🎯 Using REAL ISIC2018 data for training!")
        # Split real data
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        # Apply different transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✅ Data loaded: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    
    # Verify shapes
    images, masks = next(iter(train_loader))
    print(f"📐 Verified - Images: {images.shape}, Masks: {masks.shape}")
    print(f"📐 Verified - Image dtype: {images.dtype}, Mask dtype: {masks.dtype}")
    
    return train_loader, val_loader

def visualize_sample():
    """Visualize a sample from the dataset"""
    import matplotlib.pyplot as plt
    
    train_loader, _ = prepare_data_loaders(batch_size=1, img_size=128)
    images, masks = next(iter(train_loader))
    
    image = images[0].permute(1, 2, 0).numpy()
    mask = masks[0].squeeze().numpy()  # Remove channel dimension for display
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Segmentation Mask')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('data_sample.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Sample visualization saved as 'data_sample.png'")

def test_data_loader():
    """Test the data loader with shape verification"""
    print("🧪 Testing Data Loader...")
    
    train_loader, val_loader = prepare_data_loaders(batch_size=4, img_size=64)
    
    # Test one batch
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images: {images.shape} ({images.dtype})")
        print(f"  Masks:  {masks.shape} ({masks.dtype})")
        
        # Verify shapes are compatible
        assert images.dim() == 4, f"Images should be 4D, got {images.dim()}D"
        assert masks.dim() == 4, f"Masks should be 4D, got {masks.dim()}D"
        assert images.shape[0] == masks.shape[0], "Batch sizes don't match"
        assert images.shape[2:] == masks.shape[2:], "Spatial dimensions don't match"
        
        print("  ✅ Shape verification passed!")
        
        # Only test first batch
        if batch_idx == 0:
            break
    
    print("🎉 Data loader test completed successfully!")

if __name__ == "__main__":
    test_data_loader()
    # visualize_sample()
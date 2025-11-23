# check_data.py - Verify your data is correctly placed
import os
from glob import glob

def check_data_structure():
    print("🔍 Checking data structure...")
    
    # Check images
    image_dir = "data/ISIC2018/images"
    mask_dir = "data/ISIC2018/masks"
    
    # Count files
    image_files = glob(os.path.join(image_dir, "*.jpg"))
    mask_files = glob(os.path.join(mask_dir, "*segmentation.png"))
    
    print(f"📸 Images found: {len(image_files)}")
    print(f"🎯 Masks found: {len(mask_files)}")
    
    if len(image_files) > 0 and len(mask_files) > 0:
        print("✅ Data structure is CORRECT!")
        
        # Show first few files
        print("\n📁 Sample files:")
        for i in range(min(3, len(image_files))):
            img_name = os.path.basename(image_files[i])
            mask_name = os.path.basename(mask_files[i])
            print(f"   Image: {img_name}")
            print(f"   Mask:  {mask_name}")
            
    else:
        print("❌ Data structure issue!")
        if len(image_files) == 0:
            print("   - No images found in data/ISIC2018/images/")
        if len(mask_files) == 0:
            print("   - No masks found in data/ISIC2018/masks/")
        
        print("\n💡 Solution: Make sure you have:")
        print("   - 2594 .jpg files in data/ISIC2018/images/")
        print("   - 2594 _segmentation.png files in data/ISIC2018/masks/")

if __name__ == "__main__":
    check_data_structure()
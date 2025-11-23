# cli_interface.py - Command line interface
import argparse
import os
from predict import MedicalSegmenter

def main():
    parser = argparse.ArgumentParser(description='Medical Image Segmentation CLI')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='segmentation_result.png', help='Output path for results')
    parser.add_argument('--model', type=str, help='Path to trained model (optional)')
    
    args = parser.parse_args()
    
    print("🩺 Medical Image Segmentation")
    print("=" * 40)
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"❌ Input image not found: {args.image}")
        return
    
    # Initialize segmenter
    segmenter = MedicalSegmenter(args.model)
    
    # Process image
    try:
        print(f"📥 Input: {args.image}")
        print(f"📤 Output: {args.output}")
        print("⏳ Processing...")
        
        segmenter.visualize_prediction(args.image, args.output)
        
        print("✅ Segmentation completed!")
        print(f"💾 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error during segmentation: {e}")

if __name__ == "__main__":
    main()
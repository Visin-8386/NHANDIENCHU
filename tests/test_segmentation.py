"""
Quick test for segmentation improvements
"""
import cv2
import numpy as np
from src.data.segmentation import TextSegmenter, visualize_segmentation_detailed

# Load test image (you can replace with your own)
def test_with_canvas_image():
    """Test with a canvas-like image"""
    # Create a test image with text
    img = np.ones((300, 800, 3), dtype=np.uint8) * 255
    
    # Draw some sample text (simulate handwriting)
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    
    # Line 1: "Bite off more"
    cv2.putText(img, "Bite", (20, 50), font, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "off more", (250, 50), font, 1.5, (0, 0, 0), 3)
    
    # Line 2: "than you"
    cv2.putText(img, "than", (80, 150), font, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "you", (450, 150), font, 1.5, (0, 0, 0), 3)
    
    # Line 3: "can chew"
    cv2.putText(img, "can", (50, 250), font, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "chew", (450, 250), font, 1.5, (0, 0, 0), 3)
    
    return img

def test_with_file(image_path):
    """Test with an actual image file"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None
    return img

# Run test
if __name__ == "__main__":
    print("Testing segmentation...")
    
    # Test with generated image
    img = test_with_canvas_image()
    
    # Or test with your uploaded image
    # img = test_with_file("path_to_your_image.png")
    
    if img is not None:
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Create segmenter with optimized parameters
        segmenter = TextSegmenter(
            min_line_height=10,
            min_word_width=5,
            min_char_width=2,
            line_merge_threshold=0.3,
            word_gap_ratio=0.4
        )
        
        # Segment
        print("Segmenting image...")
        segments = segmenter.segment_full_image(gray, line_method='contours', word_method='hybrid')
        
        print(f"\n✅ Found {len(segments)} word segments")
        
        # Group by lines
        lines_info = {}
        for _, _, line_idx, word_idx in segments:
            if line_idx not in lines_info:
                lines_info[line_idx] = 0
            lines_info[line_idx] += 1
        
        print(f"📊 Lines detected: {len(lines_info)}")
        for line_idx in sorted(lines_info.keys()):
            print(f"   Line {line_idx}: {lines_info[line_idx]} words")
        
        # Visualize
        vis_img = visualize_segmentation_detailed(gray, segments)
        
        # Save result
        cv2.imwrite('segmentation_result.png', vis_img)
        print("\n💾 Saved visualization to: segmentation_result.png")
        
        # Show bounding boxes
        print("\n📦 Bounding boxes:")
        for idx, (word_img, bbox, line_idx, word_idx) in enumerate(segments):
            x, y, w, h = bbox
            print(f"   L{line_idx}W{word_idx}: x={x}, y={y}, w={w}, h={h}")

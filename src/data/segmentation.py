"""
Improved Text Segmentation with Bug Fixes
Key improvements:
1. Simplified and more robust word gap detection
2. Consistent padding strategy
3. Adaptive thresholds based on image statistics
4. Better handling of edge cases
5. Aggressive split mode for better word separation
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class TextSegmenter:
    """Enhanced text segmentation with robust algorithms"""
    
    def __init__(self, 
                 min_line_height_ratio: float = 0.02,  # 2% of image height
                 min_word_width_ratio: float = 0.01,   # 1% of image width
                 word_gap_multiplier: float = 1.2,     # Gap = 1.2x avg char width
                 force_min_gap: Optional[int] = None,  # Force minimum gap (pixels)
                 aggressive_split: bool = True,        # Use aggressive word splitting
                 # Legacy params for backward compatibility
                 min_line_height: int = None,
                 min_word_width: int = None,
                 min_char_width: int = None,
                 line_merge_threshold: float = 0.3,
                 word_gap_ratio: float = None):
        """
        Args:
            min_line_height_ratio: Minimum line height as ratio of image height
            min_word_width_ratio: Minimum word width as ratio of image width
            word_gap_multiplier: Multiplier for average char width to determine word gaps
            force_min_gap: If set, use this as minimum word gap threshold (overrides auto)
            aggressive_split: If True, use more aggressive word boundary detection
        """
        self.min_line_height_ratio = min_line_height_ratio
        self.min_word_width_ratio = min_word_width_ratio
        self.word_gap_multiplier = word_gap_multiplier
        self.force_min_gap = force_min_gap
        self.aggressive_split = aggressive_split
        self.line_merge_threshold = line_merge_threshold
        
        # Legacy absolute values (used if ratio gives too small values)
        self._min_line_height = min_line_height or 10
        self._min_word_width = min_word_width or 5
        self._min_char_width = min_char_width or 3
    
    def preprocess_for_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Robust preprocessing"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect if text is dark or light using Otsu's threshold
        _, binary_test = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_pixel_ratio = np.sum(binary_test == 0) / binary_test.size
        
        # If most pixels are dark, text is likely dark on light background
        if text_pixel_ratio < 0.5:
            # Text is dark - check if we need inversion for white bg
            mean_val = np.mean(gray)
            if mean_val < 127:
                gray = cv2.bitwise_not(gray)
        else:
            # Text is light on dark - invert to get white bg
            gray = cv2.bitwise_not(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=15, C=10
        )
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return gray, binary
    
    def segment_lines(self, image: np.ndarray, binary: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Segment lines using improved contour method"""
        h, w = image.shape[:2]
        min_line_height = max(self._min_line_height, int(h * self.min_line_height_ratio))
        
        # Dilate horizontally to connect words in same line
        kernel_width = max(30, w // 20)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        dilated = cv2.dilate(binary, kernel_h, iterations=2)
        
        # Dilate vertically slightly to merge close components
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated = cv2.dilate(dilated, kernel_v, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes
        bboxes = []
        for contour in contours:
            x, y, w_box, h_box = cv2.boundingRect(contour)
            if h_box >= min_line_height:
                bboxes.append([x, y, w_box, h_box])
        
        if not bboxes:
            return [(image, (0, 0, w, h))]
        
        # Sort by y coordinate
        bboxes.sort(key=lambda b: b[1])
        
        # Merge overlapping lines
        merged = []
        current = bboxes[0]
        
        for box in bboxes[1:]:
            overlap = min(current[1] + current[3], box[1] + box[3]) - max(current[1], box[1])
            if overlap > self.line_merge_threshold * min(current[3], box[3]):
                # Merge
                new_x = min(current[0], box[0])
                new_y = min(current[1], box[1])
                new_x2 = max(current[0] + current[2], box[0] + box[2])
                new_y2 = max(current[1] + current[3], box[1] + box[3])
                current = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
            else:
                merged.append(current)
                current = box
        merged.append(current)
        
        # Extract line images with consistent padding
        lines = []
        for x, y, w_box, h_box in merged:
            pad = max(3, int(h_box * 0.1))
            y1 = max(0, y - pad)
            y2 = min(h, y + h_box + pad)
            x1 = max(0, x - pad)
            x2 = min(w, x + w_box + pad)
            
            line_img = image[y1:y2, x1:x2]
            lines.append((line_img, (x1, y1, x2 - x1, y2 - y1)))
        
        return lines
    
    # Alias for backward compatibility
    def segment_lines_contours(self, image: np.ndarray, binary: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        return self.segment_lines(image, binary)
    
    def segment_lines_projection(self, image: np.ndarray, binary: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        return self.segment_lines(image, binary)
    
    def segment_words(self, line_image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Hybrid word segmentation using vertical projection analysis"""
        h, w = line_image.shape[:2]
        min_word_width = max(self._min_word_width, int(w * self.min_word_width_ratio))
        
        # Preprocess line
        gray, binary = self.preprocess_for_segmentation(line_image)
        
        # STRATEGY: Vertical Projection Analysis (more reliable for word gaps)
        h_proj = np.sum(binary, axis=0)  # Sum along columns
        
        # Find columns with zero or very low ink
        threshold_low = np.max(h_proj) * 0.05  # 5% of max
        gap_candidates = np.where(h_proj <= threshold_low)[0]
        
        if len(gap_candidates) < 2:
            # No clear gaps - return whole line
            return [(gray, (0, 0, w, h))]
        
        # Group consecutive gap pixels into regions
        gap_regions = []
        if len(gap_candidates) > 0:
            start = gap_candidates[0]
            prev = gap_candidates[0]
            
            for curr in gap_candidates[1:]:
                if curr - prev > 1:  # Break in gap sequence
                    gap_regions.append((start, prev, prev - start + 1))
                    start = curr
                prev = curr
            gap_regions.append((start, prev, prev - start + 1))
        
        if not gap_regions:
            return [(gray, (0, 0, w, h))]
        
        # Analyze gap sizes
        gap_widths = [g[2] for g in gap_regions]
        
        # Find word boundaries using smart threshold
        if len(gap_widths) >= 3:
            sorted_widths = sorted(gap_widths)
            
            # Look for natural break (bimodal distribution)
            max_jump = 0
            split_idx = 0
            for i in range(len(sorted_widths) - 1):
                jump = sorted_widths[i + 1] - sorted_widths[i]
                if jump > max_jump:
                    max_jump = jump
                    split_idx = i
            
            # Determine threshold
            if self.force_min_gap is not None:
                width_threshold = self.force_min_gap
            elif max_jump > sorted_widths[split_idx] * 0.3:
                # Clear bimodal distribution
                width_threshold = (sorted_widths[split_idx] + sorted_widths[split_idx + 1]) / 2
            else:
                # Use percentile approach
                if self.aggressive_split:
                    width_threshold = np.percentile(gap_widths, 50)  # Median
                else:
                    width_threshold = np.percentile(gap_widths, 65)
            
            # Ensure minimum
            width_threshold = max(width_threshold, 3)
        else:
            width_threshold = max(gap_widths) * 0.7 if gap_widths else 5
        
        # Extract word regions
        words = []
        word_start = 0
        
        for gap_start, gap_end, gap_width in gap_regions:
            if gap_width >= width_threshold:
                # This is a word boundary
                word_end = gap_start
                
                if word_end - word_start >= min_word_width:
                    # Find tight vertical bounds for this word region
                    word_region = binary[:, word_start:word_end]
                    rows_with_ink = np.where(np.sum(word_region, axis=1) > 0)[0]
                    
                    if len(rows_with_ink) > 0:
                        y_min = rows_with_ink[0]
                        y_max = rows_with_ink[-1] + 1
                        
                        # Add padding
                        pad_x = max(2, int((word_end - word_start) * 0.05))
                        pad_y = max(2, int((y_max - y_min) * 0.05))
                        
                        x1 = max(0, word_start - pad_x)
                        x2 = min(w, word_end + pad_x)
                        y1 = max(0, y_min - pad_y)
                        y2 = min(h, y_max + pad_y)
                        
                        word_img = gray[y1:y2, x1:x2]
                        words.append((word_img, (x1, y1, x2 - x1, y2 - y1)))
                
                word_start = gap_end + 1
        
        # Handle last word
        if w - word_start >= min_word_width:
            word_region = binary[:, word_start:]
            rows_with_ink = np.where(np.sum(word_region, axis=1) > 0)[0]
            
            if len(rows_with_ink) > 0:
                y_min = rows_with_ink[0]
                y_max = rows_with_ink[-1] + 1
                
                pad_x = max(2, int((w - word_start) * 0.05))
                pad_y = max(2, int((y_max - y_min) * 0.05))
                
                x1 = max(0, word_start - pad_x)
                x2 = w
                y1 = max(0, y_min - pad_y)
                y2 = min(h, y_max + pad_y)
                
                word_img = gray[y1:y2, x1:x2]
                words.append((word_img, (x1, y1, x2 - x1, y2 - y1)))
        
        return words if words else [(gray, (0, 0, w, h))]
    
    # Alias for backward compatibility
    def segment_words_in_line(self, line_image: np.ndarray, method: str = 'hybrid') -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        return self.segment_words(line_image)
    
    def segment_full_image(self, image: np.ndarray, 
                           line_method: str = 'contours',
                           word_method: str = 'hybrid') -> List[Tuple[np.ndarray, Tuple[int, int, int, int], int, int]]:
        """Complete segmentation pipeline"""
        # Preprocess
        gray, binary = self.preprocess_for_segmentation(image)
        
        # Segment lines
        lines = self.segment_lines(gray, binary)
        
        if not lines:
            lines = [(gray, (0, 0, image.shape[1], image.shape[0]))]
        
        # Segment words in each line
        all_words = []
        for line_idx, (line_img, line_bbox) in enumerate(lines):
            words = self.segment_words(line_img)
            
            if not words:
                words = [(line_img, (0, 0, line_img.shape[1], line_img.shape[0]))]
            
            for word_idx, (word_img, word_bbox) in enumerate(words):
                # Convert to global coordinates
                global_x = line_bbox[0] + word_bbox[0]
                global_y = line_bbox[1] + word_bbox[1]
                global_bbox = (global_x, global_y, word_bbox[2], word_bbox[3])
                
                all_words.append((word_img, global_bbox, line_idx, word_idx))
        
        return all_words


# Create default segmenter instance with aggressive mode
_default_segmenter = TextSegmenter(aggressive_split=True, word_gap_multiplier=1.2)


def segment_lines(image: np.ndarray, min_line_height: int = 15) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segment image into text lines
    
    Args:
        image: Binary or grayscale image
        min_line_height: Minimum height for a valid line
        
    Returns:
        List of (line_image, (x, y, w, h)) tuples
    """
    # Ensure binary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold if not binary - use adaptive threshold for better results
    if gray.max() > 1:
        # Check if background is light or dark
        mean_val = np.mean(gray)
        if mean_val < 127:
            gray = cv2.bitwise_not(gray)
        
        # Use Otsu's method for better thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = (gray * 255).astype(np.uint8)
    
    # Apply morphological operations to connect broken characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Horizontal projection
    h_projection = np.sum(binary, axis=1)
    
    # Smooth projection to reduce noise
    kernel_size = 3
    h_projection_smooth = np.convolve(h_projection, np.ones(kernel_size)/kernel_size, mode='same')
    
    # Find line boundaries with minimum gap requirement
    lines = []
    in_line = False
    start_y = 0
    min_gap = 5  # Minimum gap between lines
    gap_count = 0
    
    for i, val in enumerate(h_projection_smooth):
        if val > 0:
            if not in_line:
                start_y = i
                in_line = True
            gap_count = 0
        else:  # val == 0 (gap)
            if in_line:
                gap_count += 1
                # If gap is large enough, consider it line boundary
                if gap_count >= min_gap:
                    end_y = i - gap_count
                    if end_y - start_y >= min_line_height:
                        # Extract line with some padding
                        pad = 2
                        start_padded = max(0, start_y - pad)
                        end_padded = min(gray.shape[0], end_y + pad)
                        line_img = gray[start_padded:end_padded, :]
                        lines.append((line_img, (0, start_padded, gray.shape[1], end_padded - start_padded)))
                    in_line = False
                    gap_count = 0
    
    # Handle last line
    if in_line and gray.shape[0] - start_y >= min_line_height:
        line_img = gray[start_y:, :]
        lines.append((line_img, (0, start_y, gray.shape[1], gray.shape[0] - start_y)))
    
    return lines


def segment_words(line_image: np.ndarray, min_word_width: int = 10, min_gap: int = 5) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segment a text line into words
    
    Args:
        line_image: Grayscale image of a text line
        min_word_width: Minimum width for a valid word
        min_gap: Minimum gap between words (in pixels)
        
    Returns:
        List of (word_image, (x, y, w, h)) tuples
    """
    # Threshold if not binary
    if line_image.max() > 1:
        _, binary = cv2.threshold(line_image, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        binary = (line_image * 255).astype(np.uint8)
    
    # Vertical projection
    v_projection = np.sum(binary, axis=0)
    
    # Find word boundaries using gaps
    words = []
    in_word = False
    start_x = 0
    gap_count = 0
    
    for i, val in enumerate(v_projection):
        if val > 0:
            if not in_word:
                start_x = i
                in_word = True
            gap_count = 0
        else:  # val == 0 (gap)
            if in_word:
                gap_count += 1
                # If gap is large enough, consider it word boundary
                if gap_count >= min_gap:
                    end_x = i - gap_count
                    if end_x - start_x >= min_word_width:
                        word_img = line_image[:, start_x:end_x]
                        words.append((word_img, (start_x, 0, end_x - start_x, line_image.shape[0])))
                    in_word = False
                    gap_count = 0
    
    # Handle last word
    if in_word and line_image.shape[1] - start_x >= min_word_width:
        word_img = line_image[:, start_x:]
        words.append((word_img, (start_x, 0, line_image.shape[1] - start_x, line_image.shape[0])))
    
    return words


def segment_words_by_contours(line_image: np.ndarray, min_word_width: int = 5) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segment words using contour detection (more robust)
    
    Args:
        line_image: Grayscale image of a text line
        min_word_width: Minimum width for a valid word
        
    Returns:
        List of (word_image, (x, y, w, h)) tuples sorted left to right
    """
    # Check if background is light or dark
    mean_val = np.mean(line_image)
    if mean_val < 127:
        line_image = cv2.bitwise_not(line_image)
    
    # Threshold with Otsu's method
    if line_image.max() > 1:
        _, binary = cv2.threshold(line_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = (line_image * 255).astype(np.uint8)
    
    # Apply morphological closing to connect broken parts of characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out tiny noise
        if w >= min_word_width and h >= 5:
            bboxes.append((x, y, w, h))
    
    if not bboxes:
        return []
    
    # Merge overlapping boxes (characters in same word)
    # Use average character width to determine gap threshold
    avg_width = np.mean([b[2] for b in bboxes])
    max_gap = max(10, int(avg_width * 0.5))  # Gap = 50% of avg char width
    bboxes = merge_close_boxes(bboxes, max_gap=max_gap)
    
    # Sort by x coordinate (left to right)
    bboxes.sort(key=lambda b: b[0])
    
    # Extract word images with padding
    words = []
    for x, y, w, h in bboxes:
        # Add small padding around word
        pad = 2
        x_start = max(0, x - pad)
        y_start = max(0, y - pad)
        x_end = min(line_image.shape[1], x + w + pad)
        y_end = min(line_image.shape[0], y + h + pad)
        
        word_img = line_image[y_start:y_end, x_start:x_end]
        words.append((word_img, (x_start, y_start, x_end - x_start, y_end - y_start)))
    
    return words


def merge_close_boxes(bboxes: List[Tuple[int, int, int, int]], max_gap: int = 15) -> List[Tuple[int, int, int, int]]:
    """
    Merge bounding boxes that are close to each other (characters in same word)
    
    Args:
        bboxes: List of (x, y, w, h) tuples
        max_gap: Maximum horizontal gap to merge
        
    Returns:
        List of merged (x, y, w, h) tuples
    """
    if not bboxes:
        return []
    
    # Sort by x coordinate
    bboxes = sorted(bboxes, key=lambda b: b[0])
    
    merged = []
    current = list(bboxes[0])  # [x, y, w, h]
    
    for box in bboxes[1:]:
        x, y, w, h = box
        curr_x, curr_y, curr_w, curr_h = current
        
        # Check if boxes are close horizontally
        if x - (curr_x + curr_w) <= max_gap:
            # Merge boxes
            new_x = min(curr_x, x)
            new_y = min(curr_y, y)
            new_x2 = max(curr_x + curr_w, x + w)
            new_y2 = max(curr_y + curr_h, y + h)
            current = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
        else:
            # Save current and start new
            merged.append(tuple(current))
            current = list(box)
    
    # Add last box
    merged.append(tuple(current))
    
    return merged


def segment_text_image(image: np.ndarray, method: str = 'contours') -> List[Tuple[np.ndarray, Tuple[int, int, int, int], int, int]]:
    """
    Segment multi-line text image into individual words
    
    Args:
        image: Grayscale image with text
        method: 'contours' or 'projection'
        
    Returns:
        List of (word_image, (x, y, w, h), line_index, word_index) tuples
    """
    all_words = []
    
    # Step 1: Segment lines
    lines = segment_lines(image)
    
    # Step 2: Segment each line into words
    for line_idx, (line_img, line_bbox) in enumerate(lines):
        if method == 'contours':
            words = segment_words_by_contours(line_img)
        else:
            words = segment_words(line_img)
        
        # Add global coordinates
        for word_idx, (word_img, word_bbox) in enumerate(words):
            x, y, w, h = word_bbox
            global_x = line_bbox[0] + x
            global_y = line_bbox[1] + y
            all_words.append((word_img, (global_x, global_y, w, h), line_idx, word_idx))
    
    return all_words


def visualize_segmentation(image: np.ndarray, segments: List[Tuple[np.ndarray, Tuple[int, int, int, int], int, int]]) -> np.ndarray:
    """
    Draw bounding boxes on image for visualization
    
    Args:
        image: Original image
        segments: Output from segment_text_image()
        
    Returns:
        Image with bounding boxes drawn
    """
    # Convert to color if grayscale
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    # Define colors for different lines
    line_colors = [
        (66, 133, 244),   # Blue
        (52, 168, 83),    # Green  
        (251, 188, 4),    # Yellow
        (234, 67, 53),    # Red
        (154, 80, 164),   # Purple
        (255, 112, 67),   # Orange
    ]
    
    for word_img, (x, y, w, h), line_idx, word_idx in segments:
        color = line_colors[line_idx % len(line_colors)]
        
        # Draw rectangle with thicker border
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        
        # Add label background
        label = f"L{line_idx}W{word_idx}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis, (x, y - label_h - 6), (x + label_w + 4, y), color, -1)
        
        # Add label text
        cv2.putText(vis, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return vis


def visualize_segmentation_detailed(image: np.ndarray, 
                                     segments: List[Tuple[np.ndarray, Tuple[int, int, int, int], int, int]],
                                     texts: List[str] = None) -> np.ndarray:
    """
    Draw detailed bounding boxes with predicted text labels
    
    Args:
        image: Original image
        segments: Output from segment_text_image()
        texts: Optional list of predicted texts for each segment
        
    Returns:
        Image with bounding boxes and text labels
    """
    # Convert to color if grayscale
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    # Increase canvas size to add text annotations below
    padding_bottom = 60 if texts else 0
    if padding_bottom > 0:
        new_vis = np.ones((vis.shape[0] + padding_bottom, vis.shape[1], 3), dtype=np.uint8) * 255
        new_vis[:vis.shape[0], :, :] = vis
        vis = new_vis
    
    line_colors = [
        (66, 133, 244),   # Blue
        (52, 168, 83),    # Green  
        (251, 188, 4),    # Yellow
        (234, 67, 53),    # Red
        (154, 80, 164),   # Purple
    ]
    
    for i, (word_img, (x, y, w, h), line_idx, word_idx) in enumerate(segments):
        color = line_colors[line_idx % len(line_colors)]
        
        # Draw rectangle
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        if texts and i < len(texts):
            label = f"{texts[i]}"
        else:
            label = f"L{line_idx}W{word_idx}"
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(vis, (x, y - label_h - 4), (x + label_w + 2, y), color, -1)
        cv2.putText(vis, label, (x + 1, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    return vis


# Batch processing utilities
def prepare_batch_for_recognition(segments: List[Tuple[np.ndarray, Tuple[int, int, int, int], int, int]],
                                   preprocess_fn,
                                   target_height: int = 64,
                                   target_width: int = 256) -> Tuple[np.ndarray, List[dict]]:
    """
    Prepare a batch of word images for recognition model
    
    Args:
        segments: Output from segment_text_image()
        preprocess_fn: Preprocessing function that returns tensor
        target_height: Target height for model input
        target_width: Target width for model input
        
    Returns:
        batch_tensor: Stacked tensor of all preprocessed words
        metadata: List of metadata dicts for each word
    """
    import torch
    
    tensors = []
    metadata = []
    
    for word_img, bbox, line_idx, word_idx in segments:
        try:
            # Preprocess
            tensor = preprocess_fn(word_img, target_height=target_height, target_width=target_width, return_steps=False)
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            
            tensors.append(tensor)
            metadata.append({
                'bbox': bbox,
                'line_idx': line_idx,
                'word_idx': word_idx,
                'original_shape': word_img.shape
            })
        except Exception as e:
            print(f"Warning: Failed to preprocess word L{line_idx}W{word_idx}: {e}")
            continue
    
    if not tensors:
        return None, []
    
    # Stack tensors into batch
    batch_tensor = torch.cat(tensors, dim=0)
    
    return batch_tensor, metadata


def reconstruct_text_from_predictions(predictions: List[str], 
                                       metadata: List[dict]) -> str:
    """
    Reconstruct full text from word predictions maintaining line structure
    
    Args:
        predictions: List of predicted words
        metadata: Metadata from prepare_batch_for_recognition
        
    Returns:
        Full reconstructed text with newlines between lines
    """
    if not predictions or not metadata:
        return ""
    
    # Group by line
    lines = {}
    for pred, meta in zip(predictions, metadata):
        line_idx = meta['line_idx']
        word_idx = meta['word_idx']
        
        if line_idx not in lines:
            lines[line_idx] = []
        lines[line_idx].append((word_idx, pred))
    
    # Sort and join
    result_lines = []
    for line_idx in sorted(lines.keys()):
        words = lines[line_idx]
        words.sort(key=lambda x: x[0])  # Sort by word index
        line_text = ' '.join([w[1] for w in words])
        result_lines.append(line_text)
    
    return '\n'.join(result_lines)

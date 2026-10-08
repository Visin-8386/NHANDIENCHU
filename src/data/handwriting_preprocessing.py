"""
Preprocessing utilities for handwriting recognition
"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Union


def preprocess_word_segment(word_img: np.ndarray, 
                            target_height: int = 64, 
                            target_width: int = 256) -> torch.Tensor:
    """
    Preprocess a single word segment for model input
    Optimized version for batch processing
    
    Args:
        word_img: Grayscale word image
        target_height: Target height
        target_width: Target width
        
    Returns:
        torch.Tensor: Preprocessed image tensor [1, 1, H, W]
    """
    # Ensure grayscale
    if len(word_img.shape) == 3:
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = word_img.copy()
    
    # Ensure white background, dark text
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    
    # Resize maintaining aspect ratio
    h, w = enhanced.shape
    if h <= 0 or w <= 0:
        # Return white canvas if invalid
        canvas = np.ones((target_height, target_width), dtype=np.float32)
        return torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0)
    
    aspect = w / h
    target_aspect = target_width / target_height
    
    if aspect > target_aspect:
        new_w = target_width
        new_h = max(1, int(target_width / aspect))
    else:
        new_h = target_height
        new_w = max(1, int(target_height * aspect))
    
    new_h = min(new_h, target_height)
    new_w = min(new_w, target_width)
    
    resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad to target size (white background)
    canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Normalize to [0, 1]
    normalized = canvas.astype(np.float32) / 255.0
    
    # Convert to tensor [1, 1, H, W]
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor


def preprocess_batch(word_images: List[np.ndarray],
                     target_height: int = 64,
                     target_width: int = 256,
                     device: str = 'cpu') -> torch.Tensor:
    """
    Preprocess multiple word images as a batch
    
    Args:
        word_images: List of word images (grayscale numpy arrays)
        target_height: Target height
        target_width: Target width
        device: Device to put tensors on
        
    Returns:
        torch.Tensor: Batch tensor [N, 1, H, W]
    """
    tensors = []
    
    for word_img in word_images:
        tensor = preprocess_word_segment(word_img, target_height, target_width)
        tensors.append(tensor)
    
    if not tensors:
        return torch.zeros((0, 1, target_height, target_width), device=device)
    
    batch = torch.cat(tensors, dim=0).to(device)
    return batch


def preprocess_handwriting_image(image_np, target_height=64, target_width=256, return_steps=False):
    """
    Preprocess a handwriting image for model input - MATCHING TRAINING PREPROCESSING
    
    Args:
        image_np: Input image as numpy array (RGB or grayscale)
        target_height: Target height (default: 64)
        target_width: Target width (default: 256)
        return_steps: If True, return processing steps for visualization
    
    Returns:
        torch.Tensor: Preprocessed image tensor [1, 1, H, W]
        dict (optional): Processing steps if return_steps=True
    """
    steps = {}
    
    # Convert to grayscale if needed
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()
    
    steps['1_original'] = gray.copy()
    steps['1_original'] = gray.copy()
    
    # Check background color FIRST (before any processing)
    # If background is dark (mean < 127), we need to invert
    mean_brightness = np.mean(gray)
    needs_invert = mean_brightness < 127
    
    if needs_invert:
        # Invert: dark background → light background
        gray = cv2.bitwise_not(gray)
        steps['2_inverted'] = gray.copy()
        steps['2_note'] = f"Inverted (original mean: {mean_brightness:.1f} < 127)"
    else:
        steps['2_inverted'] = gray.copy()
        steps['2_note'] = f"Not inverted (original mean: {mean_brightness:.1f} >= 127)"
    
    # Now gray has WHITE background and DARK text
    # Enhance contrast using CLAHE (preserve grayscale details)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    steps['3_enhanced'] = enhanced.copy()
    
    # For finding contours, we need binary version (but we'll use grayscale for final output)
    _, binary_for_contours = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps['4_binary_temp'] = binary_for_contours.copy()
    
    # Find content bounding box using binary image (WHITE background, BLACK text)
    # Invert for findContours (needs black background, white objects)
    contours, _ = cv2.findContours(255 - binary_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of all text
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        
        for contour in contours:
            # Filter out very small contours (noise)
            area = cv2.contourArea(contour)
            if area < 10:  # Skip tiny noise
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # Check if we found valid contours
        if x_min == float('inf'):
            # No valid contours, use full image
            cropped = enhanced
            steps['5_cropped'] = cropped.copy()
            steps['5_bbox'] = "No valid contours - using full image"
        else:
            # Add generous padding to avoid cutting text
            padding = 20  # Increased padding
            x_min = max(0, int(x_min) - padding)
            y_min = max(0, int(y_min) - padding)
            x_max = min(enhanced.shape[1], int(x_max) + padding)
            y_max = min(enhanced.shape[0], int(y_max) + padding)
            
            # Crop the ENHANCED grayscale image (not binary)
            cropped = enhanced[y_min:y_max, x_min:x_max]
            steps['5_cropped'] = cropped.copy()
            steps['5_bbox'] = f"{x_min},{y_min} -> {x_max},{y_max}"
    else:
        cropped = enhanced
        steps['5_cropped'] = cropped.copy()
        steps['5_bbox'] = "No contours found"
    
    # Resize while maintaining aspect ratio
    h, w = cropped.shape
    
    # Validate cropped image
    if h <= 0 or w <= 0:
        print(f"⚠️ Invalid cropped dimensions: {h}x{w}, using original enhanced image")
        cropped = enhanced
        h, w = cropped.shape
    
    aspect = w / h
    target_aspect = target_width / target_height
    
    if aspect > target_aspect:
        # Width is limiting factor
        new_w = target_width
        new_h = max(1, int(target_width / aspect))  # Ensure at least 1
    else:
        # Height is limiting factor
        new_h = target_height
        new_w = max(1, int(target_height * aspect))  # Ensure at least 1
    
    # Validate new dimensions
    new_h = min(new_h, target_height)
    new_w = min(new_w, target_width)
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    steps['6_resized'] = resized.copy()
    steps['6_size'] = f"{new_w}x{new_h}"
    
    # Pad to target size (white background)
    canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    steps['7_padded'] = canvas.copy()
    steps['7_offset'] = f"x:{x_offset}, y:{y_offset}"
    
    # Normalize to [0, 1] - EXACTLY like in training
    normalized = canvas.astype(np.float32) / 255.0
    
    # Convert to tensor [1, 1, H, W]
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    if return_steps:
        return tensor, steps
    return tensor

    
    # Pad to target size
    canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
    
    # Center the image
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    steps['7_padded'] = canvas.copy()
    
    # Normalize to [0, 1]
    normalized = canvas.astype(np.float32) / 255.0
    
    # Convert to tensor [1, 1, H, W]
    tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor, steps


def extract_lines_from_image(image_np):
    """
    Extract individual text lines from an image
    
    Args:
        image_np: Input image as numpy array
    
    Returns:
        list: List of cropped line images
        dict: Processing steps
    """
    steps = {}
    
    # Convert to grayscale
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()
    
    steps['original'] = gray.copy()
    
    # Binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    
    steps['binary'] = binary.copy()
    
    # Horizontal projection profile
    horizontal_projection = np.sum(255 - binary, axis=1)


def extract_lines_from_image(image_np):
    """
    Extract individual text lines from a multi-line handwriting image - simplified version
    
    Args:
        image_np: Input image as numpy array
    
    Returns:
        list: List of line images (numpy arrays)
    """
    # Convert to grayscale if needed
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()
    
    # Simple threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed (text should be white on black for projection)
    if np.mean(binary) > 127:
        binary = 255 - binary
    
    # Horizontal projection
    horizontal_projection = np.sum(binary, axis=1)
    
    # Find line boundaries
    threshold = np.max(horizontal_projection) * 0.1
    in_line = False
    line_boundaries = []
    start = 0
    
    for i, val in enumerate(horizontal_projection):
        if not in_line and val > threshold:
            in_line = True
            start = i
        elif in_line and val <= threshold:
            in_line = False
            if i - start > 10:  # Minimum line height
                line_boundaries.append((start, i))
    
    # Extract line images
    line_images = []
    for start, end in line_boundaries:
        # Add some padding
        start = max(0, start - 5)
        end = min(gray.shape[0], end + 5)
        
        line_img = gray[start:end, :]
        line_images.append(line_img)
    
    return line_images


def image_to_base64(image_np):
    """Convert numpy image to base64 string for web display"""
    import base64
    from io import BytesIO
    from PIL import Image
    
    # Handle float images (normalize to 0-255)
    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
        image_np = (image_np * 255).astype(np.uint8)
    
    # Convert to PIL Image
    if len(image_np.shape) == 2:
        img = Image.fromarray(image_np, mode='L')
    else:
        img = Image.fromarray(image_np)
    
    # Save to bytes
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


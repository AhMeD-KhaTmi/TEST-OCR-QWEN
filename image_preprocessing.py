"""
Image Preprocessing for Financial Table Extraction
===================================================

This module provides preprocessing functions to optimize images for
Qwen3-VL 8B extraction of financial tables.

ROOT CAUSE OF FAILURE:
- Large margins dilute visual tokens
- High-contrast bold text causes attention to skip long labels
- Label column (leftmost, widest) gets deprioritized

SOLUTION:
1. Auto-crop to table content (remove margins)
2. Soften contrast to balance attention
3. Normalize visual density
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Union, Tuple, Optional


def preprocess_for_vlm(
    image: Union[Image.Image, np.ndarray],
    target_contrast: float = 0.85,
    gamma: float = 1.15,
    soften_strength: float = 0.3,
    auto_crop: bool = True,
    normalize_background: bool = True,
    target_bg_grey: int = 250,
    max_dimension: int = 1920
) -> Image.Image:
    """
    Transform high-contrast financial table images for optimal VLM extraction.
    
    This function addresses the label extraction failure in Qwen3-VL 8B by:
    1. Auto-cropping to table content (removes diluting margins)
    2. Reducing contrast to balance attention across all columns
    3. Softening bold text to prevent attention saturation
    4. Normalizing background to prevent pure-white glare
    
    Args:
        image: Input PIL Image or numpy array (BGR or RGB)
        target_contrast: Contrast reduction factor (0.8-0.95 recommended)
        gamma: Gamma correction (>1 lightens dark text, reduces boldness)
        soften_strength: Blend factor for gaussian softening (0.0-0.5)
        auto_crop: Whether to auto-detect and crop to table bounds
        normalize_background: Whether to convert pure white to light grey
        target_bg_grey: Target background grey level (240-252 recommended)
        max_dimension: Maximum width/height after processing
        
    Returns:
        Preprocessed PIL Image ready for VLM inference
    """
    
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        was_pil = True
    else:
        img_array = image.copy()
        was_pil = False
    
    # Ensure RGB format
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif not was_pil:
        # Assume BGR from OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # =========================================================================
    # STEP 1: AUTO-CROP TO TABLE CONTENT (Critical for token allocation)
    # =========================================================================
    if auto_crop:
        img_array = _auto_crop_to_content(img_array, padding=20)
    
    # =========================================================================
    # STEP 2: NORMALIZE BACKGROUND (Pure white → light grey)
    # =========================================================================
    # Pure white backgrounds cause visual "glare" and attention skipping
    if normalize_background:
        img_array = _normalize_background(img_array, target_grey=target_bg_grey)
    
    # =========================================================================
    # STEP 3: REDUCE CONTRAST (Balance attention across columns)
    # =========================================================================
    # High contrast causes model to focus on bold text, skipping regular labels
    img_array = _reduce_contrast(img_array, factor=target_contrast)
    
    # =========================================================================
    # STEP 4: GAMMA CORRECTION (Lighten dark text, reduce visual weight)
    # =========================================================================
    # Bold black text → softer grey-black, easier for model to tokenize
    img_array = _apply_gamma(img_array, gamma=gamma)
    
    # =========================================================================
    # STEP 5: SOFTEN EDGES (Reduce bold text sharpness)
    # =========================================================================
    # Mild blur reduces "visual loudness" of bold characters
    if soften_strength > 0:
        img_array = _soften_image(img_array, strength=soften_strength)
    
    # =========================================================================
    # STEP 6: RESIZE IF NEEDED (Prevent excessive visual tokens)
    # =========================================================================
    img_array = _resize_if_needed(img_array, max_dimension=max_dimension)
    
    # Convert back to PIL Image
    result = Image.fromarray(img_array)
    
    return result


def _auto_crop_to_content(img: np.ndarray, padding: int = 20) -> np.ndarray:
    """
    Automatically detect and crop to table content, removing margins.
    
    This is THE MOST IMPORTANT preprocessing step because:
    - Margins waste 30-50% of visual tokens on empty space
    - Label column gets more attention when image is tightly cropped
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Threshold to find content (inverse - content becomes white)
    _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img  # No content found, return original
    
    # Get bounding box of all content
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(img.shape[1], x_max + padding)
    y_max = min(img.shape[0], y_max + padding)
    
    # Crop
    cropped = img[y_min:y_max, x_min:x_max]
    
    # Ensure we didn't crop too aggressively
    if cropped.shape[0] < 100 or cropped.shape[1] < 100:
        return img  # Cropping failed, return original
    
    return cropped


def _normalize_background(img: np.ndarray, target_grey: int = 250) -> np.ndarray:
    """
    Convert pure white background to light grey.
    
    Pure white (255,255,255) causes visual "saturation" and attention skipping.
    A slight grey (250,250,250) is perceived identically by humans but helps VLMs.
    """
    # Create mask of near-white pixels
    white_mask = np.all(img > 252, axis=2)
    
    # Replace with target grey
    result = img.copy()
    result[white_mask] = [target_grey, target_grey, target_grey]
    
    return result


def _reduce_contrast(img: np.ndarray, factor: float = 0.85) -> np.ndarray:
    """
    Reduce image contrast by blending toward middle grey.
    
    High contrast → bold text dominates attention
    Lower contrast → more balanced attention across all columns
    """
    # Middle grey value
    mid = 128
    
    # Blend toward middle grey
    result = img.astype(np.float32)
    result = mid + factor * (result - mid)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def _apply_gamma(img: np.ndarray, gamma: float = 1.15) -> np.ndarray:
    """
    Apply gamma correction to lighten dark text.
    
    gamma > 1: Lightens dark areas (reduces boldness of black text)
    gamma < 1: Darkens (not recommended for this use case)
    
    This helps because very bold black text "saturates" visual attention.
    Slightly lightened text is easier for the model to tokenize.
    """
    # Build lookup table for efficiency
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(256)]).astype(np.uint8)
    
    return cv2.LUT(img, table)


def _soften_image(img: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """
    Apply mild Gaussian blur to soften sharp edges.
    
    Bold text has very sharp edges that "dominate" visual tokens.
    Mild softening reduces this effect without destroying readability.
    
    strength: 0.0 = no blur, 1.0 = full blur (not recommended)
    """
    if strength <= 0:
        return img
    
    # Apply mild Gaussian blur
    blurred = cv2.GaussianBlur(img, (3, 3), 0.8)
    
    # Blend original and blurred
    result = cv2.addWeighted(img, 1.0 - strength, blurred, strength, 0)
    
    return result


def _resize_if_needed(img: np.ndarray, max_dimension: int = 1920) -> np.ndarray:
    """
    Resize image if it exceeds maximum dimension.
    
    Very large images waste visual tokens on unnecessary detail.
    Financial tables are readable at 1920px max dimension.
    """
    h, w = img.shape[:2]
    
    if max(h, w) <= max_dimension:
        return img
    
    # Calculate scale factor
    scale = max_dimension / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Use INTER_AREA for downscaling (best quality)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized


# =============================================================================
# AGGRESSIVE PREPROCESSING (for very problematic images)
# =============================================================================

def preprocess_aggressive(
    image: Union[Image.Image, np.ndarray],
) -> Image.Image:
    """
    Aggressive preprocessing for images that fail with standard settings.
    
    Use this when preprocess_for_vlm() still produces empty labels.
    """
    return preprocess_for_vlm(
        image,
        target_contrast=0.75,      # More aggressive contrast reduction
        gamma=1.25,                # More aggressive lightening
        soften_strength=0.4,       # More blur
        auto_crop=True,
        normalize_background=True,
        target_bg_grey=245,        # Darker grey background
        max_dimension=1600         # Smaller to concentrate tokens
    )


# =============================================================================
# COLUMN-AWARE PREPROCESSING (experimental)
# =============================================================================

def preprocess_emphasize_labels(
    image: Union[Image.Image, np.ndarray],
    label_column_ratio: float = 0.35
) -> Image.Image:
    """
    Preprocess with special emphasis on the leftmost (label) column.
    
    This function:
    1. Applies lighter preprocessing to the label column (preserve detail)
    2. Applies heavier preprocessing to numeric columns (reduce visual weight)
    
    Args:
        image: Input image
        label_column_ratio: Assumed width of label column (0.35 = 35% of width)
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # First apply auto-crop
    img_array = _auto_crop_to_content(img_array, padding=20)
    
    h, w = img_array.shape[:2]
    split_x = int(w * label_column_ratio)
    
    # Split into label and numeric regions
    label_region = img_array[:, :split_x]
    numeric_region = img_array[:, split_x:]
    
    # Light preprocessing on labels (preserve readability)
    label_processed = _normalize_background(label_region, target_grey=252)
    label_processed = _reduce_contrast(label_processed, factor=0.92)
    
    # Heavier preprocessing on numeric columns (reduce visual dominance)
    numeric_processed = _normalize_background(numeric_region, target_grey=248)
    numeric_processed = _reduce_contrast(numeric_processed, factor=0.80)
    numeric_processed = _apply_gamma(numeric_processed, gamma=1.2)
    numeric_processed = _soften_image(numeric_processed, strength=0.35)
    
    # Recombine
    result = np.hstack([label_processed, numeric_processed])
    
    return Image.fromarray(result)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compare_preprocessing(
    image: Union[Image.Image, np.ndarray],
    save_path: str = None
) -> dict:
    """
    Generate comparison of different preprocessing settings.
    
    Useful for debugging which settings work best for your specific images.
    """
    original = Image.fromarray(np.array(image)) if isinstance(image, np.ndarray) else image
    
    results = {
        "original": original,
        "standard": preprocess_for_vlm(image),
        "aggressive": preprocess_aggressive(image),
        "label_emphasis": preprocess_emphasize_labels(image),
    }
    
    if save_path:
        # Create side-by-side comparison
        widths = [img.width for img in results.values()]
        heights = [img.height for img in results.values()]
        
        total_width = sum(widths)
        max_height = max(heights)
        
        comparison = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        x_offset = 0
        for name, img in results.items():
            comparison.paste(img, (x_offset, 0))
            x_offset += img.width
        
        comparison.save(save_path)
        print(f"Comparison saved to: {save_path}")
    
    return results


# =============================================================================
# INTEGRATION WITH EXISTING PIPELINE
# =============================================================================

def preprocess_for_qwen(image_path: str) -> Image.Image:
    """
    Simple wrapper for integration with existing pipeline.
    
    Usage:
        from image_preprocessing import preprocess_for_qwen
        
        # In your extraction function:
        preprocessed = preprocess_for_qwen("path/to/image.png")
        # Then send preprocessed image to Qwen3-VL
    """
    img = Image.open(image_path)
    return preprocess_for_vlm(img)


if __name__ == "__main__":
    # Test with sample image
    import sys
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        print(f"Testing preprocessing on: {test_image}")
        
        original = Image.open(test_image)
        print(f"Original size: {original.size}")
        
        processed = preprocess_for_vlm(original)
        print(f"Processed size: {processed.size}")
        
        # Save processed image
        output_path = test_image.replace(".png", "_preprocessed.png")
        processed.save(output_path)
        print(f"Saved to: {output_path}")
    else:
        print("Usage: python image_preprocessing.py <image_path>")

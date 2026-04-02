"""
Qwen3-VL-8B-Instruct - Optimized for RTX 4060 (8GB VRAM)
Uses 4-bit quantization with GPU-first inference
"""

import torch
import gc
import os
import tempfile
import json
import re
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from json_table_utils import extract_json_from_response

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"  # Qwen3-VL-8B with improved vision understanding
DEFAULT_MAX_IMAGE_SIZE = 1400  # Increased for single-pass extraction
DEFAULT_MAX_NEW_TOKENS = 4096  # Increased for large table extraction (was 1024)

# Token scaling thresholds
MIN_TOKENS_SMALL_TABLE = 2048   # For small images/tables
MIN_TOKENS_MEDIUM_TABLE = 4096  # For medium images/tables
MIN_TOKENS_LARGE_TABLE = 6144   # For large images/tables
MAX_RETRY_TOKENS = 8192         # Maximum tokens for retry attempts

# =============================================================================
# STEP 1: VERIFY GPU BEFORE ANYTHING
# =============================================================================

def verify_gpu():
    """Verify CUDA is available and print GPU info"""
    print("=" * 60)
    print("GPU VERIFICATION")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is NOT available!\n"
            "Fix: Reinstall PyTorch with CUDA:\n"
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
        )

    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"CUDA Available: True")
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {vram_total:.2f} GB")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print("=" * 60)

    return True

# =============================================================================
# STEP 2: VRAM MONITORING
# =============================================================================

def get_vram_usage():
    """Get current VRAM usage in GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        return allocated, reserved
    return 0, 0

def print_vram_status(stage=""):
    """Print current VRAM status"""
    allocated, reserved = get_vram_usage()
    print(f"[VRAM {stage}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# =============================================================================
# STEP 3: LOAD MODEL WITH 4-BIT QUANTIZATION
# =============================================================================

def load_model():
    """Load Qwen-VL with 4-bit quantization for 8GB VRAM (GPU-first)."""

    print("\n" + "=" * 60)
    print("LOADING MODEL WITH 4-BIT QUANTIZATION")
    print("=" * 60)

    # Clear any existing VRAM
    torch.cuda.empty_cache()
    gc.collect()
    print_vram_status("Before loading")

    # 4-bit quantization config for fast GPU inference.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16 for speed
        bnb_4bit_use_double_quant=True,        # Double quantization saves more memory
        bnb_4bit_quant_type="nf4",             # NormalFloat4 - best quality
    )

    print(f"Loading model: {MODEL_ID}")
    print("Note: Qwen3-VL-8B is configured for GPU-first execution")
    print("This may take 2-5 minutes on first run (downloading ~8GB)...")

    # Set memory limits to keep inference on GPU and avoid CPU fallback latency.
    max_memory = {
        0: "7.8GB",
    }

    # Load model with quantization on GPU.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    hf_device_map = getattr(model, "hf_device_map", {})
    cpu_mapped = {
        key: device for key, device in hf_device_map.items()
        if isinstance(device, str) and device.startswith("cpu")
    }
    if cpu_mapped:
        raise RuntimeError(
            "Model mapped some modules to CPU, which slows inference too much. "
            "Reduce image size/tokens or use a larger GPU."
        )

    print_vram_status("After model load")

    # Load processor (tokenizer + image processor)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Model loaded successfully!")
    print("=" * 60)

    return model, processor

# =============================================================================
# STEP 4: DYNAMIC EXTRACTION PROMPT (NO HARDCODED SCHEMA)
# =============================================================================

DEFAULT_EXTRACTION_PROMPT = """Extract the COMPLETE financial table from this image.

CRITICAL RULES:

1. HEADER DETECTION (MANDATORY):
   - The FIRST row of the table is the header
   - Columns MUST come ONLY from the header row
   - Read header column names EXACTLY as written (dates, labels, etc.)
   - DO NOT invent column names
   - DO NOT use placeholder names like "col1", "col2", "col3"

2. EXTRACT THE ACTUAL COLUMN NAMES:
   - Read dates: "31/12/2024", "31/12/2023", "30/06/2022"
   - Read labels: "Libellé", "Label", "Note"
   - Read variation columns: "Variation Montant", "En Montant", "Variation %", "En %"
   
3. START FROM THE TOP:
   - Extract ALL rows from TOP to BOTTOM
   - Include section headers (ACTIF, PASSIF, etc.)
   - Include first data rows (HB1, HB2, etc.)
   - Do NOT skip any rows

4. SCHEMA IS LOCKED:
   - Number of columns = number of header columns
   - Every row MUST use the SAME columns
   - If a cell is empty, use ""
   - NEVER create new columns from row data

5. ROW TYPES:
   - "section" → no numeric values (e.g., ACTIF, PASSIF EVENTUELS)
   - "data" → normal data row
   - "total" → contains "TOTAL" or "Total"

6. IGNORE:
   - Page titles (Attijari bank...)
   - Unit text (en milliers de dinars)
   - Footnotes at bottom

OUTPUT FORMAT:

{
  "columns": ["Label", "Note", "31/12/2024", "31/12/2023", "Variation Montant", "Variation %"],
  "rows": [
     {
        "type": "section",
        "Label": "PASSIFS EVENTUELS",
        "Note": "",
        "31/12/2024": "",
        "31/12/2023": "",
        "Variation Montant": "",
        "Variation %": ""
     },
     {
        "type": "data",
        "Label": "HB1 - Cautions, avals...",
        "Note": "(4.1)",
        "31/12/2024": "799 892",
        "31/12/2023": "652 772",
        "Variation Montant": "147 120",
        "Variation %": "22,5%"
     }
  ]
}

CRITICAL:
- Use REAL column names from the header (dates, not "col1")
- Start extraction from the FIRST row of the table
- Include ALL section headers
- NEVER leave Label column empty - always read the leftmost text
- Return ONLY valid JSON

7. LABEL EXTRACTION (EXTREMELY IMPORTANT):
   - Read LEFT to RIGHT for EVERY row
   - The LEFTMOST column is ALWAYS the Label column
   - Label contains French text describing the financial item
   - Labels can be LONG (up to 60+ characters)
   - Labels may end with * (asterisk) - include it
   - NEVER leave Label empty if text is visible in the leftmost column
   - Examples of valid labels:
     * "Caisse et avoirs auprès de la BCT, CCP et TGT"
     * "Créances sur les établissements bancaires et financiers *"
     * "Portefeuille-titres commercial"

8. NOTE COLUMN RULES:
   - Note column contains references like (1-1), (1-2), (2-1)
   - Some rows have NO note value - use "" for those
   - Note is NEVER a substitute for Label
   - If you see "(1-1)" in second column, the FIRST column has the Label"""

# =============================================================================
# STEP 5: INFERENCE FUNCTION
# =============================================================================

def auto_crop_to_content(img, padding=10, white_threshold=245, min_content_ratio=0.01):
    """
    Automatically crop image to content region by detecting non-white pixels.
    
    This removes:
    - Empty margins
    - Large white spaces
    - Headers/footers with minimal content
    
    Args:
        img: PIL Image
        padding: Pixels to add around detected content (avoid cutting borders)
        white_threshold: Pixels above this value (0-255) are considered white/empty
        min_content_ratio: Minimum ink density to consider a row/col as content
    
    Returns:
        Cropped PIL Image
    """
    # Convert to grayscale for analysis
    gray = np.asarray(img.convert("L"))
    
    if gray.size == 0:
        return img
    
    height, width = gray.shape
    
    # Detect non-white pixels (content)
    content_mask = gray < white_threshold
    
    # Find rows and columns with content
    row_has_content = content_mask.mean(axis=1) > min_content_ratio
    col_has_content = content_mask.mean(axis=0) > min_content_ratio
    
    # Find bounding box of content
    content_rows = np.where(row_has_content)[0]
    content_cols = np.where(col_has_content)[0]
    
    if len(content_rows) == 0 or len(content_cols) == 0:
        # No content detected, return original
        return img
    
    # Get bounds
    top = content_rows[0]
    bottom = content_rows[-1]
    left = content_cols[0]
    right = content_cols[-1]
    
    # Add padding (but stay within image bounds)
    top = max(0, top - padding)
    bottom = min(height, bottom + padding)
    left = max(0, left - padding)
    right = min(width, right + padding)
    
    # Ensure we have a valid crop region
    if right <= left or bottom <= top:
        return img
    
    # Calculate crop savings
    original_area = width * height
    cropped_area = (right - left) * (bottom - top)
    savings_percent = (1 - cropped_area / original_area) * 100
    
    if savings_percent > 5:  # Only crop if we save more than 5%
        print(f"[AUTO-CROP] {width}x{height} -> {right-left}x{bottom-top} ({savings_percent:.1f}% reduced)")
        return img.crop((left, top, right, bottom))
    else:
        return img


def detect_table_region(img, padding=15, pdf_mode=False):
    """
    Advanced table detection: find the largest rectangular content block.
    
    Useful when page has multiple content areas (header, table, footnotes).
    Focuses on the main table region.
    
    SAFEGUARDS (prevent destructive cropping):
    - Cropped height must be >= 25% of original height
    - Never remove more than 70% of image height
    - PDF mode uses relaxed detection
    
    Args:
        img: PIL Image
        padding: Pixels to add around detected region
        pdf_mode: If True, use relaxed detection (PDF pages already have table)
    
    Returns:
        Cropped PIL Image focused on table region
    """
    gray = np.asarray(img.convert("L"))
    
    if gray.size == 0:
        return img
    
    height, width = gray.shape
    original_height = height
    original_width = width
    
    # ==========================================================================
    # PDF MODE: Skip aggressive detection, PDF pages are already focused
    # ==========================================================================
    if pdf_mode:
        print(f"[TABLE-DETECT] PDF mode: skipping aggressive crop, keeping {width}x{height}")
        return img
    
    # Detect content pixels
    content_mask = gray < 240
    
    # Compute row-wise ink density
    row_density = content_mask.mean(axis=1)
    
    # Find continuous content blocks (rows with significant ink)
    in_block = False
    blocks = []
    block_start = 0
    
    for i, density in enumerate(row_density):
        if density > 0.02:  # Row has content
            if not in_block:
                block_start = i
                in_block = True
        else:
            if in_block:
                blocks.append((block_start, i, i - block_start))
                in_block = False
    
    # Close final block if still in one
    if in_block:
        blocks.append((block_start, len(row_density), len(row_density) - block_start))
    
    if not blocks:
        print(f"[TABLE-DETECT] No content blocks found, keeping original {width}x{height}")
        return img
    
    # Find the largest block (most likely the table)
    largest_block = max(blocks, key=lambda b: b[2])
    top, bottom, block_height = largest_block
    
    # ==========================================================================
    # SAFEGUARD 1: Minimum height check (must be >= 25% of original)
    # ==========================================================================
    MIN_HEIGHT_RATIO = 0.25
    if block_height < original_height * MIN_HEIGHT_RATIO:
        print(f"[TABLE-DETECT] REJECTED: block height {block_height}px < 25% of {original_height}px. "
              f"Keeping full image.")
        return img
    
    # ==========================================================================
    # SAFEGUARD 2: Maximum crop check (never remove > 70% of height)
    # ==========================================================================
    MAX_CROP_RATIO = 0.70
    height_removed = original_height - block_height
    if height_removed > original_height * MAX_CROP_RATIO:
        print(f"[TABLE-DETECT] REJECTED: would remove {height_removed}px (>{MAX_CROP_RATIO*100:.0f}% of height). "
              f"Keeping full image.")
        return img
    
    # For left/right, use column-wise analysis within the block region
    block_content = content_mask[top:bottom, :]
    col_density = block_content.mean(axis=0)
    content_cols = np.where(col_density > 0.01)[0]
    
    if len(content_cols) == 0:
        left, right = 0, width
    else:
        left = content_cols[0]
        right = content_cols[-1]
    
    # Add padding
    top = max(0, top - padding)
    bottom = min(height, bottom + padding)
    left = max(0, left - padding)
    right = min(width, right + padding)
    
    if right <= left or bottom <= top:
        print(f"[TABLE-DETECT] Invalid crop bounds, keeping original")
        return img
    
    # Calculate cropped dimensions
    cropped_width = right - left
    cropped_height = bottom - top
    
    # ==========================================================================
    # SAFEGUARD 3: Final validation - cropped height must be reasonable
    # ==========================================================================
    if cropped_height < 100:  # Absolute minimum
        print(f"[TABLE-DETECT] REJECTED: cropped height {cropped_height}px < 100px minimum. "
              f"Keeping full image.")
        return img
    
    # Calculate savings
    original_area = width * height
    cropped_area = cropped_width * cropped_height
    savings_percent = (1 - cropped_area / original_area) * 100
    
    if savings_percent > 10:  # Only use if significant savings
        print(f"[TABLE-DETECT] Cropping: {width}x{height} → {cropped_width}x{cropped_height} "
              f"({savings_percent:.1f}% reduced)")
        return img.crop((left, top, right, bottom))
    
    print(f"[TABLE-DETECT] Savings too small ({savings_percent:.1f}%), keeping original")
    return img


def crop_table_region(img, header_crop_ratio=0.08, margin_ratio=0.02):
    """
    Crop page headers and side margins while PRESERVING TABLE CONTENT.
    
    CRITICAL: Do NOT cut table headers or top data rows.
    Only remove obvious page title/logo areas.
    """
    width, height = img.size

    # Minimal margins to preserve table content
    left = int(width * margin_ratio)
    right = int(width * (1.0 - margin_ratio))
    
    # REDUCED top crop - only remove obvious page title area
    # Changed from 0.18 to 0.08 to preserve table headers
    top = int(height * header_crop_ratio)
    bottom = int(height * 0.99)

    if right <= left or bottom <= top:
        return img

    cropped = img.crop((left, top, right, bottom))

    # Refine top boundary using horizontal ink density to find table start
    # But be CONSERVATIVE - we'd rather include noise than cut table content
    gray = np.asarray(cropped.convert("L"))
    if gray.size == 0:
        return cropped

    ink_density = (gray < 220).mean(axis=1)
    content_rows = np.where(ink_density > 0.04)[0]
    if content_rows.size > 0:
        # MINIMAL refinement - only trim obvious white space
        refined_top = max(int(content_rows[0]) - 15, 0)  # Increased margin from 6 to 15
        if refined_top > 0:
            cropped = cropped.crop((0, refined_top, cropped.width, cropped.height))

    return cropped


def preprocess_image_for_model(image):
    """
    Adaptive image preprocessing for vision-language models.
    
    Normalizes image dimensions to optimize VLM inference:
    - Large images (>1600px) → downscale to ~1200px max dimension
    - Small images (<600px) → upscale slightly (1.2-1.5x)
    - Medium images → keep original
    
    This is PIXEL-based, not file-size-based, because VLMs process visual tokens
    which are directly proportional to pixel count.
    
    Args:
        image: PIL Image (any mode)
        
    Returns:
        PIL Image (RGB, resized if needed)
    """
    from PIL import ImageEnhance, ImageFilter
    
    # Thresholds for adaptive resizing
    MAX_DIMENSION = 1600      # downscale if larger
    TARGET_DIMENSION = 1200   # target for large images
    MIN_DIMENSION = 600       # upscale if smaller
    UPSCALE_TARGET = 800      # target for small images
    SHARPEN_THRESHOLD = 3000  # apply sharpening if original was very large
    
    # Convert to RGB
    image = image.convert("RGB")
    
    orig_w, orig_h = image.size
    max_dim = max(orig_w, orig_h)
    
    scale = 1.0
    action = "kept"
    
    if max_dim > MAX_DIMENSION:
        # DOWNSCALE: Large image → reduce to TARGET_DIMENSION
        scale = TARGET_DIMENSION / max_dim
        action = "downscaled"
        
    elif max_dim < MIN_DIMENSION:
        # UPSCALE: Small image → increase to UPSCALE_TARGET
        # But don't upscale too aggressively (max 1.5x)
        target_scale = UPSCALE_TARGET / max_dim
        scale = min(target_scale, 1.5)
        action = "upscaled"
    
    # Apply scaling if needed
    if scale != 1.0:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Apply slight sharpening if heavily downscaled from very large image
        if action == "downscaled" and max_dim > SHARPEN_THRESHOLD:
            sharpener = ImageEnhance.Sharpness(image)
            image = sharpener.enhance(1.1)  # Subtle sharpening
        
        final_w, final_h = image.size
        print(f"[PREPROCESS] Original: {orig_w}x{orig_h}")
        print(f"[PREPROCESS] Resized: {final_w}x{final_h} (scale={scale:.2f}, {action})")
    else:
        print(f"[PREPROCESS] Original: {orig_w}x{orig_h} (no resize needed)")
    
    return image


def preprocess_image(image_path, max_image_size=DEFAULT_MAX_IMAGE_SIZE, enable_crop=True, smart_crop=True, vlm_optimize=True, pdf_mode=False):
    """
    Preprocess image for table extraction.
    
    Pipeline:
        0. Adaptive resize (normalize pixel dimensions for VLM)
        1. Smart auto-crop (detect content boundaries)
        2. Table region detection (find main table block) - with safeguards
        3. VLM optimization (reduce contrast, soften for better label extraction)
        4. Final resize to fit model constraints
    
    Args:
        image_path: Path to input image
        max_image_size: Maximum dimension for output
        enable_crop: Enable any cropping
        smart_crop: Use smart content-based cropping (recommended)
        vlm_optimize: Apply VLM-specific preprocessing to improve label extraction
        pdf_mode: If True, use relaxed cropping (PDF pages already contain table)
    
    Returns:
        (preprocessed_path, stats_dict)
    """
    # =========================================================================
    # STEP 0: ADAPTIVE RESIZE (normalize pixel dimensions)
    # =========================================================================
    image = Image.open(image_path)
    original_dimensions = image.size
    original_size_kb = os.path.getsize(image_path) / 1024
    
    # Apply adaptive preprocessing (handles RGB conversion, resizing)
    image = preprocess_image_for_model(image)
    after_resize_dimensions = image.size
    
    resized = (original_dimensions != after_resize_dimensions)
    
    original_size = image.size
    pre_crop_size = image.size  # Track size before cropping

    # =========================================================================
    # CROPPING (with safeguards for PDF mode)
    # =========================================================================
    if enable_crop and not pdf_mode:
        if smart_crop:
            # STEP 1: Auto-crop to content (remove empty margins)
            image = auto_crop_to_content(image, padding=10)
            
            # STEP 2: Detect and focus on table region (with safeguards)
            image = detect_table_region(image, padding=15, pdf_mode=pdf_mode)
        else:
            # Legacy fixed-ratio cropping
            image = crop_table_region(image)
        
        # =================================================================
        # SAFEGUARD: Validate crop didn't destroy the image
        # =================================================================
        post_crop_size = image.size
        crop_height_ratio = post_crop_size[1] / pre_crop_size[1]
        
        if crop_height_ratio < 0.25:
            # CRITICAL: Crop removed too much, reload original
            print(f"[PREPROCESS] WARNING: Crop reduced height to {crop_height_ratio*100:.1f}%. "
                  f"Reverting to pre-crop image.")
            image = Image.open(image_path)
            image = preprocess_image_for_model(image)
    elif pdf_mode:
        print(f"[PREPROCESS] PDF mode: skipping crop, image is {image.size[0]}x{image.size[1]}")

    # STEP 3: VLM optimization - critical for label extraction
    if vlm_optimize:
        image = _optimize_for_vlm(image)

    # STEP 4: Final resize if still too large
    width, height = image.size
    
    if width > max_image_size or height > 2000:
        scale_w = max_image_size / width if width > max_image_size else 1.0
        scale_h = 2000 / height if height > 2000 else 1.0
        scale = min(scale_w, scale_h)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    preprocessed_path = tmp.name
    tmp.close()
    processed_size = image.size
    image.save(preprocessed_path, format="PNG", optimize=True)
    image.close()

    # Calculate total reduction
    orig_area = original_dimensions[0] * original_dimensions[1]
    proc_area = processed_size[0] * processed_size[1]
    reduction = (1 - proc_area / orig_area) * 100 if orig_area > 0 else 0

    return preprocessed_path, {
        "original_size": original_size,
        "processed_size": processed_size,
        "reduction_percent": round(reduction, 1),
        "vlm_optimized": vlm_optimize,
        "resized": resized,
        "original_size_kb": round(original_size_kb, 1),
        "original_dimensions": original_dimensions,
        "final_dimensions": processed_size,
    }


def _optimize_for_vlm(image):
    """
    Apply VLM-specific optimizations to improve label extraction.
    
    This addresses the issue where Qwen3-VL 8B skips label text in high-contrast images.
    
    Transforms Image A (high-contrast PDF) → Image B style (soft screen capture):
    1. Background: pure white → light grey (230)
    2. Gamma correction: 1.18 (lighten bold text)
    3. Contrast reduction: factor 0.82
    4. Sharpness reduction: factor 0.85 (soften edges)
    5. Gaussian blur: radius 1 (~3x3 kernel) - safe for "(1-1)" note text
    """
    from PIL import ImageEnhance, ImageFilter
    
    # Convert PIL to numpy for pixel operations
    img_array = np.array(image)
    
    # 1. BACKGROUND NORMALIZATION (pure white → light grey)
    # Pixels >245 are near-white; shift them to 230 (visible grey)
    white_mask = np.all(img_array > 245, axis=2)
    img_array[white_mask] = [230, 230, 230]
    
    # 2. GAMMA CORRECTION (lighten dark/bold text)
    # Gamma 1.18 = more aggressive lightening of dark pixels
    gamma = 1.18
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 
                    for i in np.arange(256)]).astype(np.uint8)
    # Apply LUT to each channel
    for c in range(3):
        img_array[:,:,c] = lut[img_array[:,:,c]]
    
    # Convert back to PIL for ImageEnhance operations
    image = Image.fromarray(img_array)
    
    # 3. CONTRAST REDUCTION (factor 0.82 = visible reduction)
    # Balances attention between bold headers and regular text
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(0.82)
    
    # 4. SHARPNESS REDUCTION (factor 0.85 = soften edges)
    # Makes bold text appear thinner/lighter
    sharpness_enhancer = ImageEnhance.Sharpness(image)
    image = sharpness_enhancer.enhance(0.85)
    
    # 5. GAUSSIAN BLUR (radius 1 ≈ 3x3 kernel) - LAST STEP
    # Softens overall appearance; radius 1 is safe for small text like "(1-1)"
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    
    return image


# =============================================================================
# CHUNKING REMOVED - SINGLE PASS EXTRACTION
# =============================================================================
# The model works best with full image processing. Chunking has been removed
# to simplify the pipeline and avoid errors caused by split processing.
# =============================================================================


def _normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9%.,()/-]", "", text)
    return text


def _is_artificial_column(col_name):
    """
    Check if a column name is an artificial placeholder.
    
    SCHEMA LOCKING: These columns should be rejected.
    """
    if not col_name:
        return True
    
    name = str(col_name).strip().lower()
    
    if not name:
        return True
    
    # Reject "col" + number patterns
    if re.match(r'^col\d+$', name):
        return True
    
    # Reject "column" + number patterns
    if re.match(r'^column[_\s]?\d+$', name):
        return True
    
    # Reject pure numbers
    if re.match(r'^\d+$', name):
        return True
    
    # Reject very short generic names
    if name in ('col', 'column', 'field', 'value', 'data', 'item'):
        return True
    
    return False


# =============================================================================
# GLOBAL SCHEMA SELECTION SYSTEM (PROBLEM 1 FIX)
# =============================================================================

# Minimum score threshold for accepting a header
HEADER_SCORE_THRESHOLD = 3

# Date column regex patterns
_DATE_PATTERN_SLASH = re.compile(r'^\d{1,2}/\d{1,2}/\d{4}$')
_DATE_PATTERN_DOT = re.compile(r'^\d{1,2}\.\d{1,2}\.\d{4}$')
_DATE_PATTERN_DASH = re.compile(r'^\d{1,2}-\d{1,2}-\d{4}$')


def _is_valid_date_column(col_name: str) -> bool:
    """Check if a column name is a valid date (dd/mm/yyyy or dd.mm.yyyy)."""
    if not col_name:
        return False
    col = str(col_name).strip()
    return bool(
        _DATE_PATTERN_SLASH.match(col) or 
        _DATE_PATTERN_DOT.match(col) or
        _DATE_PATTERN_DASH.match(col)
    )


def _is_variation_column(col_name: str) -> bool:
    """Check if column is a variation column (Montant or %)."""
    if not col_name:
        return False
    col = str(col_name).strip().lower()
    return any(kw in col for kw in ['variation', 'montant', '%', 'var', 'ecart'])


def _is_note_column(col_name: str) -> bool:
    """Check if column is a Note column."""
    if not col_name:
        return False
    col = str(col_name).strip().lower()
    return col in ('note', 'notes', 'ref', 'reference', 'renvoi')


def score_header_candidate(columns: List[str]) -> Tuple[int, Dict]:
    """
    Score a header candidate using explicit scoring rules.
    
    HEADER SCORING SYSTEM:
        +2 per valid date column (dd/mm/yyyy or dd.mm.yyyy)
        +2 if both "Montant" and "%" exist
        +1 if "Note" exists
        +1 if number of numeric columns >= 2
        -2 if no date columns
        -2 if only variation columns (no dates)
        -3 if schema has less than 3 columns
    
    Returns:
        (score, details_dict)
    """
    if not columns:
        return -10, {"reason": "empty columns"}
    
    # Clean artificial columns
    clean_cols = [c for c in columns if c and not _is_artificial_column(c)]
    
    score = 0
    details = {
        "total_columns": len(clean_cols),
        "date_columns": [],
        "variation_columns": [],
        "has_note": False,
        "has_montant": False,
        "has_percent": False,
    }
    
    # Analyze each column
    for col in clean_cols:
        if _is_valid_date_column(col):
            details["date_columns"].append(col)
        elif _is_variation_column(col):
            details["variation_columns"].append(col)
            col_lower = col.lower()
            if 'montant' in col_lower or ('variation' in col_lower and '%' not in col_lower):
                details["has_montant"] = True
            if '%' in col_lower:
                details["has_percent"] = True
        elif _is_note_column(col):
            details["has_note"] = True
    
    # Apply scoring rules
    
    # +2 per valid date column
    num_dates = len(details["date_columns"])
    score += num_dates * 2
    
    # +2 if both "Montant" and "%" exist
    if details["has_montant"] and details["has_percent"]:
        score += 2
    
    # +1 if "Note" exists
    if details["has_note"]:
        score += 1
    
    # +1 if number of numeric columns >= 2
    numeric_cols = num_dates + len(details["variation_columns"])
    if numeric_cols >= 2:
        score += 1
    
    # -2 if no date columns
    if num_dates == 0:
        score -= 2
    
    # -2 if only variation columns (no dates)
    if num_dates == 0 and len(details["variation_columns"]) > 0:
        score -= 2
    
    # -3 if schema has less than 3 columns
    if len(clean_cols) < 3:
        score -= 3
    
    details["score"] = score
    return score, details


def _row_signature(row, columns):
    label = _normalize_text(row.get("Label", ""))
    note = _normalize_text(row.get("Note", ""))
    values = []
    for col in columns:
        if col in {"Label", "Note", "type"}:
            continue
        values.append(_normalize_text(row.get(col, "")))
    joined_values = "|".join(values)
    return f"{label}|{note}|{joined_values}"


def remove_duplicate_rows(rows, columns):
    """Remove duplicate/overlap rows using exact signature and label similarity."""
    unique_rows = []
    signatures = set()

    for row in rows:
        signature = _row_signature(row, columns)
        label = _normalize_text(row.get("Label", ""))

        if signature in signatures:
            continue

        is_duplicate = False
        for existing in unique_rows:
            existing_label = _normalize_text(existing.get("Label", ""))
            if label and existing_label:
                sim = SequenceMatcher(None, label, existing_label).ratio()
                if sim >= 0.96:
                    existing_sig = _row_signature(existing, columns)
                    # When labels are almost equal and numeric content overlaps, treat as duplicate.
                    if signature.split("|")[-1] == existing_sig.split("|")[-1]:
                        is_duplicate = True
                        break

        if is_duplicate:
            continue

        signatures.add(signature)
        unique_rows.append(row)

    return unique_rows


def _columns_match(locked_col: str, row_col: str) -> bool:
    """
    Check if two column names refer to the same column.
    
    Handles minor variations in date formatting.
    """
    if not locked_col or not row_col:
        return False
    
    # Exact match
    if locked_col == row_col:
        return True
    
    # Normalize for comparison
    l1 = locked_col.strip().lower()
    l2 = row_col.strip().lower()
    
    if l1 == l2:
        return True
    
    # Date normalization: 31/12/2024 == 31.12.2024
    l1_normalized = l1.replace(".", "/")
    l2_normalized = l2.replace(".", "/")
    
    return l1_normalized == l2_normalized


def _detect_date_format(col_name: str) -> str:
    """
    Detect the date format used in a column name.
    
    Returns:
        "slash" for dd/mm/yyyy
        "dot" for dd.mm.yyyy
        "none" if not a date column
    """
    if not col_name:
        return "none"
    
    col = str(col_name).strip()
    
    # Check for slash format: dd/mm/yyyy or mm/dd/yyyy
    if re.match(r'^\d{2}/\d{2}/\d{4}$', col):
        return "slash"
    
    # Check for dot format: dd.mm.yyyy
    if re.match(r'^\d{2}\.\d{2}\.\d{4}$', col):
        return "dot"
    
    return "none"


def _validate_schema_consistency(columns: List[str]) -> Tuple[bool, List[str], str]:
    """
    Validate that all date columns use the same format.
    
    SCHEMA ISOLATION: Prevents mixing dates from different tables.
    
    Returns:
        (is_valid, cleaned_columns, dominant_format)
    """
    date_formats = {}  # format -> list of columns
    non_date_columns = []
    
    for col in columns:
        fmt = _detect_date_format(col)
        if fmt == "none":
            non_date_columns.append(col)
        else:
            if fmt not in date_formats:
                date_formats[fmt] = []
            date_formats[fmt].append(col)
    
    # If no date columns, schema is valid
    if not date_formats:
        return True, columns, "none"
    
    # If only one format, schema is valid
    if len(date_formats) == 1:
        dominant_format = list(date_formats.keys())[0]
        return True, columns, dominant_format
    
    # MIXED FORMATS DETECTED - this is the contamination bug
    print(f"[SCHEMA ISOLATION] Mixed date formats detected!")
    for fmt, cols in date_formats.items():
        print(f"  - {fmt}: {cols}")
    
    # Keep ONLY the dominant format (most columns)
    dominant_format = max(date_formats.keys(), key=lambda f: len(date_formats[f]))
    dominant_dates = date_formats[dominant_format]
    
    # Build cleaned column list
    cleaned_columns = non_date_columns.copy()
    
    # Insert date columns in their original position order
    for col in columns:
        if col in dominant_dates:
            cleaned_columns.append(col)
    
    rejected = []
    for fmt, cols in date_formats.items():
        if fmt != dominant_format:
            rejected.extend(cols)
    
    print(f"[SCHEMA ISOLATION] Keeping format '{dominant_format}': {dominant_dates}")
    print(f"[SCHEMA ISOLATION] Rejecting foreign columns: {rejected}")
    
    return False, cleaned_columns, dominant_format


# =============================================================================
# SIMPLIFIED PIPELINE - SINGLE PASS EXTRACTION
# =============================================================================
#
# SIMPLIFIED PIPELINE:
#   Image → Preprocess → Single Inference → Parse → Validate Schema → Done
#
# The alignment engine and financial validation are applied in json_table_utils.py
# after the raw extraction is complete.
#
# =============================================================================

def validate_and_clean_schema(parsed: Dict) -> Dict:
    """
    Validate extracted schema and clean artificial columns.
    
    This runs immediately after parsing to ensure schema quality.
    """
    if not isinstance(parsed, dict):
        return parsed
    
    columns = parsed.get("columns", [])
    if not isinstance(columns, list):
        return parsed
    
    # Remove artificial columns
    clean_cols = [c for c in columns if c and not _is_artificial_column(c)]
    
    # Validate date format consistency
    is_valid, validated_cols, date_format = _validate_schema_consistency(clean_cols)
    
    if not is_valid:
        print(f"[SCHEMA] Date format cleaned: {clean_cols} -> {validated_cols}")
    
    # Score the header
    score, details = score_header_candidate(validated_cols)
    print(f"[SCHEMA] Header score: {score}")
    print(f"[SCHEMA] Date columns: {details.get('date_columns', [])}")
    
    # Ensure essential columns
    if "Label" not in validated_cols:
        validated_cols.insert(0, "Label")
    if "Note" not in validated_cols:
        idx = 1 if "Label" in validated_cols else 0
        validated_cols.insert(idx, "Note")
    
    parsed["columns"] = validated_cols
    parsed["_header_score"] = score
    parsed["_date_format"] = date_format
    parsed["_schema_validated"] = True
    
    # Clean rows to match schema
    rows = parsed.get("rows", [])
    if isinstance(rows, list):
        clean_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            clean_row = {}
            for col in validated_cols:
                if col in row:
                    clean_row[col] = row[col]
                else:
                    # Try fuzzy match
                    found = False
                    for row_col, val in row.items():
                        if _columns_match(col, row_col):
                            clean_row[col] = val
                            found = True
                            break
                    if not found:
                        clean_row[col] = ""
            clean_row["type"] = row.get("type", "data")
            clean_rows.append(clean_row)
        parsed["rows"] = clean_rows
    
    return parsed


# =============================================================================
# INFERENCE FUNCTION
# =============================================================================


def run_inference(model, processor, image_path, prompt, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    """Run deterministic and efficient Qwen3-VL inference."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")
    print_vram_status("After preparing inputs")

    try:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                num_beams=1,
            )
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[CUDA ERROR] Out of memory: {e}")
        print("[CUDA ERROR] Trying to free memory and retry with fewer tokens...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Retry with half the tokens
        reduced_tokens = max(512, max_new_tokens // 2)
        print(f"[CUDA ERROR] Retrying with {reduced_tokens} tokens...")
        
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=reduced_tokens,
                do_sample=False,
                use_cache=True,
                num_beams=1,
            )
    except Exception as e:
        print(f"\n[INFERENCE ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()
    return output_text


def detect_truncation(output_text: str, parsed: Dict) -> Tuple[bool, List[str]]:
    """
    FIX 7: Detect if output was truncated due to token limit.
    
    Signs of truncation:
        - Abrupt ending (no closing brace/bracket)
        - Missing closing structure
        - Unusually low row count (< 3)
        - JSON ends mid-string
    
    Returns:
        (is_truncated, reasons)
    """
    reasons = []
    
    # Check for abrupt JSON ending
    stripped = output_text.rstrip()
    if stripped and stripped[-1] not in ('}', ']'):
        reasons.append("Output does not end with } or ]")
    
    # Check for unbalanced braces/brackets
    open_braces = output_text.count('{')
    close_braces = output_text.count('}')
    open_brackets = output_text.count('[')
    close_brackets = output_text.count(']')
    
    if open_braces > close_braces:
        reasons.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
    if open_brackets > close_brackets:
        reasons.append(f"Unbalanced brackets: {open_brackets} open, {close_brackets} close")
    
    # Check for unusually low row count
    if parsed:
        rows = parsed.get("rows", [])
        if len(rows) < 3:
            reasons.append(f"Suspiciously low row count: {len(rows)}")
    
    # Check for truncated strings (unclosed quotes)
    # Simple heuristic: odd number of unescaped quotes
    quote_count = len(re.findall(r'(?<!\\)"', output_text))
    if quote_count % 2 != 0:
        reasons.append("Odd number of quotes (possible string truncation)")
    
    is_truncated = len(reasons) > 0
    return is_truncated, reasons


def estimate_required_tokens(image_path: str, processed_size: Tuple[int, int] = None) -> int:
    """
    Dynamically estimate required max_new_tokens based on image size.
    
    Large tables need more tokens to fully generate the JSON output.
    This function scales tokens based on image dimensions.
    
    Args:
        image_path: Path to image file
        processed_size: Optional (width, height) of processed image
        
    Returns:
        Estimated max_new_tokens needed
    """
    try:
        if processed_size:
            width, height = processed_size
        else:
            with Image.open(image_path) as img:
                width, height = img.size
        
        # Calculate image area (proxy for table complexity)
        area = width * height
        
        # Estimate based on height (more rows = more tokens needed)
        # and area (larger tables = more data)
        
        if height > 1200 or area > 1_500_000:
            # Large table: needs maximum tokens
            tokens = MIN_TOKENS_LARGE_TABLE
            print(f"[TOKENS] Large image ({width}x{height}), using {tokens} tokens")
        elif height > 700 or area > 800_000:
            # Medium table
            tokens = MIN_TOKENS_MEDIUM_TABLE
            print(f"[TOKENS] Medium image ({width}x{height}), using {tokens} tokens")
        else:
            # Small table
            tokens = MIN_TOKENS_SMALL_TABLE
            print(f"[TOKENS] Small image ({width}x{height}), using {tokens} tokens")
        
        return tokens
        
    except Exception as e:
        print(f"[TOKENS] Could not estimate (using default): {e}")
        return DEFAULT_MAX_NEW_TOKENS


def should_retry_with_more_tokens(
    output_text: str,
    parsed: Dict,
    current_tokens: int
) -> Tuple[bool, int]:
    """
    Determine if we should retry with more tokens.
    
    Returns:
        (should_retry, new_token_count)
    """
    if current_tokens >= MAX_RETRY_TOKENS:
        # Already at maximum, don't retry
        return False, current_tokens
    
    # Check for truncation signs
    is_truncated, reasons = detect_truncation(output_text, parsed)
    
    if not is_truncated:
        return False, current_tokens
    
    # Check for critical truncation signs (JSON parse failure or structure issues)
    critical_truncation = any(
        "brace" in r.lower() or 
        "bracket" in r.lower() or 
        "does not end" in r.lower()
        for r in reasons
    )
    
    if critical_truncation or parsed is None:
        # Scale up by 1.5x
        new_tokens = min(int(current_tokens * 1.5), MAX_RETRY_TOKENS)
        print(f"[RETRY] Truncation detected, increasing tokens: {current_tokens} -> {new_tokens}")
        return True, new_tokens
    
    return False, current_tokens


def extract_table_from_image(
    model,
    processor,
    image_path,
    prompt=None,
    max_new_tokens=None,  # Now optional - will be auto-calculated
    max_image_size=DEFAULT_MAX_IMAGE_SIZE,
    enable_crop=True,
    pdf_mode=False,  # NEW: disable aggressive cropping for PDF pages
):
    """
    Extract table data from a financial statement image.
    
    SIMPLIFIED SINGLE-PASS PIPELINE:
        Image -> Preprocess -> Single Inference -> Parse -> Validate Schema -> Done
    
    DYNAMIC TOKEN SCALING:
        - Automatically estimates tokens needed based on image size
        - Retries with higher tokens if truncation detected
    
    SAFEGUARDS:
        - pdf_mode=True disables aggressive table detection (prevents empty outputs)
        - Fallback retry with full image if row_count < 5
    
    Args:
        model: Loaded Qwen-VL model
        processor: Qwen-VL processor
        image_path: Path to the image file
        prompt: Custom extraction prompt (optional)
        max_new_tokens: Override token limit (optional, auto-calculated if None)
        max_image_size: Maximum dimension for image
        enable_crop: Enable cropping (set False to skip all cropping)
        pdf_mode: If True, skip aggressive table detection (PDF pages already focused)

    Returns:
        Extracted JSON table data as string
    """

    print("\n" + "=" * 60)
    print("SINGLE-PASS TABLE EXTRACTION (with dynamic tokens)")
    print("=" * 60)

    print_vram_status("Before inference")

    # Clear VRAM before starting
    torch.cuda.empty_cache()
    gc.collect()

    # Use improved default prompt if none provided
    if prompt is None:
        prompt = DEFAULT_EXTRACTION_PROMPT

    preprocessed_path = None
    raw_output = None  # FIX 3: Store raw output
    
    try:
        # STEP 1: Preprocess image (resize, crop margins, VLM optimization)
        # pdf_mode disables aggressive cropping to prevent destroying tables
        preprocessed_path, stats = preprocess_image(
            image_path,
            max_image_size=max_image_size,
            enable_crop=enable_crop,
            vlm_optimize=True,  # Critical for label extraction on high-contrast images
            pdf_mode=pdf_mode,  # NEW: pass pdf_mode to disable aggressive crop
        )
        reduction = stats.get('reduction_percent', 0)
        print(f"[Preprocess] {stats['original_size']} -> {stats['processed_size']} ({reduction}% reduced)")

        # DYNAMIC TOKEN SCALING: Estimate tokens based on image size
        if max_new_tokens is None:
            current_tokens = estimate_required_tokens(image_path, stats['processed_size'])
        else:
            current_tokens = max_new_tokens
            print(f"[TOKENS] Using provided value: {current_tokens}")

        # STEP 2: Single-pass inference with dynamic tokens
        print(f"\n[Inference] Running single-pass extraction (max_tokens={current_tokens})...")
        output_text = run_inference(
            model,
            processor,
            preprocessed_path,
            prompt,
            max_new_tokens=current_tokens,
        )
        
        # FIX 3: Store raw output BEFORE any processing
        raw_output = output_text
        
        # STEP 3: Parse JSON from response
        parsed = extract_json_from_response(output_text)
        
        # RETRY LOGIC: If truncation detected, retry with more tokens
        should_retry, new_tokens = should_retry_with_more_tokens(output_text, parsed, current_tokens)
        
        if should_retry:
            print(f"\n[RETRY] Retrying extraction with {new_tokens} tokens...")
            torch.cuda.empty_cache()
            gc.collect()
            
            output_text = run_inference(
                model,
                processor,
                preprocessed_path,
                prompt,
                max_new_tokens=new_tokens,
            )
            raw_output = output_text
            parsed = extract_json_from_response(output_text)
            current_tokens = new_tokens
        
        # If still failing, try strict JSON prompt
        if parsed is None:
            print("[Parse] Parsing failed, retrying with strict JSON prompt...")
            strict_prompt = (
                "Extract the financial table. Return ONLY valid JSON.\n\n"
                "Schema: {\"columns\": [...], \"rows\": [...]}\n\n"
                "No explanation. No markdown. Just JSON."
            )
            # Use even higher tokens for retry
            retry_tokens = min(int(current_tokens * 1.5), MAX_RETRY_TOKENS)
            
            output_text = run_inference(
                model,
                processor,
                preprocessed_path,
                strict_prompt,
                max_new_tokens=retry_tokens,
            )
            
            # FIX 3: Update raw output with retry
            raw_output = output_text
            parsed = extract_json_from_response(output_text)
        
        if parsed is not None:
            # FIX 7: Check for truncation
            is_truncated, truncation_reasons = detect_truncation(output_text, parsed)
            if is_truncated:
                print(f"[WARNING] Possible truncation detected: {truncation_reasons}")
                parsed["_truncated"] = True
                parsed["_truncation_reasons"] = truncation_reasons
            
            # FIX 3: Preserve raw data BEFORE schema validation
            parsed["_raw_output"] = raw_output[:2000] if len(raw_output) > 2000 else raw_output
            parsed["_raw_columns"] = list(parsed.get("columns", []))  # Copy
            parsed["_raw_rows_count"] = len(parsed.get("rows", []))
            
            # STEP 4: Validate and clean schema
            print("\n[Schema] Validating extracted schema...")
            parsed = validate_and_clean_schema(parsed)
            
            # Add extraction metadata
            parsed["_extraction_mode"] = "single_pass"
            parsed["_image_size"] = stats["processed_size"]
            parsed["_max_tokens_used"] = current_tokens
            
            # Summary
            print(f"\n[RESULT] Schema: {parsed.get('columns', [])}")
            print(f"[RESULT] Rows: {len(parsed.get('rows', []))}")
            print(f"[RESULT] Tokens used: {current_tokens}")
            print(f"[RESULT] Header score: {parsed.get('_header_score', 'N/A')}")
            if is_truncated:
                print(f"[RESULT] TRUNCATED: {truncation_reasons}")
            
            output_text = json.dumps(parsed, ensure_ascii=False)
        else:
            # FIX 11: Return structured error on JSON failure
            print("[ERROR] Could not parse JSON from model output")
            error_result = {
                "error": "JSON parse failed",
                "error_type": "parse_failure",
                "_raw_output": raw_output[:2000] if raw_output and len(raw_output) > 2000 else raw_output,
                "_parse_failed": True,
                "columns": [],
                "rows": [],
                "_extraction_mode": "single_pass",
                "_image_size": stats["processed_size"],
                "_max_tokens_used": current_tokens
            }
            output_text = json.dumps(error_result, ensure_ascii=False)

    finally:
        if preprocessed_path and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)

    print_vram_status("After inference")
    print("\nExtraction complete!")
    print("=" * 60)

    return output_text

# =============================================================================
# STEP 6: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution flow"""

    # 1. Verify GPU
    verify_gpu()

    # 2. Load model
    model, processor = load_model()

    # 3. Test with a sample image (replace with your image path)
    test_image = "test_image.png"  # <-- CHANGE THIS to your image path

    # Check if test image exists
    import os
    if not os.path.exists(test_image):
        print(f"\nWARNING: Test image '{test_image}' not found!")
        print("To test, either:")
        print("  1. Place an image named 'test_image.png' in the current directory")
        print("  2. Modify the 'test_image' variable in this script")
        print("\nModel is loaded and ready. You can call extract_table_from_image() with your image.")

        # Interactive mode
        while True:
            user_input = input("\nEnter image path (or 'quit' to exit): ").strip()
            if user_input.lower() == 'quit':
                break
            if os.path.exists(user_input):
                result = extract_table_from_image(model, processor, user_input)
                print("\n" + "=" * 60)
                print("EXTRACTION RESULT")
                print("=" * 60)
                print(result)
            else:
                print(f"File not found: {user_input}")
    else:
        # Run extraction
        result = extract_table_from_image(model, processor, test_image)

        print("\n" + "=" * 60)
        print("EXTRACTION RESULT")
        print("=" * 60)
        print(result)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

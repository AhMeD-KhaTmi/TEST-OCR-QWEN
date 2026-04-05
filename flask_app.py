"""
Flask API for Financial Statement Table Extraction
Optimized for RTX 4060 (8GB VRAM)
Supports: Images (PNG, JPG) and PDF files with page selection
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import torch
import time
import io
import base64
import json
import re
import tempfile
import uuid
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

# PDF processing
import fitz  # PyMuPDF

# Import our extraction functions
from run_qwen_vl import (
    load_model,
    extract_table_from_image,
    get_vram_usage,
    DEFAULT_MAX_NEW_TOKENS,
    extract_table_title_sync,  # TABLE TITLE DETECTION
)
from json_table_utils import (
    extract_json_from_response,
    validate_table_json,
    post_process_extraction
)

# PDF handler for unified pipeline
from pdf_handler import (
    is_pdf_supported,
    get_pdf_info_from_bytes,
    parse_page_selection,
    extract_pdf_pages_from_bytes,
    merge_page_results,
    generate_pdf_thumbnails,
    estimate_processing_time
)

# Financial table detector for page recommendations
from financial_table_detector import (
    detect_financial_statements,
    get_financial_pages,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# PDF rendering and OCR quality settings tuned for 8GB VRAM.
PDF_DPI_LEVELS = [216, 264, 300]  # ~3x to 4.16x scaling (72 DPI base)
PDF_OCR_MAX_IMAGE_SIZE = 1700
PDF_MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS
PDF_FALLBACK_MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS
# Table detection threshold - LOWERED for text-based financial tables
# Financial statements often don't have visible grid lines, so we rely more on
# numeric density than line detection. User-selected pages should always be processed.
TABLE_DETECTION_MIN_SCORE = 0.02  # Very low threshold - trust user selection

# Image extraction budgets tuned to avoid row truncation on long tables.
IMAGE_MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS
IMAGE_RETRY_MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS
IMAGE_RESCUE_MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS

# Focused prompt for PDF pages to keep generation shorter and more stable.
PDF_EXTRACTION_PROMPT = """Extract the COMPLETE main financial table from this page as valid JSON only.

Rules:
1. Return one table only (the primary statement table).
2. Keep all column names exactly as shown.
3. Keep values exactly as shown.
4. Row type must be one of: section, data, total.
5. Do not include narrative text, titles, footers, or notes paragraphs.
6. Include every visible table row from first to last row.
7. Do not stop after 10 rows; continue until the final row.
8. Do not summarize rows and do not apply any row limit.

Output schema:
{
    \"columns\": [\"Label\", \"Note\", \"...\"],
    \"rows\": [
        {
            \"type\": \"section|data|total\",
            \"Label\": \"...\",
            \"Note\": \"...\"
        }
    ]
}
"""

PDF_FALLBACK_PROMPT = """Extract the COMPLETE financial statement table from this image and return valid JSON only.
Preserve column headers and values exactly.
Include all visible rows from top to bottom (do not stop early).
Do not summarize rows and do not apply any row limit.
Output schema:
{
    "columns": ["Label", "Note", "..."],
    "rows": [{"type": "section|data|total", "Label": "...", "Note": "..."}]
}
"""

# =============================================================================
# GLOBAL MODEL (load once at startup)
# =============================================================================

model = None
processor = None
model_loaded = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _get_upload_size(file_storage):
    """Best-effort upload size in bytes for logging/debug."""
    try:
        stream = file_storage.stream
        current_pos = stream.tell()
        stream.seek(0, os.SEEK_END)
        size = int(stream.tell())
        stream.seek(current_pos, os.SEEK_SET)
        return size
    except Exception:
        return None


def _parse_selected_pages(pages_str):
    """Parse selected pages from JSON array or comma-separated input."""
    pages_str = (pages_str or '').strip()
    if not pages_str:
        raise ValueError('No pages selected')

    if pages_str.startswith('['):
        parsed = json.loads(pages_str)
        if not isinstance(parsed, list):
            raise ValueError('Pages must be a list of page numbers')
        pages = [int(p) for p in parsed]
    else:
        pages = [int(p.strip()) for p in pages_str.split(',') if p.strip()]

    pages = sorted(set(pages))
    if not pages:
        raise ValueError('No valid page numbers found')
    return pages


def _render_page_to_pil(page, dpi):
    """Render a PDF page to PIL image at target DPI."""
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    del pix
    return image


def _laplacian_variance(gray_array):
    """Approximate blur via Laplacian variance using NumPy only."""
    arr = gray_array.astype(np.float32)
    lap = (
        -4.0 * arr[1:-1, 1:-1]
        + arr[:-2, 1:-1]
        + arr[2:, 1:-1]
        + arr[1:-1, :-2]
        + arr[1:-1, 2:]
    )
    return float(np.var(lap)) if lap.size else 0.0


def _image_quality_metrics(image):
    """Compute quality metrics used by the auto-quality optimizer."""
    gray = image.convert('L')
    arr = np.asarray(gray)

    width, height = image.size
    longest_side = max(width, height)
    shortest_side = min(width, height)
    contrast_std = float(arr.std())
    blur_score = _laplacian_variance(arr)

    low_resolution = longest_side < 1800 or shortest_side < 1100
    low_contrast = contrast_std < 38.0
    blurry = blur_score < 60.0

    return {
        'width': width,
        'height': height,
        'longest_side': longest_side,
        'contrast_std': round(contrast_std, 2),
        'blur_score': round(blur_score, 2),
        'low_resolution': low_resolution,
        'low_contrast': low_contrast,
        'blurry': blurry,
        'low_quality': low_resolution or low_contrast or blurry,
    }


def enhance_image(image, apply_threshold=False):
    """Enhance rendered PDF page for OCR: grayscale, contrast, and optional threshold."""
    gray = image.convert('L')
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.25)

    # A light median filter helps remove scan noise without destroying table lines.
    gray = gray.filter(ImageFilter.MedianFilter(size=3))

    if apply_threshold:
        arr = np.asarray(gray)
        # Conservative threshold keeps line structure and avoids over-binarization.
        threshold = int(np.clip(arr.mean() - 8, 90, 185))
        bw = (arr > threshold).astype(np.uint8) * 255
        gray = Image.fromarray(bw, mode='L')

    return gray.convert('RGB')


def detect_table_page(page, image):
    """Heuristic table detector based on visual line density and numeric text density."""
    text = page.get_text("text") or ""
    total_chars = max(len(text), 1)
    numeric_chars = len(re.findall(r'\d', text))
    numeric_density = numeric_chars / total_chars

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    numeric_lines = 0
    for ln in lines:
        if len(re.findall(r'\d', ln)) >= 4:
            numeric_lines += 1
    numeric_line_ratio = numeric_lines / max(len(lines), 1)

    preview = image.convert('L')
    if max(preview.size) > 1200:
        ratio = 1200 / max(preview.size)
        preview = preview.resize((int(preview.size[0] * ratio), int(preview.size[1] * ratio)), Image.Resampling.BILINEAR)

    arr = np.asarray(preview)
    binary = arr < 150
    row_dark_ratio = binary.mean(axis=1)
    col_dark_ratio = binary.mean(axis=0)

    horizontal_line_score = float(np.mean(row_dark_ratio > 0.65))
    vertical_line_score = float(np.mean(col_dark_ratio > 0.65))
    line_score = min(1.0, (horizontal_line_score * 4.0) + (vertical_line_score * 4.0))

    numeric_density_score = min(1.0, numeric_density / 0.22)
    numeric_line_score = min(1.0, numeric_line_ratio / 0.35)

    table_score = (0.45 * numeric_density_score) + (0.35 * line_score) + (0.20 * numeric_line_score)
    is_table = table_score >= TABLE_DETECTION_MIN_SCORE

    metrics = {
        'table_score': round(table_score, 3),
        'numeric_density': round(numeric_density, 3),
        'numeric_line_ratio': round(numeric_line_ratio, 3),
        'line_score': round(line_score, 3),
    }
    return is_table, metrics


def pdf_to_images(pdf_bytes, selected_pages):
    """
    Convert selected PDF pages to OCR-ready PNG images with auto-quality optimization.

    Returns:
        dict containing page_count and per-page image metadata.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)

    invalid_pages = [p for p in selected_pages if p < 1 or p > page_count]
    if invalid_pages:
        doc.close()
        raise ValueError(f'Invalid page numbers: {invalid_pages}. PDF has {page_count} pages.')

    results = []

    try:
        for page_num in selected_pages:
            page = doc.load_page(page_num - 1)

            best_image = None
            best_quality = None
            best_dpi = PDF_DPI_LEVELS[0]

            # Auto-quality optimizer: re-render at higher DPI when needed.
            for dpi in PDF_DPI_LEVELS:
                candidate = _render_page_to_pil(page, dpi)
                quality = _image_quality_metrics(candidate)

                best_image = candidate
                best_quality = quality
                best_dpi = dpi

                if not quality['low_quality']:
                    break

            enhanced = enhance_image(
                best_image,
                apply_threshold=best_quality['low_contrast']
            )

            # Keep image details for OCR while respecting VRAM constraints.
            if max(enhanced.size) > PDF_OCR_MAX_IMAGE_SIZE:
                ratio = PDF_OCR_MAX_IMAGE_SIZE / max(enhanced.size)
                new_size = (int(enhanced.size[0] * ratio), int(enhanced.size[1] * ratio))
                enhanced = enhanced.resize(new_size, Image.Resampling.LANCZOS)

            table_detected, table_metrics = detect_table_page(page, enhanced)

            raw_tmp = tempfile.NamedTemporaryFile(
                mode='wb',
                suffix=f'_p{page_num}_raw.png',
                dir=UPLOAD_FOLDER,
                delete=False
            )
            raw_path = raw_tmp.name
            raw_tmp.close()

            ocr_tmp = tempfile.NamedTemporaryFile(
                mode='wb',
                suffix=f'_p{page_num}_ocr.png',
                dir=UPLOAD_FOLDER,
                delete=False
            )
            ocr_path = ocr_tmp.name
            ocr_tmp.close()

            best_image.save(raw_path, format='PNG', optimize=True)
            enhanced.save(ocr_path, format='PNG', optimize=True)

            results.append({
                'page': page_num,
                'raw_image_path': raw_path,
                'ocr_image_path': ocr_path,
                'dpi': best_dpi,
                'quality': best_quality,
                'table_detected': table_detected,
                'table_metrics': table_metrics,
            })

    finally:
        doc.close()

    return {
        'page_count': page_count,
        'pages': results,
    }


def infer_table_type_from_content(parsed_json):
    """
    SAFETY FALLBACK: Infer table type from row content when title detection fails.
    
    Rules:
    - If rows contain "RESULTAT", "PRODUIT", "CHARGES" → income_statement
    - If rows contain "ACTIF", "PASSIF" → balance_sheet
    - If rows contain "ENGAGEMENT", "HORS BILAN" → off_balance
    
    Args:
        parsed_json: Extracted table JSON with rows
        
    Returns:
        Inferred table_type string or None
    """
    if not parsed_json or not isinstance(parsed_json, dict):
        return None
    
    rows = parsed_json.get('rows', [])
    if not rows:
        return None
    
    # Collect all text from labels (first column typically)
    all_text = []
    for row in rows:
        if isinstance(row, dict):
            # Check "label" field
            label = row.get('label', '')
            if label:
                all_text.append(str(label).upper())
            # Also check first value in row
            for key, val in row.items():
                if isinstance(val, str):
                    all_text.append(val.upper())
                    break
        elif isinstance(row, list) and row:
            all_text.append(str(row[0]).upper())
    
    combined_text = ' '.join(all_text)
    
    # Income statement indicators
    income_keywords = ['RESULTAT', 'PRODUIT', 'CHARGES', 'EXPLOITATION', 'BENEFICE', 'PERTE']
    income_score = sum(1 for kw in income_keywords if kw in combined_text)
    
    # Balance sheet indicators
    balance_keywords = ['ACTIF', 'PASSIF', 'CAPITAUX', 'IMMOBILISATIONS', 'CREANCES', 'DETTES']
    balance_score = sum(1 for kw in balance_keywords if kw in combined_text)
    
    # Off-balance indicators
    off_balance_keywords = ['ENGAGEMENT', 'HORS BILAN', 'GARANTIE', 'CAUTION']
    off_balance_score = sum(1 for kw in off_balance_keywords if kw in combined_text)
    
    # Return highest score type (minimum 2 matches required)
    scores = {
        'income_statement': income_score,
        'balance_sheet': balance_score,
        'off_balance': off_balance_score,
    }
    
    best_type = max(scores, key=scores.get)
    if scores[best_type] >= 2:
        return best_type
    
    return None


def process_pages(pdf_bytes, selected_pages):
    """Process selected PDF pages sequentially through OCR with robust fallbacks."""
    conversion = pdf_to_images(pdf_bytes, selected_pages)
    page_artifacts = conversion['pages']
    results = []

    for idx, artifact in enumerate(page_artifacts):
        page_num = artifact['page']
        page_start_time = time.time()

        print(f"[PDF] Processing page {page_num} ({idx + 1}/{len(page_artifacts)})...", flush=True)

        # NOTE: We no longer skip pages based on table detection score.
        # If the user (or smart detector) selected this page, we ALWAYS try to extract.
        # The table_metrics are logged for debugging but don't block extraction.
        if not artifact['table_detected']:
            print(
                f"[PDF] Page {page_num}: low table score ({artifact['table_metrics']['table_score']:.3f}) "
                f"- proceeding anyway (user-selected)",
                flush=True
            )

        try:
            torch.cuda.empty_cache()
            
            # =================================================================
            # STEP 1: PER-PAGE TITLE DETECTION (BEFORE table extraction)
            # Each page gets its own title - NO SHARED TITLES
            # =================================================================
            title_info = {"table_name": "UNKNOWN", "table_type": None, "title_detection_success": False}
            raw_image_path = artifact.get('raw_image_path')
            if raw_image_path and os.path.exists(raw_image_path):
                title_info = extract_table_title_sync(model, processor, raw_image_path)
                print(
                    f"[PAGE {page_num}] Title detected: {title_info.get('table_name')!r} "
                    f"(type: {title_info.get('table_type')})",
                    flush=True,
                )
            
            # =================================================================
            # STEP 2: TABLE EXTRACTION
            # =================================================================
            raw_result = extract_table_from_image(
                model,
                processor,
                artifact['ocr_image_path'],
                prompt=PDF_EXTRACTION_PROMPT,
                max_new_tokens=PDF_MAX_NEW_TOKENS,
                max_image_size=PDF_OCR_MAX_IMAGE_SIZE,
                pdf_mode=True,  # CRITICAL: Disable aggressive cropping for PDF
            )

            parsed_json = extract_json_from_response(raw_result)
            fallback_used = False

            # =============================================================
            # FALLBACK RETRY: If row_count < 5, retry with no crop
            # =============================================================
            row_count = len(parsed_json.get('rows', [])) if parsed_json else 0
            
            if not parsed_json or row_count < 5:
                # Fallback path uses raw render and slightly higher token budget.
                fallback_used = True
                print(f"[PDF] Page {page_num}: Low row count ({row_count}), retrying with no crop...")
                raw_result = extract_table_from_image(
                    model,
                    processor,
                    artifact['raw_image_path'],
                    prompt=PDF_FALLBACK_PROMPT,
                    max_new_tokens=PDF_FALLBACK_MAX_NEW_TOKENS,
                    max_image_size=PDF_OCR_MAX_IMAGE_SIZE,
                    pdf_mode=True,  # CRITICAL: Disable aggressive cropping for PDF
                    enable_crop=False,  # Disable ALL cropping for retry
                )
                parsed_json = extract_json_from_response(raw_result)

            if not parsed_json:
                print(f"[PDF] Page {page_num} raw model output preview:\n{raw_result[:1200]}", flush=True)
                raise ValueError('Model response did not contain valid JSON table output')

            parsed_json = post_process_extraction(parsed_json)
            is_valid, validation_errors = validate_table_json(parsed_json)
            inference_time = time.time() - page_start_time
            
            # =================================================================
            # STEP 3: INJECT TABLE TITLE INTO PARSED JSON (PER-PAGE)
            # =================================================================
            page_table_name = title_info.get('table_name', 'UNKNOWN')
            page_table_type = title_info.get('table_type')
            
            # SAFETY FALLBACK: If title detection failed, infer from content
            if page_table_name == 'UNKNOWN' or page_table_type is None:
                inferred_type = infer_table_type_from_content(parsed_json)
                if inferred_type:
                    page_table_type = inferred_type
                    print(
                        f"[PAGE {page_num}] Content-based fallback: inferred type={inferred_type}",
                        flush=True,
                    )
            
            if isinstance(parsed_json, dict):
                parsed_json['table_name'] = page_table_name
                parsed_json['table_type'] = page_table_type
            
            # HARDENING #5: Extract confidence/unreliable flags for frontend
            validation_info = parsed_json.get('_validation', {}) if isinstance(parsed_json, dict) else {}
            confidence_info = validation_info.get('confidence', {})
            overall_confidence = confidence_info.get('overall', 1.0)
            is_unreliable = validation_info.get('unreliable', False)
            charts_enabled = validation_info.get('charts_enabled', True)

            results.append({
                'page': page_num,
                'success': True,
                'parsed_json': parsed_json,
                'extraction_meta': parsed_json.get('meta', {}) if isinstance(parsed_json, dict) else {},
                'json_valid': is_valid,
                'validation_errors': validation_errors if validation_errors else [],
                'fallback_used': fallback_used,
                'inference_time_seconds': round(inference_time, 2),
                'table_metrics': artifact['table_metrics'],
                'quality': artifact['quality'],
                'render_dpi': artifact['dpi'],
                # HARDENING #5: Confidence enforcement for frontend
                'confidence': round(overall_confidence, 3),
                'unreliable': is_unreliable,
                'charts_enabled': charts_enabled,
                # TABLE TITLE DETECTION (PER-PAGE)
                'table_name': page_table_name,
                'table_type': page_table_type,
            })

        except Exception as e:
            results.append({
                'page': page_num,
                'success': False,
                'error': f'Extraction failed: {e}',
                'table_metrics': artifact['table_metrics'],
                'quality': artifact['quality'],
                'render_dpi': artifact['dpi'],
                'table_name': 'UNKNOWN',
                'table_type': None,
            })

        finally:
            torch.cuda.empty_cache()
            for path_key in ('raw_image_path', 'ocr_image_path'):
                tmp_path = artifact.get(path_key)
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # NO global table_name/table_type - each page has its own in results[]
    return {
        'page_count': conversion['page_count'],
        'results': results,
    }

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    allocated, reserved = get_vram_usage()

    return jsonify({
        'status': 'ready' if model_loaded else 'loading',
        'model_loaded': model_loaded,
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'vram_allocated_gb': round(allocated, 2),
        'vram_reserved_gb': round(reserved, 2),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/extract', methods=['POST'])
def extract():
    """Extract table data from financial statement image"""

    if not model_loaded:
        return jsonify({'error': 'Model not loaded yet. Please wait.'}), 503

    # Check if file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    # Get custom prompt if provided
    custom_prompt = request.form.get('prompt', None)

    # Save file temporarily with a collision-proof name
    request_id = uuid.uuid4().hex[:12]
    filename = secure_filename(file.filename)
    timestamp_ms = int(time.time() * 1000)
    temp_filename = f"{timestamp_ms}_{request_id}_{filename}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    upload_size = _get_upload_size(file)

    try:
        print(
            f"[IMAGE][{request_id}] incoming filename={filename!r} size_bytes={upload_size} "
            f"temp_file={temp_filename!r}",
            flush=True,
        )
        file.save(temp_path)

        # =================================================================
        # STEP 0: TABLE TITLE DETECTION (BEFORE table extraction)
        # Extract title from top 25% of image header region
        # =================================================================
        title_info = extract_table_title_sync(model, processor, temp_path)
        print(
            f"[IMAGE][{request_id}] title_detection table_name={title_info.get('table_name')!r} "
            f"table_type={title_info.get('table_type')} success={title_info.get('title_detection_success')}",
            flush=True,
        )

        # Extract table data
        start_time = time.time()
        raw_result = extract_table_from_image(
            model,
            processor,
            temp_path,
            prompt=custom_prompt,
            max_new_tokens=IMAGE_MAX_NEW_TOKENS,
            max_image_size=1300,
            enable_crop=False,
        )

        # Try to parse JSON from response
        parsed_json = extract_json_from_response(raw_result)
        is_valid = False
        validation_errors = []

        # Fallback retry for cases where fast mode produces non-JSON output.
        if not parsed_json:
            print(f"[IMAGE] Raw model output preview (pass-1):\n{raw_result[:1200]}", flush=True)
            strict_json_prompt = (
                (custom_prompt or "Extract the financial table from this image.")
                + "\n\nIMPORTANT: Return ONLY valid JSON. No markdown. No explanation. "
                + "Schema: {\"columns\": [...], \"rows\": [...]}"
            )
            raw_result = extract_table_from_image(
                model,
                processor,
                temp_path,
                prompt=strict_json_prompt,
                max_new_tokens=IMAGE_RETRY_MAX_NEW_TOKENS,
                max_image_size=1300,
                enable_crop=False,
            )
            parsed_json = extract_json_from_response(raw_result)

        if not parsed_json:
            print(f"[IMAGE] Raw model output preview (pass-2):\n{raw_result[:1200]}", flush=True)
            chunk_retry_prompt = (
                (custom_prompt or "Extract the financial table from this image.")
                + "\n\nFor long tables, include all rows and keep consistent columns across the full output. "
                + "Return ONLY valid JSON with keys: columns, rows."
            )
            raw_result = extract_table_from_image(
                model,
                processor,
                temp_path,
                prompt=chunk_retry_prompt,
                max_new_tokens=IMAGE_RESCUE_MAX_NEW_TOKENS,
                max_image_size=1400,
                enable_crop=True,
            )
            parsed_json = extract_json_from_response(raw_result)

        inference_time = time.time() - start_time

        if parsed_json:
            parsed_json = post_process_extraction(parsed_json)
            is_valid, validation_errors = validate_table_json(parsed_json)
            
            # =================================================================
            # INJECT TABLE TITLE INTO PARSED JSON
            # =================================================================
            if isinstance(parsed_json, dict):
                parsed_json['table_name'] = title_info.get('table_name', 'UNKNOWN')
                parsed_json['table_type'] = title_info.get('table_type')

        # Get VRAM usage after inference
        allocated, reserved = get_vram_usage()
        
        # HARDENING #5: Extract confidence/unreliable flags for frontend
        validation_info = parsed_json.get('_validation', {}) if isinstance(parsed_json, dict) else {}
        confidence_info = validation_info.get('confidence', {})
        overall_confidence = confidence_info.get('overall', 1.0)
        is_unreliable = validation_info.get('unreliable', False)
        charts_enabled = validation_info.get('charts_enabled', True)

        response_data = {
            'success': True,
            'raw_text': raw_result,
            'inference_time_seconds': round(inference_time, 2),
            'vram_used_gb': round(allocated, 2),
            'filename': filename,
            'request_id': request_id,
            'uploaded_file_size_bytes': upload_size,
            # HARDENING #5: Confidence enforcement for frontend
            'confidence': round(overall_confidence, 3),
            'unreliable': is_unreliable,
            'charts_enabled': charts_enabled,
            # TABLE TITLE DETECTION
            'table_name': title_info.get('table_name', 'UNKNOWN'),
            'table_type': title_info.get('table_type'),
        }

        # Add parsed JSON if available
        if parsed_json:
            response_data['parsed_json'] = parsed_json
            response_data['extraction_meta'] = parsed_json.get('meta', {}) if isinstance(parsed_json, dict) else {}
            response_data['json_valid'] = is_valid
            if validation_errors:
                response_data['validation_errors'] = validation_errors
        else:
            print(f"[IMAGE] Raw model output preview (final failure):\n{raw_result[:1200]}", flush=True)
            response_data['json_valid'] = False
            response_data['validation_errors'] = ['Could not extract valid JSON from response']
            response_data['success'] = False
            response_data['error'] = 'Model could not return valid table JSON for this image'
            # Mark as unreliable when extraction fails
            response_data['unreliable'] = True
            response_data['charts_enabled'] = False

        extracted_rows = (
            response_data.get('parsed_json', {})
            .get('meta', {})
            .get('row_count')
            if isinstance(response_data.get('parsed_json'), dict)
            else None
        )
        print(
            f"[IMAGE][{request_id}] completed success={response_data['success']} "
            f"rows={extracted_rows} inference_time={response_data['inference_time_seconds']}s "
            f"confidence={overall_confidence:.2f} unreliable={is_unreliable}",
            flush=True,
        )

        if response_data['success']:
            return jsonify(response_data)
        return jsonify(response_data), 422

    except Exception as e:
        print(f"[IMAGE][{request_id}] failed error={str(e)}", flush=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'request_id': request_id,
        }), 500

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/extract-batch', methods=['POST'])
def extract_batch():
    """Extract from multiple images"""

    if not model_loaded:
        return jsonify({'error': 'Model not loaded yet'}), 503

    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')

    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400

    results = []
    total_start_time = time.time()

    for file in files:
        if not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'success': False,
                'error': 'Invalid file type'
            })
            continue

        filename = secure_filename(file.filename)
        request_id = uuid.uuid4().hex[:10]
        timestamp_ms = int(time.time() * 1000)
        temp_filename = f"{timestamp_ms}_{request_id}_{filename}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)

        try:
            file.save(temp_path)

            start_time = time.time()
            result = extract_table_from_image(
                model,
                processor,
                temp_path,
                max_new_tokens=IMAGE_MAX_NEW_TOKENS,
                max_image_size=1300,
                enable_crop=False,
            )
            inference_time = time.time() - start_time

            results.append({
                'filename': filename,
                'success': True,
                'result': result,
                'inference_time_seconds': round(inference_time, 2)
            })

        except Exception as e:
            results.append({
                'filename': filename,
                'success': False,
                'error': str(e)
            })

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Clear VRAM between images to prevent accumulation
        torch.cuda.empty_cache()

    total_time = time.time() - total_start_time

    return jsonify({
        'total_processed': len(results),
        'total_time_seconds': round(total_time, 2),
        'results': results
    })

@app.route('/prompts', methods=['GET'])
def get_prompts():
    """Get predefined prompts for different financial statements"""
    return jsonify({
        'balance_sheet': """Extract the balance sheet from this image:
- Assets (Current & Non-Current)
- Liabilities (Current & Long-term)
- Equity
Format as: Account Name | Current Year | Prior Year""",

        'income_statement': """Extract the income statement:
- Revenue
- Operating Expenses
- Net Income
Preserve all line items and amounts exactly.""",

        'cash_flow': """Extract cash flow statement:
- Operating Activities
- Investing Activities
- Financing Activities
Show all inflows and outflows.""",

        'general_table': """Analyze this financial statement image and extract all table data.
For each table found:
1. Identify the table title/header
2. Extract all rows and columns
3. Preserve numerical values exactly as shown
4. Format as structured data"""
    })

# =============================================================================
# PDF ENDPOINTS
# =============================================================================

@app.route('/pdf/info', methods=['POST'])
def pdf_info():
    """
    Get PDF information (page count, thumbnails) with SMART PAGE RECOMMENDATIONS.
    
    Returns recommended pages for extraction based on financial statement detection:
    - balance_sheet: pages containing balance sheet/BILAN
    - income_statement: pages containing income statement/RESULTAT
    - cashflow: pages containing cash flow statement
    """

    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400

    file = request.files['pdf']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'File must be a PDF'}), 400

    temp_path = None
    try:
        # Read PDF into memory
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        page_count = len(doc)
        thumbnails = []

        # Generate thumbnails for each page
        for page_num in range(page_count):
            page = doc[page_num]

            # Render at low resolution for thumbnail (72 DPI)
            mat = fitz.Matrix(0.3, 0.3)  # 30% scale
            pix = page.get_pixmap(matrix=mat)

            # Convert to base64
            img_bytes = pix.tobytes("png")
            b64_img = base64.b64encode(img_bytes).decode('utf-8')

            thumbnails.append({
                'page': page_num + 1,
                'width': pix.width,
                'height': pix.height,
                'thumbnail': f"data:image/png;base64,{b64_img}"
            })

        doc.close()
        
        # =================================================================
        # SMART PAGE RECOMMENDATIONS
        # Detect financial tables and recommend pages for extraction
        # =================================================================
        recommended_pages = {
            "balance_sheet": [],
            "income_statement": [],
            "cashflow": [],
            "all_recommended": [],
        }
        
        try:
            # Save PDF to temp file for detector (needs file path)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                temp_path = tmp.name
                tmp.write(pdf_bytes)
            
            # Run financial statement detection
            detection_result = detect_financial_statements(temp_path, verbose=False)
            
            recommended_pages["balance_sheet"] = detection_result.balance_sheet_pages or []
            recommended_pages["income_statement"] = detection_result.income_statement_pages or []
            recommended_pages["cashflow"] = detection_result.cashflow_pages or []
            
            # Combine all recommended pages (unique, sorted)
            all_pages = set()
            all_pages.update(recommended_pages["balance_sheet"])
            all_pages.update(recommended_pages["income_statement"])
            all_pages.update(recommended_pages["cashflow"])
            recommended_pages["all_recommended"] = sorted(all_pages)
            
            print(
                f"[PDF INFO] Page recommendations: "
                f"balance_sheet={recommended_pages['balance_sheet']} "
                f"income_statement={recommended_pages['income_statement']} "
                f"cashflow={recommended_pages['cashflow']}",
                flush=True
            )
            
        except Exception as detect_error:
            print(f"[PDF INFO] Page detection failed (non-critical): {detect_error}", flush=True)
            # Detection failure is non-critical - return empty recommendations

        return jsonify({
            'success': True,
            'filename': file.filename,
            'page_count': page_count,
            'thumbnails': thumbnails,
            # PAGE RECOMMENDATIONS
            'recommended_pages': recommended_pages,
            'has_recommendations': len(recommended_pages["all_recommended"]) > 0,
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/pdf/extract', methods=['POST'])
def pdf_extract():
    """Extract tables from selected PDF pages"""

    if not model_loaded:
        return jsonify({'error': 'Model not loaded yet. Please wait.'}), 503

    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400

    file = request.files['pdf']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Get selected pages (comma-separated or JSON array)
    pages_str = request.form.get('pages', '')

    if not pages_str:
        return jsonify({'error': 'No pages selected'}), 400

    try:
        selected_pages = _parse_selected_pages(pages_str)
        pdf_bytes = file.read()

        total_start_time = time.time()
        torch.cuda.empty_cache()

        print(f"\n[PDF] Starting extraction of {len(selected_pages)} selected page(s)...", flush=True)

        processed = process_pages(pdf_bytes, selected_pages)
        results = processed['results']

        total_time = time.time() - total_start_time
        allocated, reserved = get_vram_usage()

        print(f"[PDF] All {len(results)} pages processed in {total_time:.1f}s total", flush=True)

        return jsonify({
            'success': True,
            'filename': file.filename,
            'page_count': processed['page_count'],
            'pages_processed': len(results),
            'total_time_seconds': round(total_time, 2),
            'vram_used_gb': round(allocated, 2),
            'pages': results
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/pdf/page-image/<int:page_num>', methods=['POST'])
def get_page_image(page_num):
    """Get a specific page as full-resolution image"""

    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400

    file = request.files['pdf']

    try:
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if page_num < 1 or page_num > len(doc):
            doc.close()
            return jsonify({'error': f'Invalid page number. PDF has {len(doc)} pages.'}), 400

        page = doc[page_num - 1]

        # Render at 150 DPI for preview (balance between quality and size)
        mat = fitz.Matrix(150/72, 150/72)
        pix = page.get_pixmap(matrix=mat)

        img_bytes = pix.tobytes("png")
        doc.close()

        return send_file(
            io.BytesIO(img_bytes),
            mimetype='image/png',
            as_attachment=False
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# UNIFIED PDF EXTRACTION (merged output)
# =============================================================================

@app.route('/pdf/extract-unified', methods=['POST'])
def pdf_extract_unified():
    """
    Extract tables from PDF and return UNIFIED merged result.
    
    This endpoint:
    1. Converts selected pages to images
    2. Processes each through existing image pipeline
    3. Merges all results into single JSON
    4. Removes duplicates and preserves order
    
    Request:
        - pdf: PDF file
        - pages: Page selection (optional, default=all)
            - "1,3,5" or "[1,3,5]" or "1-5" or None
        - deduplicate: Remove duplicate rows (default=true)
    
    Response:
        {
            "success": true,
            "columns": [...],
            "rows": [...],
            "pages_processed": N,
            "failed_pages": [...],
            "total_rows": N
        }
    """
    if not model_loaded:
        return jsonify({'error': 'Model not loaded yet. Please wait.'}), 503

    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400

    file = request.files['pdf']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Get parameters
    pages_str = request.form.get('pages', '')
    deduplicate = request.form.get('deduplicate', 'true').lower() != 'false'

    try:
        pdf_bytes = file.read()
        
        # Get PDF info
        pdf_info = get_pdf_info_from_bytes(pdf_bytes, file.filename)
        total_pages = pdf_info['page_count']
        
        # Parse page selection (empty = all pages)
        if pages_str.strip():
            selected_pages = parse_page_selection(pages_str, total_pages)
        else:
            selected_pages = list(range(1, total_pages + 1))
        
        print(f"\n[PDF Unified] Processing {len(selected_pages)} pages from '{file.filename}'...")
        total_start_time = time.time()
        torch.cuda.empty_cache()

        # Step 1: Convert pages to images (in-memory)
        page_images = extract_pdf_pages_from_bytes(pdf_bytes, selected_pages)
        
        # Step 2: Process each page through EXISTING image pipeline
        page_results = []
        
        for idx, page_info in enumerate(page_images):
            page_num = page_info['page_number']
            pil_image = page_info['image']
            
            print(f"[PDF Unified] Processing page {page_num} ({idx+1}/{len(page_images)})...", flush=True)
            page_start = time.time()
            
            # Save PIL image to temp file for existing pipeline
            tmp = tempfile.NamedTemporaryFile(
                suffix=f'_unified_p{page_num}.png',
                dir=UPLOAD_FOLDER,
                delete=False
            )
            tmp_path = tmp.name
            tmp.close()
            
            try:
                pil_image.save(tmp_path, format='PNG', optimize=True)
                
                # Use EXISTING extraction function (no duplication)
                # pdf_mode=True disables aggressive table cropping
                raw_result = extract_table_from_image(
                    model,
                    processor,
                    tmp_path,
                    prompt=PDF_EXTRACTION_PROMPT,
                    max_new_tokens=PDF_MAX_NEW_TOKENS,
                    max_image_size=PDF_OCR_MAX_IMAGE_SIZE,
                    pdf_mode=True,  # CRITICAL: Prevent destructive cropping
                )
                
                # Parse JSON
                parsed_json = extract_json_from_response(raw_result)
                
                # =============================================================
                # FALLBACK RETRY: If row_count < 5, retry with full image
                # =============================================================
                row_count = len(parsed_json.get('rows', [])) if parsed_json else 0
                
                if not parsed_json or row_count < 5:
                    print(f"[PDF Unified] Page {page_num}: Low row count ({row_count}), retrying with fallback...")
                    raw_result = extract_table_from_image(
                        model,
                        processor,
                        tmp_path,
                        prompt=PDF_FALLBACK_PROMPT,
                        max_new_tokens=PDF_FALLBACK_MAX_NEW_TOKENS,
                        max_image_size=PDF_OCR_MAX_IMAGE_SIZE,
                        pdf_mode=True,  # CRITICAL: Prevent destructive cropping
                        enable_crop=False,  # Disable ALL cropping for retry
                    )
                    parsed_json = extract_json_from_response(raw_result)
                
                if parsed_json:
                    parsed_json = post_process_extraction(parsed_json)
                
                inference_time = time.time() - page_start
                
                page_results.append({
                    'page_number': page_num,
                    'success': parsed_json is not None,
                    'parsed_json': parsed_json,
                    'inference_time': round(inference_time, 2),
                    'quality': page_info['quality'],
                    'error': None if parsed_json else 'Failed to parse JSON'
                })
                
                if parsed_json:
                    row_count = len(parsed_json.get('rows', []))
                    print(f"[PDF Unified] Page {page_num}: {row_count} rows in {inference_time:.1f}s")
                else:
                    print(f"[PDF Unified] Page {page_num}: Failed to parse")
                    
            except Exception as e:
                page_results.append({
                    'page_number': page_num,
                    'success': False,
                    'parsed_json': None,
                    'error': str(e),
                    'quality': page_info.get('quality')
                })
                print(f"[PDF Unified] Page {page_num}: Error - {e}")
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                torch.cuda.empty_cache()
        
        # Step 3: Merge all results
        merged = merge_page_results(page_results, deduplicate=deduplicate)
        
        total_time = time.time() - total_start_time
        allocated, reserved = get_vram_usage()
        
        # Build response
        response = {
            'success': True,
            'filename': file.filename,
            'columns': merged['columns'],
            'rows': merged['rows'],
            'pages_processed': merged['pages_processed'],
            'total_pages_in_pdf': total_pages,
            'failed_pages': merged['failed_pages'],
            'total_rows': len(merged['rows']),
            'deduplicated': deduplicate,
            '_coverage_complete': merged['_coverage_complete'],
            'total_time_seconds': round(total_time, 2),
            'vram_used_gb': round(allocated, 2)
        }
        
        print(f"[PDF Unified] Complete: {merged['pages_processed']} pages, {len(merged['rows'])} rows in {total_time:.1f}s")
        
        return jsonify(response)

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# STARTUP
# =============================================================================

def load_model_on_startup():
    """Load model when Flask starts"""
    global model, processor, model_loaded

    print("\n" + "=" * 60)
    print("LOADING QWEN-VL MODEL")
    print("=" * 60)
    print("This may take 30-60 seconds on first run...")

    try:
        model, processor = load_model()
        model_loaded = True
        print("\n" + "=" * 60)
        print("MODEL LOADED SUCCESSFULLY!")
        print("API is ready to accept requests")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        print("API will not be able to process requests.\n")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Load model before starting server
    load_model_on_startup()

    print("\n" + "=" * 60)
    print("STARTING FLASK SERVER")
    print("=" * 60)
    print("API Endpoints:")
    print("  GET  /health         - Check API status")
    print("  POST /extract        - Extract from single image")
    print("  POST /extract-batch  - Extract from multiple images")
    print("  GET  /prompts        - Get predefined prompts")
    print("  POST /pdf/info       - Get PDF page count & thumbnails")
    print("  POST /pdf/extract    - Extract from selected PDF pages (per-page results)")
    print("  POST /pdf/extract-unified - Extract & merge into unified JSON")
    print("\nExample usage:")
    print('  curl -X POST -F "image=@statement.png" http://localhost:5000/extract')
    print('  curl -X POST -F "pdf=@report.pdf" -F "pages=1,2,3" http://localhost:5000/pdf/extract')
    print('  curl -X POST -F "pdf=@report.pdf" http://localhost:5000/pdf/extract-unified')
    print("=" * 60 + "\n")

    # Run Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=False,     # Don't use debug mode with GPU models
        threaded=True    # Handle multiple requests
    )

"""
PDF Handler Module - Convert PDF pages to images for table extraction.

This module provides PDF → Image conversion that integrates seamlessly with
the existing image extraction pipeline. It does NOT implement new extraction
logic - it only transforms input format.

Key Features:
- Page selection (specific pages or all)
- Auto-quality rendering (DPI optimization)
- Multi-page result merging
- Duplicate row detection
- Order preservation (page + vertical position)
"""

import os
import io
import tempfile
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
import numpy as np

# PDF library - using PyMuPDF (already installed)
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[WARNING] PyMuPDF not installed. PDF support disabled. Install with: pip install PyMuPDF")


# =============================================================================
# CONFIGURATION
# =============================================================================

PDF_DPI_LEVELS = [200, 250, 300]       # Auto-quality DPI levels
PDF_MAX_IMAGE_WIDTH = 1500              # Max width for OCR (pixels)
PDF_MAX_FILE_SIZE_KB = 200              # Max size before compression
PDF_THUMBNAIL_SCALE = 0.25              # Thumbnail scale factor


# =============================================================================
# PDF INFO & VALIDATION
# =============================================================================

def is_pdf_supported() -> bool:
    """Check if PDF processing is available."""
    return PDF_SUPPORT


def get_pdf_info(pdf_path: str) -> Dict:
    """
    Get PDF metadata without processing.
    
    Returns:
        dict with page_count, filename, file_size_bytes
    """
    if not PDF_SUPPORT:
        raise RuntimeError("PDF support not available. Install PyMuPDF.")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    try:
        return {
            'filename': os.path.basename(pdf_path),
            'page_count': len(doc),
            'file_size_bytes': os.path.getsize(pdf_path)
        }
    finally:
        doc.close()


def get_pdf_info_from_bytes(pdf_bytes: bytes, filename: str = "document.pdf") -> Dict:
    """Get PDF metadata from bytes."""
    if not PDF_SUPPORT:
        raise RuntimeError("PDF support not available. Install PyMuPDF.")
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return {
            'filename': filename,
            'page_count': len(doc),
            'file_size_bytes': len(pdf_bytes)
        }
    finally:
        doc.close()


# =============================================================================
# PAGE SELECTION
# =============================================================================

def parse_page_selection(pages_input, total_pages: int) -> List[int]:
    """
    Parse page selection input into list of valid page numbers.
    
    Args:
        pages_input: Can be:
            - None or empty string → all pages
            - List of ints: [1, 3, 5]
            - Comma-separated string: "1,3,5"
            - JSON array string: "[1, 3, 5]"
            - Range string: "1-5" or "2-4,6,8-10"
        total_pages: Total pages in PDF
        
    Returns:
        Sorted list of valid page numbers (1-indexed)
    """
    import json
    
    # None or empty → all pages
    if pages_input is None or (isinstance(pages_input, str) and not pages_input.strip()):
        return list(range(1, total_pages + 1))
    
    # Already a list
    if isinstance(pages_input, list):
        pages = [int(p) for p in pages_input]
    
    # String input
    elif isinstance(pages_input, str):
        pages_str = pages_input.strip()
        
        # JSON array
        if pages_str.startswith('['):
            try:
                parsed = json.loads(pages_str)
                pages = [int(p) for p in parsed]
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Invalid JSON page array: {e}")
        
        # Comma/range separated
        else:
            pages = []
            for part in pages_str.split(','):
                part = part.strip()
                if '-' in part:
                    # Range: "2-5"
                    try:
                        start, end = part.split('-', 1)
                        start, end = int(start.strip()), int(end.strip())
                        pages.extend(range(start, end + 1))
                    except ValueError:
                        raise ValueError(f"Invalid page range: {part}")
                else:
                    try:
                        pages.append(int(part))
                    except ValueError:
                        raise ValueError(f"Invalid page number: {part}")
    else:
        raise ValueError(f"Invalid pages input type: {type(pages_input)}")
    
    # Validate and deduplicate
    valid_pages = sorted(set(p for p in pages if 1 <= p <= total_pages))
    
    if not valid_pages:
        raise ValueError(f"No valid pages in selection. PDF has {total_pages} pages.")
    
    invalid = [p for p in pages if p < 1 or p > total_pages]
    if invalid:
        print(f"[PDF] Warning: Skipping invalid page numbers: {invalid}")
    
    return valid_pages


# =============================================================================
# PDF → IMAGE CONVERSION
# =============================================================================

def _render_page_to_pil(page, dpi: int) -> Image.Image:
    """Render a single PDF page to PIL Image at specified DPI."""
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    del pixmap
    return image


def _calculate_image_quality(image: Image.Image) -> Dict:
    """Calculate quality metrics for rendered image."""
    width, height = image.size
    gray = np.asarray(image.convert('L'))
    
    # Contrast (standard deviation of pixel values)
    contrast_std = float(gray.std())
    
    # Sharpness (Laplacian variance approximation)
    arr = gray.astype(np.float32)
    if arr.shape[0] > 2 and arr.shape[1] > 2:
        lap = (
            -4.0 * arr[1:-1, 1:-1]
            + arr[:-2, 1:-1]
            + arr[2:, 1:-1]
            + arr[1:-1, :-2]
            + arr[1:-1, 2:]
        )
        sharpness = float(np.var(lap)) if lap.size else 0.0
    else:
        sharpness = 0.0
    
    return {
        'width': width,
        'height': height,
        'contrast_std': round(contrast_std, 2),
        'sharpness_score': round(sharpness, 2),
        'is_low_quality': contrast_std < 35 or sharpness < 50
    }


def _preprocess_for_ocr(image: Image.Image, max_width: int = PDF_MAX_IMAGE_WIDTH) -> Image.Image:
    """
    Preprocess rendered PDF page for OCR.
    
    - Convert to RGB
    - Resize if too large
    - Normalize contrast
    """
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if too large (maintain aspect ratio)
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        print(f"[PDF] Resized: {image.width}x{image.height} (ratio={ratio:.2f})")
    
    return image


def render_pdf_page(
    doc,
    page_num: int,
    auto_quality: bool = True
) -> Tuple[Image.Image, Dict]:
    """
    Render a single PDF page to PIL Image with auto-quality optimization.
    
    Args:
        doc: Open PyMuPDF document
        page_num: Page number (1-indexed)
        auto_quality: If True, try multiple DPI levels
        
    Returns:
        Tuple of (PIL Image, quality metrics dict)
    """
    page = doc.load_page(page_num - 1)  # 0-indexed
    
    best_image = None
    best_quality = None
    best_dpi = PDF_DPI_LEVELS[0]
    
    dpi_levels = PDF_DPI_LEVELS if auto_quality else [PDF_DPI_LEVELS[0]]
    
    for dpi in dpi_levels:
        image = _render_page_to_pil(page, dpi)
        quality = _calculate_image_quality(image)
        
        best_image = image
        best_quality = quality
        best_dpi = dpi
        
        # Stop if quality is acceptable
        if not quality['is_low_quality']:
            break
    
    # Preprocess for OCR
    processed = _preprocess_for_ocr(best_image, PDF_MAX_IMAGE_WIDTH)
    
    best_quality['render_dpi'] = best_dpi
    best_quality['final_width'] = processed.width
    best_quality['final_height'] = processed.height
    
    return processed, best_quality


def extract_pdf_pages(
    pdf_path: str,
    selected_pages: Optional[List[int]] = None,
    output_dir: Optional[str] = None
) -> List[Dict]:
    """
    Extract selected pages from PDF as images.
    
    Args:
        pdf_path: Path to PDF file
        selected_pages: List of page numbers (1-indexed) or None for all
        output_dir: Directory to save images (optional, uses temp if None)
        
    Returns:
        List of dicts with page info and image paths
    """
    if not PDF_SUPPORT:
        raise RuntimeError("PDF support not available.")
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Parse page selection
    if selected_pages is None:
        pages_to_extract = list(range(1, total_pages + 1))
    else:
        pages_to_extract = parse_page_selection(selected_pages, total_pages)
    
    results = []
    
    try:
        for page_num in pages_to_extract:
            print(f"[PDF] Rendering page {page_num}/{total_pages}...", flush=True)
            
            image, quality = render_pdf_page(doc, page_num)
            
            # Save to file if output_dir provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                image_path = os.path.join(output_dir, f"page_{page_num:03d}.png")
            else:
                # Use temp file
                tmp = tempfile.NamedTemporaryFile(
                    suffix=f'_page{page_num}.png',
                    delete=False
                )
                image_path = tmp.name
                tmp.close()
            
            image.save(image_path, format='PNG', optimize=True)
            
            results.append({
                'page_number': page_num,
                'image_path': image_path,
                'image': image,  # Keep PIL image for direct use
                'quality': quality,
                'width': image.width,
                'height': image.height
            })
    finally:
        doc.close()
    
    return results


def extract_pdf_pages_from_bytes(
    pdf_bytes: bytes,
    selected_pages: Optional[List[int]] = None
) -> List[Dict]:
    """
    Extract selected pages from PDF bytes as images (in-memory).
    
    Returns:
        List of dicts with page info and PIL images (no file paths)
    """
    if not PDF_SUPPORT:
        raise RuntimeError("PDF support not available.")
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    
    pages_to_extract = parse_page_selection(selected_pages, total_pages)
    
    results = []
    
    try:
        for page_num in pages_to_extract:
            print(f"[PDF] Rendering page {page_num}/{total_pages}...", flush=True)
            
            image, quality = render_pdf_page(doc, page_num)
            
            results.append({
                'page_number': page_num,
                'image': image,
                'quality': quality,
                'width': image.width,
                'height': image.height
            })
    finally:
        doc.close()
    
    return results


# =============================================================================
# MULTI-PAGE RESULT MERGING
# =============================================================================

def _compute_row_hash(row: Dict, label_column: str = 'Label') -> str:
    """
    Compute hash for a row to detect duplicates.
    Uses label + all numeric values.
    """
    label = str(row.get(label_column, row.get('label', ''))).strip().lower()
    
    # Collect all numeric-looking values
    values = []
    for key, val in row.items():
        if key.lower() in ('type', 'label', 'note'):
            continue
        val_str = str(val).strip()
        # Keep if it looks numeric
        if any(c.isdigit() for c in val_str):
            values.append(val_str)
    
    combined = label + '|' + '|'.join(sorted(values))
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def merge_page_results(
    page_results: List[Dict],
    deduplicate: bool = True
) -> Dict:
    """
    Merge extraction results from multiple PDF pages into unified output.
    
    Args:
        page_results: List of per-page extraction results
        deduplicate: Remove duplicate rows across pages
        
    Returns:
        Unified JSON with columns, rows, metadata
    """
    if not page_results:
        return {
            'columns': [],
            'rows': [],
            'pages_processed': 0,
            'failed_pages': [],
            '_coverage_complete': True
        }
    
    # Collect results
    all_columns = set()
    merged_rows = []
    failed_pages = []
    seen_hashes = set()
    label_column = 'Label'
    
    for pr in page_results:
        page_num = pr.get('page_number', pr.get('page', 0))
        
        # Handle failures
        if not pr.get('success', True):
            failed_pages.append({
                'page': page_num,
                'error': pr.get('error', 'Unknown error')
            })
            continue
        
        # Get parsed JSON
        parsed = pr.get('parsed_json', pr.get('data', {}))
        if not parsed:
            continue
        
        # Collect columns
        columns = parsed.get('columns', [])
        for col in columns:
            all_columns.add(col)
        
        # Detect label column
        for col in columns:
            if col.lower() in ('label', 'libellé', 'désignation', 'description'):
                label_column = col
                break
        
        # Process rows
        rows = parsed.get('rows', [])
        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            
            # Add page metadata
            row['_page'] = page_num
            row['_row_index'] = row_idx
            
            # Duplicate detection
            if deduplicate:
                row_hash = _compute_row_hash(row, label_column)
                if row_hash in seen_hashes:
                    continue
                seen_hashes.add(row_hash)
            
            merged_rows.append(row)
    
    # Sort by page, then row index (preserves order)
    merged_rows.sort(key=lambda r: (r.get('_page', 0), r.get('_row_index', 0)))
    
    # Clean up internal metadata
    for row in merged_rows:
        row.pop('_page', None)
        row.pop('_row_index', None)
    
    # Determine column order (preserve from first page, add new ones at end)
    final_columns = []
    for pr in page_results:
        parsed = pr.get('parsed_json', pr.get('data', {}))
        if parsed and parsed.get('columns'):
            for col in parsed['columns']:
                if col not in final_columns:
                    final_columns.append(col)
    
    # Add any columns we might have missed
    for col in sorted(all_columns):
        if col not in final_columns:
            final_columns.append(col)
    
    return {
        'columns': final_columns,
        'rows': merged_rows,
        'pages_processed': len(page_results) - len(failed_pages),
        'total_pages_attempted': len(page_results),
        'failed_pages': failed_pages,
        '_coverage_complete': len(failed_pages) == 0
    }


# =============================================================================
# PIPELINE INTEGRATION (uses existing extract_table_from_image)
# =============================================================================

def process_pdf_with_pipeline(
    pdf_path: str,
    extract_fn,  # The existing extraction function
    model,
    processor,
    selected_pages: Optional[List[int]] = None,
    **extract_kwargs
) -> Dict:
    """
    Process PDF through the existing image extraction pipeline.
    
    This function:
    1. Converts PDF pages to images
    2. Sends each image through extract_fn (existing pipeline)
    3. Merges results
    
    Args:
        pdf_path: Path to PDF file
        extract_fn: The existing extract_table_from_image function
        model: Loaded model
        processor: Model processor
        selected_pages: Pages to process (None = all)
        **extract_kwargs: Additional args for extract_fn
        
    Returns:
        Unified extraction result
    """
    import time
    
    # Step 1: Convert PDF to images
    print(f"\n[PDF Pipeline] Converting PDF to images...")
    page_images = extract_pdf_pages(pdf_path, selected_pages)
    print(f"[PDF Pipeline] Extracted {len(page_images)} page images")
    
    # Step 2: Process each page through existing pipeline
    page_results = []
    
    for idx, page_info in enumerate(page_images):
        page_num = page_info['page_number']
        image_path = page_info['image_path']
        
        print(f"\n[PDF Pipeline] Processing page {page_num} ({idx+1}/{len(page_images)})...")
        start_time = time.time()
        
        try:
            # Use existing extraction function
            raw_result = extract_fn(
                model,
                processor,
                image_path,
                **extract_kwargs
            )
            
            # Parse JSON from response (import from json_table_utils)
            from json_table_utils import extract_json_from_response, post_process_extraction
            
            parsed = extract_json_from_response(raw_result)
            if parsed:
                parsed = post_process_extraction(parsed)
            
            inference_time = time.time() - start_time
            
            page_results.append({
                'page_number': page_num,
                'success': parsed is not None,
                'parsed_json': parsed,
                'raw_text': raw_result[:500] if not parsed else None,
                'inference_time': round(inference_time, 2),
                'quality': page_info['quality']
            })
            
            if parsed:
                row_count = len(parsed.get('rows', []))
                print(f"[PDF Pipeline] Page {page_num}: Extracted {row_count} rows in {inference_time:.1f}s")
            else:
                print(f"[PDF Pipeline] Page {page_num}: Failed to parse JSON")
                
        except Exception as e:
            page_results.append({
                'page_number': page_num,
                'success': False,
                'error': str(e),
                'quality': page_info['quality']
            })
            print(f"[PDF Pipeline] Page {page_num}: Error - {e}")
        
        finally:
            # Clean up temp image
            if os.path.exists(image_path):
                os.remove(image_path)
    
    # Step 3: Merge results
    print(f"\n[PDF Pipeline] Merging results from {len(page_results)} pages...")
    merged = merge_page_results(page_results)
    
    # Add metadata
    merged['source_file'] = os.path.basename(pdf_path)
    merged['total_rows'] = len(merged['rows'])
    
    successful = sum(1 for r in page_results if r.get('success'))
    print(f"[PDF Pipeline] Complete: {successful}/{len(page_results)} pages successful, {merged['total_rows']} total rows")
    
    return merged


# =============================================================================
# THUMBNAIL GENERATION
# =============================================================================

def generate_pdf_thumbnails(
    pdf_bytes: bytes,
    scale: float = PDF_THUMBNAIL_SCALE
) -> List[Dict]:
    """
    Generate thumbnails for all PDF pages.
    
    Returns:
        List of dicts with page number, dimensions, and base64 image
    """
    import base64
    
    if not PDF_SUPPORT:
        raise RuntimeError("PDF support not available.")
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    thumbnails = []
    
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            img_bytes = pix.tobytes("png")
            b64_img = base64.b64encode(img_bytes).decode('utf-8')
            
            thumbnails.append({
                'page': page_num + 1,
                'width': pix.width,
                'height': pix.height,
                'thumbnail': f"data:image/png;base64,{b64_img}"
            })
            
            del pix
    finally:
        doc.close()
    
    return thumbnails


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cleanup_temp_images(page_results: List[Dict]):
    """Clean up any temporary image files from page results."""
    for pr in page_results:
        image_path = pr.get('image_path')
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass


def estimate_processing_time(page_count: int, avg_seconds_per_page: float = 15.0) -> Dict:
    """Estimate total processing time for PDF."""
    total_seconds = page_count * avg_seconds_per_page
    
    return {
        'page_count': page_count,
        'avg_seconds_per_page': avg_seconds_per_page,
        'estimated_total_seconds': total_seconds,
        'estimated_minutes': round(total_seconds / 60, 1)
    }

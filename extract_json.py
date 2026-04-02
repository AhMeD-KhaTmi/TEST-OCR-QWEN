"""
Enhanced Financial Statement Extractor with JSON Output
Uses the production-tested OCR prompt and automatically parses JSON
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

from run_qwen_vl import load_model, extract_table_from_image, get_vram_usage
from json_table_utils import (
    extract_json_from_response,
    validate_table_json,
    convert_to_csv,
    convert_to_excel,
    print_table_summary,
    pretty_print_table,
    post_process_extraction  # NEW: Post-processing pipeline
)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "extracted_tables"
SAVE_RAW_RESPONSE = True  # Save raw model output for debugging
SAVE_JSON = True
SAVE_CSV = True
SAVE_EXCEL = True  # Requires pandas and openpyxl

# =============================================================================
# ENHANCED EXTRACTION FUNCTION
# =============================================================================

def extract_and_parse_table(
    model,
    processor,
    image_path: str,
    output_dir: str = OUTPUT_DIR,
    save_formats: dict = None
):
    """
    Extract table from image and automatically parse JSON output

    Args:
        model: Loaded Qwen-VL model
        processor: Qwen-VL processor
        image_path: Path to financial statement image
        output_dir: Directory to save outputs
        save_formats: Dict of format flags (json, csv, excel, raw)

    Returns:
        Dict with extraction results and parsed data
    """

    if save_formats is None:
        save_formats = {
            'json': SAVE_JSON,
            'csv': SAVE_CSV,
            'excel': SAVE_EXCEL,
            'raw': SAVE_RAW_RESPONSE
        }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename
    base_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 70)
    print(f"EXTRACTING: {Path(image_path).name}")
    print("=" * 70)

    # Extract using the default JSON prompt
    raw_response = extract_table_from_image(model, processor, image_path)

    result = {
        'success': False,
        'image': image_path,
        'timestamp': datetime.now().isoformat(),
        'raw_response': raw_response,
        'parsed_json': None,
        'validation_errors': [],
        'saved_files': []
    }

    # Save raw response if requested
    if save_formats.get('raw', False):
        raw_path = Path(output_dir) / f"{base_name}_{timestamp}_raw.txt"
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(raw_response)
        result['saved_files'].append(str(raw_path))
        print(f"\n[OK] Raw response saved: {raw_path.name}")

    # Try to extract JSON
    print("\n[*] Parsing JSON from response...")
    parsed_json = extract_json_from_response(raw_response)

    if parsed_json is None:
        print("[ERROR] Could not extract valid JSON from response")
        print("\nRaw response preview:")
        print("-" * 70)
        print(raw_response[:500])
        print("-" * 70)
        result['validation_errors'].append("Failed to extract JSON from response")
        return result

    print("[OK] JSON extracted successfully")

    # POST-PROCESS: Fix common issues (column shift, headers, types)
    print("[*] Applying post-processing fixes...")
    parsed_json = post_process_extraction(parsed_json)
    print("[OK] Post-processing complete")

    # Validate structure
    print("[*] Validating table structure...")
    is_valid, errors = validate_table_json(parsed_json)

    result['parsed_json'] = parsed_json
    result['validation_errors'] = errors

    if not is_valid:
        print("[WARN] Validation warnings:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("[OK] Table structure is valid")

    # Print summary
    print_table_summary(parsed_json)

    # Print preview
    print("\n[*] Table Preview (first 10 rows):")
    pretty_print_table(parsed_json, max_rows=10)

    # Save JSON
    if save_formats.get('json', False):
        json_path = Path(output_dir) / f"{base_name}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        result['saved_files'].append(str(json_path))
        print(f"[OK] JSON saved: {json_path.name}")

    # Save CSV
    if save_formats.get('csv', False):
        csv_path = Path(output_dir) / f"{base_name}_{timestamp}.csv"
        if convert_to_csv(parsed_json, str(csv_path)):
            result['saved_files'].append(str(csv_path))
            print(f"[OK] CSV saved: {csv_path.name}")
        else:
            print("[ERROR] Failed to save CSV")

    # Save Excel
    if save_formats.get('excel', False):
        excel_path = Path(output_dir) / f"{base_name}_{timestamp}.xlsx"
        if convert_to_excel(parsed_json, str(excel_path)):
            result['saved_files'].append(str(excel_path))
            print(f"[OK] Excel saved: {excel_path.name}")
        else:
            print("[WARN] Excel save failed (may need: pip install pandas openpyxl)")

    # Get VRAM usage
    allocated, reserved = get_vram_usage()
    result['vram_gb'] = round(allocated, 2)

    result['success'] = True

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Files saved to: {output_dir}/")
    print(f"VRAM used: {allocated:.2f} GB")
    print("=" * 70 + "\n")

    return result

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""

    print("\n" + "=" * 70)
    print("ENHANCED FINANCIAL STATEMENT EXTRACTOR")
    print("Extracts tables as JSON with validation and multi-format export")
    print("=" * 70)

    # Check command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image
        image_path = "test_image.png"

        if not os.path.exists(image_path):
            print("\n[WARN] No image specified!")
            print("\nUsage:")
            print(f"  python {Path(__file__).name} <image_path>")
            print("\nExample:")
            print(f"  python {Path(__file__).name} financial_statement.png")
            print("\nOr edit this script and set image_path variable.")

            # Interactive mode
            while True:
                user_input = input("\nEnter image path (or 'quit'): ").strip()

                if user_input.lower() == 'quit':
                    return

                if os.path.exists(user_input):
                    image_path = user_input
                    break
                else:
                    print(f"[ERROR] File not found: {user_input}")

    # Verify image exists
    if not os.path.exists(image_path):
        print(f"\n[ERROR] Image not found: {image_path}")
        return

    # Load model
    print("\n[*] Loading model...")
    model, processor = load_model()

    # Extract and parse
    result = extract_and_parse_table(model, processor, image_path)

    if result['success']:
        print("\n[OK] SUCCESS!")
        # Handle both old ('table') and new ('rows') formats
        rows = result['parsed_json'].get('rows', result['parsed_json'].get('table', []))
        print(f"Extracted {len(rows)} rows")
        print(f"Saved {len(result['saved_files'])} file(s)")

        if result['validation_errors']:
            print(f"\n[WARN] {len(result['validation_errors'])} validation warning(s)")

    else:
        print("\n[ERROR] EXTRACTION FAILED")
        if result['validation_errors']:
            print("\nErrors:")
            for error in result['validation_errors']:
                print(f"  - {error}")

    # Save execution log
    log_path = Path(OUTPUT_DIR) / f"extraction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        # Remove raw response from log to save space
        log_result = result.copy()
        if 'raw_response' in log_result:
            log_result['raw_response'] = f"<truncated {len(result['raw_response'])} chars>"

        json.dump(log_result, f, indent=2, ensure_ascii=False)

    print(f"\n[*] Execution log saved: {log_path}")

if __name__ == "__main__":
    main()

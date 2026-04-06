"""
Test extraction pipeline with images in the project folder.

USAGE:
    cd "c:\\Users\\THOURAYA\\test qwen"
    .\\venv\\Scripts\\activate
    python test_images.py

Or test a single image:
    python test_images.py attijari_statement.png
"""
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_qwen_vl import load_model, extract_table_from_image, extract_table_title_sync
from json_table_utils import extract_json_from_response, post_process_extraction

def test_image(image_path, model, processor):
    """Test extraction on a single image."""
    print(f"\n{'='*70}")
    print(f"TESTING: {os.path.basename(image_path)}")
    print('='*70)
    
    try:
        # Extract table title first
        print("\n[1] Extracting table title...")
        title_info = extract_table_title_sync(model, processor, image_path)
        print(f"    Title: {title_info.get('table_name', 'UNKNOWN')}")
        print(f"    Type: {title_info.get('table_type', 'unknown')}")
        
        # Extract table data
        print("\n[2] Extracting table data...")
        raw_result = extract_table_from_image(
            model, processor, image_path,
            enable_crop=True,
            pdf_mode=False,
            add_grid=False
        )
        
        # Parse JSON from response
        print("\n[3] Parsing JSON...")
        parsed = extract_json_from_response(raw_result)
        
        if not parsed:
            print("    ERROR: Could not parse JSON from response")
            return None
        
        # Add title info
        parsed['table_name'] = title_info.get('table_name', 'UNKNOWN')
        parsed['table_type'] = title_info.get('table_type', 'unknown')
        
        # Post-process (alignment engine + validation)
        print("\n[4] Running alignment engine + validation...")
        result = post_process_extraction(parsed)
        
        # Print summary
        print("\n" + "-"*50)
        print("RESULTS SUMMARY:")
        print("-"*50)
        print(f"  Table Name: {result.get('table_name', 'UNKNOWN')}")
        print(f"  Table Type: {result.get('table_type', 'unknown')}")
        print(f"  Columns: {result.get('columns', [])}")
        print(f"  Row Count: {len(result.get('rows', []))}")
        print(f"  Confidence: {result.get('confidence', 0)}")
        
        # Check financial validation
        meta = result.get('meta', {})
        fv = meta.get('financial_validation', {})
        if fv:
            print(f"  Validation Valid: {fv.get('is_valid', False)}")
            print(f"  Charts Enabled: {fv.get('charts_enabled', False)}")
            print(f"  Errors: {fv.get('error_count', 0)}")
            print(f"  Warnings: {fv.get('warning_count', 0)}")
        
        # Print first 3 rows
        rows = result.get('rows', [])
        if rows:
            print("\n  First 3 rows:")
            for i, row in enumerate(rows[:3]):
                label = row.get('Label', row.get('label', ''))[:50]
                note = row.get('Note', row.get('note', ''))
                row_type = row.get('type', '')
                print(f"    [{i}] {row_type:8} | Note: {note:6} | {label}")
        
        return result
        
    except Exception as e:
        print(f"    ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # Check for command line argument (single image)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    if len(sys.argv) > 1:
        # Test single image
        img_name = sys.argv[1]
        if not os.path.exists(os.path.join(project_dir, img_name)):
            print(f"ERROR: Image not found: {img_name}")
            return
        images = [img_name]
    else:
        # Find all PNG images
        images = [f for f in os.listdir(project_dir) if f.endswith('.png')]
    
    print(f"Found {len(images)} images to test:")
    for img in images:
        print(f"  - {img}")
    
    # Load model
    print("\nLoading model...")
    model, processor = load_model()
    print("Model loaded!")
    
    # Test each image
    results = {}
    for img_name in images:
        img_path = os.path.join(project_dir, img_name)
        result = test_image(img_path, model, processor)
        results[img_name] = result
        
        # Save individual result to JSON
        if result:
            output_name = img_name.rsplit('.', 1)[0] + '_result.json'
            output_path = os.path.join(project_dir, output_name)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"    Saved: {output_name}")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    success = 0
    for img_name, result in results.items():
        if result:
            rows = len(result.get('rows', []))
            conf = result.get('confidence', 0)
            valid = result.get('meta', {}).get('financial_validation', {}).get('is_valid', False)
            status = "OK" if rows > 5 and conf >= 0.6 else "WARN"
            print(f"  {status:4} | {img_name:40} | Rows: {rows:3} | Conf: {conf:.2f} | Valid: {valid}")
            if rows > 5:
                success += 1
        else:
            print(f"  FAIL | {img_name:40} | Could not extract")
    
    print(f"\nSuccess rate: {success}/{len(images)} ({100*success/len(images):.0f}%)")


if __name__ == "__main__":
    main()

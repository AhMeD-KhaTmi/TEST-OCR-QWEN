"""
Quick test to verify inference doesn't hang with the safeguards.
"""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

# Test with attijari_statement.png
TEST_IMAGE = "attijari_statement.png"

def main():
    # Check if image exists
    if not os.path.exists(TEST_IMAGE):
        print(f"[ERROR] Test image not found: {TEST_IMAGE}")
        return
    
    # Check image size
    with Image.open(TEST_IMAGE) as img:
        w, h = img.size
        area = w * h
        print(f"[INFO] Image size: {w}x{h} (area={area:,} pixels)")
        
        # Predict token count
        if h > 1200 or area > 1_500_000:
            expected_tokens = 4096
            size_class = "LARGE"
        elif h > 700 or area > 800_000:
            expected_tokens = 2560
            size_class = "MEDIUM"
        else:
            expected_tokens = 1536
            size_class = "SMALL"
        
        print(f"[INFO] Expected size class: {size_class}")
        print(f"[INFO] Expected token count: {expected_tokens}")
    
    # Load model and run extraction
    print("\n[INFO] Loading model...")
    from run_qwen_vl import load_model, extract_table_from_image
    
    model, processor = load_model()
    
    print(f"\n[INFO] Running extraction on {TEST_IMAGE}...")
    print("[INFO] With anti-hang safeguards: repetition_penalty=1.1, dynamic tokens")
    
    import time
    start = time.time()
    
    result = extract_table_from_image(
        model, 
        processor, 
        TEST_IMAGE,
        max_new_tokens=None,  # Auto-calculate
    )
    
    elapsed = time.time() - start
    
    print(f"\n[SUCCESS] Extraction completed in {elapsed:.1f}s")
    print(f"[INFO] Rows extracted: {len(result.get('rows', []))}")
    print(f"[INFO] Columns: {result.get('columns', [])}")

if __name__ == "__main__":
    main()

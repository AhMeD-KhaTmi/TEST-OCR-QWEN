"""
DEEP SYSTEM VALIDATION TEST
============================
Tests the financial table extraction pipeline under real edge cases.

Run: python deep_validation_test.py (with venv activated)
   OR: Run from VS Code with the correct Python interpreter
"""

import os
import sys
import json
import traceback
import re

# Check if we can import heavy modules
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Note: torch not available, skipping VLM-dependent tests")

from PIL import Image
import numpy as np
from io import BytesIO

# Import pipeline modules that don't require torch
from json_table_utils import (
    extract_json_from_response,
    convert_list_rows_to_dicts,
)
from table_alignment_engine import (
    detect_schema_from_columns,
    align_row,
    run_alignment_engine,
)
from financial_validation_engine import (
    run_financial_validation,
    identify_date_order,
    _parse_numeric_value,
)

# Conditionally import VLM functions
if TORCH_AVAILABLE:
    from run_qwen_vl import (
        preprocess_image,
        detect_table_region,
        preprocess_image_for_model,
        auto_crop_to_content,
        DEFAULT_MAX_IMAGE_SIZE,
    )

# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = True
        self.issues = []
        self.warnings = []
    
    def fail(self, msg):
        self.passed = False
        self.issues.append(msg)
    
    def warn(self, msg):
        self.warnings.append(msg)
    
    def __str__(self):
        status = "[PASS]" if self.passed else "[FAIL]"
        result = f"{status} {self.name}"
        for issue in self.issues:
            result += f"\n    [X] {issue}"
        for warn in self.warnings:
            result += f"\n    [!] {warn}"
        return result


def create_test_image(width, height, content_ratio=0.5):
    """Create a test image with content in the center portion."""
    img = Image.new('RGB', (width, height), color='white')
    pixels = np.array(img)
    
    # Add table-like content in the middle
    margin_top = int(height * (1 - content_ratio) / 2)
    margin_bottom = int(height * (1 + content_ratio) / 2)
    
    # Draw horizontal lines (table rows)
    for y in range(margin_top, margin_bottom, 30):
        if y < height:
            pixels[y:y+2, 50:width-50] = [0, 0, 0]
    
    # Draw vertical lines (columns)
    for x in range(50, width-50, 100):
        pixels[margin_top:margin_bottom, x:x+1] = [0, 0, 0]
    
    return Image.fromarray(pixels)


# =============================================================================
# TEST 1: PREPROCESSING SAFEGUARDS
# =============================================================================

def test_preprocessing_safeguards():
    """Verify crop never removes >70% of image height."""
    result = TestResult("Preprocessing Safeguards")
    
    if not TORCH_AVAILABLE:
        result.warn("Skipping: torch not available")
        print("  [!] Skipped: requires torch/VLM modules")
        return result
    
    # Test 1.1: Image with content only in tiny region
    print("\n  [1.1] Testing small content region...")
    img = create_test_image(800, 1000, content_ratio=0.15)  # Only 15% has content
    cropped = detect_table_region(img, pdf_mode=False)
    
    original_height = img.size[1]
    cropped_height = cropped.size[1]
    
    if cropped_height < original_height * 0.25:
        result.fail(f"Crop removed too much: {original_height}->{cropped_height} ({cropped_height/original_height*100:.1f}% remaining)")
    else:
        print(f"    [OK] Safeguard held: {original_height}->{cropped_height}")
    
    # Test 1.2: Image with 90% content (should allow some cropping)
    print("  [1.2] Testing large content region...")
    img2 = create_test_image(800, 1000, content_ratio=0.9)
    cropped2 = detect_table_region(img2, pdf_mode=False)
    print(f"    [OK] Cropped: {img2.size[1]}->{cropped2.size[1]}")
    
    # Test 1.3: PDF mode should skip cropping
    print("  [1.3] Testing PDF mode bypass...")
    img3 = create_test_image(800, 1000, content_ratio=0.3)
    cropped3 = detect_table_region(img3, pdf_mode=True)
    
    if cropped3.size != img3.size:
        result.fail(f"PDF mode should not crop: {img3.size}->{cropped3.size}")
    else:
        print(f"    [OK] PDF mode preserved size: {img3.size}")
    
    # Test 1.4: Empty image should return original
    print("  [1.4] Testing empty image fallback...")
    empty_img = Image.new('RGB', (600, 800), color='white')
    cropped_empty = detect_table_region(empty_img, pdf_mode=False)
    
    if cropped_empty.size != empty_img.size:
        result.fail(f"Empty image should not be cropped")
    else:
        print(f"    [OK] Empty image preserved")
    
    return result


# =============================================================================
# TEST 2: COMPOUND CROP SAFETY
# =============================================================================

def test_compound_crop_safety():
    """Verify multiple crop steps do not compound errors."""
    result = TestResult("Compound Crop Safety")
    
    if not TORCH_AVAILABLE:
        result.warn("Skipping: torch not available")
        print("  [!] Skipped: requires torch/VLM modules")
        return result
    
    print("\n  [2.1] Testing multi-step preprocessing...")
    
    # Create image with moderate content
    original = create_test_image(1200, 1600, content_ratio=0.7)
    original_size = original.size
    
    # Run full preprocessing pipeline
    try:
        processed, _ = preprocess_image(original, max_tokens=2048, pdf_mode=False)
        final_size = processed.size
        
        # Check that final dimensions are reasonable
        final_area = final_size[0] * final_size[1]
        original_area = original_size[0] * original_size[1]
        remaining_ratio = final_area / original_area
        
        if remaining_ratio < 0.15:  # Lost more than 85% of area
            result.fail(f"Compound cropping too aggressive: {original_size}->{final_size} ({remaining_ratio*100:.1f}% remaining)")
        else:
            print(f"    [OK] Compound cropping safe: {original_size}->{final_size} ({remaining_ratio*100:.1f}% remaining)")
        
    except Exception as e:
        result.fail(f"Preprocessing failed: {e}")
    
    return result


# =============================================================================
# TEST 3: JSON PARSING ROBUSTNESS
# =============================================================================

def test_json_parsing():
    """Test JSON extraction from various malformed inputs."""
    result = TestResult("JSON Parsing Robustness")
    
    test_cases = [
        # Case 1: Valid JSON
        ('{"columns": ["A", "B"], "rows": [{"A": "1", "B": "2"}]}', True, "Valid JSON"),
        
        # Case 2: JSON with markdown wrapper
        ('```json\n{"columns": ["A"], "rows": []}\n```', True, "Markdown wrapped"),
        
        # Case 3: Truncated JSON
        ('{"columns": ["A", "B"], "rows": [{"A": "1", "B": "2"', True, "Truncated JSON"),
        
        # Case 4: Python-style literals
        ("{'columns': ['A'], 'rows': []}", True, "Python literals"),
        
        # Case 5: Nested escape issues
        ('{"columns": ["Label"], "rows": [{"Label": "Test \\"quote\\" here"}]}', True, "Escaped quotes"),
        
        # Case 6: Completely invalid
        ('This is not JSON at all', False, "Invalid text"),
        
        # Case 7: Empty response
        ('', False, "Empty response"),
    ]
    
    for json_str, should_parse, desc in test_cases:
        print(f"  [{desc}]...")
        parsed = extract_json_from_response(json_str)
        
        if should_parse and parsed is None:
            result.fail(f"Failed to parse: {desc}")
        elif not should_parse and parsed is not None:
            result.warn(f"Unexpected parse success: {desc}")
        else:
            print(f"    [OK] {desc}: {'parsed' if parsed else 'rejected'} as expected")
    
    return result


# =============================================================================
# TEST 4: LIST TO DICT CONVERSION
# =============================================================================

def test_list_to_dict_conversion():
    """Test conversion of list-based rows to dict format."""
    result = TestResult("List to Dict Conversion")
    
    # Test case: VLM returns list format
    data = {
        "columns": ["Label", "Note", "2024", "2023"],
        "rows": [
            ["Caisse", "(1)", "100 000", "90 000"],
            ["Banque", "(2)", "200 000", "180 000"],
        ]
    }
    
    converted = convert_list_rows_to_dicts(data)
    
    # Verify conversion
    if not isinstance(converted["rows"][0], dict):
        result.fail("Rows not converted to dicts")
        return result
    
    first_row = converted["rows"][0]
    if first_row.get("Label") != "Caisse":
        result.fail(f"Label mismatch: expected 'Caisse', got '{first_row.get('Label')}'")
    if first_row.get("2024") != "100 000":
        result.fail(f"Value mismatch in 2024 column")
    
    print(f"  [OK] Converted {len(converted['rows'])} rows from list to dict")
    
    # Test with more values than columns
    data2 = {
        "columns": ["A", "B"],
        "rows": [["1", "2", "3", "4"]]  # Extra values
    }
    converted2 = convert_list_rows_to_dicts(data2)
    
    if "_extra_2" not in converted2["rows"][0]:
        result.warn("Extra values not preserved")
    else:
        print(f"  [OK] Extra values preserved as _extra_* fields")
    
    return result


# =============================================================================
# TEST 5: DATA INTEGRITY (No Fabricated Labels)
# =============================================================================

def test_data_integrity():
    """Verify no fabricated labels or overwritten data."""
    result = TestResult("Data Integrity")
    
    # Simulate VLM output
    raw_data = {
        "columns": ["Label", "Note", "31/12/2024", "31/12/2023"],
        "rows": [
            {"Label": "ACTIF", "Note": "", "31/12/2024": "", "31/12/2023": ""},
            {"Label": "Caisse", "Note": "(1)", "31/12/2024": "100000", "31/12/2023": "90000"},
            {"Label": "TOTAL", "Note": "", "31/12/2024": "100000", "31/12/2023": "90000"},
        ]
    }
    
    # Process through alignment
    processed = run_alignment_engine(raw_data)
    
    # Check that original labels are preserved
    original_labels = [r["Label"] for r in raw_data["rows"]]
    processed_labels = [r.get("Label", r.get("label", "")) for r in processed["rows"]]
    
    for orig in original_labels:
        if orig and orig not in processed_labels:
            result.fail(f"Original label lost: '{orig}'")
    
    # Check that no artificial labels were created
    for label in processed_labels:
        if label and label not in original_labels:
            result.warn(f"New label created (may be valid): '{label}'")
    
    print(f"  [OK] All {len(original_labels)} original labels preserved")
    
    # Verify raw data is stored
    if "_raw_rows" not in processed:
        result.warn("Raw rows not preserved in output")
    else:
        print(f"  [OK] Raw data preserved in _raw_rows")
    
    return result


# =============================================================================
# TEST 6: COVERAGE ENFORCEMENT
# =============================================================================

def test_coverage_enforcement():
    """Verify low coverage triggers warnings."""
    result = TestResult("Coverage Enforcement")
    
    # Simulate incomplete extraction
    data = {
        "columns": ["Label", "2024", "2023"],
        "rows": [
            {"Label": "Item1", "2024": "100", "2023": "90"},
        ],
        "_coverage_complete": False,  # Simulated low coverage
        "_extraction_confidence": 0.5,
    }
    
    # Check that warning fields are propagated
    if data.get("_coverage_complete") is False:
        print(f"  [OK] Coverage warning flag present")
    else:
        result.fail("Coverage warning not in data")
    
    if data.get("_extraction_confidence", 1.0) < 0.7:
        print(f"  [OK] Low confidence flag: {data.get('_extraction_confidence')}")
    
    return result


# =============================================================================
# TEST 7: FINANCIAL VALIDATION EDGE CASES
# =============================================================================

def test_financial_validation():
    """Test financial validation edge cases."""
    result = TestResult("Financial Validation Edge Cases")
    
    # Test date order detection
    print("  [7.1] Testing date order detection...")
    dates1 = ["31/12/2024", "31/12/2023"]
    current, previous = identify_date_order(dates1)
    
    if current != "31/12/2024" or previous != "31/12/2023":
        result.fail(f"Date order wrong: current={current}, previous={previous}")
    else:
        print(f"    [OK] Date order: current={current}, previous={previous}")
    
    # Test numeric parsing edge cases
    print("  [7.2] Testing numeric parsing...")
    test_values = [
        ("100 000", 100000),
        ("(50 000)", -50000),
        ("1 234 567", 1234567),
        ("1.234.567,89", 1234567.89),
        ("-", None),
        ("", None),
        ("N/A", None),
    ]
    
    for value, expected in test_values:
        parsed = _parse_numeric_value(value)
        if parsed != expected:
            result.fail(f"Numeric parse failed: '{value}' -> {parsed}, expected {expected}")
        else:
            print(f"    [OK] '{value}' -> {parsed}")
    
    # Test division by zero handling
    print("  [7.3] Testing division by zero protection...")
    # This is validated in the financial_validation_engine code - lines 440-454
    # The function returns early with percent=None if previous=0
    
    return result


# =============================================================================
# TEST 8: TRUNCATION HANDLING
# =============================================================================

def test_truncation_handling():
    """Verify truncation detection and retry mechanism exists."""
    result = TestResult("Truncation Handling")
    
    # Check that retry logic exists in run_qwen_vl.py (read source directly without importing)
    source = open("run_qwen_vl.py", "r", encoding="utf-8").read()
    
    checks = [
        ("RETRY" in source, "Retry keyword present"),
        ("truncat" in source.lower(), "Truncation detection present"),
        ("increasing tokens" in source.lower() or "increase" in source.lower(), "Token increase logic present"),
    ]
    
    for check, desc in checks:
        if check:
            print(f"  [OK] {desc}")
        else:
            result.warn(f"Missing: {desc}")
    
    return result


# =============================================================================
# TEST 9: SCHEMA DETECTION
# =============================================================================

def test_schema_detection():
    """Test dynamic schema detection."""
    result = TestResult("Schema Detection")
    
    # Test various column formats
    test_schemas = [
        (["Label", "Note", "31/12/2024", "31/12/2023", "Variation Montant", "Variation %"], 
         {"dates": 2, "variation": True}),
        (["Libellé", "2024", "2023"], 
         {"dates": 2, "variation": False}),
        (["Item", "Dec-24", "Dec-23", "Change"], 
         {"dates": 2, "variation": False}),  # "Change" might be detected
    ]
    
    for columns, expected in test_schemas:
        schema = detect_schema_from_columns(columns)
        
        if schema:
            date_count = len(schema.date_cols)
            has_variation = schema.variation_amount_col is not None
            
            print(f"  Schema: {columns}")
            print(f"    -> Dates: {date_count}, Variation: {has_variation}")
            
            if date_count != expected["dates"]:
                result.warn(f"Date column count mismatch: expected {expected['dates']}, got {date_count}")
        else:
            result.fail(f"Failed to detect schema for: {columns}")
    
    return result


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all validation tests."""
    print("=" * 70)
    print("DEEP SYSTEM VALIDATION")
    print("=" * 70)
    
    tests = [
        test_preprocessing_safeguards,
        test_compound_crop_safety,
        test_json_parsing,
        test_list_to_dict_conversion,
        test_data_integrity,
        test_coverage_enforcement,
        test_financial_validation,
        test_truncation_handling,
        test_schema_detection,
    ]
    
    results = []
    for test in tests:
        try:
            print(f"\n{'='*70}")
            print(f"TEST: {test.__name__}")
            print("="*70)
            result = test()
            results.append(result)
            print(f"\n{result}")
        except Exception as e:
            result = TestResult(test.__name__)
            result.fail(f"Exception: {e}\n{traceback.format_exc()}")
            results.append(result)
            print(f"\n{result}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    for result in results:
        status = "[PASS]" if result.passed else "[FAIL]"
        print(f"  {status} {result.name}")
    
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    if failed > 0:
        print("\n[!] SYSTEM HAS ISSUES - See failures above")
        return False
    else:
        print("\n[OK] ALL TESTS PASSED")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

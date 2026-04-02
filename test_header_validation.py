#!/usr/bin/env python3
"""
Test suite for Header Semantic Validation.

Tests:
1. Multi-row header normalization (CRITICAL: must not lose date columns)
2. Column name normalization
3. Header confidence scoring
4. Header structure validation (SAFE MODE)
5. Fallback schema inference from data rows
6. Robust date column detection (regex-based)
7. Schema merging
"""

import sys
sys.path.insert(0, r"c:\Users\THOURAYA\test qwen")

from table_alignment_engine import (
    normalize_column_name,
    merge_hierarchical_headers,
    compute_header_confidence,
    validate_header_structure,
    infer_schema_from_data_rows,
    detect_schema_from_columns,
    ColumnSchema,
    HEADER_CONFIDENCE_THRESHOLD,
    _is_date_column_name,
    _merge_schemas,
)


def test_robust_date_detection():
    """Test robust date column detection via regex."""
    print("\n=== TEST: Robust Date Detection ===")
    
    test_cases = [
        ("31/12/2024", True),
        ("31.12.2024", True),
        ("30/06/2023", True),
        ("1/1/2024", True),
        ("01-12-2024", True),
        ("Label", False),
        ("Variation", False),
        ("Montant", False),
        ("%", False),
        ("123456", False),
    ]
    
    passed = 0
    for col_name, expected in test_cases:
        result = _is_date_column_name(col_name)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"  {status} _is_date_column_name('{col_name}') = {result} (expected: {expected})")
    
    print(f"\n  Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_column_name_normalization():
    """Test that column names are properly normalized."""
    print("\n=== TEST: Column Name Normalization ===")
    
    test_cases = [
        ("Montant", "Variation Montant"),
        ("En %", "Variation %"),
        ("Libellé", "Label"),
        ("Notes", "Note"),
        ("en montant", "Variation Montant"),
        ("%", "Variation %"),
        ("31/12/2024", "31/12/2024"),  # Dates unchanged
        ("TOTAL ACTIF", "TOTAL ACTIF"),  # Custom labels unchanged
    ]
    
    passed = 0
    for input_name, expected in test_cases:
        result = normalize_column_name(input_name)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"  {status} normalize_column_name('{input_name}') = '{result}' (expected: '{expected}')")
    
    print(f"\n  Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_hierarchical_header_merge_preserves_dates():
    """CRITICAL TEST: Ensure hierarchical header merge preserves date columns."""
    print("\n=== TEST: Hierarchical Header Merge (Preserves Dates) ===")
    
    # Test case 1: Full header with dates
    headers1 = ["Label", "Note", "31/12/2024", "31/12/2023", "Variation", "Montant", "%"]
    merged1 = merge_hierarchical_headers(headers1)
    
    print(f"  Input:    {headers1}")
    print(f"  Merged:   {merged1}")
    
    # Check that date columns are preserved
    date_preserved = "31/12/2024" in merged1 and "31/12/2023" in merged1
    has_var_montant = "Variation Montant" in merged1
    has_var_pct = "Variation %" in merged1
    
    print(f"  - Dates preserved: {date_preserved}")
    print(f"  - Has Variation Montant: {has_var_montant}")
    print(f"  - Has Variation %: {has_var_pct}")
    
    test1_pass = date_preserved and has_var_montant and has_var_pct
    print(f"  Result: {'✓ PASS' if test1_pass else '✗ FAIL'}")
    
    # Test case 2: Problematic case - Variation followed by sub-headers
    headers2 = ["Libellé", "30/06/2024", "30/06/2023", "Variation", "Montant", "%"]
    merged2 = merge_hierarchical_headers(headers2)
    
    print(f"\n  Input:    {headers2}")
    print(f"  Merged:   {merged2}")
    
    # Check dates are preserved
    date_preserved2 = "30/06/2024" in merged2 and "30/06/2023" in merged2
    print(f"  - Dates preserved: {date_preserved2}")
    
    test2_pass = date_preserved2
    print(f"  Result: {'✓ PASS' if test2_pass else '✗ FAIL'}")
    
    # Test case 3: Edge case - only variation sub-headers, no dates
    headers3 = ["Variation", "Montant", "%"]
    merged3 = merge_hierarchical_headers(headers3)
    
    print(f"\n  Input:    {headers3}")
    print(f"  Merged:   {merged3}")
    
    # Should become ["Variation Montant", "Variation %"]
    test3_pass = "Variation Montant" in merged3 and "Variation %" in merged3
    print(f"  Result: {'✓ PASS' if test3_pass else '✗ FAIL'}")
    
    return test1_pass and test2_pass and test3_pass


def test_header_confidence_safe_mode():
    """Test that confidence scoring uses SAFE MODE (minimum score for numeric columns)."""
    print("\n=== TEST: Header Confidence (Safe Mode) ===")
    
    # Schema with only date columns (no label)
    date_only_schema = ColumnSchema(
        label_col=None,
        note_col=None,
        date_cols=["31/12/2024", "31/12/2023"],
        variation_amount_col=None,
        variation_percent_col=None,
        unresolved_cols=[],
    )
    date_score = compute_header_confidence(date_only_schema)
    print(f"  Date-only schema confidence: {date_score:.2f}")
    print(f"    - Expected: >= 0.3 (SAFE MODE minimum)")
    date_pass = date_score >= 0.3
    print(f"    - Result: {'✓ PASS' if date_pass else '✗ FAIL'}")
    
    # Full schema
    good_schema = ColumnSchema(
        label_col="Label",
        note_col="Note",
        date_cols=["31/12/2024", "31/12/2023"],
        variation_amount_col="Variation Montant",
        variation_percent_col="Variation %",
        unresolved_cols=[],
    )
    good_score = compute_header_confidence(good_schema)
    print(f"\n  Good schema confidence: {good_score:.2f}")
    print(f"    - Expected: >= 0.8")
    good_pass = good_score >= 0.8
    print(f"    - Result: {'✓ PASS' if good_pass else '✗ FAIL'}")
    
    return date_pass and good_pass


def test_header_structure_permissive():
    """Test that header validation is permissive (SAFE MODE)."""
    print("\n=== TEST: Header Structure Validation (Permissive) ===")
    
    # Schema with only date columns - should be VALID
    date_only = ColumnSchema(
        label_col=None,
        note_col=None,
        date_cols=["31/12/2024"],
        variation_amount_col=None,
        variation_percent_col=None,
        unresolved_cols=[],
    )
    is_valid, issues = validate_header_structure(date_only)
    print(f"  Date-only schema: is_valid={is_valid}, issues={issues}")
    date_pass = is_valid  # Should be valid because it has numeric columns
    print(f"    - Result: {'✓ PASS' if date_pass else '✗ FAIL'}")
    
    # Schema with only variation percent - should be VALID
    pct_only = ColumnSchema(
        label_col=None,
        note_col=None,
        date_cols=[],
        variation_amount_col=None,
        variation_percent_col="Variation %",
        unresolved_cols=[],
    )
    is_valid2, issues2 = validate_header_structure(pct_only)
    print(f"\n  Percent-only schema: is_valid={is_valid2}, issues={issues2}")
    pct_pass = is_valid2
    print(f"    - Result: {'✓ PASS' if pct_pass else '✗ FAIL'}")
    
    # Schema with NO numeric columns - should be INVALID
    no_numeric = ColumnSchema(
        label_col="Label",
        note_col=None,
        date_cols=[],
        variation_amount_col=None,
        variation_percent_col=None,
        unresolved_cols=[],
    )
    is_valid3, issues3 = validate_header_structure(no_numeric)
    print(f"\n  No-numeric schema: is_valid={is_valid3}, issues={issues3}")
    no_num_pass = not is_valid3  # Should be invalid
    print(f"    - Result: {'✓ PASS' if no_num_pass else '✗ FAIL'}")
    
    return date_pass and pct_pass and no_num_pass


def test_fallback_schema_inference():
    """Test fallback schema inference from data rows."""
    print("\n=== TEST: Fallback Schema Inference ===")
    
    # Simulate data rows with clear column types
    rows = [
        {"Libellé": "Caisse", "Note": "H1", "31/12/2024": "1,234,567", "31/12/2023": "1,111,222", "Var %": "11.1%"},
        {"Libellé": "Banque", "Note": "H2", "31/12/2024": "2,345,678", "31/12/2023": "2,222,333", "Var %": "5.5%"},
        {"Libellé": "Total", "Note": "", "31/12/2024": "3,580,245", "31/12/2023": "3,333,555", "Var %": "7.4%"},
    ]
    
    schema = infer_schema_from_data_rows(rows, num_rows=3)
    
    if schema:
        print(f"  Inferred schema:")
        print(f"    - Label col: {schema.label_col}")
        print(f"    - Note col: {schema.note_col}")
        print(f"    - Date cols: {schema.date_cols}")
        print(f"    - Var % col: {schema.variation_percent_col}")
        print(f"    - Confidence: {schema.confidence_score:.2f}")
        
        # Should detect date columns by name
        has_dates = len(schema.date_cols) >= 2
        print(f"    - Has 2+ date columns: {has_dates}")
        
        test_pass = has_dates
        print(f"    - Result: {'✓ PASS' if test_pass else '✗ FAIL'}")
        return test_pass
    else:
        print("  Failed to infer schema from data rows")
        print("    - Result: ✗ FAIL")
        return False


def test_full_schema_detection_with_multirow():
    """Test full schema detection with multi-row headers."""
    print("\n=== TEST: Full Schema Detection (Multi-Row Headers) ===")
    
    # Simulated multi-row header that caused failures
    columns = ["Libellé", "Note", "31/12/2024", "31/12/2023", "Variation", "Montant", "%"]
    rows = [
        {"Libellé": "Caisse", "Note": "H1", "31/12/2024": "1,234,567", "31/12/2023": "1,111,222"},
    ]
    
    schema = detect_schema_from_columns(columns, rows=rows, use_fallback=True)
    
    print(f"  Input columns: {columns}")
    print(f"  Detected schema:")
    print(f"    - Label col: {schema.label_col}")
    print(f"    - Date cols: {schema.date_cols}")
    print(f"    - Var amount: {schema.variation_amount_col}")
    print(f"    - Var %: {schema.variation_percent_col}")
    print(f"    - Confidence: {schema.confidence_score:.2f}")
    
    # CRITICAL: Date columns must be preserved
    has_dates = len(schema.date_cols) >= 2
    above_threshold = schema.confidence_score >= HEADER_CONFIDENCE_THRESHOLD
    
    print(f"    - Has 2+ dates: {has_dates}")
    print(f"    - Above threshold: {above_threshold}")
    
    test_pass = has_dates and above_threshold
    print(f"    - Result: {'✓ PASS' if test_pass else '✗ FAIL'}")
    return test_pass


def test_schema_merging():
    """Test schema merging combines best of both schemas."""
    print("\n=== TEST: Schema Merging ===")
    
    # Primary schema: has label and note
    primary = ColumnSchema(
        label_col="Label",
        note_col="Note",
        date_cols=["31/12/2024"],
        variation_amount_col=None,
        variation_percent_col=None,
        unresolved_cols=[],
    )
    
    # Fallback schema: has dates and variation
    fallback = ColumnSchema(
        label_col=None,
        note_col=None,
        date_cols=["31/12/2024", "31/12/2023"],
        variation_amount_col="Variation Montant",
        variation_percent_col="Variation %",
        unresolved_cols=[],
    )
    
    merged = _merge_schemas(primary, fallback)
    
    print(f"  Primary: label={primary.label_col}, dates={primary.date_cols}")
    print(f"  Fallback: dates={fallback.date_cols}, var_amt={fallback.variation_amount_col}")
    print(f"  Merged:")
    print(f"    - Label: {merged.label_col}")
    print(f"    - Dates: {merged.date_cols}")
    print(f"    - Var amount: {merged.variation_amount_col}")
    print(f"    - Var %: {merged.variation_percent_col}")
    print(f"    - Confidence: {merged.confidence_score:.2f}")
    
    # Check merged has best of both
    has_label = merged.label_col == "Label"
    has_both_dates = len(merged.date_cols) >= 2
    has_var = merged.variation_amount_col is not None
    
    test_pass = has_label and has_both_dates and has_var
    print(f"    - Result: {'✓ PASS' if test_pass else '✗ FAIL'}")
    return test_pass


def main():
    """Run all tests."""
    print("=" * 60)
    print("HEADER SEMANTIC VALIDATION TEST SUITE (SAFE MODE)")
    print("=" * 60)
    
    results = []
    
    results.append(("Robust Date Detection", test_robust_date_detection()))
    results.append(("Column Name Normalization", test_column_name_normalization()))
    results.append(("Header Merge Preserves Dates", test_hierarchical_header_merge_preserves_dates()))
    results.append(("Confidence Safe Mode", test_header_confidence_safe_mode()))
    results.append(("Structure Validation Permissive", test_header_structure_permissive()))
    results.append(("Fallback Schema Inference", test_fallback_schema_inference()))
    results.append(("Full Detection Multi-Row", test_full_schema_detection_with_multirow()))
    results.append(("Schema Merging", test_schema_merging()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

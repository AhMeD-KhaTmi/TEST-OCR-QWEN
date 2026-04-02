"""
Test suite for Financial Validation Engine
============================================

Tests all 7 validation components:
1. Variation Amount Validation
2. Percentage Validation
3. Negative Value Consistency
4. Total Validation
5. Note Consistency
6. Zero & Edge Case Handling
7. Final Sanity Check
"""

import json
import sys
from typing import Dict, List

# Import the validation engine
from financial_validation_engine import (
    validate_financial_table,
    run_financial_validation,
    ValidationResult,
    _parse_numeric_value,
    _parse_percent_value,
    _format_number,
    _format_percent,
)


def print_test_result(test_name: str, passed: bool, details: str = ""):
    """Print formatted test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")
    if details and not passed:
        print(f"         {details}")


def test_variation_amount_validation():
    """Test 1: Variation Amount Validation."""
    print("\n1. VARIATION AMOUNT VALIDATION")
    print("-" * 50)
    
    test_table = {
        "columns": ["Label", "Note", "31/12/2024", "31/12/2023",
                    "Variation Montant", "Variation %"],
        "rows": [
            {
                "type": "data",
                "Label": "Test Item",
                "Note": "(1)",
                "31/12/2024": "500000",
                "31/12/2023": "400000",
                "Variation Montant": "90000",  # ERROR: should be 100000
                "Variation %": "25%"
            },
        ]
    }
    
    validated, result = validate_financial_table(test_table)
    row = validated["rows"][0]
    
    # Check if variation was corrected
    var_val = _parse_numeric_value(row["Variation Montant"])
    passed = var_val == 100000
    print_test_result(
        "Corrects incorrect variation amount (90000 → 100000)",
        passed,
        f"Got: {row['Variation Montant']}"
    )
    
    return passed


def test_percentage_validation():
    """Test 2: Percentage Validation."""
    print("\n2. PERCENTAGE VALIDATION")
    print("-" * 50)
    
    test_table = {
        "columns": ["Label", "Note", "31/12/2024", "31/12/2023",
                    "Variation Montant", "Variation %"],
        "rows": [
            {
                "type": "data",
                "Label": "Test Item",
                "Note": "(1)",
                "31/12/2024": "500000",
                "31/12/2023": "400000",
                "Variation Montant": "100000",
                "Variation %": "15%"  # ERROR: should be 25%
            },
        ]
    }
    
    validated, result = validate_financial_table(test_table)
    row = validated["rows"][0]
    
    # Check if percentage was corrected
    pct_val = _parse_percent_value(row["Variation %"])
    passed = pct_val is not None and abs(pct_val - 25.0) < 0.5
    print_test_result(
        "Corrects incorrect percentage (15% → 25%)",
        passed,
        f"Got: {row['Variation %']}"
    )
    
    return passed


def test_negative_value_consistency():
    """Test 3: Negative Value Consistency."""
    print("\n3. NEGATIVE VALUE CONSISTENCY")
    print("-" * 50)
    
    test_table = {
        "columns": ["Label", "Note", "31/12/2024", "31/12/2023",
                    "Variation Montant", "Variation %"],
        "rows": [
            {
                "type": "data",
                "Label": "Declining Asset",
                "Note": "(1)",
                "31/12/2024": "200000",
                "31/12/2023": "300000",
                "Variation Montant": "(100000)",  # Negative (correct)
                "Variation %": "33.3%"  # ERROR: should be negative
            },
        ]
    }
    
    validated, result = validate_financial_table(test_table)
    row = validated["rows"][0]
    
    # Check if percentage sign was corrected
    pct_val = _parse_percent_value(row["Variation %"])
    passed = pct_val is not None and pct_val < 0
    print_test_result(
        "Corrects percentage sign to match variation (positive → negative)",
        passed,
        f"Got: {row['Variation %']}"
    )
    
    return passed


def test_total_validation():
    """Test 4: Total Validation."""
    print("\n4. TOTAL VALIDATION")
    print("-" * 50)
    
    test_table = {
        "columns": ["Label", "Note", "31/12/2024", "31/12/2023",
                    "Variation Montant", "Variation %"],
        "rows": [
            {
                "type": "data",
                "Label": "Item A",
                "Note": "(1)",
                "31/12/2024": "100000",
                "31/12/2023": "80000",
                "Variation Montant": "20000",
                "Variation %": "25%"
            },
            {
                "type": "data",
                "Label": "Item B",
                "Note": "(2)",
                "31/12/2024": "200000",
                "31/12/2023": "150000",
                "Variation Montant": "50000",
                "Variation %": "33.3%"
            },
            {
                "type": "total",
                "Label": "TOTAL",
                "Note": "",
                "31/12/2024": "299000",  # ERROR: should be 300000
                "31/12/2023": "230000",
                "Variation Montant": "70000",
                "Variation %": "30%"
            },
        ]
    }
    
    validated, result = validate_financial_table(test_table)
    total_row = validated["rows"][2]
    
    # Check if total was corrected
    total_val = _parse_numeric_value(total_row["31/12/2024"])
    passed = total_val == 300000
    print_test_result(
        "Corrects incorrect total sum (299000 → 300000)",
        passed,
        f"Got: {total_row['31/12/2024']}"
    )
    
    return passed


def test_note_consistency():
    """Test 5: Note Consistency."""
    print("\n5. NOTE CONSISTENCY")
    print("-" * 50)
    
    test_table = {
        "columns": ["Label", "Note", "31/12/2024", "31/12/2023",
                    "Variation Montant", "Variation %"],
        "rows": [
            {
                "type": "data",
                "Label": "Item A",
                "Note": "(1)",
                "31/12/2024": "100000",
                "31/12/2023": "80000",
                "Variation Montant": "20000",
                "Variation %": "25%"
            },
            {
                "type": "data",
                "Label": "Item B",
                "Note": "(1)",  # DUPLICATE note
                "31/12/2024": "200000",
                "31/12/2023": "150000",
                "Variation Montant": "50000",
                "Variation %": "33.3%"
            },
        ]
    }
    
    validated, result = validate_financial_table(test_table)
    
    # Check if duplicate note was flagged
    has_warning = any("Duplicate" in w for w in result.warnings)
    print_test_result(
        "Detects duplicate note references",
        has_warning,
        f"Warnings: {result.warnings}"
    )
    
    return has_warning


def test_zero_edge_cases():
    """Test 6: Zero & Edge Case Handling."""
    print("\n6. ZERO & EDGE CASE HANDLING")
    print("-" * 50)
    
    all_passed = True
    
    # Test 6a: Equal values
    test_table_a = {
        "columns": ["Label", "Note", "31/12/2024", "31/12/2023",
                    "Variation Montant", "Variation %"],
        "rows": [
            {
                "type": "data",
                "Label": "Unchanged Item",
                "Note": "(1)",
                "31/12/2024": "100000",
                "31/12/2023": "100000",  # Same as current
                "Variation Montant": "5000",  # ERROR: should be 0
                "Variation %": "5%"  # ERROR: should be 0%
            },
        ]
    }
    
    validated_a, result_a = validate_financial_table(test_table_a)
    row_a = validated_a["rows"][0]
    
    var_val = _parse_numeric_value(row_a["Variation Montant"])
    passed_a = var_val == 0
    print_test_result(
        "Equal values: variation amount = 0",
        passed_a,
        f"Got: {row_a['Variation Montant']}"
    )
    all_passed = all_passed and passed_a
    
    pct_val = _parse_percent_value(row_a["Variation %"])
    passed_b = pct_val == 0
    print_test_result(
        "Equal values: variation percent = 0%",
        passed_b,
        f"Got: {row_a['Variation %']}"
    )
    all_passed = all_passed and passed_b
    
    # Test 6b: Division by zero (previous = 0)
    test_table_b = {
        "columns": ["Label", "Note", "31/12/2024", "31/12/2023",
                    "Variation Montant", "Variation %"],
        "rows": [
            {
                "type": "data",
                "Label": "New Item",
                "Note": "(1)",
                "31/12/2024": "100000",
                "31/12/2023": "0",  # Zero previous
                "Variation Montant": "100000",
                "Variation %": "100%"  # Should be cleared (div by zero)
            },
        ]
    }
    
    validated_b, result_b = validate_financial_table(test_table_b)
    row_b = validated_b["rows"][0]
    
    pct_empty = row_b["Variation %"] == ""
    print_test_result(
        "Division by zero: percent cleared when previous = 0",
        pct_empty,
        f"Got: '{row_b['Variation %']}'"
    )
    all_passed = all_passed and pct_empty
    
    return all_passed


def test_final_sanity_check():
    """Test 7: Final Sanity Check."""
    print("\n7. FINAL SANITY CHECK")
    print("-" * 50)
    
    # Test: Raw number in percent field
    test_table = {
        "columns": ["Label", "Note", "31/12/2024", "31/12/2023",
                    "Variation Montant", "Variation %"],
        "rows": [
            {
                "type": "data",
                "Label": "Test Item",
                "Note": "(1)",
                "31/12/2024": "500000",
                "31/12/2023": "400000",
                "Variation Montant": "100000",
                "Variation %": "25"  # Missing % sign
            },
        ]
    }
    
    validated, result = validate_financial_table(test_table)
    row = validated["rows"][0]
    
    # Check if % was added
    has_percent = "%" in row["Variation %"]
    print_test_result(
        "Adds missing % sign to percent field",
        has_percent,
        f"Got: {row['Variation %']}"
    )
    
    return has_percent


def test_format_functions():
    """Test helper formatting functions."""
    print("\n8. FORMAT FUNCTIONS")
    print("-" * 50)
    
    all_passed = True
    
    # Test number formatting
    tests = [
        (100000, "100 000"),
        (-50000, "(50 000)"),
        (0, "0"),
        (1234567, "1 234 567"),
    ]
    
    for num, expected in tests:
        result = _format_number(num)
        # Remove trailing decimals for comparison
        result_clean = result.rstrip('0').rstrip('.') if '.' in result else result
        expected_clean = expected.rstrip('0').rstrip('.') if '.' in expected else expected
        passed = result_clean == expected_clean or result == expected
        print_test_result(f"format_number({num})", passed, f"Expected '{expected}', got '{result}'")
        all_passed = all_passed and passed
    
    # Test percent formatting
    pct_tests = [
        (25.0, "25.0%"),
        (-10.5, "(10.5%)"),
        (0.0, "0.0%"),
    ]
    
    for pct, expected in pct_tests:
        result = _format_percent(pct)
        passed = result == expected
        print_test_result(f"format_percent({pct})", passed, f"Expected '{expected}', got '{result}'")
        all_passed = all_passed and passed
    
    return all_passed


def run_all_tests():
    """Run all test suites."""
    print("=" * 70)
    print("FINANCIAL VALIDATION ENGINE - TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    results["1. Variation Amount"] = test_variation_amount_validation()
    results["2. Percentage"] = test_percentage_validation()
    results["3. Sign Consistency"] = test_negative_value_consistency()
    results["4. Total Validation"] = test_total_validation()
    results["5. Note Consistency"] = test_note_consistency()
    results["6. Edge Cases"] = test_zero_edge_cases()
    results["7. Sanity Check"] = test_final_sanity_check()
    results["8. Format Functions"] = test_format_functions()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

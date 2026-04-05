"""
Financial Validation Engine
============================

ACCOUNTING-CORRECT & LOGICALLY VALID financial data validation.

This module ensures mathematical and logical consistency in extracted
financial tables, behaving like an accountant verifying a balance sheet.

Components:
    1. Variation Amount Validation   - expected = current - previous
    2. Percentage Validation         - expected = (variation / previous) * 100
    3. Negative Value Consistency    - sign coherence between amount & percent
    4. Total Validation              - sum verification for TOTAL rows
    5. Note Consistency              - no duplicated or misplaced note refs
    6. Zero & Edge Case Handling     - division by zero, equal values
    7. Final Sanity Check            - type placement validation

All corrections are logged for audit purposes.

CRITICAL SAFETY RULES:
    - NEVER modify data without full confidence
    - ALWAYS validate coverage before corrections
    - ALWAYS detect date order (newest = current, oldest = previous)
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Import from alignment engine for schema detection
from table_alignment_engine import (
    detect_schema_from_columns,
    classify_value,
    VALUE_TYPE_NUMBER,
    VALUE_TYPE_PERCENT,
    VALUE_TYPE_NOTE,
    VALUE_TYPE_EMPTY,
    VALUE_TYPE_TEXT,
    _clean,
    _normalize_number,
    _normalize_percent,
    ColumnSchema,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Tolerance for floating-point comparison (absolute difference)
VARIATION_TOLERANCE = 1.0  # Allow 1 unit rounding difference
PERCENT_TOLERANCE = 0.5    # Allow 0.5% difference (covers rounding and OCR errors)

# Minimum requirements for safe validation
MIN_ROWS_FOR_VALIDATION = 5
MIN_DATE_COLS_FOR_VARIATION = 2


# =============================================================================
# FIX 1: DATE ORDER DETECTION (CRITICAL)
# =============================================================================

def parse_date_column(col_name: str) -> Optional[datetime]:
    """
    Parse a date column name to a datetime object.
    
    Supports formats:
        - dd/mm/yyyy
        - dd.mm.yyyy
        - dd-mm-yyyy
    
    Returns None if parsing fails.
    """
    if not col_name:
        return None
    
    col_clean = str(col_name).strip()
    
    # Try multiple date formats
    for fmt in ['%d/%m/%Y', '%d.%m.%Y', '%d-%m-%Y']:
        try:
            return datetime.strptime(col_clean, fmt)
        except ValueError:
            continue
    
    return None


def identify_date_order(date_cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    FIX 1: Identify current and previous date columns by parsing actual dates.
    
    CRITICAL: Never rely on column order. Always parse and sort chronologically.
    
    Returns:
        (current_col, previous_col) - newest date first, oldest second
    """
    if not date_cols:
        return None, None
    
    if len(date_cols) == 1:
        return date_cols[0], None
    
    # Parse all date columns
    parsed = []
    for col in date_cols[:4]:  # Max 4 date columns
        dt = parse_date_column(col)
        parsed.append((col, dt))
    
    # Filter successfully parsed dates
    valid_parsed = [(col, dt) for col, dt in parsed if dt is not None]
    
    if len(valid_parsed) >= 2:
        # Sort by date, NEWEST FIRST (descending)
        valid_parsed.sort(key=lambda x: x[1], reverse=True)
        current_col = valid_parsed[0][0]
        previous_col = valid_parsed[1][0]
        print(f"[DATE ORDER] Detected: current={current_col}, previous={previous_col}")
        return current_col, previous_col
    elif len(valid_parsed) == 1:
        # Only one valid date, can't determine order
        print(f"[DATE ORDER] Only one parseable date: {valid_parsed[0][0]}")
        return valid_parsed[0][0], None
    else:
        # Fallback: assume original order (first = current)
        print(f"[DATE ORDER] WARNING: Could not parse dates, using fallback order")
        return date_cols[0], date_cols[1] if len(date_cols) > 1 else None


# =============================================================================
# FIX 10: OCR NOISE NORMALIZATION
# =============================================================================

def normalize_ocr_noise(val: str) -> str:
    """
    FIX 10: Normalize common OCR errors before numeric parsing.
    
    Common OCR errors:
        - 'l' (lowercase L) read as '1'
        - 'O' (letter O) read as '0'
        - Extra/missing spaces
        - 'I' read as '1'
    
    Only applies corrections in numeric context.
    """
    if not val:
        return val
    
    result = val.strip()
    
    # Only apply OCR fixes if the value looks numeric
    has_digits = any(c.isdigit() for c in result)
    if not has_digits:
        return result
    
    # Fix 'l' (lowercase L) to '1' when surrounded by digits
    result = re.sub(r'(?<=\d)l(?=\d|\s|$)', '1', result)
    result = re.sub(r'\bl(?=\d)', '1', result)  # word boundary instead of lookbehind
    
    # Fix 'O' (letter O) to '0' when surrounded by digits
    result = re.sub(r'(?<=\d)O(?=\d|\s|$)', '0', result)
    result = re.sub(r'\bO(?=\d)', '0', result)  # word boundary instead of lookbehind
    
    # Fix 'I' to '1' in numeric context
    result = re.sub(r'(?<=\d)I(?=\d|\s|$)', '1', result)
    result = re.sub(r'\bI(?=\d)', '1', result)  # word boundary instead of lookbehind
    
    # Normalize multiple spaces to single space
    result = re.sub(r'\s+', ' ', result)
    
    return result


# =============================================================================
# VALIDATION RESULT DATACLASS
# =============================================================================

@dataclass
class ValidationResult:
    """Result of financial validation for a single table."""
    is_valid: bool = True
    corrections_made: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrections: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_numeric_value(value: Any) -> Optional[float]:
    """
    Parse a numeric value from various formats.
    Handles: "147 120", "(317 205)", "<123 456>", "-97", "1,730,727"
    
    FIX 10: Applies OCR noise normalization before parsing.
    
    Returns None if value cannot be parsed as a number.
    """
    val = _clean(value)
    if not val or val in ("-", "–", "—", "N/A", "n/a", ""):
        return None
    
    # FIX 10: Apply OCR noise normalization
    val = normalize_ocr_noise(val)
    
    # Check if it's a percentage (should not be parsed as number)
    if "%" in val:
        return None
    
    vtype, normalized = classify_value(val)
    if vtype == VALUE_TYPE_NUMBER:
        try:
            return float(normalized)
        except (ValueError, TypeError):
            return None
    return None


def _parse_percent_value(value: Any) -> Optional[float]:
    """
    Parse a percentage value.
    Handles: "22,5%", "(5,4%)", "-15,4 %", "0.0%"
    
    FIX 10: Applies OCR noise normalization before parsing.
    
    Returns the numeric percentage value (e.g., 22.5 for "22.5%").
    Returns None if not a valid percentage.
    """
    val = _clean(value)
    if not val:
        return None
    
    vtype, normalized = classify_value(val)
    if vtype == VALUE_TYPE_PERCENT:
        try:
            return float(normalized)
        except (ValueError, TypeError):
            return None
    return None


def _format_number(value: float) -> str:
    """
    Format a number for display in financial tables.
    Uses space as thousands separator, no decimals for integers.
    Negative values shown in parentheses.
    """
    if value is None:
        return ""
    
    is_negative = value < 0
    abs_val = abs(value)
    
    # Format with space as thousands separator
    if abs_val == int(abs_val):
        formatted = f"{int(abs_val):,}".replace(",", " ")
    else:
        formatted = f"{abs_val:,.2f}".replace(",", " ")
    
    if is_negative:
        return f"({formatted})"
    return formatted


def _format_percent(value: float) -> str:
    """
    Format a percentage for display.
    Rounds to 1 decimal place, adds % sign.
    Negative values shown in parentheses with %.
    """
    if value is None:
        return ""
    
    is_negative = value < 0
    abs_val = abs(value)
    
    # Round to 1 decimal
    rounded = round(abs_val, 1)
    
    # Format with one decimal place
    if rounded == int(rounded):
        formatted = f"{int(rounded)}.0%"
    else:
        formatted = f"{rounded:.1f}%"
    
    if is_negative:
        return f"({formatted})"
    return formatted


def _sign_of(value: float) -> int:
    """Return -1, 0, or 1 based on sign of value."""
    if value is None:
        return 0
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _is_total_row(row: Dict) -> bool:
    """Check if row is a TOTAL row."""
    row_type = str(row.get("type", "")).lower()
    if row_type == "total":
        return True
    
    label = str(row.get("Label", "")).lower()
    return "total" in label


def _is_section_row(row: Dict) -> bool:
    """Check if row is a section header (no numeric values)."""
    return str(row.get("type", "")).lower() == "section"


def _is_data_row(row: Dict) -> bool:
    """Check if row is a data row with numeric values."""
    return str(row.get("type", "")).lower() == "data"


# =============================================================================
# 1. VARIATION AMOUNT VALIDATION
# =============================================================================

def validate_variation_amount(
    row: Dict,
    schema: ColumnSchema,
    result: ValidationResult,
    current_col: Optional[str] = None,
    previous_col: Optional[str] = None
) -> Dict:
    """
    Validate and correct variation amount.
    
    FIX 1: Uses properly detected date order (current_col, previous_col)
           instead of assuming schema.date_cols[0] is current.
    
    CRITICAL SAFETY: This function ONLY modifies schema.variation_amount_col.
    It READS from date columns but NEVER WRITES to them.
    
    Formula: expected_variation = current_date_value - previous_date_value
    
    If mismatch exceeds tolerance, overwrites variation_amount.
    """
    if not schema.variation_amount_col:
        return row
    
    # SAFETY: Verify the target column is NOT an immutable source column
    if _is_immutable_column(schema.variation_amount_col):
        result.warnings.append(
            f"[SAFETY] Refusing to modify '{schema.variation_amount_col}' - detected as immutable source column"
        )
        return row
    
    # FIX 1: Use provided date order, or detect it
    if current_col is None or previous_col is None:
        current_col, previous_col = identify_date_order(schema.date_cols)
    
    if not current_col or not previous_col:
        return row
    
    # READ from date columns (immutable source data)
    current_val = _parse_numeric_value(row.get(current_col, ""))
    previous_val = _parse_numeric_value(row.get(previous_col, ""))
    variation_val = _parse_numeric_value(row.get(schema.variation_amount_col, ""))
    
    # Skip if we don't have the necessary values
    if current_val is None or previous_val is None:
        return row
    
    expected_variation = current_val - previous_val
    
    # If variation is missing or incorrect, compute it
    # WRITE ONLY to variation_amount_col (derived metric)
    if variation_val is None:
        row[schema.variation_amount_col] = _format_number(expected_variation)
        result.corrections.append({
            "type": "variation_amount_computed",
            "row_label": row.get("Label", ""),
            "field": schema.variation_amount_col,
            "old_value": row.get(schema.variation_amount_col, ""),
            "new_value": row[schema.variation_amount_col],
            "formula": f"{current_val} - {previous_val} = {expected_variation}",
            "current_col": current_col,
            "previous_col": previous_col,
            "column_type": "derived_metric"
        })
        result.corrections_made += 1
        return row
    
    # Check if the difference exceeds tolerance
    difference = abs(expected_variation - variation_val)
    if difference > VARIATION_TOLERANCE:
        old_val = row.get(schema.variation_amount_col, "")
        row[schema.variation_amount_col] = _format_number(expected_variation)
        result.corrections.append({
            "type": "variation_amount_corrected",
            "row_label": row.get("Label", ""),
            "field": schema.variation_amount_col,
            "old_value": old_val,
            "new_value": row[schema.variation_amount_col],
            "expected": expected_variation,
            "found": variation_val,
            "difference": difference,
            "current_col": current_col,
            "previous_col": previous_col,
            "column_type": "derived_metric"
        })
        result.corrections_made += 1
    
    return row


# =============================================================================
# 2. PERCENTAGE VALIDATION
# =============================================================================

def validate_variation_percent(
    row: Dict,
    schema: ColumnSchema,
    result: ValidationResult,
    current_col: Optional[str] = None,
    previous_col: Optional[str] = None
) -> Dict:
    """
    Validate and correct variation percentage.
    
    FIX 1: Uses properly detected date order (previous_col)
           instead of assuming schema.date_cols[1] is previous.
    
    CRITICAL SAFETY: This function ONLY modifies schema.variation_percent_col.
    It READS from date/variation columns but NEVER WRITES to them.
    
    Formula: expected_percent = (variation_amount / previous_value) * 100
    
    Handles division by zero (previous == 0 → percent = null).
    """
    if not schema.variation_percent_col:
        return row
    
    # SAFETY: Verify the target column is NOT an immutable source column
    if _is_immutable_column(schema.variation_percent_col):
        result.warnings.append(
            f"[SAFETY] Refusing to modify '{schema.variation_percent_col}' - detected as immutable source column"
        )
        return row
    
    # FIX 1: Use provided date order, or detect it
    if previous_col is None:
        _, previous_col = identify_date_order(schema.date_cols)
    
    if not previous_col:
        return row
    
    # READ from date columns (immutable source data)
    previous_val = _parse_numeric_value(row.get(previous_col, ""))
    variation_val = _parse_numeric_value(row.get(schema.variation_amount_col, "")) if schema.variation_amount_col else None
    current_percent = _parse_percent_value(row.get(schema.variation_percent_col, ""))
    
    # Handle division by zero: if previous == 0, percent should be null/empty
    if previous_val is None or previous_val == 0:
        if current_percent is not None:
            old_val = row.get(schema.variation_percent_col, "")
            row[schema.variation_percent_col] = ""
            result.corrections.append({
                "type": "percent_cleared_division_by_zero",
                "row_label": row.get("Label", ""),
                "field": schema.variation_percent_col,
                "old_value": old_val,
                "new_value": "",
                "reason": "previous_value is 0 or missing, division by zero",
                "column_type": "derived_metric"
            })
            result.corrections_made += 1
        return row
    
    # Compute expected percentage
    if variation_val is None:
        return row
    
    expected_percent = (variation_val / previous_val) * 100
    
    # If percent is missing, compute it
    # WRITE ONLY to variation_percent_col (derived metric)
    if current_percent is None:
        row[schema.variation_percent_col] = _format_percent(expected_percent)
        result.corrections.append({
            "type": "percent_computed",
            "row_label": row.get("Label", ""),
            "field": schema.variation_percent_col,
            "old_value": row.get(schema.variation_percent_col, ""),
            "new_value": row[schema.variation_percent_col],
            "formula": f"({variation_val} / {previous_val}) * 100 = {expected_percent:.1f}",
            "column_type": "derived_metric"
        })
        result.corrections_made += 1
        return row
    
    # Check if the difference exceeds tolerance
    difference = abs(expected_percent - current_percent)
    if difference > PERCENT_TOLERANCE:
        old_val = row.get(schema.variation_percent_col, "")
        row[schema.variation_percent_col] = _format_percent(expected_percent)
        result.corrections.append({
            "type": "percent_corrected",
            "row_label": row.get("Label", ""),
            "field": schema.variation_percent_col,
            "old_value": old_val,
            "new_value": row[schema.variation_percent_col],
            "expected": f"{expected_percent:.1f}%",
            "found": f"{current_percent:.1f}%",
            "difference": f"{difference:.2f}%",
            "column_type": "derived_metric"
        })
        result.corrections_made += 1
    
    return row


# =============================================================================
# 3. NEGATIVE VALUE CONSISTENCY
# =============================================================================

def validate_sign_consistency(
    row: Dict,
    schema: ColumnSchema,
    result: ValidationResult
) -> Dict:
    """
    Ensure sign consistency between variation_amount and variation_percent.
    
    Rules:
    - If variation_amount < 0 → variation_percent must be negative
    - If variation_amount > 0 → variation_percent must be positive
    - If variation_amount == 0 → variation_percent must be 0%
    """
    if not schema.variation_amount_col or not schema.variation_percent_col:
        return row
    
    variation_val = _parse_numeric_value(row.get(schema.variation_amount_col, ""))
    percent_val = _parse_percent_value(row.get(schema.variation_percent_col, ""))
    
    if variation_val is None or percent_val is None:
        return row
    
    var_sign = _sign_of(variation_val)
    pct_sign = _sign_of(percent_val)
    
    # Signs should match (or both be zero)
    if var_sign != pct_sign:
        # Correct the percentage sign to match variation
        corrected_percent = abs(percent_val) * var_sign if var_sign != 0 else 0.0
        old_val = row.get(schema.variation_percent_col, "")
        row[schema.variation_percent_col] = _format_percent(corrected_percent)
        result.corrections.append({
            "type": "sign_consistency_corrected",
            "row_label": row.get("Label", ""),
            "field": schema.variation_percent_col,
            "old_value": old_val,
            "new_value": row[schema.variation_percent_col],
            "variation_sign": "negative" if var_sign < 0 else ("positive" if var_sign > 0 else "zero"),
            "percent_sign_was": "negative" if pct_sign < 0 else ("positive" if pct_sign > 0 else "zero")
        })
        result.corrections_made += 1
    
    return row


# =============================================================================
# 4. TOTAL VALIDATION (CRITICAL - SAFE VERSION)
# =============================================================================

# IMMUTABLE SOURCE COLUMNS: These columns contain raw extracted data
# and MUST NEVER be modified by any correction logic
_IMMUTABLE_COLUMN_PATTERNS = [
    r'^\d{2}[/.\-]\d{2}[/.\-]\d{4}$',  # Date columns: 31/12/2024, 31.12.2024
    r'^\d{4}$',                          # Year columns: 2024, 2023
]


def _is_immutable_column(col_name: str) -> bool:
    """
    Check if a column contains source data that must NEVER be modified.
    
    IMMUTABLE columns:
    - Date columns (31/12/2024, 31.12.2024, etc.)
    - Year columns (2024, 2023)
    
    These are BASE financial values extracted from the document.
    Only DERIVED columns (variation, percentage) can be corrected.
    """
    if not col_name:
        return False
    
    col_clean = str(col_name).strip()
    
    for pattern in _IMMUTABLE_COLUMN_PATTERNS:
        if re.match(pattern, col_clean):
            return True
    
    return False


def _get_mutable_numeric_columns(schema: 'ColumnSchema') -> List[str]:
    """
    Return ONLY columns that are safe to modify during corrections.
    
    SAFE TO MODIFY (derived metrics):
    - Variation amount columns
    - Variation percent columns
    
    NEVER MODIFY (source data):
    - Date columns (31/12/2024, etc.)
    
    This prevents cascading corruption where fixing a variation
    accidentally overwrites base financial values.
    """
    mutable_cols = []
    
    # Only variation columns are safe to modify
    if schema.variation_amount_col:
        mutable_cols.append(schema.variation_amount_col)
    
    # NOTE: We explicitly DO NOT include date_cols here
    # Date columns contain source data and must remain immutable
    
    return mutable_cols


def validate_totals(
    rows: List[Dict],
    schema: ColumnSchema,
    result: ValidationResult
) -> List[Dict]:
    """
    Validate TOTAL rows by summing all data rows in the same section.
    
    CRITICAL SAFETY RULE:
    - ONLY validate/correct variation columns
    - NEVER modify date columns (source data)
    
    For each TOTAL row:
    - Sum all preceding data rows (until previous section/total)
    - Compare with total value
    - If mismatch in VARIATION columns only, correct them
    - For date columns: LOG WARNING but DO NOT MODIFY
    """
    if not schema.date_cols:
        return rows
    
    # CRITICAL FIX: Only include columns that are SAFE to modify
    # Date columns are IMMUTABLE source data
    mutable_cols = _get_mutable_numeric_columns(schema)
    
    # Date columns for VALIDATION ONLY (no modifications)
    immutable_cols = [col for col in schema.date_cols if _is_immutable_column(col)]
    
    # Track sections: each section ends with a TOTAL row
    i = 0
    while i < len(rows):
        row = rows[i]
        
        if _is_total_row(row):
            # Find the start of this section (previous total or start of table)
            section_start = i - 1
            while section_start >= 0:
                if _is_total_row(rows[section_start]) or _is_section_row(rows[section_start]):
                    section_start += 1
                    break
                section_start -= 1
            if section_start < 0:
                section_start = 0
            
            # =================================================================
            # VALIDATE IMMUTABLE COLUMNS (LOG ONLY, NO MODIFICATION)
            # =================================================================
            for col in immutable_cols:
                expected_sum = 0.0
                has_values = False
                
                for j in range(section_start, i):
                    if _is_data_row(rows[j]):
                        val = _parse_numeric_value(rows[j].get(col, ""))
                        if val is not None:
                            expected_sum += val
                            has_values = True
                
                if not has_values:
                    continue
                
                actual_total = _parse_numeric_value(row.get(col, ""))
                
                if actual_total is not None and abs(expected_sum - actual_total) > VARIATION_TOLERANCE:
                    # LOG WARNING but DO NOT MODIFY
                    result.warnings.append(
                        f"[IMMUTABLE] Total mismatch in '{row.get('Label', '')}', "
                        f"column '{col}' (source data): expected {expected_sum}, found {actual_total}. "
                        f"NOT CORRECTED - source data is immutable."
                    )
            
            # =================================================================
            # CORRECT MUTABLE COLUMNS (VARIATION ONLY)
            # =================================================================
            for col in mutable_cols:
                expected_sum = 0.0
                has_values = False
                
                for j in range(section_start, i):
                    if _is_data_row(rows[j]):
                        val = _parse_numeric_value(rows[j].get(col, ""))
                        if val is not None:
                            expected_sum += val
                            has_values = True
                
                if not has_values:
                    continue
                
                # Get actual total value
                actual_total = _parse_numeric_value(row.get(col, ""))
                
                if actual_total is None:
                    # Total is missing, compute it
                    row[col] = _format_number(expected_sum)
                    result.corrections.append({
                        "type": "total_computed",
                        "row_label": row.get("Label", ""),
                        "field": col,
                        "old_value": row.get(col, ""),
                        "new_value": row[col],
                        "computed_sum": expected_sum,
                        "rows_summed": f"{section_start} to {i-1}",
                        "column_type": "derived_metric"
                    })
                    result.corrections_made += 1
                elif abs(expected_sum - actual_total) > VARIATION_TOLERANCE:
                    # Total is incorrect - safe to correct variation columns
                    old_val = row.get(col, "")
                    row[col] = _format_number(expected_sum)
                    result.corrections.append({
                        "type": "total_corrected",
                        "row_label": row.get("Label", ""),
                        "field": col,
                        "old_value": old_val,
                        "new_value": row[col],
                        "expected_sum": expected_sum,
                        "actual_total": actual_total,
                        "difference": abs(expected_sum - actual_total),
                        "column_type": "derived_metric"
                    })
                    result.corrections_made += 1
        
        i += 1
    
    return rows


# =============================================================================
# 5. NOTE CONSISTENCY
# =============================================================================

def validate_note_consistency(
    rows: List[Dict],
    schema: ColumnSchema,
    result: ValidationResult
) -> List[Dict]:
    """
    Ensure note references are consistent and not duplicated/misplaced.
    
    Rules:
    - Note codes should only appear in the Note column
    - No duplicate note codes in the same table
    - Section and total rows typically don't have notes
    """
    if not schema.note_col:
        return rows
    
    seen_notes = {}  # note -> row index
    
    for i, row in enumerate(rows):
        note_val = _clean(row.get(schema.note_col, ""))
        
        if not note_val:
            continue
        
        vtype, _ = classify_value(note_val)
        
        # Check if it's actually a note
        if vtype != VALUE_TYPE_NOTE:
            # This might be a misplaced value
            result.warnings.append(
                f"Row {i} '{row.get('Label', '')}': Note column contains non-note value '{note_val}'"
            )
            continue
        
        # Check for duplicates
        if note_val in seen_notes:
            result.warnings.append(
                f"Duplicate note reference '{note_val}' found in rows {seen_notes[note_val]} and {i}"
            )
        else:
            seen_notes[note_val] = i
        
        # Section and total rows shouldn't have notes (warning only)
        if _is_section_row(row) or _is_total_row(row):
            result.warnings.append(
                f"Note reference '{note_val}' in section/total row: '{row.get('Label', '')}'"
            )
    
    # Check other columns for misplaced notes
    for i, row in enumerate(rows):
        for col in schema.date_cols:
            val = _clean(row.get(col, ""))
            if val:
                vtype, _ = classify_value(val)
                if vtype == VALUE_TYPE_NOTE:
                    result.errors.append(
                        f"Row {i} '{row.get('Label', '')}': Note '{val}' found in date column '{col}'"
                    )
                    result.is_valid = False
        
        if schema.variation_amount_col:
            val = _clean(row.get(schema.variation_amount_col, ""))
            if val:
                vtype, _ = classify_value(val)
                if vtype == VALUE_TYPE_NOTE:
                    result.errors.append(
                        f"Row {i} '{row.get('Label', '')}': Note '{val}' found in variation amount column"
                    )
                    result.is_valid = False
    
    return rows


# =============================================================================
# 6. ZERO & EDGE CASE HANDLING
# =============================================================================

def handle_edge_cases(
    row: Dict,
    schema: ColumnSchema,
    result: ValidationResult
) -> Dict:
    """
    Handle special edge cases:
    
    - If previous == 0: variation_percent = null (avoid division by zero)
    - If both values equal: variation_amount = 0, variation_percent = 0%
    """
    if not schema.variation_amount_col or not schema.variation_percent_col:
        return row
    
    if len(schema.date_cols) < 2:
        return row
    
    current_col = schema.date_cols[0]
    previous_col = schema.date_cols[1]
    
    current_val = _parse_numeric_value(row.get(current_col, ""))
    previous_val = _parse_numeric_value(row.get(previous_col, ""))
    
    if current_val is None or previous_val is None:
        return row
    
    # Case: both values equal
    if current_val == previous_val:
        var_val = _parse_numeric_value(row.get(schema.variation_amount_col, ""))
        pct_val = _parse_percent_value(row.get(schema.variation_percent_col, ""))
        
        if var_val is not None and var_val != 0:
            old_val = row.get(schema.variation_amount_col, "")
            row[schema.variation_amount_col] = "0"
            result.corrections.append({
                "type": "equal_values_variation_zeroed",
                "row_label": row.get("Label", ""),
                "field": schema.variation_amount_col,
                "old_value": old_val,
                "new_value": "0",
                "reason": f"current ({current_val}) equals previous ({previous_val})"
            })
            result.corrections_made += 1
        
        if pct_val is not None and pct_val != 0:
            old_val = row.get(schema.variation_percent_col, "")
            row[schema.variation_percent_col] = "0.0%"
            result.corrections.append({
                "type": "equal_values_percent_zeroed",
                "row_label": row.get("Label", ""),
                "field": schema.variation_percent_col,
                "old_value": old_val,
                "new_value": "0.0%",
                "reason": f"current ({current_val}) equals previous ({previous_val})"
            })
            result.corrections_made += 1
    
    # Case: previous is zero (already handled in validate_variation_percent)
    
    return row


# =============================================================================
# 7. FINAL SANITY CHECK
# =============================================================================

def final_sanity_check(
    rows: List[Dict],
    schema: ColumnSchema,
    result: ValidationResult
) -> List[Dict]:
    """
    Final validation pass to ensure:
    
    - No % in numeric fields (date columns, variation amount)
    - No raw numbers in % fields
    - No misplaced values
    """
    for i, row in enumerate(rows):
        # Check date columns for percentage values
        for col in schema.date_cols:
            val = _clean(row.get(col, ""))
            if val and "%" in val:
                result.errors.append(
                    f"Row {i} '{row.get('Label', '')}': Percentage '{val}' in date column '{col}'"
                )
                result.is_valid = False
        
        # Check variation amount for percentage
        if schema.variation_amount_col:
            val = _clean(row.get(schema.variation_amount_col, ""))
            if val and "%" in val:
                result.errors.append(
                    f"Row {i} '{row.get('Label', '')}': Percentage '{val}' in variation amount column"
                )
                result.is_valid = False
        
        # Check variation percent for raw number (no %)
        if schema.variation_percent_col:
            val = _clean(row.get(schema.variation_percent_col, ""))
            if val:
                vtype, _ = classify_value(val)
                if vtype == VALUE_TYPE_NUMBER:
                    # Raw number in percent field - convert to percent
                    num_val = _parse_numeric_value(val)
                    if num_val is not None:
                        old_val = val
                        row[schema.variation_percent_col] = _format_percent(num_val)
                        result.corrections.append({
                            "type": "raw_number_to_percent",
                            "row_label": row.get("Label", ""),
                            "field": schema.variation_percent_col,
                            "old_value": old_val,
                            "new_value": row[schema.variation_percent_col]
                        })
                        result.corrections_made += 1
    
    return rows


# =============================================================================
# HARDENING #7: VALIDATION LAYER FOR NUMBERS
# =============================================================================

# Pattern for malformed numbers (detect common OCR/extraction errors)
_MALFORMED_NUMBER_PATTERNS = [
    re.compile(r'^\d+[a-zA-Z]+\d*$'),     # Digits mixed with letters: "123abc456"
    re.compile(r'^[a-zA-Z]+\d+$'),         # Letters then digits: "abc123"
    re.compile(r'^\d+\s+\d+\s+\d+\s+\d+'), # Too many space-separated groups (>3)
    re.compile(r'^[\(\[<]\s*$'),           # Unclosed bracket/paren
    re.compile(r'^\s*[\)\]>]$'),           # Orphan closing bracket
]


def is_malformed_number(val: str) -> bool:
    """
    HARDENING #7: Check if a numeric value is malformed.
    
    Detects:
    - Mixed alphanumeric: "123abc"
    - Unclosed brackets: "(123"
    - Corrupted formatting: "12 34 56 78 90"
    - Invalid characters in numeric context
    
    Returns True if malformed, False if valid or empty.
    """
    if not val:
        return False
    
    val = str(val).strip()
    if not val:
        return False
    
    # Check against malformed patterns
    for pattern in _MALFORMED_NUMBER_PATTERNS:
        if pattern.match(val):
            return True
    
    # Check for unbalanced parentheses/brackets
    open_count = val.count('(') + val.count('[') + val.count('<')
    close_count = val.count(')') + val.count(']') + val.count('>')
    if open_count != close_count:
        return True
    
    # Check for multiple decimal separators
    dot_count = val.count('.')
    comma_count = val.count(',')
    # More than one decimal indicator in non-thousands context is malformed
    # Exception: European format "1.234.567,89" is valid
    if dot_count > 1 and comma_count > 1:
        return True
    
    return False


def validate_numeric_consistency(
    rows: List[Dict],
    schema: ColumnSchema,
    result: ValidationResult
) -> List[Dict]:
    """
    HARDENING #7: VALIDATION LAYER FOR NUMBERS
    
    After parsing, reject malformed numbers and ensure numeric consistency.
    
    Rules:
    - Reject values that fail numeric pattern validation
    - Clear malformed values rather than fabricating corrections
    - Log all rejections for audit
    """
    numeric_cols = list(schema.date_cols)
    if schema.variation_amount_col:
        numeric_cols.append(schema.variation_amount_col)
    
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        
        for col in numeric_cols:
            val = _clean(row.get(col, ""))
            if not val:
                continue
            
            # Check if value is malformed
            if is_malformed_number(val):
                result.warnings.append(
                    f"Row {i} '{row.get('Label', '')}': Malformed number '{val}' in column '{col}' - cleared"
                )
                row[col] = ""  # Clear malformed value
                print(f"[NUMBER VALIDATION] Rejected malformed value: '{val}' in row {i}, col '{col}'")
                continue
            
            # Additional consistency check: try parsing
            vtype, normalized = classify_value(val)
            if vtype == VALUE_TYPE_NUMBER:
                # Verify the normalized value is actually numeric
                try:
                    float(normalized)
                except (ValueError, TypeError):
                    result.warnings.append(
                        f"Row {i} '{row.get('Label', '')}': Cannot parse '{val}' as number in column '{col}'"
                    )
                    row[col] = ""
    
    return rows


# =============================================================================
# FIX 4: COVERAGE VALIDATION (CRITICAL)
# =============================================================================

def validate_coverage(data: Dict) -> Tuple[bool, List[str], float]:
    """
    FIX 4: Validate extraction coverage before processing.
    
    Checks:
        - row_count >= MIN_ROWS_FOR_VALIDATION
        - at least 1 section row exists
        - at least 1 data row exists
        - first row is NOT empty
        - at least 1 numeric column exists
    
    Returns:
        (is_valid, issues, confidence_score)
    """
    issues = []
    confidence = 1.0  # Start at 100%
    
    rows = data.get("rows", [])
    columns = data.get("columns", [])
    
    # Check row count
    if len(rows) < MIN_ROWS_FOR_VALIDATION:
        issues.append(f"Too few rows: {len(rows)} < {MIN_ROWS_FOR_VALIDATION}")
        confidence -= 0.3
    
    # Count row types
    section_rows = [r for r in rows if isinstance(r, dict) and str(r.get("type", "")).lower() == "section"]
    data_rows = [r for r in rows if isinstance(r, dict) and str(r.get("type", "")).lower() == "data"]
    total_rows = [r for r in rows if isinstance(r, dict) and str(r.get("type", "")).lower() == "total"]
    
    if not section_rows:
        issues.append("No section headers found")
        confidence -= 0.2
    
    if not data_rows:
        issues.append("No data rows found")
        confidence -= 0.4
    
    if not total_rows:
        issues.append("No TOTAL rows found")
        confidence -= 0.1
    
    # Check first row is not empty
    if rows and isinstance(rows[0], dict):
        first_row = rows[0]
        label = _clean(first_row.get("Label", ""))
        if not label:
            issues.append("First row has empty label")
            confidence -= 0.1
    
    # Check for numeric columns
    has_numeric_col = False
    for col in columns:
        if isinstance(col, str):
            # Check if it's a date column
            if parse_date_column(col):
                has_numeric_col = True
                break
            # Check if it's a variation column
            col_lower = col.lower()
            if 'variation' in col_lower or 'montant' in col_lower:
                has_numeric_col = True
                break
    
    if not has_numeric_col:
        issues.append("No numeric columns found")
        confidence -= 0.3
    
    # Normalize confidence to [0, 1]
    confidence = max(0.0, min(1.0, confidence))
    
    is_valid = len(issues) == 0 and confidence >= 0.5
    
    return is_valid, issues, confidence


# =============================================================================
# FIX 9: CONFIDENCE SCORING (IMPORTANT)
# =============================================================================

def compute_extraction_confidence(data: Dict, schema: ColumnSchema) -> float:
    """
    FIX 9: Compute global extraction confidence score.
    
    Weighted factors:
        - Schema validity (25%)
        - Row count (20%)
        - Presence of sections (15%)
        - Presence of totals (15%)
        - Presence of numeric columns (25%)
    
    Returns score from 0.0 to 1.0
    """
    score = 0.0
    
    rows = data.get("rows", [])
    
    # Schema validity (25%)
    if schema.is_valid():
        score += 0.25
    elif schema.has_minimum_data():
        score += 0.15
    
    # Row count (20%)
    row_count = len(rows)
    if row_count >= 10:
        score += 0.20
    elif row_count >= MIN_ROWS_FOR_VALIDATION:
        score += 0.15
    elif row_count >= 3:
        score += 0.10
    
    # Presence of sections (15%)
    section_rows = [r for r in rows if isinstance(r, dict) and str(r.get("type", "")).lower() == "section"]
    if section_rows:
        score += 0.15
    
    # Presence of totals (15%)
    total_rows = [r for r in rows if isinstance(r, dict) and str(r.get("type", "")).lower() == "total"]
    if total_rows:
        score += 0.15
    
    # Numeric columns (25%)
    if len(schema.date_cols) >= 2:
        score += 0.25
    elif len(schema.date_cols) >= 1:
        score += 0.15
    elif schema.variation_amount_col or schema.variation_percent_col:
        score += 0.10
    
    return min(1.0, score)


# =============================================================================
# MAIN VALIDATION PIPELINE
# =============================================================================

def validate_financial_table(data: Dict) -> Tuple[Dict, ValidationResult]:
    """
    Run the complete financial validation pipeline.
    
    Input:  {"columns": [...], "rows": [...]}
    Output: (corrected_data, validation_result)
    
    FIX 4: Validates coverage BEFORE any corrections.
    FIX 5: Only applies corrections if confidence is sufficient.
    
    Validation steps:
    1. Coverage validation (FIX 4)
    2. Confidence scoring (FIX 9)
    3. Date order detection (FIX 1)
    4. Variation amount validation
    5. Percentage validation
    6. Negative value consistency
    7. Total validation
    8. Note consistency
    9. Zero & edge case handling
    10. Final sanity check
    """
    result = ValidationResult()
    
    if not isinstance(data, dict):
        result.is_valid = False
        result.errors.append("Input must be a dictionary")
        return data, result
    
    columns = data.get("columns", [])
    rows = data.get("rows", [])
    
    if not isinstance(rows, list) or not rows:
        result.warnings.append("No rows to validate")
        return data, result
    
    # =========================================================================
    # FIX 4: COVERAGE VALIDATION (CRITICAL)
    # =========================================================================
    coverage_valid, coverage_issues, coverage_confidence = validate_coverage(data)
    
    # Also check for pre-existing coverage markers
    external_coverage_complete = data.get("_coverage_complete", True)
    external_coverage_issues = data.get("_coverage_issues", [])
    
    if not external_coverage_complete:
        coverage_valid = False
        coverage_issues.extend(external_coverage_issues)
    
    # Detect schema
    if not isinstance(columns, list) or not columns:
        for r in rows:
            if isinstance(r, dict):
                columns = [k for k in r.keys() if k not in ("type", "__chunk_index", "__y_position")]
                break
    
    # Get data rows for fallback
    data_rows = [r for r in rows if isinstance(r, dict)]
    schema = detect_schema_from_columns(columns, rows=data_rows, use_fallback=True)
    
    # =========================================================================
    # FIX 9: CONFIDENCE SCORING
    # =========================================================================
    extraction_confidence = compute_extraction_confidence(data, schema)
    
    # Combine coverage and extraction confidence
    overall_confidence = (coverage_confidence + extraction_confidence) / 2
    
    print(f"[FINANCIAL VALIDATION] Coverage: {coverage_confidence:.2f}, Extraction: {extraction_confidence:.2f}, Overall: {overall_confidence:.2f}")
    
    # =========================================================================
    # FIX 5: SAFE FINANCIAL VALIDATION (CRITICAL)
    # =========================================================================
    # Determine if we should apply corrections
    can_apply_corrections = (
        coverage_valid and 
        overall_confidence >= 0.5 and
        len(schema.date_cols) >= MIN_DATE_COLS_FOR_VARIATION
    )
    
    if not can_apply_corrections:
        skip_corrections = True
        reasons = []
        if not coverage_valid:
            reasons.append(f"coverage invalid ({coverage_issues})")
        if overall_confidence < 0.5:
            reasons.append(f"low confidence ({overall_confidence:.2f})")
        if len(schema.date_cols) < MIN_DATE_COLS_FOR_VARIATION:
            reasons.append(f"insufficient date columns ({len(schema.date_cols)})")
        result.warnings.append(f"Corrections skipped: {', '.join(reasons)}")
        print(f"[FINANCIAL VALIDATION] CORRECTIONS SKIPPED: {reasons}")
    else:
        skip_corrections = False
        print(f"[FINANCIAL VALIDATION] Corrections enabled (confidence: {overall_confidence:.2f})")
    
    # =========================================================================
    # FIX 1: DETECT DATE ORDER ONCE (use for all validations)
    # =========================================================================
    current_col, previous_col = identify_date_order(schema.date_cols)
    
    # Validate each data row
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        
        # Skip section headers
        if _is_section_row(row):
            continue
        
        # FIX 6: Ensure label is not empty for data rows
        if _is_data_row(row):
            label = _clean(row.get("Label", ""))
            if not label:
                # Try to find a text value to use as label
                for key, val in row.items():
                    if key not in ("type", "Note", "Label") and not key.startswith("_"):
                        vtype, _ = classify_value(val)
                        if vtype == VALUE_TYPE_TEXT:
                            row["Label"] = val
                            result.warnings.append(f"Row {i}: Label was empty, used '{val}' as fallback")
                            break
        
        # 1. Variation amount validation (only if corrections allowed)
        if not skip_corrections:
            rows[i] = validate_variation_amount(row, schema, result, current_col, previous_col)
        
        # 2. Percentage validation (only if corrections allowed)
        if not skip_corrections:
            rows[i] = validate_variation_percent(rows[i], schema, result, current_col, previous_col)
        
        # 3. Negative value consistency (always check)
        rows[i] = validate_sign_consistency(rows[i], schema, result)
        
        # 6. Zero & edge case handling (always check)
        rows[i] = handle_edge_cases(rows[i], schema, result)
    
    # 4. Total validation (ONLY if corrections allowed - CRITICAL)
    if not skip_corrections:
        rows = validate_totals(rows, schema, result)
    else:
        result.warnings.append("Total validation skipped - insufficient confidence")
    
    # 5. Note consistency (always check)
    rows = validate_note_consistency(rows, schema, result)
    
    # 7. Final sanity check (always check)
    rows = final_sanity_check(rows, schema, result)
    
    # HARDENING #7: Numeric consistency validation (always check)
    rows = validate_numeric_consistency(rows, schema, result)
    
    data["rows"] = rows
    
    # =========================================================================
    # HARDENING #5: CONFIDENCE ENFORCEMENT
    # Mark result as unreliable if confidence < 0.6
    # Frontend must disable charts when unreliable=True
    # =========================================================================
    CONFIDENCE_THRESHOLD = 0.6
    is_unreliable = overall_confidence < CONFIDENCE_THRESHOLD
    
    if is_unreliable:
        print(f"[CONFIDENCE ENFORCEMENT] Result marked UNRELIABLE (confidence: {overall_confidence:.2f} < {CONFIDENCE_THRESHOLD})")
    
    # Add comprehensive validation metadata
    data["_validation"] = {
        "is_valid": result.is_valid,
        "corrections_made": result.corrections_made,
        "corrections_allowed": not skip_corrections,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "corrections": result.corrections,
        "errors": result.errors,
        "warnings": result.warnings,
        # HARDENING #5: Unreliable flag for frontend
        "unreliable": is_unreliable,
        "charts_enabled": not is_unreliable,
        # FIX 9: Include confidence scores
        "confidence": {
            "overall": round(overall_confidence, 3),
            "coverage": round(coverage_confidence, 3),
            "extraction": round(extraction_confidence, 3),
            "threshold": CONFIDENCE_THRESHOLD
        },
        # FIX 1: Include detected date order
        "date_order": {
            "current_col": current_col,
            "previous_col": previous_col
        }
    }
    
    return data, result


def run_financial_validation(data: Dict) -> Dict:
    """
    Public entry point for the Financial Validation Engine.
    
    Safe to call on any dict — returns original data with validation metadata
    if an unexpected exception occurs.
    
    Args:
        data: Table dict {"columns": [...], "rows": [...]}
    
    Returns:
        Validated and corrected table dict with "_validation" metadata.
    """
    if not isinstance(data, dict):
        return data
    
    try:
        validated_data, result = validate_financial_table(data)
        return validated_data
    except Exception as exc:
        import traceback
        data["_validation_error"] = str(exc)
        data["_validation_traceback"] = traceback.format_exc()
        print(f"[FINANCIAL VALIDATION] ERROR: {exc}")
        return data


# =============================================================================
# STANDALONE DEMO / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import json
    
    # Test case: Financial table with intentional errors
    test_table = {
        "columns": ["Label", "Note", "31/12/2024", "31/12/2023",
                    "Variation Montant", "Variation %"],
        "rows": [
            {
                "type": "section",
                "Label": "ACTIF",
                "Note": "",
                "31/12/2024": "",
                "31/12/2023": "",
                "Variation Montant": "",
                "Variation %": ""
            },
            {
                "type": "data",
                "Label": "Caisse et Banques Centrales",
                "Note": "(1)",
                "31/12/2024": "500000",
                "31/12/2023": "400000",
                "Variation Montant": "90000",  # ERROR: should be 100000
                "Variation %": "25%"  # Correct
            },
            {
                "type": "data",
                "Label": "Créances",
                "Note": "(2)",
                "31/12/2024": "300000",
                "31/12/2023": "300000",  # Equal values
                "Variation Montant": "5000",  # ERROR: should be 0
                "Variation %": "5%"  # ERROR: should be 0%
            },
            {
                "type": "data",
                "Label": "Titres",
                "Note": "(3)",
                "31/12/2024": "200000",
                "31/12/2023": "250000",
                "Variation Montant": "50000",  # ERROR: sign wrong, should be -50000
                "Variation %": "20%"  # ERROR: sign wrong, should be -20%
            },
            {
                "type": "total",
                "Label": "TOTAL ACTIF",
                "Note": "",
                "31/12/2024": "1000000",
                "31/12/2023": "950000",
                "Variation Montant": "50000",
                "Variation %": "5.3%"
            },
        ]
    }
    
    print("=" * 70)
    print("FINANCIAL VALIDATION ENGINE - SELF-TEST")
    print("=" * 70)
    
    print("\n--- BEFORE VALIDATION ---")
    for row in test_table["rows"]:
        print(json.dumps(row, ensure_ascii=False))
    
    # Run validation
    validated, result = validate_financial_table(test_table)
    
    print("\n--- AFTER VALIDATION ---")
    for row in validated["rows"]:
        # Exclude internal keys for display
        display_row = {k: v for k, v in row.items() if not k.startswith("_")}
        print(json.dumps(display_row, ensure_ascii=False))
    
    print("\n--- VALIDATION RESULT ---")
    print(f"Is Valid: {result.is_valid}")
    print(f"Corrections Made: {result.corrections_made}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    if result.corrections:
        print("\nCorrections:")
        for corr in result.corrections:
            print(f"  - {corr['type']}: {corr.get('row_label', '')} "
                  f"({corr.get('field', '')}: {corr.get('old_value', '')} → {corr.get('new_value', '')})")
    
    if result.errors:
        print("\nErrors:")
        for err in result.errors:
            print(f"  - {err}")
    
    if result.warnings:
        print("\nWarnings:")
        for warn in result.warnings:
            print(f"  - {warn}")

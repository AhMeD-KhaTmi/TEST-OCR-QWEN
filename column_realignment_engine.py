"""
Column Realignment Engine - Fixes Structural Corruption in Financial Tables
=============================================================================

ROOT CAUSE OF CORRUPTION:
    1. OCR extracts numbers correctly but assigns to WRONG columns
    2. Small integers (note refs: 4, 12) get classified as financial values
    3. Financial values (90637660) get classified as variation
    4. Result: entire row shifts left/right by 1-2 positions

THIS ENGINE FIXES:
    1. COLUMN TYPE DETECTION - classify each column semantically
    2. NOTE VS NUMBER DISAMBIGUATION - small integers in context of millions = notes
    3. COLUMN REALIGNMENT - swap misplaced values automatically
    4. VARIATION RECOMPUTATION - compute variation instead of trusting extraction
    5. CONFIDENCE ADJUSTMENT - drop confidence when corruption detected

Author: Financial Pipeline Debugging System
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics


# =============================================================================
# CONFIGURATION
# =============================================================================

# Threshold for distinguishing notes from financial values
# If a numeric value is < this threshold AND surrounding values are > this * 1000,
# it's likely a note reference, not a financial value
NOTE_VS_FINANCIAL_THRESHOLD = 50

# Minimum ratio: if smallest number is < 1/10000 of median, it's likely a note
# CONSERVATIVE: Reduced from 0.001 to 0.0001 to avoid false positives
NOTE_MAGNITUDE_RATIO = 0.0001

# Minimum median magnitude to trigger note detection
# Values must be significantly large (at least 50,000) for note detection to apply
MIN_MEDIAN_FOR_NOTE_DETECTION = 50000

# Percentage of values that must be numeric for a column to be classified as numeric
NUMERIC_COLUMN_THRESHOLD = 0.70

# Cash flow detection keywords
CASHFLOW_KEYWORDS = [
    "flux", "trésorerie", "tresorerie", "encaissements", "décaissements",
    "decaissements", "cash", "flow", "liquidités", "liquidites"
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ColumnTypeInfo:
    """Classification of a single column."""
    column_name: str
    column_type: str  # "label", "note", "numeric", "percent", "empty", "mixed"
    value_count: int
    numeric_count: int
    note_candidate_count: int  # small integers that could be notes
    text_count: int
    empty_count: int
    median_magnitude: float  # median of numeric values (for note detection)
    confidence: float


@dataclass
class RealignmentResult:
    """Result of column realignment."""
    realigned_rows: List[Dict]
    corrections_applied: int
    column_shifts_detected: int
    note_corrections: int
    confidence_penalty: float
    issues: List[str]
    

# =============================================================================
# COLUMN TYPE DETECTION
# =============================================================================

def _parse_numeric_value(val: str) -> Optional[float]:
    """
    Parse a numeric string to float.
    Handles: spaces, parens (negative), European format.
    """
    if not val or not isinstance(val, str):
        return None
    
    val = val.strip()
    if not val or val in ("-", "–", "—", "N/A", "n/a", ""):
        return None
    
    # Check if it's a percentage (skip these)
    if "%" in val:
        return None
    
    # Check for negative (parens or angle brackets)
    is_negative = bool(re.search(r"[(<]", val))
    
    # Remove all non-digit characters except dots and commas
    cleaned = re.sub(r"[^\d.,]", "", val)
    
    if not cleaned:
        return None
    
    # Handle European format (1.234.567,89)
    euro_pattern = re.match(r"^(\d{1,3}(?:\.\d{3})+),(\d{1,2})$", cleaned)
    if euro_pattern:
        integer_part = euro_pattern.group(1).replace(".", "")
        decimal_part = euro_pattern.group(2)
        cleaned = integer_part + "." + decimal_part
    else:
        # Standard handling
        euro_thousands = re.match(r"^(\d{1,3}(?:\.\d{3})+)$", cleaned)
        if euro_thousands:
            cleaned = cleaned.replace(".", "")
        else:
            # Comma as decimal if followed by 1-2 digits at end
            if re.search(r",\d{1,2}$", cleaned) and "." not in cleaned:
                cleaned = cleaned[:-3].replace(",", "") + "." + cleaned[-2:]
            else:
                cleaned = cleaned.replace(",", "")
    
    try:
        result = float(cleaned)
        return -abs(result) if is_negative else result
    except ValueError:
        return None


def _is_note_pattern(val: str) -> bool:
    """Check if value matches note reference patterns: (1), 1-2, III.1, 7 1, etc."""
    if not val:
        return False
    
    val = str(val).strip()
    
    # Explicit note patterns
    note_patterns = [
        r"^\(\s*[1-9]\s*\)$",                    # (1) through (9)
        r"^\(\s*\d{1,2}[.\-]\d{1,2}\s*\)$",     # (1-2), (4.1)
        r"^[IVXLCDM]{1,6}\.\d{1,2}$",           # III.1, IV.2
        r"^\d{1,2}[.\-]\d{1,2}$",               # 1-2, 1.2, 7.1
        # Space-separated notes: "7 1", "7 11" - BOTH parts must be 1-50
        r"^(?:[1-9]|[1-4][0-9]|50)\s(?:[1-9]|[1-4][0-9]|50)$",
    ]
    
    for pattern in note_patterns:
        if re.match(pattern, val, re.IGNORECASE):
            return True
    
    return False


def _is_small_integer(val: str, threshold: int = NOTE_VS_FINANCIAL_THRESHOLD) -> bool:
    """
    Check if value is a small integer that could be a note reference.
    
    CRITICAL: Must NOT match thousand-separated financial values like "10 892".
    Only matches plain small integers like "1", "12", "50".
    """
    if not val:
        return False
    
    val = str(val).strip()
    
    # If value contains space, check if it's a thousand-separator pattern
    # "10 892" = financial number (NOT a note)
    # "7 1" = note (handled by _is_note_pattern)
    if " " in val:
        # Check if it looks like a thousand-separated number
        # Pattern: "XXX XXX" or "XX XXX" or "X XXX XXX"
        parts = val.split()
        if len(parts) >= 2:
            # If ANY part is > 3 digits or if total value would be > threshold*100
            # it's likely a financial number, not a note
            all_digits = all(p.replace(",", "").replace(".", "").isdigit() for p in parts)
            if all_digits:
                combined = int("".join(p.replace(",", "").replace(".", "") for p in parts))
                if combined > threshold:
                    return False  # It's a financial number
        return False  # Space-separated notes should be detected by _is_note_pattern
    
    parsed = _parse_numeric_value(val)
    if parsed is None:
        return False
    
    # Must be a positive integer
    if parsed <= 0 or parsed != int(parsed):
        return False
    
    # Must be small
    return parsed <= threshold


def detect_column_types(rows: List[Dict], skip_cols: set = None) -> Dict[str, ColumnTypeInfo]:
    """
    COLUMN TYPE DETECTION (CRITICAL)
    
    For each column, classify as:
        - label: mostly text
        - note: small integers (1-30), sparse
        - numeric: >70% numeric values, magnitude > 100
        - percent: contains % sign
        - empty: mostly empty
        - mixed: unclear
    
    Returns:
        Dict mapping column name to ColumnTypeInfo
    """
    if skip_cols is None:
        skip_cols = {"type", "__chunk_index", "__y_position", "_row_corrected", "_alignment_corrected"}
    
    if not rows:
        return {}
    
    # Collect all column names
    all_cols = set()
    for row in rows:
        if isinstance(row, dict):
            all_cols.update(row.keys())
    all_cols -= skip_cols
    
    results = {}
    
    for col in all_cols:
        values = []
        for row in rows:
            if isinstance(row, dict) and col in row:
                values.append(row[col])
        
        if not values:
            results[col] = ColumnTypeInfo(
                column_name=col,
                column_type="empty",
                value_count=0,
                numeric_count=0,
                note_candidate_count=0,
                text_count=0,
                empty_count=0,
                median_magnitude=0.0,
                confidence=1.0
            )
            continue
        
        # Classify each value
        numeric_vals = []
        note_candidates = []
        text_count = 0
        empty_count = 0
        percent_count = 0
        
        for val in values:
            val_str = str(val).strip() if val else ""
            
            if not val_str or val_str in ("-", "–", "—"):
                empty_count += 1
                continue
            
            # Check for percentage
            if "%" in val_str:
                percent_count += 1
                continue
            
            # Check for explicit note pattern
            if _is_note_pattern(val_str):
                note_candidates.append(val_str)
                continue
            
            # Check for small integer (potential note)
            if _is_small_integer(val_str):
                parsed = _parse_numeric_value(val_str)
                note_candidates.append(val_str)
                numeric_vals.append(parsed)  # Also track as numeric for magnitude check
                continue
            
            # Check for numeric
            parsed = _parse_numeric_value(val_str)
            if parsed is not None:
                numeric_vals.append(abs(parsed))
                continue
            
            # Text
            if any(c.isalpha() for c in val_str):
                text_count += 1
        
        # Calculate statistics
        total_values = len(values)
        total_non_empty = total_values - empty_count
        
        median_magnitude = 0.0
        if numeric_vals:
            median_magnitude = statistics.median(numeric_vals) if len(numeric_vals) > 0 else 0.0
        
        # Determine column type
        col_type = "mixed"
        confidence = 0.5
        
        # High percent count = percent column
        if percent_count > total_non_empty * 0.5:
            col_type = "percent"
            confidence = percent_count / total_non_empty if total_non_empty > 0 else 0.0
        
        # High text count = label column
        elif text_count > total_non_empty * 0.5:
            col_type = "label"
            confidence = text_count / total_non_empty if total_non_empty > 0 else 0.0
        
        # Note column: mostly small integers, low magnitude
        elif len(note_candidates) > total_non_empty * 0.5 and median_magnitude < 100:
            col_type = "note"
            confidence = len(note_candidates) / total_non_empty if total_non_empty > 0 else 0.0
        
        # Numeric column: >70% numeric with high magnitude
        elif len(numeric_vals) > total_non_empty * NUMERIC_COLUMN_THRESHOLD:
            col_type = "numeric"
            confidence = len(numeric_vals) / total_non_empty if total_non_empty > 0 else 0.0
        
        # Empty column
        elif empty_count > total_values * 0.8:
            col_type = "empty"
            confidence = 1.0
        
        results[col] = ColumnTypeInfo(
            column_name=col,
            column_type=col_type,
            value_count=total_values,
            numeric_count=len(numeric_vals),
            note_candidate_count=len(note_candidates),
            text_count=text_count,
            empty_count=empty_count,
            median_magnitude=median_magnitude,
            confidence=confidence
        )
    
    return results


# =============================================================================
# DATE COLUMN NAME DETECTION
# =============================================================================

def _is_date_column_name(col_name: str) -> bool:
    """
    Check if column name looks like a date (e.g., 31/12/2022, 2022, Dec 2022).
    """
    if not col_name:
        return False
    
    col_lower = col_name.lower().strip()
    
    # Full date patterns: 31/12/2022, 31-12-2022, 2022-12-31
    date_patterns = [
        r"^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$",  # 31/12/2022
        r"^\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}$",    # 2022-12-31
        r"^\d{4}$",                                  # 2022
        r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)",
    ]
    
    for pattern in date_patterns:
        if re.match(pattern, col_lower):
            return True
    
    return False


# =============================================================================
# NOTE VS FINANCIAL VALUE DISAMBIGUATION
# =============================================================================

def detect_misplaced_notes(
    row: Dict,
    column_types: Dict[str, ColumnTypeInfo],
    skip_cols: set = None
) -> List[Tuple[str, str, str]]:
    """
    Detect values that are likely notes but placed in numeric columns.
    
    CONSERVATIVE APPROACH: Only flag values that are VERY clearly notes.
    False positives cause more harm than missed note corrections.
    
    Returns:
        List of (column_name, current_value, suggested_role) tuples
    """
    if skip_cols is None:
        skip_cols = {"type", "__chunk_index", "__y_position", "_row_corrected"}
    
    misplacements = []
    
    # Find median magnitude of numeric values in this row
    numeric_values = []
    for col, val in row.items():
        if col in skip_cols:
            continue
        parsed = _parse_numeric_value(str(val) if val else "")
        if parsed is not None and abs(parsed) > NOTE_VS_FINANCIAL_THRESHOLD:
            numeric_values.append(abs(parsed))
    
    if not numeric_values:
        return []
    
    median_magnitude = statistics.median(numeric_values)
    
    # SAFETY: Only apply note detection when median is very large
    # This prevents false positives on tables with smaller numbers
    if median_magnitude < MIN_MEDIAN_FOR_NOTE_DETECTION:
        return []
    
    # Check each value
    for col, val in row.items():
        if col in skip_cols:
            continue
        
        col_type = column_types.get(col)
        if not col_type:
            continue
        
        val_str = str(val).strip() if val else ""
        if not val_str:
            continue
        
        # Skip percentages
        if "%" in val_str:
            continue
        
        # FIRST CHECK: Does it match explicit note pattern?
        # If yes, it's definitely a note regardless of position
        if _is_note_pattern(val_str):
            if col_type.column_type in ("numeric",) or _is_date_column_name(col):
                misplacements.append((col, val_str, "note"))
            continue
        
        parsed = _parse_numeric_value(val_str)
        if parsed is None:
            continue
        
        # SECOND CHECK: Plain small integer in a numeric column
        # BUT: Be very conservative - only flag if BOTH conditions met:
        #   1. Value is truly small (1-50) with no thousand separators
        #   2. Median is VERY large (>50,000)
        if col_type.column_type == "numeric":
            # Only consider plain small integers (no spaces, no formatting)
            if _is_small_integer(val_str) and " " not in val_str:
                if median_magnitude > MIN_MEDIAN_FOR_NOTE_DETECTION:
                    # Additional check: ratio must be VERY small
                    if abs(parsed) / median_magnitude < NOTE_MAGNITUDE_RATIO:
                        misplacements.append((col, val_str, "note"))
        
        # DATE-like columns: same conservative approach
        elif _is_date_column_name(col):
            if _is_small_integer(val_str) and " " not in val_str:
                if median_magnitude > MIN_MEDIAN_FOR_NOTE_DETECTION:
                    if abs(parsed) / median_magnitude < NOTE_MAGNITUDE_RATIO:
                        misplacements.append((col, val_str, "note"))
    
    return misplacements


# =============================================================================
# COLUMN REALIGNMENT
# =============================================================================

def _get_date_like_columns(row: Dict, skip_cols: set) -> List[str]:
    """Get all columns with date-like names, sorted by date order."""
    date_cols = []
    for col in row.keys():
        if col not in skip_cols and _is_date_column_name(col):
            date_cols.append(col)
    
    # Sort by date (assuming format dd/mm/yyyy or yyyy)
    def date_sort_key(col):
        # Try to extract year for sorting
        year_match = re.search(r'(\d{4})', col)
        if year_match:
            return int(year_match.group(1))
        return 9999
    
    return sorted(date_cols, key=date_sort_key, reverse=True)  # Most recent first


def realign_row(
    row: Dict,
    column_types: Dict[str, ColumnTypeInfo],
    schema_cols: Dict[str, str] = None,  # mapping: role -> column_name
    skip_cols: set = None
) -> Tuple[Dict, List[str]]:
    """
    COLUMN REALIGNMENT ENGINE
    
    If misplaced values are detected:
        - Shift numeric values to correct columns
        - Move small integers to Note column
        - Recompute variation if needed
    
    Returns:
        (realigned_row, list_of_corrections)
    """
    if skip_cols is None:
        skip_cols = {"type", "__chunk_index", "__y_position", "_row_corrected"}
    
    corrections = []
    
    # Detect misplaced notes
    misplacements = detect_misplaced_notes(row, column_types, skip_cols)
    
    if not misplacements:
        return row.copy(), []
    
    realigned = row.copy()
    
    # Find the actual Note column by name (not by detected type!)
    note_col = None
    note_col_names = ["Note", "Notes", "note", "notes", "NOTE", "NOTES", "Réf.", "Ref", "Réf"]
    for col in row.keys():
        if col in note_col_names:
            note_col = col
            break
    
    # If no explicit Note column, try schema
    if not note_col and schema_cols and "note" in schema_cols:
        note_col = schema_cols["note"]
    
    # Get date-like columns for shifting
    date_cols = _get_date_like_columns(row, skip_cols)
    
    # Also get any columns that look like variations
    variation_cols = [col for col in row.keys() if col not in skip_cols and 
                      any(x in col.lower() for x in ["variation", "écart", "mouvement"])]
    
    # Build the ordered list of value columns (what should contain numbers)
    # Order: Note (empty/small int), Date1 (most recent), Date2 (older), Variation
    value_columns_order = []
    if note_col:
        value_columns_order.append(note_col)
    value_columns_order.extend(date_cols)
    value_columns_order.extend(variation_cols)
    
    # Now process misplacements
    for col, val, suggested_role in misplacements:
        if suggested_role == "note":
            # Check if this is a date column with a misplaced note value
            if col in date_cols and note_col:
                current_note = realigned.get(note_col, "")
                
                # Only shift if Note column is empty/dash
                if not current_note or current_note in ("", "-", "—"):
                    # Move the note value to Note column
                    realigned[note_col] = val
                    
                    # Find position in value columns
                    try:
                        col_idx = value_columns_order.index(col)
                    except ValueError:
                        col_idx = -1
                    
                    if col_idx > 0:  # Must be after Note column
                        # Shift values LEFT: each column gets value from the next
                        # Example: Note="" | 2022="4" | 2021="90M" | Var="83M"
                        # Becomes: Note="4" | 2022="90M" | 2021="83M" | Var=""
                        for i in range(col_idx, len(value_columns_order) - 1):
                            curr = value_columns_order[i]
                            next_ = value_columns_order[i + 1]
                            realigned[curr] = realigned.get(next_, "")
                        
                        # Last column becomes empty
                        realigned[value_columns_order[-1]] = ""
                        
                        corrections.append(f"Shifted row: '{val}' moved to Note, values shifted left")
                    else:
                        # Just clear the source column
                        realigned[col] = ""
                        corrections.append(f"Moved '{val}' from {col} to Note column")
    
    if corrections:
        realigned["_row_corrected"] = True
    
    return realigned, corrections
    
    return realigned, corrections


def realign_columns(
    rows: List[Dict],
    schema_cols: Dict[str, str] = None,
    skip_cols: set = None
) -> RealignmentResult:
    """
    Apply column realignment to all rows.
    
    Returns:
        RealignmentResult with realigned rows and statistics
    """
    if not rows:
        return RealignmentResult(
            realigned_rows=[],
            corrections_applied=0,
            column_shifts_detected=0,
            note_corrections=0,
            confidence_penalty=0.0,
            issues=[]
        )
    
    # Step 1: Detect column types
    column_types = detect_column_types(rows, skip_cols)
    
    # Log column type detection
    print("[REALIGNMENT] Column type detection:")
    for col, info in column_types.items():
        print(f"  {col}: {info.column_type} (median={info.median_magnitude:.0f}, conf={info.confidence:.2f})")
    
    # Step 2: Realign each row
    realigned_rows = []
    total_corrections = 0
    total_shifts = 0
    total_note_corrections = 0
    all_issues = []
    
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            realigned_rows.append(row)
            continue
        
        realigned, corrections = realign_row(row, column_types, schema_cols, skip_cols)
        realigned_rows.append(realigned)
        
        if corrections:
            total_corrections += len(corrections)
            for corr in corrections:
                if "Note column" in corr:
                    total_note_corrections += 1
                else:
                    total_shifts += 1
            all_issues.extend([f"Row {i}: {c}" for c in corrections])
    
    # Calculate confidence penalty
    total_rows = len([r for r in rows if isinstance(r, dict)])
    if total_rows > 0:
        correction_rate = total_corrections / total_rows
        confidence_penalty = min(0.5, correction_rate * 0.5)  # Max 50% penalty
    else:
        confidence_penalty = 0.0
    
    return RealignmentResult(
        realigned_rows=realigned_rows,
        corrections_applied=total_corrections,
        column_shifts_detected=total_shifts,
        note_corrections=total_note_corrections,
        confidence_penalty=confidence_penalty,
        issues=all_issues
    )


# =============================================================================
# VARIATION RECOMPUTATION
# =============================================================================

def recompute_variation(
    row: Dict,
    current_year_col: str,
    previous_year_col: str,
    variation_col: str = "Variation Montant"
) -> Tuple[Optional[float], bool]:
    """
    REBUILD VARIATION - Do NOT trust extracted variation.
    
    Compute: variation = current_year - previous_year
    
    Returns:
        (computed_variation, is_corrected)
    """
    current_val = _parse_numeric_value(str(row.get(current_year_col, "")))
    previous_val = _parse_numeric_value(str(row.get(previous_year_col, "")))
    
    if current_val is None or previous_val is None:
        return None, False
    
    computed = current_val - previous_val
    
    # Check extracted variation
    extracted_val = _parse_numeric_value(str(row.get(variation_col, "")))
    
    if extracted_val is not None:
        # Check if extracted matches computed (within 1% tolerance)
        if abs(computed) > 0:
            error_rate = abs(computed - extracted_val) / abs(computed)
            if error_rate > 0.01:  # More than 1% error
                return computed, True
    
    return computed, False


# =============================================================================
# SECTION-BASED VALIDATION
# =============================================================================

def validate_section_totals(
    rows: List[Dict],
    label_col: str,
    numeric_cols: List[str],
    tolerance: float = 0.01
) -> List[Dict]:
    """
    SECTION-BASED VALIDATION
    
    For each "total" row:
        - Sum ONLY rows within same section
        - Respect "Moins :" as negative
    
    Returns:
        List of validation issues with expected vs actual values
    """
    issues = []
    
    # Identify section boundaries and total rows
    current_section = []
    section_start_idx = 0
    
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        
        label = str(row.get(label_col, "")).strip().upper()
        row_type = str(row.get("type", "")).lower()
        
        # Check if this is a total row
        is_total = (
            row_type == "total" or
            "TOTAL" in label or
            label.startswith("TOTAL")
        )
        
        if is_total:
            # Validate this total against section sum
            for num_col in numeric_cols:
                expected_sum = 0.0
                
                for j in range(section_start_idx, i):
                    section_row = rows[j]
                    if not isinstance(section_row, dict):
                        continue
                    
                    section_label = str(section_row.get(label_col, "")).strip().upper()
                    val = _parse_numeric_value(str(section_row.get(num_col, "")))
                    
                    if val is not None:
                        # Check for "Moins :" prefix - treat as negative
                        if section_label.startswith("MOINS"):
                            val = -abs(val)
                        expected_sum += val
                
                # Compare with actual total
                actual_total = _parse_numeric_value(str(row.get(num_col, "")))
                
                if actual_total is not None and abs(expected_sum) > 0:
                    error = abs(actual_total - expected_sum)
                    error_rate = error / abs(expected_sum)
                    
                    if error_rate > tolerance:
                        issues.append({
                            "row_index": i,
                            "label": label,
                            "column": num_col,
                            "expected": expected_sum,
                            "actual": actual_total,
                            "error_rate": error_rate
                        })
            
            # Reset section
            current_section = []
            section_start_idx = i + 1
        else:
            current_section.append(row)
    
    return issues


# =============================================================================
# CASH FLOW DETECTION
# =============================================================================

def detect_cashflow_table(rows: List[Dict], label_col: str = "Label") -> bool:
    """
    CASH FLOW DETECTION
    
    If label contains "Flux", "Trésorerie", etc. → classify as cash_flow
    """
    keyword_count = 0
    total_rows = 0
    
    for row in rows:
        if not isinstance(row, dict):
            continue
        
        total_rows += 1
        label = str(row.get(label_col, "")).lower()
        
        for keyword in CASHFLOW_KEYWORDS:
            if keyword in label:
                keyword_count += 1
                break
    
    # If >20% of rows contain cash flow keywords, it's a cash flow table
    if total_rows > 0 and keyword_count / total_rows > 0.2:
        return True
    
    return False


# =============================================================================
# MAIN INTEGRATION FUNCTION
# =============================================================================

def run_column_realignment(
    data: Dict,
    verbose: bool = True
) -> Dict:
    """
    MAIN ENTRY POINT - Run full column realignment pipeline.
    
    Pipeline:
        1. Detect column types
        2. Find misaligned values (notes in numeric columns)
        3. Realign columns
        4. Recompute variations
        5. Validate section totals
        6. Adjust confidence
    
    Returns:
        Modified data dict with:
            - realigned rows
            - updated confidence
            - validation results
    """
    if not data:
        return data
    
    rows = data.get("rows", [])
    if not rows:
        return data
    
    if verbose:
        print("[COLUMN REALIGNMENT] Starting pipeline...")
    
    # Step 1: Column type detection
    column_types = detect_column_types(rows)
    
    if verbose:
        print("[COLUMN REALIGNMENT] Detected column types:")
        for col, info in column_types.items():
            print(f"  {col}: {info.column_type} (magnitude={info.median_magnitude:.0f})")
    
    # Find schema columns
    schema_cols = {}
    label_col = None
    note_col = None
    numeric_cols = []
    
    for col, info in column_types.items():
        if info.column_type == "label":
            label_col = col
            schema_cols["label"] = col
        elif info.column_type == "note":
            note_col = col
            schema_cols["note"] = col
        elif info.column_type == "numeric":
            numeric_cols.append(col)
    
    # Sort numeric columns by median magnitude (descending) - largest first
    numeric_cols_sorted = sorted(
        numeric_cols,
        key=lambda c: column_types[c].median_magnitude,
        reverse=True
    )
    
    # Step 2: Realign columns
    result = realign_columns(rows, schema_cols)
    
    if verbose:
        print(f"[COLUMN REALIGNMENT] Applied {result.corrections_applied} corrections")
        print(f"[COLUMN REALIGNMENT] Column shifts: {result.column_shifts_detected}")
        print(f"[COLUMN REALIGNMENT] Note corrections: {result.note_corrections}")
    
    # Update rows
    data["rows"] = result.realigned_rows
    
    # Step 3: Validate section totals
    if label_col and len(numeric_cols) >= 1:
        validation_issues = validate_section_totals(
            result.realigned_rows,
            label_col,
            numeric_cols[:2]  # Validate first 2 numeric columns
        )
        
        if verbose and validation_issues:
            print(f"[COLUMN REALIGNMENT] Found {len(validation_issues)} total mismatches")
            for issue in validation_issues[:3]:
                print(f"  {issue['label']}: expected={issue['expected']:.0f}, actual={issue['actual']:.0f}")
    else:
        validation_issues = []
    
    # Step 4: Cash flow detection
    if label_col and detect_cashflow_table(result.realigned_rows, label_col):
        if verbose:
            print("[COLUMN REALIGNMENT] Detected CASH FLOW table")
        data["table_type"] = "cash_flow"
    
    # Step 5: Adjust confidence
    current_confidence = data.get("confidence", 1.0)
    
    # Apply penalties
    confidence_penalty = result.confidence_penalty
    
    # Additional penalty for validation issues
    if validation_issues:
        confidence_penalty += min(0.3, len(validation_issues) * 0.1)
    
    new_confidence = max(0.0, current_confidence - confidence_penalty)
    
    # Mark as unreliable if confidence < 0.6
    if new_confidence < 0.6:
        data["unreliable"] = True
        if verbose:
            print(f"[COLUMN REALIGNMENT] Marked as UNRELIABLE (confidence={new_confidence:.2f})")
    
    data["confidence"] = new_confidence
    
    # Add metadata
    data["_realignment"] = {
        "corrections_applied": result.corrections_applied,
        "column_shifts": result.column_shifts_detected,
        "note_corrections": result.note_corrections,
        "validation_issues": len(validation_issues),
        "confidence_penalty": confidence_penalty,
        "issues": result.issues[:10]  # First 10 issues
    }
    
    if verbose:
        print(f"[COLUMN REALIGNMENT] Final confidence: {new_confidence:.2f}")
    
    return data

"""
Safe Pipeline Improvements - Incremental Fixes for Financial OCR
================================================================

This module provides SAFE, INCREMENTAL improvements to the financial OCR pipeline.

DESIGN PRINCIPLES:
    1. ALL changes are additive - no breaking changes
    2. Functions return metadata - original data untouched unless explicitly enabled
    3. Conservative thresholds - better to miss corrections than corrupt data
    4. Full audit trail - all decisions are logged

PHASES IMPLEMENTED:
    Phase 1: Enhanced Column Type Detection (semantic understanding)
    Phase 2: Safe Column Validation (detect corruption without modifying)
    Phase 3: Controlled Realignment (fix ONLY obvious cases)
    Phase 4: Improved Total Validation (section-aware)
    Phase 5: Meaningful Confidence Scoring
    Phase 6: Cash Flow Table Detection
    Phase 7: Safety/Reliability Flags

Author: Safe Refactor System
Version: 1.0
"""

import re
import statistics
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# PHASE 1: ENHANCED COLUMN TYPE DETECTION
# =============================================================================

class ColumnRole(Enum):
    """Semantic roles for columns in financial tables."""
    LABEL = "label"
    NOTE = "note"
    DATE_CURRENT = "date_current"
    DATE_PREVIOUS = "date_previous"
    DATE_OTHER = "date_other"
    VARIATION_AMOUNT = "variation_amount"
    VARIATION_PERCENT = "variation_percent"
    NUMERIC = "numeric"
    EMPTY = "empty"
    UNKNOWN = "unknown"


@dataclass
class EnhancedColumnInfo:
    """Enhanced column classification with confidence and reasoning."""
    column_name: str
    detected_role: ColumnRole
    confidence: float  # 0.0 to 1.0
    reasoning: str     # Human-readable explanation
    
    # Statistics
    total_values: int
    non_empty_values: int
    numeric_values: int
    text_values: int
    note_pattern_matches: int
    percent_values: int
    
    # Derived metrics
    numeric_ratio: float
    text_ratio: float
    median_magnitude: float
    is_date_header: bool


@dataclass 
class ColumnTypeResult:
    """Result of enhanced column type detection."""
    columns: Dict[str, EnhancedColumnInfo]
    detected_date_order: List[str]  # Chronologically ordered date columns
    has_variation_columns: bool
    schema_quality: float  # 0.0 to 1.0
    issues: List[str]


# -----------------------------------------------------------------------------
# Date Column Detection (Improved)
# -----------------------------------------------------------------------------

# Extended date patterns
_DATE_PATTERNS = [
    # Full dates: 31/12/2024, 31.12.2024, 31-12-2024
    re.compile(r'^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})$'),
    # Year only: 2024, 2023
    re.compile(r'^(20\d{2})$'),
    # Exercice format: "Exercice 2024", "Ex. 2024"
    re.compile(r'^(?:exercice|ex\.?)\s*(20\d{2})$', re.IGNORECASE),
    # Month-Year: "Dec 2024", "Décembre 2024"
    re.compile(r'^(?:jan|fév|mar|avr|mai|juin|juil|aoû|sep|oct|nov|déc|january|february|march|april|may|june|july|august|september|october|november|december)[a-zéû]*\.?\s*(20\d{2})$', re.IGNORECASE),
]

def _extract_year_from_column(col_name: str) -> Optional[int]:
    """Extract year from column name for chronological ordering."""
    if not col_name:
        return None
    
    col_clean = str(col_name).strip()
    
    # Try each pattern
    for pattern in _DATE_PATTERNS:
        match = pattern.match(col_clean)
        if match:
            groups = match.groups()
            # Find the 4-digit year group
            for g in groups:
                if g and len(g) == 4 and g.isdigit():
                    return int(g)
    
    # Fallback: look for any 4-digit year
    year_match = re.search(r'(20\d{2})', col_clean)
    if year_match:
        return int(year_match.group(1))
    
    return None


def _is_date_column_name_enhanced(col_name: str) -> Tuple[bool, Optional[int]]:
    """
    Enhanced date column detection.
    
    Returns:
        (is_date, extracted_year)
    """
    if not col_name:
        return False, None
    
    col_clean = str(col_name).strip()
    
    for pattern in _DATE_PATTERNS:
        if pattern.match(col_clean):
            year = _extract_year_from_column(col_clean)
            return True, year
    
    return False, None


# -----------------------------------------------------------------------------
# Note Pattern Detection (Enhanced)
# -----------------------------------------------------------------------------

# Extended note patterns (more comprehensive than original)
_ENHANCED_NOTE_PATTERNS = [
    # Parenthesized with separator: (4.1), (1-2), (III.1), (A.1)
    re.compile(r'^\(\s*(?:[IVXLCDM]{1,6}|[A-Za-z]?[\d]{1,3})[.\-][\d]{1,3}\s*\)$', re.IGNORECASE),
    # Single-digit footnote: (1) through (9)
    re.compile(r'^\(\s*[1-9]\s*\)$'),
    # Roman numeral prefix: III.1, IV.2
    re.compile(r'^[IVXLCDM]{2,6}\.[\d]{1,3}$', re.IGNORECASE),
    # Bare numeric dash/dot: 1-2, 1.2
    re.compile(r'^[\d]{1,3}[.\-][\d]{1,3}$'),
    # Double-digit in parens without separator (for note ranges): (10), (11)
    re.compile(r'^\(\s*[1-4][0-9]\s*\)$'),  # (10) through (49)
    # Asterisk notes
    re.compile(r'^\*+$'),
    # Dagger/cross notes
    re.compile(r'^[†‡§]+$'),
]

def _is_note_pattern_enhanced(val: str) -> bool:
    """Enhanced note pattern detection."""
    if not val:
        return False
    
    val = str(val).strip()
    if not val:
        return False
    
    for pattern in _ENHANCED_NOTE_PATTERNS:
        if pattern.match(val):
            return True
    
    return False


# -----------------------------------------------------------------------------
# Value Classification Helpers
# -----------------------------------------------------------------------------

def _parse_numeric_safe(val: str) -> Optional[float]:
    """
    Safely parse numeric value.
    Returns None if not a valid number.
    """
    if not val or not isinstance(val, str):
        return None
    
    val = val.strip()
    if not val or val in ("-", "–", "—", "N/A", "n/a", ""):
        return None
    
    # Skip percentages
    if "%" in val:
        return None
    
    # Check for negative (parens or angle brackets)
    is_negative = bool(re.search(r'[(<]', val))
    
    # Remove all non-digit characters except dots and commas
    cleaned = re.sub(r'[^\d.,]', '', val)
    
    if not cleaned:
        return None
    
    # Handle European format (1.234.567,89)
    euro_pattern = re.match(r'^(\d{1,3}(?:\.\d{3})+),(\d{1,2})$', cleaned)
    if euro_pattern:
        integer_part = euro_pattern.group(1).replace('.', '')
        decimal_part = euro_pattern.group(2)
        cleaned = integer_part + '.' + decimal_part
    else:
        # Standard handling
        euro_thousands = re.match(r'^(\d{1,3}(?:\.\d{3})+)$', cleaned)
        if euro_thousands:
            cleaned = cleaned.replace('.', '')
        else:
            # Comma as decimal if followed by 1-2 digits at end
            if re.search(r',\d{1,2}$', cleaned) and '.' not in cleaned:
                cleaned = cleaned[:-3].replace(',', '') + '.' + cleaned[-2:]
            else:
                cleaned = cleaned.replace(',', '')
    
    try:
        result = float(cleaned)
        return -abs(result) if is_negative else result
    except ValueError:
        return None


def _is_small_integer_safe(val: str, threshold: int = 50) -> bool:
    """
    Check if value is a small integer (potential note reference).
    
    SAFE VERSION: Rejects thousand-separated values like "10 892".
    """
    if not val:
        return False
    
    val = str(val).strip()
    
    # If contains space, NOT a small integer (likely thousand-separated)
    if ' ' in val:
        return False
    
    parsed = _parse_numeric_safe(val)
    if parsed is None:
        return False
    
    # Must be positive integer
    if parsed <= 0 or parsed != int(parsed):
        return False
    
    return parsed <= threshold


# -----------------------------------------------------------------------------
# PHASE 1 MAIN FUNCTION: detect_column_types_enhanced
# -----------------------------------------------------------------------------

def detect_column_types_enhanced(
    rows: List[Dict],
    columns: Optional[List[str]] = None,
    skip_cols: Optional[set] = None
) -> ColumnTypeResult:
    """
    PHASE 1: Enhanced Column Type Detection
    
    Provides semantic understanding of columns WITHOUT modifying data.
    Results are attached as metadata for downstream phases.
    
    Args:
        rows: List of row dictionaries
        columns: Optional explicit column list (otherwise inferred from rows)
        skip_cols: Columns to ignore (internal metadata columns)
    
    Returns:
        ColumnTypeResult with detailed column analysis
    """
    if skip_cols is None:
        skip_cols = {"type", "__chunk_index", "__y_position", "_row_corrected", 
                     "_alignment_corrected", "_alignment_errors"}
    
    issues = []
    
    # Collect all column names
    if columns:
        all_cols = [c for c in columns if c not in skip_cols]
    else:
        all_cols = set()
        for row in rows:
            if isinstance(row, dict):
                all_cols.update(row.keys())
        all_cols = list(all_cols - skip_cols)
    
    if not all_cols:
        return ColumnTypeResult(
            columns={},
            detected_date_order=[],
            has_variation_columns=False,
            schema_quality=0.0,
            issues=["No columns detected"]
        )
    
    column_infos: Dict[str, EnhancedColumnInfo] = {}
    date_columns_with_years: List[Tuple[str, Optional[int]]] = []
    variation_columns: List[str] = []
    
    for col in all_cols:
        # Gather values for this column
        values = []
        for row in rows:
            if isinstance(row, dict) and col in row:
                values.append(row[col])
        
        total_values = len(values)
        
        if total_values == 0:
            column_infos[col] = EnhancedColumnInfo(
                column_name=col,
                detected_role=ColumnRole.EMPTY,
                confidence=1.0,
                reasoning="No values found in column",
                total_values=0,
                non_empty_values=0,
                numeric_values=0,
                text_values=0,
                note_pattern_matches=0,
                percent_values=0,
                numeric_ratio=0.0,
                text_ratio=0.0,
                median_magnitude=0.0,
                is_date_header=False
            )
            continue
        
        # Classify each value
        numeric_vals = []
        text_count = 0
        empty_count = 0
        percent_count = 0
        note_pattern_count = 0
        small_int_count = 0
        
        for val in values:
            val_str = str(val).strip() if val else ""
            
            if not val_str or val_str in ("-", "–", "—"):
                empty_count += 1
                continue
            
            # Check percentage first (contains %)
            if "%" in val_str:
                percent_count += 1
                continue
            
            # Check explicit note pattern
            if _is_note_pattern_enhanced(val_str):
                note_pattern_count += 1
                continue
            
            # Check small integer (potential note)
            if _is_small_integer_safe(val_str):
                small_int_count += 1
                # Also count as numeric for statistics
                parsed = _parse_numeric_safe(val_str)
                if parsed is not None:
                    numeric_vals.append(abs(parsed))
                continue
            
            # Check numeric
            parsed = _parse_numeric_safe(val_str)
            if parsed is not None:
                numeric_vals.append(abs(parsed))
                continue
            
            # Text (has alphabetic characters)
            if any(c.isalpha() for c in val_str):
                text_count += 1
        
        non_empty = total_values - empty_count
        numeric_ratio = len(numeric_vals) / non_empty if non_empty > 0 else 0.0
        text_ratio = text_count / non_empty if non_empty > 0 else 0.0
        median_magnitude = statistics.median(numeric_vals) if numeric_vals else 0.0
        
        # Check if column header is a date
        is_date_header, year = _is_date_column_name_enhanced(col)
        
        # Determine role based on evidence
        role = ColumnRole.UNKNOWN
        confidence = 0.5
        reasoning = ""
        
        # Priority 1: Date column (by header name)
        if is_date_header:
            role = ColumnRole.DATE_OTHER  # Will be refined later
            confidence = 0.95
            reasoning = f"Column header matches date pattern (year: {year})"
            date_columns_with_years.append((col, year))
        
        # Priority 2: Variation columns (by header name)
        elif _is_variation_column_name(col):
            if "%" in col.lower() or "percent" in col.lower() or "pct" in col.lower():
                role = ColumnRole.VARIATION_PERCENT
                confidence = 0.9
                reasoning = "Column header indicates variation percentage"
            else:
                role = ColumnRole.VARIATION_AMOUNT
                confidence = 0.9
                reasoning = "Column header indicates variation amount"
            variation_columns.append(col)
        
        # Priority 3: Percentage column (>50% values contain %)
        elif percent_count > non_empty * 0.5:
            role = ColumnRole.VARIATION_PERCENT
            confidence = percent_count / non_empty if non_empty > 0 else 0.0
            reasoning = f"{percent_count}/{non_empty} values contain %"
            variation_columns.append(col)
        
        # Priority 4: Label column (>50% text, first text-heavy column)
        elif text_ratio > 0.5:
            role = ColumnRole.LABEL
            confidence = text_ratio
            reasoning = f"{text_count}/{non_empty} values are text"
        
        # Priority 5: Note column (note patterns OR mostly small integers with low magnitude)
        elif note_pattern_count > non_empty * 0.3:
            role = ColumnRole.NOTE
            confidence = note_pattern_count / non_empty if non_empty > 0 else 0.0
            reasoning = f"{note_pattern_count}/{non_empty} values match note patterns"
        
        elif small_int_count > non_empty * 0.5 and median_magnitude < 100:
            role = ColumnRole.NOTE
            confidence = 0.7
            reasoning = f"Mostly small integers (median={median_magnitude:.0f})"
        
        # Priority 6: Numeric column (>70% numeric with higher magnitude)
        elif numeric_ratio > 0.7:
            role = ColumnRole.NUMERIC
            confidence = numeric_ratio
            reasoning = f"{len(numeric_vals)}/{non_empty} values are numeric (median={median_magnitude:,.0f})"
        
        # Priority 7: Empty column
        elif empty_count > total_values * 0.8:
            role = ColumnRole.EMPTY
            confidence = 1.0
            reasoning = f"{empty_count}/{total_values} values are empty"
        
        else:
            role = ColumnRole.UNKNOWN
            confidence = 0.3
            reasoning = "Could not determine role with confidence"
            issues.append(f"Column '{col}' has unclear role")
        
        column_infos[col] = EnhancedColumnInfo(
            column_name=col,
            detected_role=role,
            confidence=confidence,
            reasoning=reasoning,
            total_values=total_values,
            non_empty_values=non_empty,
            numeric_values=len(numeric_vals),
            text_values=text_count,
            note_pattern_matches=note_pattern_count,
            percent_values=percent_count,
            numeric_ratio=numeric_ratio,
            text_ratio=text_ratio,
            median_magnitude=median_magnitude,
            is_date_header=is_date_header
        )
    
    # Order date columns chronologically (newest first)
    date_columns_with_years.sort(key=lambda x: x[1] if x[1] else 0, reverse=True)
    detected_date_order = [col for col, _ in date_columns_with_years]
    
    # Assign date roles (current/previous)
    if len(detected_date_order) >= 2:
        column_infos[detected_date_order[0]].detected_role = ColumnRole.DATE_CURRENT
        column_infos[detected_date_order[1]].detected_role = ColumnRole.DATE_PREVIOUS
    elif len(detected_date_order) == 1:
        column_infos[detected_date_order[0]].detected_role = ColumnRole.DATE_CURRENT
    
    # Calculate schema quality
    has_label = any(c.detected_role == ColumnRole.LABEL for c in column_infos.values())
    has_dates = len(detected_date_order) > 0
    avg_confidence = sum(c.confidence for c in column_infos.values()) / len(column_infos) if column_infos else 0.0
    
    schema_quality = (
        0.3 * (1.0 if has_label else 0.0) +
        0.3 * min(len(detected_date_order) / 2.0, 1.0) +
        0.2 * (1.0 if variation_columns else 0.0) +
        0.2 * avg_confidence
    )
    
    if not has_dates:
        issues.append("No date columns detected")
    if not has_label:
        issues.append("No label column detected")
    
    return ColumnTypeResult(
        columns=column_infos,
        detected_date_order=detected_date_order,
        has_variation_columns=len(variation_columns) > 0,
        schema_quality=schema_quality,
        issues=issues
    )


def _is_variation_column_name(col_name: str) -> bool:
    """Check if column name indicates variation/change."""
    if not col_name:
        return False
    col_lower = col_name.lower()
    variation_keywords = ['variation', 'var', 'écart', 'ecart', 'mouvement', 
                          'change', 'delta', 'diff', 'evolution']
    return any(kw in col_lower for kw in variation_keywords)


# =============================================================================
# PHASE 2: SAFE COLUMN VALIDATION
# =============================================================================

@dataclass
class ValidationWarning:
    """A detected issue that may indicate corruption."""
    severity: str  # "high", "medium", "low"
    row_index: int
    column: str
    value: str
    expected_type: str
    actual_type: str
    message: str


@dataclass
class ColumnValidationResult:
    """Result of safe column validation."""
    is_consistent: bool
    warnings: List[ValidationWarning]
    anomaly_count: int
    row_structure_issues: int
    confidence_penalty: float  # 0.0 to 1.0 reduction


def validate_column_consistency(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    threshold_ratio: float = 0.1  # Max 10% anomalies before flagging
) -> ColumnValidationResult:
    """
    PHASE 2: Safe Column Validation
    
    Detects obvious corruption WITHOUT modifying data.
    
    Checks:
        - Numeric values inside note columns
        - Note-like values inside numeric columns
        - Inconsistent row structure (missing required columns)
    
    Args:
        rows: Row data
        column_types: Result from Phase 1
        threshold_ratio: Maximum anomaly ratio before flagging inconsistency
    
    Returns:
        ColumnValidationResult with warnings (data not modified)
    """
    warnings: List[ValidationWarning] = []
    row_structure_issues = 0
    
    # Get expected columns by role
    note_cols = [name for name, info in column_types.columns.items() 
                 if info.detected_role == ColumnRole.NOTE]
    numeric_cols = [name for name, info in column_types.columns.items()
                    if info.detected_role in (ColumnRole.DATE_CURRENT, ColumnRole.DATE_PREVIOUS, 
                                               ColumnRole.DATE_OTHER, ColumnRole.NUMERIC,
                                               ColumnRole.VARIATION_AMOUNT)]
    
    # Calculate median magnitude for numeric columns (for anomaly detection)
    all_numeric_values = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for col in numeric_cols:
            val = _parse_numeric_safe(str(row.get(col, "")))
            if val is not None:
                all_numeric_values.append(abs(val))
    
    median_magnitude = statistics.median(all_numeric_values) if all_numeric_values else 0.0
    
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            row_structure_issues += 1
            continue
        
        row_type = str(row.get("type", "")).lower()
        
        # Skip section headers (they don't have numeric values)
        if row_type == "section":
            continue
        
        # Check note columns for misplaced large numbers
        for col in note_cols:
            val_str = str(row.get(col, "")).strip()
            if not val_str or val_str in ("-", "–", "—"):
                continue
            
            parsed = _parse_numeric_safe(val_str)
            if parsed is not None and abs(parsed) > 100:
                # Large number in note column = suspicious
                warnings.append(ValidationWarning(
                    severity="high",
                    row_index=i,
                    column=col,
                    value=val_str,
                    expected_type="note",
                    actual_type="large_number",
                    message=f"Large number ({parsed:,.0f}) found in note column"
                ))
        
        # Check numeric columns for misplaced notes
        for col in numeric_cols:
            val_str = str(row.get(col, "")).strip()
            if not val_str or val_str in ("-", "–", "—"):
                continue
            
            # Skip percentages in percent columns
            if "%" in val_str:
                continue
            
            # Check for note patterns
            if _is_note_pattern_enhanced(val_str):
                warnings.append(ValidationWarning(
                    severity="high",
                    row_index=i,
                    column=col,
                    value=val_str,
                    expected_type="numeric",
                    actual_type="note_pattern",
                    message=f"Note pattern '{val_str}' found in numeric column"
                ))
                continue
            
            # Check for small integers that might be misplaced notes
            # CONSERVATIVE: Only flag if median is very large (>10,000)
            parsed = _parse_numeric_safe(val_str)
            if parsed is not None and median_magnitude > 10000:
                if abs(parsed) < 50 and abs(parsed) / median_magnitude < 0.0001:
                    warnings.append(ValidationWarning(
                        severity="medium",
                        row_index=i,
                        column=col,
                        value=val_str,
                        expected_type="numeric",
                        actual_type="possible_note",
                        message=f"Small integer ({parsed:.0f}) in column with median {median_magnitude:,.0f}"
                    ))
        
        # Check row structure (all expected columns present)
        missing_cols = [col for col in column_types.columns.keys() 
                        if col not in row and col not in ("type",)]
        if missing_cols:
            row_structure_issues += 1
    
    # Calculate metrics
    total_cells = len(rows) * len(column_types.columns)
    anomaly_count = len(warnings)
    anomaly_ratio = anomaly_count / total_cells if total_cells > 0 else 0.0
    
    is_consistent = anomaly_ratio <= threshold_ratio
    
    # Calculate confidence penalty
    high_severity_count = sum(1 for w in warnings if w.severity == "high")
    confidence_penalty = min(0.5, (high_severity_count * 0.05) + (anomaly_ratio * 0.3))
    
    return ColumnValidationResult(
        is_consistent=is_consistent,
        warnings=warnings,
        anomaly_count=anomaly_count,
        row_structure_issues=row_structure_issues,
        confidence_penalty=confidence_penalty
    )


# =============================================================================
# PHASE 3: CONTROLLED REALIGNMENT (SAFE MODE)
# =============================================================================

@dataclass
class RealignmentCorrection:
    """Record of a single realignment correction."""
    row_index: int
    source_column: str
    target_column: str
    value: str
    reason: str
    confidence: float


@dataclass
class SafeRealignmentResult:
    """Result of controlled realignment."""
    corrected_rows: List[Dict]
    corrections: List[RealignmentCorrection]
    skipped_ambiguous: int
    confidence_after: float


def controlled_realignment(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    validation_result: ColumnValidationResult,
    strict_mode: bool = True
) -> SafeRealignmentResult:
    """
    PHASE 3: Controlled Realignment
    
    Fixes ONLY obvious misalignment cases where ALL conditions are met:
        1. Value < 100 AND neighboring values > 10,000
        2. Column types clearly mismatch
        3. High confidence anomaly from validation
    
    Args:
        rows: Row data
        column_types: Result from Phase 1
        validation_result: Result from Phase 2
        strict_mode: If True, only fix highest-confidence cases
    
    Returns:
        SafeRealignmentResult with corrected rows (original rows not modified)
    """
    # Deep copy rows to avoid modifying original
    import copy
    corrected_rows = copy.deepcopy(rows)
    corrections: List[RealignmentCorrection] = []
    skipped_ambiguous = 0
    
    # Get note column (target for misplaced notes)
    note_cols = [name for name, info in column_types.columns.items() 
                 if info.detected_role == ColumnRole.NOTE]
    note_col = note_cols[0] if note_cols else None
    
    # Process only high-severity warnings with note patterns
    for warning in validation_result.warnings:
        if warning.severity != "high":
            if strict_mode:
                skipped_ambiguous += 1
                continue
        
        row_idx = warning.row_index
        if row_idx >= len(corrected_rows):
            continue
        
        row = corrected_rows[row_idx]
        if not isinstance(row, dict):
            continue
        
        # ONLY fix note-pattern-in-numeric cases
        if warning.actual_type == "note_pattern" and note_col:
            source_col = warning.column
            value = warning.value
            
            # Additional safety check: Note column should be empty or "-"
            current_note = str(row.get(note_col, "")).strip()
            if current_note and current_note not in ("", "-", "–", "—"):
                # Note column already has a value - DON'T overwrite
                skipped_ambiguous += 1
                continue
            
            # Calculate confidence for this correction
            source_info = column_types.columns.get(source_col)
            if not source_info:
                skipped_ambiguous += 1
                continue
            
            # Require high median magnitude in source column
            if source_info.median_magnitude < 10000:
                skipped_ambiguous += 1
                continue
            
            correction_confidence = min(
                source_info.confidence,
                0.9 if _is_note_pattern_enhanced(value) else 0.5
            )
            
            if strict_mode and correction_confidence < 0.7:
                skipped_ambiguous += 1
                continue
            
            # Apply correction
            row[note_col] = value
            row[source_col] = ""  # Clear the source
            
            corrections.append(RealignmentCorrection(
                row_index=row_idx,
                source_column=source_col,
                target_column=note_col,
                value=value,
                reason=f"Note pattern moved from numeric column (median={source_info.median_magnitude:,.0f})",
                confidence=correction_confidence
            ))
    
    # Calculate final confidence
    if corrections:
        avg_correction_confidence = sum(c.confidence for c in corrections) / len(corrections)
        confidence_after = 1.0 - validation_result.confidence_penalty + (avg_correction_confidence * 0.1)
    else:
        confidence_after = 1.0 - validation_result.confidence_penalty
    
    return SafeRealignmentResult(
        corrected_rows=corrected_rows,
        corrections=corrections,
        skipped_ambiguous=skipped_ambiguous,
        confidence_after=min(1.0, max(0.0, confidence_after))
    )


# =============================================================================
# PHASE 4: IMPROVED TOTAL VALIDATION
# =============================================================================

@dataclass
class SectionInfo:
    """Information about a detected section."""
    start_row: int
    end_row: int  # Exclusive
    section_name: str
    total_row: Optional[int]
    data_rows: List[int]
    is_subtotal: bool


@dataclass
class TotalValidationResult:
    """Result of improved total validation."""
    sections: List[SectionInfo]
    validation_passed: bool
    total_errors: List[Dict]
    total_warnings: List[str]


def detect_sections_improved(rows: List[Dict]) -> List[SectionInfo]:
    """
    Detect sections in financial tables.
    
    Handles:
        - Section headers (type="section")
        - Total rows (type="total" or label contains "TOTAL")
        - Sub-totals (intermediate totals)
        - "Moins" (negative) rows
    """
    sections: List[SectionInfo] = []
    current_section_start = 0
    current_section_name = ""
    current_data_rows = []
    
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        
        row_type = str(row.get("type", "")).lower()
        label = str(row.get("Label", "")).upper()
        
        # Detect section headers
        if row_type == "section":
            # Close previous section if exists
            if current_data_rows or current_section_name:
                sections.append(SectionInfo(
                    start_row=current_section_start,
                    end_row=i,
                    section_name=current_section_name,
                    total_row=None,
                    data_rows=current_data_rows,
                    is_subtotal=False
                ))
            
            current_section_start = i
            current_section_name = str(row.get("Label", ""))
            current_data_rows = []
        
        # Detect total rows
        elif row_type == "total" or "TOTAL" in label:
            is_subtotal = "SOUS" in label or "SUB" in label
            
            sections.append(SectionInfo(
                start_row=current_section_start,
                end_row=i + 1,
                section_name=current_section_name,
                total_row=i,
                data_rows=current_data_rows.copy(),
                is_subtotal=is_subtotal
            ))
            
            current_section_start = i + 1
            current_data_rows = []
        
        # Track data rows
        elif row_type == "data" or row_type == "":
            current_data_rows.append(i)
    
    # Close final section
    if current_data_rows:
        sections.append(SectionInfo(
            start_row=current_section_start,
            end_row=len(rows),
            section_name=current_section_name,
            total_row=None,
            data_rows=current_data_rows,
            is_subtotal=False
        ))
    
    return sections


def validate_totals_improved(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    tolerance: float = 2.0
) -> TotalValidationResult:
    """
    PHASE 4: Improved Total Validation
    
    Validates totals by:
        1. Detecting sections properly
        2. Summing child rows within same section
        3. Handling "Moins" (negative) rows
        4. Supporting multiple periods
    
    Args:
        rows: Row data
        column_types: Result from Phase 1
        tolerance: Allowed difference for total validation
    
    Returns:
        TotalValidationResult (data not modified, only validated)
    """
    sections = detect_sections_improved(rows)
    total_errors: List[Dict] = []
    total_warnings: List[str] = []
    
    # Get numeric columns to validate
    numeric_cols = [name for name, info in column_types.columns.items()
                    if info.detected_role in (ColumnRole.DATE_CURRENT, ColumnRole.DATE_PREVIOUS,
                                               ColumnRole.DATE_OTHER, ColumnRole.VARIATION_AMOUNT)]
    
    for section in sections:
        if section.total_row is None:
            continue
        
        if not section.data_rows:
            total_warnings.append(f"Section '{section.section_name}' has total but no data rows")
            continue
        
        total_row = rows[section.total_row]
        
        for col in numeric_cols:
            expected_sum = 0.0
            has_values = False
            
            for data_row_idx in section.data_rows:
                data_row = rows[data_row_idx]
                label = str(data_row.get("Label", "")).upper()
                
                val = _parse_numeric_safe(str(data_row.get(col, "")))
                if val is not None:
                    # Handle "Moins" rows (should be subtracted)
                    if "MOINS" in label or "LESS" in label:
                        expected_sum -= abs(val)
                    else:
                        expected_sum += val
                    has_values = True
            
            if not has_values:
                continue
            
            actual_total = _parse_numeric_safe(str(total_row.get(col, "")))
            
            if actual_total is None:
                total_errors.append({
                    "section": section.section_name,
                    "column": col,
                    "expected": expected_sum,
                    "actual": None,
                    "error": "Missing total value"
                })
            elif abs(expected_sum - actual_total) > tolerance:
                total_errors.append({
                    "section": section.section_name,
                    "column": col,
                    "expected": expected_sum,
                    "actual": actual_total,
                    "difference": abs(expected_sum - actual_total),
                    "error": "Total mismatch"
                })
    
    return TotalValidationResult(
        sections=sections,
        validation_passed=len(total_errors) == 0,
        total_errors=total_errors,
        total_warnings=total_warnings
    )


# =============================================================================
# PHASE 5: MEANINGFUL CONFIDENCE SCORING
# =============================================================================

@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of confidence score components."""
    schema_validity: float      # 0.0 to 1.0 (30%)
    column_consistency: float   # 0.0 to 1.0 (30%)
    validation_pass_rate: float # 0.0 to 1.0 (20%)
    coverage: float             # 0.0 to 1.0 (20%)
    
    # Penalties
    misalignment_penalty: float
    validation_error_penalty: float
    
    final_score: float


def compute_meaningful_confidence(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    validation_result: ColumnValidationResult,
    total_validation: TotalValidationResult
) -> ConfidenceBreakdown:
    """
    PHASE 5: Meaningful Confidence Scoring
    
    Computes confidence based on actual correctness indicators:
        - schema_validity (30%): Column detection confidence
        - column_consistency (30%): No type mismatches
        - validation_pass_rate (20%): Total validation success
        - coverage (20%): Non-empty rows with all required columns
    
    Penalties applied for:
        - Detected misalignment (reduces score)
        - Validation errors (reduces score)
    """
    # Schema validity (30%)
    schema_validity = column_types.schema_quality
    
    # Column consistency (30%)
    column_consistency = 1.0 - validation_result.confidence_penalty
    
    # Validation pass rate (20%)
    if total_validation.sections:
        sections_with_totals = sum(1 for s in total_validation.sections if s.total_row is not None)
        if sections_with_totals > 0:
            validation_pass_rate = 1.0 - (len(total_validation.total_errors) / (sections_with_totals * len(column_types.detected_date_order) or 1))
        else:
            validation_pass_rate = 1.0  # No totals to validate
    else:
        validation_pass_rate = 0.5  # Uncertain
    validation_pass_rate = max(0.0, validation_pass_rate)
    
    # Coverage (20%)
    total_rows = len(rows)
    data_rows = sum(1 for r in rows if isinstance(r, dict) and str(r.get("type", "")).lower() not in ("section",))
    
    non_empty_labels = sum(1 for r in rows if isinstance(r, dict) and str(r.get("Label", "")).strip())
    coverage = non_empty_labels / total_rows if total_rows > 0 else 0.0
    
    # Calculate penalties
    misalignment_penalty = validation_result.confidence_penalty
    validation_error_penalty = min(0.3, len(total_validation.total_errors) * 0.05)
    
    # Compute final score
    base_score = (
        0.30 * schema_validity +
        0.30 * column_consistency +
        0.20 * validation_pass_rate +
        0.20 * coverage
    )
    
    final_score = max(0.0, base_score - misalignment_penalty - validation_error_penalty)
    
    return ConfidenceBreakdown(
        schema_validity=schema_validity,
        column_consistency=column_consistency,
        validation_pass_rate=validation_pass_rate,
        coverage=coverage,
        misalignment_penalty=misalignment_penalty,
        validation_error_penalty=validation_error_penalty,
        final_score=final_score
    )


# =============================================================================
# PHASE 6: CASH FLOW TABLE DETECTION
# =============================================================================

CASHFLOW_KEYWORDS = [
    "flux", "trésorerie", "tresorerie", "encaissements", "décaissements",
    "decaissements", "cash", "flow", "liquidités", "liquidites",
    "operating activities", "investing activities", "financing activities",
    "activités d'exploitation", "activités d'investissement", "activités de financement"
]

BALANCE_SHEET_KEYWORDS = [
    "actif", "passif", "bilan", "balance", "asset", "liability",
    "capitaux propres", "equity", "immobilisations", "fixed assets"
]

INCOME_STATEMENT_KEYWORDS = [
    "résultat", "resultat", "income", "profit", "loss", "revenue",
    "produits", "charges", "expense", "operating income"
]


def detect_table_type_enhanced(rows: List[Dict], title: Optional[str] = None) -> Dict:
    """
    PHASE 6: Enhanced Table Type Detection
    
    Detects:
        - balance_sheet
        - income_statement
        - cash_flow
        - off_balance (hors bilan)
        - unknown
    """
    # Collect all text from labels
    all_labels = []
    for row in rows:
        if isinstance(row, dict):
            label = str(row.get("Label", "")).lower()
            if label:
                all_labels.append(label)
    
    combined_text = " ".join(all_labels)
    if title:
        combined_text = title.lower() + " " + combined_text
    
    # Score each type
    scores = {
        "cash_flow": sum(1 for kw in CASHFLOW_KEYWORDS if kw in combined_text),
        "balance_sheet": sum(1 for kw in BALANCE_SHEET_KEYWORDS if kw in combined_text),
        "income_statement": sum(1 for kw in INCOME_STATEMENT_KEYWORDS if kw in combined_text),
    }
    
    # Check for off-balance sheet
    if "hors bilan" in combined_text or "engagement" in combined_text:
        return {
            "table_type": "off_balance",
            "confidence": 0.9,
            "scores": scores
        }
    
    # Return highest scoring type (minimum 2 matches)
    best_type = max(scores, key=scores.get)
    if scores[best_type] >= 2:
        return {
            "table_type": best_type,
            "confidence": min(0.95, 0.5 + scores[best_type] * 0.1),
            "scores": scores
        }
    
    return {
        "table_type": "unknown",
        "confidence": 0.0,
        "scores": scores
    }


# =============================================================================
# PHASE 7: SAFETY/RELIABILITY FLAGS
# =============================================================================

@dataclass
class ReliabilityAssessment:
    """Overall reliability assessment of extraction."""
    is_reliable: bool
    unreliable_reasons: List[str]
    confidence_score: float
    recommended_action: str  # "accept", "review", "reject"


def assess_reliability(
    column_types: ColumnTypeResult,
    validation_result: ColumnValidationResult,
    total_validation: TotalValidationResult,
    confidence: ConfidenceBreakdown
) -> ReliabilityAssessment:
    """
    PHASE 7: Safety Layer
    
    Determines if extraction should be flagged as unreliable.
    
    Flags unreliable if:
        - Severe misalignment (>5 high-severity warnings)
        - Validation failure (>2 total mismatches)
        - Missing key columns (no dates or no label)
        - Very low confidence (<0.3)
    """
    unreliable_reasons = []
    
    # Check for severe misalignment
    high_severity_warnings = sum(1 for w in validation_result.warnings if w.severity == "high")
    if high_severity_warnings > 5:
        unreliable_reasons.append(f"Severe misalignment: {high_severity_warnings} high-severity anomalies")
    
    # Check for validation failures
    if len(total_validation.total_errors) > 2:
        unreliable_reasons.append(f"Validation failed: {len(total_validation.total_errors)} total mismatches")
    
    # Check for missing key columns
    if not column_types.detected_date_order:
        unreliable_reasons.append("Missing date columns")
    
    has_label = any(info.detected_role == ColumnRole.LABEL for info in column_types.columns.values())
    if not has_label:
        unreliable_reasons.append("Missing label column")
    
    # Check confidence threshold
    if confidence.final_score < 0.3:
        unreliable_reasons.append(f"Very low confidence: {confidence.final_score:.2f}")
    
    # Determine reliability and recommended action
    is_reliable = len(unreliable_reasons) == 0
    
    if confidence.final_score >= 0.7 and is_reliable:
        recommended_action = "accept"
    elif confidence.final_score >= 0.4 or len(unreliable_reasons) <= 1:
        recommended_action = "review"
    else:
        recommended_action = "reject"
    
    return ReliabilityAssessment(
        is_reliable=is_reliable,
        unreliable_reasons=unreliable_reasons,
        confidence_score=confidence.final_score,
        recommended_action=recommended_action
    )


# =============================================================================
# UNIFIED PIPELINE ENTRY POINT
# =============================================================================

@dataclass
class SafePipelineResult:
    """Complete result from safe pipeline improvements."""
    # Phase 1
    column_types: ColumnTypeResult
    
    # Phase 2
    validation_result: ColumnValidationResult
    
    # Phase 3
    realignment_result: Optional[SafeRealignmentResult]
    
    # Phase 4
    total_validation: TotalValidationResult
    
    # Phase 5
    confidence: ConfidenceBreakdown
    
    # Phase 6
    table_type: Dict
    
    # Phase 7
    reliability: ReliabilityAssessment
    
    # Final output
    corrected_rows: List[Dict]
    metadata: Dict


def run_safe_pipeline(
    data: Dict,
    apply_corrections: bool = True,
    strict_mode: bool = True
) -> SafePipelineResult:
    """
    Run the complete safe pipeline improvements.
    
    This is the main entry point that runs all phases in order.
    
    Args:
        data: Extracted table data {"columns": [...], "rows": [...]}
        apply_corrections: If True, apply Phase 3 corrections
        strict_mode: If True, only apply highest-confidence corrections
    
    Returns:
        SafePipelineResult with all phase results
    """
    columns = data.get("columns", [])
    rows = data.get("rows", [])
    title = data.get("table_name", data.get("title", None))
    
    # Phase 1: Column Type Detection
    column_types = detect_column_types_enhanced(rows, columns)
    
    # Phase 2: Safe Validation
    validation_result = validate_column_consistency(rows, column_types)
    
    # Phase 3: Controlled Realignment (optional)
    if apply_corrections:
        realignment_result = controlled_realignment(
            rows, column_types, validation_result, strict_mode
        )
        corrected_rows = realignment_result.corrected_rows
    else:
        realignment_result = None
        corrected_rows = rows
    
    # Phase 4: Total Validation
    total_validation = validate_totals_improved(corrected_rows, column_types)
    
    # Phase 5: Confidence Scoring
    confidence = compute_meaningful_confidence(
        corrected_rows, column_types, validation_result, total_validation
    )
    
    # Phase 6: Table Type Detection
    table_type = detect_table_type_enhanced(corrected_rows, title)
    
    # Phase 7: Reliability Assessment
    reliability = assess_reliability(
        column_types, validation_result, total_validation, confidence
    )
    
    # Build metadata
    metadata = {
        "_safe_pipeline_version": "1.0",
        "_confidence_score": confidence.final_score,
        "_confidence_breakdown": {
            "schema_validity": confidence.schema_validity,
            "column_consistency": confidence.column_consistency,
            "validation_pass_rate": confidence.validation_pass_rate,
            "coverage": confidence.coverage
        },
        "_column_types": {
            name: {
                "role": info.detected_role.value,
                "confidence": info.confidence,
                "reasoning": info.reasoning
            }
            for name, info in column_types.columns.items()
        },
        "_detected_date_order": column_types.detected_date_order,
        "_table_type": table_type["table_type"],
        "_is_reliable": reliability.is_reliable,
        "_unreliable_reasons": reliability.unreliable_reasons,
        "_recommended_action": reliability.recommended_action,
        "_corrections_applied": len(realignment_result.corrections) if realignment_result else 0,
        "_validation_warnings": len(validation_result.warnings),
        "_total_errors": len(total_validation.total_errors)
    }
    
    return SafePipelineResult(
        column_types=column_types,
        validation_result=validation_result,
        realignment_result=realignment_result,
        total_validation=total_validation,
        confidence=confidence,
        table_type=table_type,
        reliability=reliability,
        corrected_rows=corrected_rows,
        metadata=metadata
    )


# =============================================================================
# INTEGRATION HELPER - Attach to existing pipeline
# =============================================================================

def enhance_extraction_result(data: Dict, apply_corrections: bool = True) -> Dict:
    """
    Integration helper: Run safe pipeline and attach metadata to existing result.
    
    This function can be called at the end of the existing pipeline to add
    the new improvements without breaking existing functionality.
    
    Args:
        data: Existing extraction result {"columns": [...], "rows": [...]}
        apply_corrections: If True, replace rows with corrected version
    
    Returns:
        Enhanced data dict with metadata and optionally corrected rows
    """
    if not isinstance(data, dict) or "rows" not in data:
        return data
    
    try:
        result = run_safe_pipeline(data, apply_corrections=apply_corrections)
        
        # Attach metadata
        data["_safe_pipeline"] = result.metadata
        
        # Optionally replace rows
        if apply_corrections and result.realignment_result:
            data["rows"] = result.corrected_rows
            data["_corrections_log"] = [
                {
                    "row": c.row_index,
                    "from": c.source_column,
                    "to": c.target_column,
                    "value": c.value,
                    "reason": c.reason
                }
                for c in result.realignment_result.corrections
            ]
        
        # Add reliability flag at top level for easy access
        data["_confidence"] = result.confidence.final_score
        data["_is_reliable"] = result.reliability.is_reliable
        data["_recommended_action"] = result.reliability.recommended_action
        
        if not result.reliability.is_reliable:
            data["_unreliable"] = True
            data["_unreliable_reasons"] = result.reliability.unreliable_reasons
        
        return data
    
    except Exception as e:
        # SAFETY: Never break existing pipeline
        data["_safe_pipeline_error"] = str(e)
        return data

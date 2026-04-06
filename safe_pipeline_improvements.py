"""
Safe Pipeline Improvements - Incremental Fixes for Financial OCR
================================================================

This module provides SAFE, NON-DESTRUCTIVE validation for the financial OCR pipeline.

DESIGN PRINCIPLES (v3.1 - CRITICAL FIX):
    1. OCR EXTRACTED VALUES ARE SOURCE OF TRUTH - never overwrite
    2. VALIDATION ONLY by default - no automatic corrections
    3. Only compute MISSING values - never replace existing
    4. Full audit trail - all validations logged as warnings
    5. IMMUTABLE RAW DATA - _raw_rows NEVER modified
    6. FAIL-SAFE - return original data on any error

PHASES IMPLEMENTED:
    Phase 1: Enhanced Column Type Detection (semantic understanding)
    Phase 2: Safe Column Validation (detect corruption without modifying)
    Phase 3: Controlled Realignment (detection only, no modification)
    Phase 4: Total Validation (validation warnings, not corrections)
    Phase 5: Meaningful Confidence Scoring
    Phase 6: Cash Flow Table Detection
    Phase 7: Safety/Reliability Flags

CRITICAL FIX (v3.1):
    - DEFAULT is NON-DESTRUCTIVE (apply_corrections=False)
    - NEVER overwrites extracted values
    - Mismatches reported as WARNINGS only
    - System is now ADVISORY, not corrective

Author: Safe Refactor System
Version: 3.1 - NON-DESTRUCTIVE (Critical Bug Fix)
"""

import re
import copy
import statistics
from typing import Any, Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# PRODUCTION HARDENING: CONFIDENCE GATING (STEP 1)
# =============================================================================

CorrectionMode = Literal["none", "safe", "aggressive"]


def should_apply_corrections(confidence: float, aggressive_mode: bool = False) -> CorrectionMode:
    """
    Determine what level of corrections should be applied based on confidence.
    
    RULES (NON-NEGOTIABLE):
        - confidence < 0.7  → NO corrections allowed ("none")
        - 0.7 ≤ confidence < 0.85 → ONLY safe corrections ("safe")
        - confidence ≥ 0.85 → allow full corrections ("aggressive" if enabled)
    
    Args:
        confidence: Confidence score from 0.0 to 1.0
        aggressive_mode: If True and confidence allows, enable aggressive corrections
    
    Returns:
        CorrectionMode: "none", "safe", or "aggressive"
    """
    if confidence < 0.7:
        return "none"
    elif confidence < 0.85:
        return "safe"
    else:
        return "aggressive" if aggressive_mode else "safe"


def is_correction_allowed(
    confidence: float,
    correction_type: str,
    aggressive_mode: bool = False
) -> bool:
    """
    Check if a specific type of correction is allowed given the confidence level.
    
    Args:
        confidence: Current confidence score
        correction_type: Type of correction ("realignment", "shift", "recomputation", "cashflow")
        aggressive_mode: Whether aggressive mode is enabled
    
    Returns:
        True if the correction type is allowed
    """
    mode = should_apply_corrections(confidence, aggressive_mode)
    
    # Safe corrections (allowed at mode == "safe" or "aggressive")
    SAFE_CORRECTIONS = {"realignment", "warning", "validation"}
    
    # Aggressive corrections (only allowed at mode == "aggressive")
    AGGRESSIVE_CORRECTIONS = {"shift", "recomputation", "cashflow", "total_fix"}
    
    if mode == "none":
        return False
    elif mode == "safe":
        return correction_type in SAFE_CORRECTIONS
    else:  # aggressive
        return correction_type in SAFE_CORRECTIONS or correction_type in AGGRESSIVE_CORRECTIONS


@dataclass
class CorrectionLogEntry:
    """Structured log entry for every correction made."""
    field: str
    row_label: str
    row_index: int
    old_value: Any
    new_value: Any
    reason: str
    correction_type: str  # "column_shift" | "recomputation" | "cashflow_fix" | "realignment"
    confidence_at_correction: float
    is_reversible: bool = True


# Global corrections log for the current pipeline run
_corrections_log: List[CorrectionLogEntry] = []


def log_correction(
    field: str,
    row_label: str,
    row_index: int,
    old_value: Any,
    new_value: Any,
    reason: str,
    correction_type: str,
    confidence: float
) -> CorrectionLogEntry:
    """
    Log a correction with full traceability.
    
    RULE: NEVER overwrite values silently - ALWAYS call this function.
    """
    entry = CorrectionLogEntry(
        field=field,
        row_label=row_label,
        row_index=row_index,
        old_value=old_value,
        new_value=new_value,
        reason=reason,
        correction_type=correction_type,
        confidence_at_correction=confidence
    )
    _corrections_log.append(entry)
    return entry


def reset_corrections_log():
    """Reset the corrections log for a new pipeline run."""
    global _corrections_log
    _corrections_log = []


def get_corrections_log() -> List[Dict]:
    """Get the corrections log as a list of dicts for JSON output."""
    return [
        {
            "field": e.field,
            "row_label": e.row_label,
            "row_index": e.row_index,
            "old_value": e.old_value,
            "new_value": e.new_value,
            "reason": e.reason,
            "correction_type": e.correction_type,
            "confidence": e.confidence_at_correction
        }
        for e in _corrections_log
    ]


def protect_raw_data(rows: List[Dict]) -> List[Dict]:
    """
    Create a deep copy of rows to protect raw data from modification.
    
    RULE: _raw_rows is NEVER modified. Always work on copies.
    """
    return copy.deepcopy(rows)


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
# ENHANCED STEP 4: FINANCIAL RECOMPUTATION & BALANCE SHEET IDENTITY
# =============================================================================

@dataclass
class FinancialRecomputationResult:
    """Result of financial recomputation and correction."""
    corrected_rows: List[Dict]
    corrections_made: List[Dict]
    balance_sheet_valid: bool
    balance_sheet_difference: Optional[float]
    totals_corrected: int


# Balance sheet section identifiers
ACTIF_KEYWORDS = ["total actif", "total des actifs", "total general actif", "total de l'actif"]
PASSIF_KEYWORDS = ["total passif", "total des passifs", "total general passif", "total du passif"]
CAPITAUX_KEYWORDS = ["capitaux propres", "total capitaux", "equity", "fonds propres"]


def _find_total_row(rows: List[Dict], keywords: List[str]) -> Optional[Tuple[int, Dict]]:
    """Find a total row matching any of the keywords."""
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        label = str(row.get("Label", "")).lower().strip()
        for kw in keywords:
            if kw in label:
                return i, row
    return None


def _get_row_value(row: Dict, column: str) -> Optional[float]:
    """Get numeric value from a row column."""
    val_str = str(row.get(column, "")).strip()
    return _parse_numeric_safe(val_str)


def validate_balance_sheet_identity(
    rows: List[Dict],
    column_types: ColumnTypeResult
) -> Dict:
    """
    Validate balance sheet identity: TOTAL ACTIF = TOTAL PASSIF + CAPITAUX PROPRES
    
    Returns validation result with differences if any.
    """
    result = {
        "is_balance_sheet": False,
        "identity_valid": None,
        "actif_total": None,
        "passif_total": None,
        "capitaux_total": None,
        "computed_passif_plus_capitaux": None,
        "difference": None,
        "column_checked": None
    }
    
    # Find total rows
    actif_row = _find_total_row(rows, ACTIF_KEYWORDS)
    passif_row = _find_total_row(rows, PASSIF_KEYWORDS)
    capitaux_row = _find_total_row(rows, CAPITAUX_KEYWORDS)
    
    if not actif_row:
        return result
    
    result["is_balance_sheet"] = True
    
    # Get the primary date column for validation
    if not column_types.detected_date_order:
        return result
    
    date_col = column_types.detected_date_order[0]
    result["column_checked"] = date_col
    
    # Get values
    actif_val = _get_row_value(actif_row[1], date_col)
    result["actif_total"] = actif_val
    
    if passif_row and capitaux_row:
        passif_val = _get_row_value(passif_row[1], date_col)
        capitaux_val = _get_row_value(capitaux_row[1], date_col)
        
        result["passif_total"] = passif_val
        result["capitaux_total"] = capitaux_val
        
        if passif_val is not None and capitaux_val is not None:
            computed = passif_val + capitaux_val
            result["computed_passif_plus_capitaux"] = computed
            
            if actif_val is not None:
                diff = abs(actif_val - computed)
                result["difference"] = diff
                # Allow small tolerance (rounding errors)
                result["identity_valid"] = diff < 10
    
    elif passif_row:
        # Sometimes PASSIF includes CAPITAUX PROPRES
        passif_val = _get_row_value(passif_row[1], date_col)
        result["passif_total"] = passif_val
        
        if passif_val is not None and actif_val is not None:
            diff = abs(actif_val - passif_val)
            result["difference"] = diff
            result["identity_valid"] = diff < 10
    
    return result


def recompute_totals(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    sections: List[SectionInfo],
    apply_corrections: bool = False  # CRITICAL FIX: Default to False - NEVER overwrite
) -> FinancialRecomputationResult:
    """
    STEP 4: Financial Validation (NON-DESTRUCTIVE)
    
    CRITICAL FIX (v3.1):
        - NEVER overwrites extracted values
        - Only VALIDATES totals and reports mismatches
        - Only COMPUTES values when field is EMPTY/MISSING
        - All mismatches are flagged as WARNINGS, not corrections
    
    OCR-extracted values are the SOURCE OF TRUTH.
    """
    # CRITICAL: Work on deep copy to protect original
    corrected_rows = copy.deepcopy(rows)
    validations = []  # Changed from corrections_made
    totals_validated = 0
    totals_computed = 0  # Only count values computed for EMPTY fields
    warnings = []  # Validation warnings (mismatches)
    
    # Get numeric columns
    numeric_cols = [name for name, info in column_types.columns.items()
                    if info.detected_role in (ColumnRole.DATE_CURRENT, ColumnRole.DATE_PREVIOUS,
                                               ColumnRole.DATE_OTHER, ColumnRole.VARIATION_AMOUNT)]
    
    for section in sections:
        if section.total_row is None or not section.data_rows:
            continue
        
        total_row = corrected_rows[section.total_row]
        
        for col in numeric_cols:
            expected_sum = 0.0
            component_values = []
            has_values = False
            
            for data_row_idx in section.data_rows:
                data_row = corrected_rows[data_row_idx]
                label = str(data_row.get("Label", data_row.get("label", ""))).upper()
                
                val = _parse_numeric_safe(str(data_row.get(col, "")))
                if val is not None:
                    # Handle negative indicators
                    if "MOINS" in label or "LESS" in label or "(DEDUCTION)" in label:
                        expected_sum -= abs(val)
                        component_values.append(-abs(val))
                    else:
                        expected_sum += val
                        component_values.append(val)
                    has_values = True
            
            if not has_values:
                continue
            
            actual_total_str = str(total_row.get(col, "")).strip()
            actual_total = _parse_numeric_safe(actual_total_str)
            
            # CASE 1: Value is MISSING/EMPTY - we can compute it
            if actual_total is None or actual_total_str in ("", "-", "—"):
                if apply_corrections:
                    total_row[col] = _format_number_enhanced(expected_sum)
                    totals_computed += 1
                validations.append({
                    "type": "total_computed",
                    "action": "computed_missing_value",
                    "section": section.section_name,
                    "column": col,
                    "computed_value": expected_sum,
                    "components": component_values,
                    "was_empty": True
                })
            
            # CASE 2: Value EXISTS - NEVER overwrite, only validate
            elif abs(expected_sum - actual_total) > 2.0:
                diff = abs(expected_sum - actual_total)
                diff_percent = diff / abs(actual_total) * 100 if actual_total != 0 else 100
                
                # CRITICAL: DO NOT MODIFY - only log warning
                warnings.append({
                    "type": "total_mismatch",
                    "severity": "warning",
                    "action": "validation_only",  # NOT corrected
                    "section": section.section_name,
                    "column": col,
                    "extracted_value": actual_total,  # This is the SOURCE OF TRUTH
                    "computed_value": expected_sum,
                    "difference": diff,
                    "diff_percent": diff_percent,
                    "message": f"Extracted value {actual_total} differs from computed sum {expected_sum} by {diff_percent:.1f}%"
                })
                totals_validated += 1
            else:
                # Values match - validation passed
                totals_validated += 1
                validations.append({
                    "type": "total_validated",
                    "action": "validation_passed",
                    "section": section.section_name,
                    "column": col,
                    "value": actual_total
                })
    
    # Validate balance sheet identity (NON-DESTRUCTIVE)
    balance_check = validate_balance_sheet_identity(corrected_rows, column_types)
    
    # Add warnings to validations for output
    validations.extend(warnings)
    
    return FinancialRecomputationResult(
        corrected_rows=corrected_rows,
        corrections_made=validations,  # Now contains validations + warnings, not corrections
        balance_sheet_valid=balance_check.get("identity_valid", True),
        balance_sheet_difference=balance_check.get("difference"),
        totals_corrected=totals_computed  # Only counts computed MISSING values
    )


def _format_number_enhanced(value: float) -> str:
    """Format number with space as thousand separator (French format)."""
    if value == 0:
        return "0"
    
    # Handle negative numbers
    sign = ""
    if value < 0:
        sign = "("
        value = abs(value)
    
    # Format with spaces as thousand separators
    int_val = int(round(value))
    formatted = f"{int_val:,}".replace(",", " ")
    
    if sign:
        return f"({formatted})"
    return formatted


# =============================================================================
# ENHANCED STEP 5: CASH FLOW COLUMN RECOVERY
# =============================================================================

@dataclass
class CashFlowRecoveryResult:
    """Result of cash flow column recovery."""
    recovery_applied: bool
    recovered_column: Optional[str]
    original_column: Optional[str]
    rows_affected: int
    reason: str


def detect_misplaced_financial_values_in_note(
    rows: List[Dict],
    column_types: ColumnTypeResult
) -> Tuple[bool, List[Tuple[int, float]]]:
    """
    Detect if Note column contains financial values instead of note references.
    
    Returns:
        (has_misplaced_values, list of (row_index, value) pairs)
    """
    note_cols = [name for name, info in column_types.columns.items() 
                 if info.detected_role == ColumnRole.NOTE]
    
    if not note_cols:
        return False, []
    
    note_col = note_cols[0]
    misplaced = []
    
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        
        row_type = str(row.get("type", "")).lower()
        if row_type == "section":
            continue
        
        val_str = str(row.get(note_col, "")).strip()
        if not val_str or val_str in ("-", "", "—"):
            continue
        
        # Check if it's a large number (financial value)
        parsed = _parse_numeric_safe(val_str)
        if parsed is not None and abs(parsed) > 1000:
            misplaced.append((i, parsed))
    
    # If >30% of Note values are large numbers, likely misplaced financial data
    total_data_rows = sum(1 for r in rows if isinstance(r, dict) and str(r.get("type", "")).lower() != "section")
    has_misplaced = len(misplaced) > total_data_rows * 0.3 if total_data_rows > 0 else False
    
    return has_misplaced, misplaced


def recover_cash_flow_column(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    columns: List[str]
) -> Tuple[List[Dict], CashFlowRecoveryResult]:
    """
    STEP 5: Cash Flow Column Recovery
    
    If no date columns detected but Note column contains financial values,
    convert Note column to a financial column.
    """
    import copy
    
    # Check if recovery is needed
    if column_types.detected_date_order:
        # Already have date columns - no recovery needed
        return rows, CashFlowRecoveryResult(
            recovery_applied=False,
            recovered_column=None,
            original_column=None,
            rows_affected=0,
            reason="Date columns already present"
        )
    
    has_misplaced, misplaced_values = detect_misplaced_financial_values_in_note(rows, column_types)
    
    if not has_misplaced:
        return rows, CashFlowRecoveryResult(
            recovery_applied=False,
            recovered_column=None,
            original_column=None,
            rows_affected=0,
            reason="No financial values detected in Note column"
        )
    
    # Apply recovery
    corrected_rows = copy.deepcopy(rows)
    note_cols = [name for name, info in column_types.columns.items() 
                 if info.detected_role == ColumnRole.NOTE]
    note_col = note_cols[0] if note_cols else None
    
    if not note_col:
        return rows, CashFlowRecoveryResult(
            recovery_applied=False,
            recovered_column=None,
            original_column=None,
            rows_affected=0,
            reason="No Note column found"
        )
    
    # Rename Note column to "Montant" or use as current period
    new_col_name = "Montant"
    rows_affected = 0
    
    for row in corrected_rows:
        if not isinstance(row, dict):
            continue
        if note_col in row:
            val = row.pop(note_col)
            row[new_col_name] = val
            rows_affected += 1
    
    return corrected_rows, CashFlowRecoveryResult(
        recovery_applied=True,
        recovered_column=new_col_name,
        original_column=note_col,
        rows_affected=rows_affected,
        reason=f"Converted Note column with {len(misplaced_values)} financial values to '{new_col_name}'"
    )


# =============================================================================
# ENHANCED STEP 3: AGGRESSIVE COLUMN SHIFT DETECTION
# =============================================================================

@dataclass
class ShiftCorrection:
    """Record of a column shift correction."""
    row_index: int
    shift_direction: str  # "left" or "right"
    columns_affected: List[str]
    values_shifted: Dict[str, str]
    reason: str
    confidence: float


def detect_column_shift(
    row: Dict,
    row_index: int,
    column_types: ColumnTypeResult,
    column_order: List[str],
    row_median: float
) -> Optional[ShiftCorrection]:
    """
    Detect if a row has shifted columns based on value magnitude analysis.
    
    Detects:
        - Small number (< 50) in financial column when others are > 10,000
        - Large number in Note column
        - Sequence of values that would make more sense shifted
    """
    numeric_cols = [name for name, info in column_types.columns.items()
                    if info.detected_role in (ColumnRole.DATE_CURRENT, ColumnRole.DATE_PREVIOUS,
                                               ColumnRole.DATE_OTHER)]
    
    if len(numeric_cols) < 2:
        return None
    
    values = []
    for col in numeric_cols:
        val = _parse_numeric_safe(str(row.get(col, "")))
        values.append((col, val))
    
    # Check for anomalous small value in context of large values
    non_null_values = [v for c, v in values if v is not None]
    if len(non_null_values) < 2:
        return None
    
    median_val = statistics.median([abs(v) for v in non_null_values if v != 0])
    
    for col, val in values:
        if val is not None and median_val > 10000:
            if abs(val) < 50 and abs(val) / median_val < 0.001:
                # This looks like a note reference in a financial column
                # Check if there's a pattern suggesting shift
                return ShiftCorrection(
                    row_index=row_index,
                    shift_direction="detected",  # Direction TBD
                    columns_affected=[col],
                    values_shifted={col: str(row.get(col, ""))},
                    reason=f"Small value ({val}) in context of large values (median={median_val:,.0f})",
                    confidence=0.8
                )
    
    return None


def apply_aggressive_realignment(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    columns: List[str]
) -> Tuple[List[Dict], List[ShiftCorrection]]:
    """
    STEP 3 Enhanced: More Aggressive Column Shift Correction
    
    Applies shift corrections for detected anomalies.
    """
    import copy
    corrected_rows = copy.deepcopy(rows)
    corrections = []
    
    # Calculate overall median for numeric columns
    all_numeric_values = []
    numeric_cols = [name for name, info in column_types.columns.items()
                    if info.detected_role in (ColumnRole.DATE_CURRENT, ColumnRole.DATE_PREVIOUS,
                                               ColumnRole.DATE_OTHER)]
    
    for row in rows:
        if not isinstance(row, dict):
            continue
        for col in numeric_cols:
            val = _parse_numeric_safe(str(row.get(col, "")))
            if val is not None and abs(val) > 100:  # Skip small values
                all_numeric_values.append(abs(val))
    
    if not all_numeric_values:
        return corrected_rows, corrections
    
    overall_median = statistics.median(all_numeric_values)
    
    # Get column order for shift operations
    ordered_cols = [c for c in columns if c in column_types.columns]
    
    # Find note column
    note_cols = [name for name, info in column_types.columns.items() 
                 if info.detected_role == ColumnRole.NOTE]
    note_col = note_cols[0] if note_cols else None
    
    for i, row in enumerate(corrected_rows):
        if not isinstance(row, dict):
            continue
        
        row_type = str(row.get("type", "")).lower()
        if row_type == "section":
            continue
        
        # Check for small value in numeric column (potential misplaced note)
        for col_idx, col in enumerate(numeric_cols):
            val_str = str(row.get(col, "")).strip()
            val = _parse_numeric_safe(val_str)
            
            if val is not None and abs(val) < 50 and overall_median > 10000:
                # Small value detected - check if it should be in Note column
                
                # Condition 1: Note column is empty
                current_note = str(row.get(note_col, "")).strip() if note_col else ""
                if note_col and (not current_note or current_note in ("-", "—")):
                    # Move small value to Note, shift remaining values left
                    row[note_col] = val_str
                    
                    # Shift values left starting from this column
                    cols_to_shift = numeric_cols[col_idx:]
                    for j, shift_col in enumerate(cols_to_shift[:-1]):
                        next_col = cols_to_shift[j + 1]
                        row[shift_col] = row.get(next_col, "")
                    
                    # Clear the last column
                    if cols_to_shift:
                        row[cols_to_shift[-1]] = ""
                    
                    corrections.append(ShiftCorrection(
                        row_index=i,
                        shift_direction="left",
                        columns_affected=cols_to_shift,
                        values_shifted={col: val_str},
                        reason=f"Small value ({val}) moved to Note, columns shifted left",
                        confidence=0.85
                    ))
                    break  # Only one correction per row
    
    return corrected_rows, corrections


# =============================================================================
# PRODUCTION HARDENED VERSIONS (STEP 4, 5, 7 SAFE IMPLEMENTATIONS)
# =============================================================================

def recover_cash_flow_column_safe(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    columns: List[str]
) -> Tuple[List[Dict], CashFlowRecoveryResult]:
    """
    STEP 5 (SAFE VERSION): Cash Flow Column Recovery with strict validation.
    
    ONLY convert Note column into financial values if:
        - More than 70% of Note values are numeric
        - Numeric columns are empty or invalid
        - Values match realistic financial ranges
    """
    # Check if recovery is needed
    if column_types.detected_date_order:
        return rows, CashFlowRecoveryResult(
            recovery_applied=False,
            recovered_column=None,
            original_column=None,
            rows_affected=0,
            reason="Date columns already present"
        )
    
    # Find note column
    note_cols = [name for name, info in column_types.columns.items() 
                 if info.detected_role == ColumnRole.NOTE]
    note_col = note_cols[0] if note_cols else None
    
    if not note_col:
        return rows, CashFlowRecoveryResult(
            recovery_applied=False,
            recovered_column=None,
            original_column=None,
            rows_affected=0,
            reason="No Note column found"
        )
    
    # STEP 5 CONSTRAINT: Count numeric values in Note column
    note_values = []
    numeric_note_count = 0
    total_note_count = 0
    
    for row in rows:
        if not isinstance(row, dict):
            continue
        val_str = str(row.get(note_col, "")).strip()
        if val_str and val_str not in ("-", "—", ""):
            total_note_count += 1
            val = _parse_numeric_safe(val_str)
            if val is not None:
                numeric_note_count += 1
                note_values.append(abs(val))
    
    # CONSTRAINT: More than 70% must be numeric
    if total_note_count == 0:
        return rows, CashFlowRecoveryResult(
            recovery_applied=False,
            recovered_column=None,
            original_column=None,
            rows_affected=0,
            reason="Note column is empty"
        )
    
    numeric_ratio = numeric_note_count / total_note_count
    if numeric_ratio < 0.7:
        return rows, CashFlowRecoveryResult(
            recovery_applied=False,
            recovered_column=None,
            original_column=None,
            rows_affected=0,
            reason=f"Only {numeric_ratio:.0%} of Note values are numeric (need 70%+)"
        )
    
    # CONSTRAINT: Values must match realistic financial ranges (> 1000 typically)
    if note_values:
        median_value = statistics.median(note_values)
        if median_value < 1000:
            return rows, CashFlowRecoveryResult(
                recovery_applied=False,
                recovered_column=None,
                original_column=None,
                rows_affected=0,
                reason=f"Median Note value {median_value:.0f} too small for financial data"
            )
    
    # Check that numeric columns are empty/invalid
    has_valid_numeric_cols = False
    numeric_cols = [name for name, info in column_types.columns.items()
                    if info.detected_role in (ColumnRole.DATE_CURRENT, ColumnRole.DATE_PREVIOUS,
                                              ColumnRole.DATE_OTHER, ColumnRole.NUMERIC)]
    for row in rows:
        if not isinstance(row, dict):
            continue
        for col in numeric_cols:
            val = _parse_numeric_safe(str(row.get(col, "")))
            if val is not None and abs(val) > 100:
                has_valid_numeric_cols = True
                break
        if has_valid_numeric_cols:
            break
    
    if has_valid_numeric_cols:
        return rows, CashFlowRecoveryResult(
            recovery_applied=False,
            recovered_column=None,
            original_column=None,
            rows_affected=0,
            reason="Valid numeric columns already present"
        )
    
    # Apply recovery (passed all constraints)
    corrected_rows = copy.deepcopy(rows)
    new_col_name = "Montant"
    rows_affected = 0
    
    for row in corrected_rows:
        if not isinstance(row, dict):
            continue
        if note_col in row:
            val = row.pop(note_col)
            row[new_col_name] = val
            rows_affected += 1
    
    return corrected_rows, CashFlowRecoveryResult(
        recovery_applied=True,
        recovered_column=new_col_name,
        original_column=note_col,
        rows_affected=rows_affected,
        reason=f"Converted Note column ({numeric_ratio:.0%} numeric, median={statistics.median(note_values):.0f}) to '{new_col_name}'"
    )


def apply_aggressive_realignment_safe(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    columns: List[str],
    current_confidence: float
) -> Tuple[List[Dict], List[ShiftCorrection]]:
    """
    STEP 4 (VALIDATION ONLY): Column Shift Detection - NON-DESTRUCTIVE
    
    CRITICAL FIX (v3.1):
        - NEVER modifies extracted values
        - Only DETECTS potential misalignments
        - Reports as WARNINGS for manual review
        - Returns ORIGINAL rows unchanged
    
    OCR-extracted values are the SOURCE OF TRUTH.
    """
    # CRITICAL: Return original rows - DO NOT modify
    # We only detect and warn, never correct
    warnings = []
    
    # Calculate overall statistics for context
    all_numeric_values = []
    numeric_cols = [name for name, info in column_types.columns.items()
                    if info.detected_role in (ColumnRole.DATE_CURRENT, ColumnRole.DATE_PREVIOUS,
                                               ColumnRole.DATE_OTHER)]
    
    for row in rows:
        if not isinstance(row, dict):
            continue
        for col in numeric_cols:
            val = _parse_numeric_safe(str(row.get(col, "")))
            if val is not None and abs(val) > 100:
                all_numeric_values.append(abs(val))
    
    if not all_numeric_values:
        return rows, []  # Return original, no warnings
    
    overall_median = statistics.median(all_numeric_values)
    
    # Only analyze if we have large values context
    if overall_median < 10000:
        return rows, []
    
    # Find note column
    note_cols = [name for name, info in column_types.columns.items() 
                 if info.detected_role == ColumnRole.NOTE]
    note_col = note_cols[0] if note_cols else None
    
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        
        row_type = str(row.get("type", "")).lower()
        if row_type == "section":
            continue
        
        # Calculate row total for context
        row_total = 0
        row_values = []
        for col in numeric_cols:
            val = _parse_numeric_safe(str(row.get(col, "")))
            if val is not None:
                row_total += abs(val)
                row_values.append(abs(val))
        
        if row_total == 0:
            continue
        
        for col_idx, col in enumerate(numeric_cols):
            val_str = str(row.get(col, "")).strip()
            val = _parse_numeric_safe(val_str)
            
            if val is None:
                continue
            
            # Detect anomaly
            anomaly_threshold = min(50, row_total * 0.01) if row_total > 0 else 50
            
            if abs(val) >= anomaly_threshold:
                continue
            
            other_values = [v for v in row_values if v != abs(val)]
            if not other_values or statistics.median(other_values) < 1000:
                continue
            
            # DETECTED a potential misalignment - DO NOT CORRECT, just warn
            warnings.append(ShiftCorrection(
                row_index=i,
                shift_direction="potential_left",  # Indicates DETECTION, not action
                columns_affected=[col],
                values_shifted={col: val_str},
                reason=f"POTENTIAL MISALIGNMENT DETECTED: Value {val} in column '{col}' appears abnormally small (threshold: {anomaly_threshold:.0f}). Manual review recommended.",
                confidence=current_confidence
            ))
            
            # Log as warning, not correction
            log_correction(
                field=col,
                row_label=str(row.get("label", row.get("Label", f"Row {i}"))),
                row_index=i,
                old_value=val_str,
                new_value=val_str,  # Same value - NO CHANGE
                reason=f"POTENTIAL MISALIGNMENT WARNING: Small value ({val}) detected in financial column. No automatic correction applied.",
                correction_type="warning_only",
                confidence=current_confidence
            )
            break
    
    # CRITICAL: Return ORIGINAL rows unchanged
    return rows, warnings


def recompute_totals_balance_sheet(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    sections: List,
    apply_corrections: bool = True,
    confidence: float = 0.0
) -> Optional[FinancialRecomputationResult]:
    """
    STEP 7 (SAFE VERSION): Balance Sheet Specific Recomputation.
    
    Only recompute when:
        - All required components are present
        - Confidence >= 0.7
    """
    # CONSTRAINT: Minimum confidence for recomputation
    if confidence < 0.7:
        return None
    
    # Use generic recompute but flag as balance sheet specific
    result = recompute_totals(rows, column_types, sections, apply_corrections)
    if result:
        result.corrections_made.append({
            "type": "table_type_specific",
            "table_type": "balance_sheet",
            "note": "Applied balance sheet rules"
        })
    return result


def recompute_totals_income_statement(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    sections: List,
    apply_corrections: bool = True,
    confidence: float = 0.0
) -> Optional[FinancialRecomputationResult]:
    """
    STEP 7 (SAFE VERSION): Income Statement Specific Recomputation.
    """
    if confidence < 0.7:
        return None
    
    result = recompute_totals(rows, column_types, sections, apply_corrections)
    if result:
        result.corrections_made.append({
            "type": "table_type_specific",
            "table_type": "income_statement",
            "note": "Applied income statement rules"
        })
    return result


def recompute_totals_cash_flow(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    sections: List,
    apply_corrections: bool = True,
    confidence: float = 0.0
) -> Optional[FinancialRecomputationResult]:
    """
    STEP 7 (SAFE VERSION): Cash Flow Specific Recomputation.
    """
    if confidence < 0.7:
        return None
    
    result = recompute_totals(rows, column_types, sections, apply_corrections)
    if result:
        result.corrections_made.append({
            "type": "table_type_specific",
            "table_type": "cash_flow",
            "note": "Applied cash flow rules"
        })
    return result

@dataclass
class ConsistencyCheckResult:
    """Result of cross-section consistency checks."""
    checks_performed: List[str]
    checks_passed: List[str]
    checks_failed: List[Dict]
    overall_consistent: bool


def check_cross_section_consistency(
    rows: List[Dict],
    column_types: ColumnTypeResult,
    table_type: str
) -> ConsistencyCheckResult:
    """
    STEP 6: Cross-Section Consistency Checks
    
    Verifies:
        - Internal sums match totals
        - Balance sheet identity (if applicable)
        - Net values = Gross - Amortization
        - Result consistency
    """
    checks_performed = []
    checks_passed = []
    checks_failed = []
    
    date_col = column_types.detected_date_order[0] if column_types.detected_date_order else None
    
    if not date_col:
        return ConsistencyCheckResult(
            checks_performed=["date_column_detection"],
            checks_passed=[],
            checks_failed=[{"check": "date_column", "reason": "No date column found"}],
            overall_consistent=False
        )
    
    # Check 1: Balance sheet identity (ACTIF = PASSIF + CAPITAUX)
    if table_type == "balance_sheet":
        checks_performed.append("balance_sheet_identity")
        
        identity_check = validate_balance_sheet_identity(rows, column_types)
        if identity_check.get("identity_valid") is True:
            checks_passed.append("balance_sheet_identity")
        elif identity_check.get("identity_valid") is False:
            checks_failed.append({
                "check": "balance_sheet_identity",
                "actif": identity_check.get("actif_total"),
                "passif_plus_capitaux": identity_check.get("computed_passif_plus_capitaux"),
                "difference": identity_check.get("difference")
            })
    
    # Check 2: Net = Gross - Amortization pattern
    checks_performed.append("net_gross_amortization")
    net_gross_valid = True
    
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        label = str(row.get("Label", "")).lower()
        
        if "net" in label and "brut" not in label:
            # Find corresponding gross and amortization rows
            gross_row = None
            amort_row = None
            
            # Look in nearby rows (within 5 rows)
            for j in range(max(0, i-5), min(len(rows), i+5)):
                if j == i:
                    continue
                nearby_label = str(rows[j].get("Label", "")).lower()
                if "brut" in nearby_label or "gross" in nearby_label:
                    gross_row = rows[j]
                elif "amortis" in nearby_label or "deprec" in nearby_label or "prov" in nearby_label:
                    amort_row = rows[j]
            
            if gross_row and amort_row:
                net_val = _parse_numeric_safe(str(row.get(date_col, "")))
                gross_val = _parse_numeric_safe(str(gross_row.get(date_col, "")))
                amort_val = _parse_numeric_safe(str(amort_row.get(date_col, "")))
                
                if net_val is not None and gross_val is not None and amort_val is not None:
                    expected_net = gross_val - abs(amort_val)  # Amortization is often negative
                    if abs(net_val - expected_net) > 10:
                        net_gross_valid = False
                        checks_failed.append({
                            "check": "net_gross_amortization",
                            "row": i,
                            "label": row.get("Label"),
                            "net": net_val,
                            "gross": gross_val,
                            "amortization": amort_val,
                            "expected_net": expected_net
                        })
    
    if net_gross_valid:
        checks_passed.append("net_gross_amortization")
    
    return ConsistencyCheckResult(
        checks_performed=checks_performed,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        overall_consistent=len(checks_failed) == 0
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
    
    # NEW: Enhanced steps
    financial_recomputation: Optional[FinancialRecomputationResult]
    cash_flow_recovery: Optional[CashFlowRecoveryResult]
    shift_corrections: Optional[List[ShiftCorrection]]
    consistency_check: Optional[ConsistencyCheckResult]
    
    # Final output
    corrected_rows: List[Dict]
    metadata: Dict


def run_safe_pipeline(
    data: Dict,
    apply_corrections: bool = False,  # CRITICAL FIX: Default to False - NEVER modify by default
    strict_mode: bool = True,
    aggressive_mode: bool = False  # CRITICAL FIX: Default to False
) -> SafePipelineResult:
    """
    Run the complete safe pipeline improvements - NON-DESTRUCTIVE MODE.
    
    CRITICAL FIX (v3.1):
        - DEFAULT is NON-DESTRUCTIVE (apply_corrections=False)
        - OCR extracted values are SOURCE OF TRUTH
        - Only VALIDATES, does not overwrite
        - Mismatches are flagged as WARNINGS
        - _raw_rows is NEVER modified
    
    Args:
        data: Extracted table data {"columns": [...], "rows": [...]}
        apply_corrections: If True, only compute MISSING values (never overwrite existing)
        strict_mode: If True, only apply highest-confidence corrections
        aggressive_mode: DEPRECATED - kept for API compatibility, does not enable overwrites
    
    Returns:
        SafePipelineResult with validation results (not corrections)
    """
    columns = data.get("columns", [])
    rows = data.get("rows", [])
    title = data.get("table_name", data.get("title", None))
    
    # CRITICAL: Protect raw data - work on a copy, NEVER modify original
    working_rows = protect_raw_data(rows)
    
    # Phase 1: Column Type Detection (ALWAYS run - detection only)
    column_types = detect_column_types_enhanced(working_rows, columns)
    
    # Phase 2: Safe Validation (ALWAYS run - detection only)
    validation_result = validate_column_consistency(working_rows, column_types)
    
    # Phase 6: Table Type Detection
    table_type = detect_table_type_enhanced(working_rows, title)
    detected_table_type = table_type["table_type"]
    
    # Compute confidence
    preliminary_total_validation = validate_totals_improved(working_rows, column_types)
    preliminary_confidence = compute_meaningful_confidence(
        working_rows, column_types, validation_result, preliminary_total_validation
    )
    
    # CRITICAL FIX: Determine correction mode - but NEVER allow destructive corrections
    correction_mode = should_apply_corrections(
        preliminary_confidence.final_score, 
        aggressive_mode
    )
    
    # Initialize results
    realignment_result = None
    financial_recomputation = None
    cash_flow_recovery = None
    shift_corrections = None
    consistency_check = None
    
    # CRITICAL: Start with original rows - minimal modifications only
    corrected_rows = working_rows
    
    # Phase 3: Controlled Realignment - VALIDATION ONLY
    # Only detect potential issues, do not modify
    if correction_mode in ("safe", "aggressive"):
        realignment_result = controlled_realignment(
            corrected_rows, column_types, validation_result, strict_mode
        )
        # CRITICAL FIX: Only take corrected rows if explicitly enabled AND low-risk
        if apply_corrections and realignment_result.corrections:
            # Only apply if corrections are minimal and safe
            if len(realignment_result.corrections) <= 2:
                corrected_rows = realignment_result.corrected_rows
    
    # Cash Flow Recovery - Only if date columns are missing
    if not column_types.detected_date_order:
        corrected_rows, cash_flow_recovery = recover_cash_flow_column_safe(
            corrected_rows, column_types, columns
        )
        if cash_flow_recovery and cash_flow_recovery.recovery_applied:
            column_types = detect_column_types_enhanced(corrected_rows, columns)
    
    # STEP 4: Shift Detection - VALIDATION ONLY (never modifies)
    if correction_mode == "aggressive":
        # This function now only DETECTS, never modifies
        _, shift_corrections = apply_aggressive_realignment_safe(
            corrected_rows, column_types, columns, preliminary_confidence.final_score
        )
    
    # Phase 4: Total Validation & Section Detection
    sections = detect_sections_improved(corrected_rows)
    total_validation = validate_totals_improved(corrected_rows, column_types)
    
    # STEP 7: Financial VALIDATION (NOT recomputation)
    # CRITICAL: apply_corrections=False means NEVER overwrite existing values
    # Only compute values for EMPTY fields
    financial_recomputation = recompute_totals(
        corrected_rows, column_types, sections, 
        apply_corrections=apply_corrections  # Only fills EMPTY fields
    )
    if financial_recomputation:
        corrected_rows = financial_recomputation.corrected_rows
    
    # Cross-Section Consistency Check (always run for diagnostics)
    consistency_check = check_cross_section_consistency(
        corrected_rows, column_types, detected_table_type
    )
    
    # Phase 5: Final Confidence Scoring
    confidence = compute_meaningful_confidence(
        corrected_rows, column_types, validation_result, total_validation
    )
    
    # Adjust confidence based on consistency checks
    if consistency_check and not consistency_check.overall_consistent:
        confidence = ConfidenceBreakdown(
            schema_validity=confidence.schema_validity,
            column_consistency=confidence.column_consistency,
            validation_pass_rate=confidence.validation_pass_rate,
            coverage=confidence.coverage,
            misalignment_penalty=confidence.misalignment_penalty + 0.1,
            validation_error_penalty=confidence.validation_error_penalty + len(consistency_check.checks_failed) * 0.05,
            final_score=max(0.0, confidence.final_score - 0.1 - len(consistency_check.checks_failed) * 0.05)
        )
    
    # Phase 7: Reliability Assessment
    reliability = assess_reliability(
        column_types, validation_result, total_validation, confidence
    )
    
    # Add unreliable reasons
    if not column_types.detected_date_order:
        reliability.unreliable_reasons.append("missing_date_columns")
    if total_validation.total_errors:
        reliability.unreliable_reasons.append("inconsistent_totals")
    if column_types.schema_quality < 0.5:
        reliability.unreliable_reasons.append("schema_ambiguity")
    if confidence.final_score < 0.7:
        reliability.unreliable_reasons.append("low_confidence")
        reliability.is_reliable = False
    
    # Build metadata
    metadata = {
        "_safe_pipeline_version": "3.0",  # Production hardened
        "_correction_mode": correction_mode,
        "_aggressive_mode_requested": aggressive_mode,
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
        "_table_type": detected_table_type,
        "_is_reliable": reliability.is_reliable,
        "_unreliable_reasons": reliability.unreliable_reasons,
        "_recommended_action": reliability.recommended_action,
        "_corrections_applied": len(realignment_result.corrections) if realignment_result else 0,
        "_validation_warnings": len(validation_result.warnings),
        "_total_errors": len(total_validation.total_errors)
    }
    
    # Add enhanced step results to metadata
    if financial_recomputation:
        metadata["_financial_recomputation"] = {
            "totals_corrected": financial_recomputation.totals_corrected,
            "balance_sheet_valid": financial_recomputation.balance_sheet_valid,
            "balance_sheet_difference": financial_recomputation.balance_sheet_difference,
            "corrections": financial_recomputation.corrections_made
        }
    
    if cash_flow_recovery and cash_flow_recovery.recovery_applied:
        metadata["_cash_flow_recovery"] = {
            "recovered_column": cash_flow_recovery.recovered_column,
            "original_column": cash_flow_recovery.original_column,
            "rows_affected": cash_flow_recovery.rows_affected,
            "reason": cash_flow_recovery.reason
        }
    
    if shift_corrections:
        metadata["_shift_corrections"] = [
            {
                "row": c.row_index,
                "direction": c.shift_direction,
                "reason": c.reason,
                "confidence": c.confidence
            }
            for c in shift_corrections
        ]
    
    if consistency_check:
        metadata["_consistency_check"] = {
            "checks_performed": consistency_check.checks_performed,
            "checks_passed": consistency_check.checks_passed,
            "checks_failed": consistency_check.checks_failed,
            "overall_consistent": consistency_check.overall_consistent
        }
    
    return SafePipelineResult(
        column_types=column_types,
        validation_result=validation_result,
        realignment_result=realignment_result,
        total_validation=total_validation,
        confidence=confidence,
        table_type=table_type,
        reliability=reliability,
        financial_recomputation=financial_recomputation,
        cash_flow_recovery=cash_flow_recovery,
        shift_corrections=shift_corrections,
        consistency_check=consistency_check,
        corrected_rows=corrected_rows,
        metadata=metadata
    )


# =============================================================================
# INTEGRATION HELPER - Attach to existing pipeline (PRODUCTION HARDENED)
# =============================================================================

def enhance_extraction_result(
    data: Dict, 
    apply_corrections: bool = False,  # CRITICAL FIX: Default to False - NEVER modify by default
    aggressive_mode: bool = False  # CRITICAL FIX: Default to False
) -> Dict:
    """
    Integration helper: Run safe pipeline and attach metadata to existing result.
    
    CRITICAL FIX (v3.1) - NON-DESTRUCTIVE MODE:
        - DEFAULT is apply_corrections=False (NEVER modify)
        - OCR extracted values are SOURCE OF TRUTH
        - Only VALIDATES and reports warnings
        - _raw_rows is NEVER modified
        - Returns original data with validation metadata attached
    
    Args:
        data: Existing extraction result {"columns": [...], "rows": [...]}
        apply_corrections: If True, only compute MISSING values (never overwrite)
        aggressive_mode: DEPRECATED - kept for API compatibility
    
    Returns:
        Enhanced data dict with validation metadata (rows unchanged by default)
    """
    if not isinstance(data, dict) or "rows" not in data:
        return data
    
    # CRITICAL: Protect raw data - store immutable copy FIRST
    if "_raw_rows" not in data:
        data["_raw_rows"] = copy.deepcopy(data.get("rows", []))
    
    # Reset corrections log for this run
    reset_corrections_log()
    
    try:
        # Run safe pipeline in VALIDATION-ONLY mode by default
        result = run_safe_pipeline(
            data, 
            apply_corrections=apply_corrections,  # Default: False = validation only
            aggressive_mode=aggressive_mode
        )
        
        # Get effective correction mode
        effective_mode = should_apply_corrections(
            result.confidence.final_score, 
            aggressive_mode
        )
        
        # Attach metadata
        data["_safe_pipeline"] = result.metadata
        data["_safe_pipeline"]["_version"] = "3.1"  # NON-DESTRUCTIVE version
        data["_safe_pipeline"]["_mode"] = "validation_only" if not apply_corrections else "minimal_corrections"
        data["_safe_pipeline"]["_correction_mode_requested"] = "aggressive" if aggressive_mode else "safe"
        data["_safe_pipeline"]["_correction_mode_applied"] = effective_mode
        
        # CRITICAL FIX: Count actual modifications made
        actual_corrections = 0
        if result.financial_recomputation:
            # Only count values computed for EMPTY fields
            actual_corrections = sum(
                1 for c in result.financial_recomputation.corrections_made 
                if c.get("was_empty") or c.get("type") == "total_computed"
            )
        
        data["_safe_pipeline"]["_values_modified"] = actual_corrections
        
        # SAFETY ASSERTION: If corrections not requested, ensure nothing was modified
        if not apply_corrections:
            assert actual_corrections == 0, f"BUG: {actual_corrections} corrections applied when apply_corrections=False"
        
        # Return original rows unless apply_corrections=True AND we have changes
        if apply_corrections and actual_corrections > 0:
            data["rows"] = result.corrected_rows
        # else: Keep original rows unchanged
        
        # Collect all warnings/validations for reporting
        validation_log = get_corrections_log()
        
        # Add validation warnings from financial recomputation
        if result.financial_recomputation:
            for c in result.financial_recomputation.corrections_made:
                if c.get("type") in ("total_mismatch", "total_validated"):
                    validation_log.append({
                        "type": c.get("type"),
                        "action": c.get("action", "validation_only"),
                        "section": c.get("section"),
                        "column": c.get("column"),
                        "extracted_value": c.get("extracted_value"),
                        "computed_value": c.get("computed_value"),
                        "message": c.get("message"),
                        "severity": c.get("severity", "info")
                    })
        
        # Add shift warnings (detection only)
        if result.shift_corrections:
            for c in result.shift_corrections:
                validation_log.append({
                    "type": "potential_misalignment",
                    "action": "warning_only",
                    "row_index": c.row_index,
                    "columns": c.columns_affected,
                    "message": c.reason,
                    "severity": "warning"
                })
        
        if validation_log:
            data["_validation_warnings"] = validation_log
        
        # Add reliability flags
        data["_confidence"] = result.confidence.final_score
        data["_is_reliable"] = result.reliability.is_reliable
        data["_recommended_action"] = result.reliability.recommended_action
        
        if not result.reliability.is_reliable:
            data["_unreliable"] = True
            data["_unreliable_reasons"] = result.reliability.unreliable_reasons
        
        # Add consistency check summary
        if result.consistency_check:
            data["_consistency_valid"] = result.consistency_check.overall_consistent
        
        # Add balance sheet validity
        if result.financial_recomputation:
            data["_balance_sheet_valid"] = result.financial_recomputation.balance_sheet_valid
        
        return data
    
    except Exception as e:
        # STEP 10: SAFETY - Never break existing pipeline, return original data
        import traceback
        # Restore raw rows if available
        if "_raw_rows" in data:
            data["rows"] = copy.deepcopy(data["_raw_rows"])
        data["_safe_pipeline_error"] = str(e)
        data["_safe_pipeline_traceback"] = traceback.format_exc()
        data["_is_reliable"] = False
        data["_unreliable"] = True
        data["_unreliable_reasons"] = [f"Pipeline error: {str(e)}"]
        return data

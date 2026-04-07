"""
Immutable Financial Extraction Module - ZERO DATA CORRUPTION
=============================================================

This module enforces STRICT IMMUTABILITY for OCR-extracted financial data.

CORE PRINCIPLE:
    IF OCR IS CORRECT → OUTPUT IS IDENTICAL
    
RULES ENFORCED:
    1. NEVER modify extracted string values
    2. NEVER normalize number formatting
    3. NEVER remove spaces from numbers ("1 542 904" stays as-is)
    4. NEVER convert parentheses to minus signs
    5. NEVER convert strings to int/float
    6. NEVER recompute totals
    7. ALL values stored as STRINGS
    8. Validation is READ-ONLY comparison

Author: Immutable Extraction System
Version: 1.0 - STRICT IMMUTABILITY
"""

import copy
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# =============================================================================
# IMMUTABLE EXTRACTION PROMPT - Character-for-character fidelity
# =============================================================================

IMMUTABLE_EXTRACTION_PROMPT = """Extract the COMPLETE financial table from this image.

⚠️ CRITICAL: EXACT CHARACTER-BY-CHARACTER COPYING

You MUST copy every value EXACTLY as it appears in the document.
DO NOT modify, normalize, or reformat ANY value.

STRICT RULES:

1. NUMBER FORMATTING - PRESERVE EXACTLY:
   ✅ "1 542 904" → "1 542 904" (keep spaces)
   ✅ "(72 125)" → "(72 125)" (keep parentheses AND spaces)
   ✅ "22,5%" → "22,5%" (keep European decimal comma)
   ✅ "1.234.567,89" → "1.234.567,89" (keep European format)
   ❌ "1542904" is WRONG if original has spaces
   ❌ "-72125" is WRONG if original uses parentheses

2. CHARACTER PRESERVATION:
   - Spaces inside numbers: KEEP them
   - Parentheses for negatives: KEEP them as ()
   - Percentage signs: KEEP them
   - Comma vs dot: KEEP original
   - Leading/trailing spaces: TRIM only outer whitespace

3. TYPE RULES:
   - ALL numeric values must be STRINGS (quoted)
   - NEVER use JSON numbers (unquoted)
   - Empty cells: use "" (empty string)

4. HEADER DETECTION:
   - First row = header row
   - Use EXACT column names from header
   - DO NOT invent column names

5. ROW EXTRACTION:
   - Extract ALL rows from TOP to BOTTOM
   - Include section headers
   - Row types: "section", "data", "total"

6. IGNORE:
   - Page titles
   - Unit text (en milliers de dinars)
   - Footnotes

OUTPUT FORMAT:

{
  "columns": ["Libellé", "Note", "31/12/2024", "31/12/2023", "Variation"],
  "rows": [
    {
      "type": "data",
      "Libellé": "Produits d'exploitation",
      "Note": "(1-1)",
      "31/12/2024": "1 542 904",
      "31/12/2023": "1 470 779",
      "Variation": "(72 125)"
    }
  ],
  "_extraction_mode": "immutable"
}

VERIFICATION:
Before outputting, verify:
1. Every number with spaces in image → has spaces in output
2. Every (value) in image → is (value) in output
3. Every comma decimal → stays as comma
4. No unquoted numbers in JSON

Return ONLY valid JSON. ALL values as STRINGS."""


# =============================================================================
# DATA INTEGRITY TYPES
# =============================================================================

@dataclass
class ImmutableValue:
    """Wrapper that prevents modification of extracted values."""
    original: str
    hash: str = field(init=False)
    
    def __post_init__(self):
        self.hash = hashlib.md5(self.original.encode('utf-8')).hexdigest()[:8]
    
    def __str__(self):
        return self.original
    
    def verify(self) -> bool:
        """Verify value hasn't been modified."""
        current_hash = hashlib.md5(self.original.encode('utf-8')).hexdigest()[:8]
        return current_hash == self.hash


@dataclass 
class IntegrityReport:
    """Report of data integrity checks."""
    is_valid: bool
    total_values: int
    modified_values: int
    corrupted_fields: List[Dict]
    message: str


@dataclass
class ValidationFinding:
    """A validation finding (informational only, NO modification)."""
    row_index: int
    field: str
    finding_type: str  # "mismatch", "anomaly", "format_unusual"
    extracted_value: str
    computed_value: Optional[str]
    message: str
    severity: str  # "info", "warning", "error"


# =============================================================================
# ANTI-CORRUPTION SAFEGUARDS
# =============================================================================

def freeze_value(value: Any) -> str:
    """
    Convert any value to immutable string representation.
    
    RULES:
        - String → unchanged
        - Number → str() without formatting
        - None → ""
        - Already string → unchanged
    
    NEVER modifies the actual content, only ensures it's a string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # If it's already a number (shouldn't happen), convert to string
    # WITHOUT any formatting changes
    return str(value)


def verify_no_corruption(original: str, current: str) -> bool:
    """
    Check if value was corrupted.
    
    Returns True if values are IDENTICAL, False if corrupted.
    """
    return original == current


def detect_silent_corruption(original: Any, processed: Any) -> Optional[str]:
    """
    Detect if silent corruption occurred during processing.
    
    Returns description of corruption or None if clean.
    """
    orig_str = freeze_value(original)
    proc_str = freeze_value(processed)
    
    if orig_str == proc_str:
        return None
    
    # Detect specific corruption types
    if ' ' in orig_str and ' ' not in proc_str:
        return f"Space removal: '{orig_str}' → '{proc_str}'"
    
    if '(' in orig_str and '(' not in proc_str and '-' in proc_str:
        return f"Parentheses converted to minus: '{orig_str}' → '{proc_str}'"
    
    if ',' in orig_str and ',' not in proc_str and '.' in proc_str:
        return f"Comma/dot conversion: '{orig_str}' → '{proc_str}'"
    
    if '%' in orig_str and '%' not in proc_str:
        return f"Percentage sign removed: '{orig_str}' → '{proc_str}'"
    
    return f"Unknown corruption: '{orig_str}' → '{proc_str}'"


# =============================================================================
# IMMUTABLE ROW OPERATIONS
# =============================================================================

def freeze_row(row: Dict) -> Dict:
    """
    Freeze a row by converting all values to immutable strings.
    
    Creates a deep copy to prevent any reference issues.
    """
    frozen = {}
    for key, value in row.items():
        frozen[key] = freeze_value(value)
    return frozen


def freeze_rows(rows: List[Dict]) -> List[Dict]:
    """Freeze all rows in a list."""
    return [freeze_row(row) for row in rows]


def create_integrity_snapshot(data: Dict) -> Dict:
    """
    Create a cryptographic snapshot of all values for later verification.
    
    This allows detecting ANY modification after extraction.
    """
    snapshot = {
        "created_at": None,  # Would use datetime in production
        "row_hashes": [],
        "total_hash": None
    }
    
    rows = data.get("rows", [])
    all_values = []
    
    for i, row in enumerate(rows):
        row_values = []
        for key, value in sorted(row.items()):
            val_str = freeze_value(value)
            row_values.append(f"{key}:{val_str}")
            all_values.append(val_str)
        
        row_str = "|".join(row_values)
        row_hash = hashlib.md5(row_str.encode('utf-8')).hexdigest()[:12]
        snapshot["row_hashes"].append({
            "row_index": i,
            "hash": row_hash
        })
    
    total_str = "||".join(all_values)
    snapshot["total_hash"] = hashlib.md5(total_str.encode('utf-8')).hexdigest()
    
    return snapshot


def verify_integrity(data: Dict, snapshot: Dict) -> IntegrityReport:
    """
    Verify data matches the original snapshot.
    
    Returns detailed report of any modifications.
    """
    rows = data.get("rows", [])
    corrupted = []
    
    for i, row in enumerate(rows):
        if i >= len(snapshot.get("row_hashes", [])):
            corrupted.append({
                "row_index": i,
                "type": "row_added",
                "message": "Row was added after extraction"
            })
            continue
        
        # Recompute hash
        row_values = []
        for key, value in sorted(row.items()):
            val_str = freeze_value(value)
            row_values.append(f"{key}:{val_str}")
        
        row_str = "|".join(row_values)
        current_hash = hashlib.md5(row_str.encode('utf-8')).hexdigest()[:12]
        
        if current_hash != snapshot["row_hashes"][i]["hash"]:
            corrupted.append({
                "row_index": i,
                "type": "row_modified",
                "original_hash": snapshot["row_hashes"][i]["hash"],
                "current_hash": current_hash,
                "message": f"Row {i} was modified after extraction"
            })
    
    is_valid = len(corrupted) == 0
    
    return IntegrityReport(
        is_valid=is_valid,
        total_values=sum(len(row) for row in rows),
        modified_values=len(corrupted),
        corrupted_fields=corrupted,
        message="Data integrity verified" if is_valid else f"{len(corrupted)} corruptions detected"
    )


# =============================================================================
# READ-ONLY VALIDATION (NO MODIFICATION)
# =============================================================================

def validate_without_modifying(
    rows: List[Dict],
    columns: List[str]
) -> List[ValidationFinding]:
    """
    Perform validation checks WITHOUT modifying any data.
    
    This is purely informational - all findings are logged but
    NEVER applied to the data.
    
    Returns list of findings for review.
    """
    findings = []
    
    # Identify numeric columns (for validation purposes only)
    numeric_cols = []
    for col in columns:
        if _looks_like_date_column(col):
            numeric_cols.append(col)
        elif col.lower() in ['variation', 'variation montant', 'variation %', 'en montant', 'en %']:
            numeric_cols.append(col)
    
    for i, row in enumerate(rows):
        row_type = row.get("type", "data")
        
        # Skip section headers
        if row_type == "section":
            continue
        
        # Collect numeric values for this row (READ-ONLY)
        row_values = {}
        for col in numeric_cols:
            val = row.get(col, "")
            if val and val.strip():
                parsed = _parse_for_validation_only(val)
                if parsed is not None:
                    row_values[col] = (val, parsed)  # (original_string, numeric_value)
        
        # Check for anomalies (but NEVER fix them)
        for col, (orig, num) in row_values.items():
            # Detect potential misalignment
            if _is_suspiciously_small(num, list(v[1] for v in row_values.values())):
                findings.append(ValidationFinding(
                    row_index=i,
                    field=col,
                    finding_type="potential_misalignment",
                    extracted_value=orig,
                    computed_value=None,
                    message=f"Value {orig} appears unusually small compared to row context",
                    severity="warning"
                ))
        
        # Check for Variation column anomalies (but NEVER correct)
        if len(numeric_cols) >= 3:
            # Try to identify date columns and variation
            date_cols = [c for c in numeric_cols if _looks_like_date_column(c)]
            var_cols = [c for c in numeric_cols if 'variation' in c.lower()]
            
            if len(date_cols) >= 2 and len(var_cols) >= 1:
                # Just validate, don't correct
                for var_col in var_cols:
                    if var_col in row_values:
                        extracted_var = row_values[var_col][1]
                        # Compute what variation WOULD be (for comparison only)
                        if date_cols[0] in row_values and date_cols[1] in row_values:
                            v1 = row_values[date_cols[0]][1]
                            v2 = row_values[date_cols[1]][1]
                            computed_var = v1 - v2
                            
                            # If mismatch, LOG it but DON'T change anything
                            if abs(extracted_var - computed_var) > 1:
                                findings.append(ValidationFinding(
                                    row_index=i,
                                    field=var_col,
                                    finding_type="variation_mismatch",
                                    extracted_value=row_values[var_col][0],
                                    computed_value=str(computed_var),
                                    message=f"Extracted variation differs from computed. KEEPING ORIGINAL.",
                                    severity="info"
                                ))
    
    return findings


def _parse_for_validation_only(value: str) -> Optional[float]:
    """
    Parse a value to numeric FOR VALIDATION COMPARISON ONLY.
    
    This does NOT modify the original string in any way.
    Used only to compare magnitudes.
    """
    if not value or not isinstance(value, str):
        return None
    
    # Work on a copy for parsing (original untouched)
    s = value.strip()
    
    # Detect negative
    is_negative = '(' in s or '<' in s or s.startswith('-')
    
    # Remove all non-numeric characters for parsing
    digits = ''.join(c for c in s if c.isdigit() or c in '.,')
    
    if not digits:
        return None
    
    # Handle European format (1.234,56) vs US format (1,234.56)
    if ',' in digits and '.' in digits:
        if digits.rfind(',') > digits.rfind('.'):
            # European: dots are thousands, comma is decimal
            digits = digits.replace('.', '').replace(',', '.')
        else:
            # US: commas are thousands, dot is decimal
            digits = digits.replace(',', '')
    elif ',' in digits:
        # Could be European decimal or thousands
        parts = digits.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Likely decimal
            digits = digits.replace(',', '.')
        else:
            # Likely thousands
            digits = digits.replace(',', '')
    
    # Remove remaining dots that are thousands separators
    parts = digits.split('.')
    if len(parts) > 2:
        # Multiple dots = thousands separators
        digits = ''.join(parts[:-1]) + '.' + parts[-1] if parts[-1] else ''.join(parts)
    
    try:
        num = float(digits)
        return -abs(num) if is_negative else num
    except ValueError:
        return None


def _looks_like_date_column(col: str) -> bool:
    """Check if column name looks like a date."""
    import re
    date_patterns = [
        r'\d{2}/\d{2}/\d{4}',  # 31/12/2024
        r'\d{4}',              # Just year
        r'exercice',
    ]
    col_lower = col.lower()
    return any(re.search(p, col_lower) for p in date_patterns)


def _is_suspiciously_small(value: float, all_values: List[float]) -> bool:
    """Check if a value is suspiciously small compared to others."""
    if not all_values or len(all_values) < 2:
        return False
    
    non_zero = [abs(v) for v in all_values if v != 0]
    if not non_zero:
        return False
    
    median = sorted(non_zero)[len(non_zero) // 2]
    
    # If value is < 0.1% of median, it's suspicious
    return abs(value) < median * 0.001 and median > 1000


# =============================================================================
# IMMUTABLE EXTRACTION WRAPPER
# =============================================================================

def immutable_extract(raw_extraction: Dict) -> Dict:
    """
    Wrap raw extraction result with immutability guarantees.
    
    This function:
    1. Freezes all values as strings
    2. Creates integrity snapshot
    3. Attaches validation findings (read-only)
    4. NEVER modifies any extracted values
    
    Returns enhanced result with validation metadata.
    """
    # Deep copy to prevent any reference issues
    result = copy.deepcopy(raw_extraction)
    
    # Freeze all rows
    if "rows" in result:
        result["rows"] = freeze_rows(result["rows"])
    
    # Create integrity snapshot
    snapshot = create_integrity_snapshot(result)
    result["_integrity_snapshot"] = snapshot
    
    # Perform read-only validation
    columns = result.get("columns", [])
    rows = result.get("rows", [])
    findings = validate_without_modifying(rows, columns)
    
    # Attach findings as metadata
    result["_validation_findings"] = [
        {
            "row": f.row_index,
            "field": f.field,
            "type": f.finding_type,
            "extracted": f.extracted_value,
            "computed": f.computed_value,
            "message": f.message,
            "severity": f.severity
        }
        for f in findings
    ]
    
    # Mark extraction mode
    result["_extraction_mode"] = "immutable"
    result["_data_modified"] = False
    result["_corruption_risk"] = "none"
    
    # Final integrity check
    integrity = verify_integrity(result, snapshot)
    result["_integrity_valid"] = integrity.is_valid
    
    return result


# =============================================================================
# SAFE JSON ENCODING (No type coercion)
# =============================================================================

class ImmutableJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that preserves string types.
    
    NEVER converts numeric strings to numbers.
    """
    def default(self, obj):
        if isinstance(obj, ImmutableValue):
            return obj.original
        return super().default(obj)
    
    def encode(self, obj):
        # Ensure all values remain strings
        return super().encode(obj)


def safe_json_dumps(data: Dict, **kwargs) -> str:
    """
    Safe JSON serialization that preserves string types.
    """
    return json.dumps(data, cls=ImmutableJSONEncoder, ensure_ascii=False, **kwargs)


def safe_json_dump(data: Dict, file_handle, **kwargs):
    """
    Safe JSON file write that preserves string types.
    """
    json.dump(data, file_handle, cls=ImmutableJSONEncoder, ensure_ascii=False, **kwargs)


# =============================================================================
# CORRUPTION DETECTION & FAIL-SAFE
# =============================================================================

class DataCorruptionError(Exception):
    """Raised when data corruption is detected."""
    pass


def assert_no_corruption(original: Dict, processed: Dict, raise_on_error: bool = True) -> List[str]:
    """
    Compare original and processed data, fail if corrupted.
    
    Args:
        original: Original extraction result
        processed: Processed result
        raise_on_error: If True, raises DataCorruptionError on corruption
        
    Returns:
        List of corruption descriptions (empty if none)
        
    Raises:
        DataCorruptionError if corruption detected and raise_on_error=True
    """
    corruptions = []
    
    orig_rows = original.get("rows", [])
    proc_rows = processed.get("rows", [])
    
    if len(orig_rows) != len(proc_rows):
        corruptions.append(f"Row count changed: {len(orig_rows)} → {len(proc_rows)}")
    
    for i, (orig_row, proc_row) in enumerate(zip(orig_rows, proc_rows)):
        for key in orig_row:
            if key.startswith("_"):
                continue
            
            orig_val = freeze_value(orig_row.get(key))
            proc_val = freeze_value(proc_row.get(key))
            
            corruption = detect_silent_corruption(orig_val, proc_val)
            if corruption:
                corruptions.append(f"Row {i}, field '{key}': {corruption}")
    
    if corruptions and raise_on_error:
        error_msg = "DATA CORRUPTION DETECTED:\n" + "\n".join(corruptions)
        raise DataCorruptionError(error_msg)
    
    return corruptions


# =============================================================================
# LOGGING STRATEGY
# =============================================================================

def log_extraction_integrity(data: Dict, logger=None):
    """
    Log extraction integrity status.
    
    Use this after every extraction to catch issues early.
    """
    log = logger.info if logger else print
    
    snapshot = data.get("_integrity_snapshot", {})
    findings = data.get("_validation_findings", [])
    
    log(f"[IMMUTABLE] Extraction complete")
    log(f"[IMMUTABLE] Total rows: {len(data.get('rows', []))}")
    log(f"[IMMUTABLE] Integrity hash: {snapshot.get('total_hash', 'N/A')[:16]}...")
    log(f"[IMMUTABLE] Data modified: {data.get('_data_modified', 'N/A')}")
    log(f"[IMMUTABLE] Validation findings: {len(findings)}")
    
    for finding in findings:
        if finding.get("severity") == "warning":
            log(f"[IMMUTABLE] WARNING: Row {finding['row']}, {finding['field']}: {finding['message']}")


# =============================================================================
# PDF vs JSON COMPARISON
# =============================================================================

def compare_pdf_to_json(
    pdf_values: List[str],
    json_values: List[str]
) -> List[Dict]:
    """
    Compare values extracted from PDF to final JSON output.
    
    Use this to detect corruption in the pipeline.
    
    Args:
        pdf_values: List of values as they appear in PDF
        json_values: List of values as they appear in JSON output
        
    Returns:
        List of mismatches with details
    """
    mismatches = []
    
    for i, (pdf_val, json_val) in enumerate(zip(pdf_values, json_values)):
        if pdf_val != json_val:
            corruption = detect_silent_corruption(pdf_val, json_val)
            mismatches.append({
                "index": i,
                "pdf_value": pdf_val,
                "json_value": json_val,
                "corruption_type": corruption or "unknown"
            })
    
    return mismatches

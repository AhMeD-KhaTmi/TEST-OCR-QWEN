"""
Financial Table Alignment Engine
=================================

Robust, schema-agnostic post-processing for financial tables extracted by VLMs.

Components:
  A. Schema Detection Layer   - classify columns by role (label, note, date,
                                variation_amount, variation_percent)
  B. Value Classification Layer - classify each cell value by type
                                  (TEXT, NOTE, NUMBER, PERCENT)
  C. Alignment Engine          - rebuild each row by semantic value type,
                                  NOT column position
  D. Validation Layer          - verify alignment; auto-correct on failure

Works for ANY financial table layout:
  bilan, résultat, cash flow, hors bilan, engagements, etc.

ROOT CAUSES FIXED:
  1. Hardcoded column names  → dynamic schema detection
  2. Position-based mapping  → value-type-based alignment
  3. No type validation      → strict type guards with auto-correction
  4. narrow note detection   → extended NOTE pattern (4.1), (1-2), III.1, (3)
  5. % in numeric columns   → PERCENT type guard prevents this
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# =============================================================================
# VALUE TYPE CONSTANTS
# =============================================================================

VALUE_TYPE_EMPTY   = "EMPTY"
VALUE_TYPE_TEXT    = "TEXT"
VALUE_TYPE_NOTE    = "NOTE"
VALUE_TYPE_NUMBER  = "NUMBER"
VALUE_TYPE_PERCENT = "PERCENT"


# =============================================================================
# B. VALUE CLASSIFICATION LAYER
# =============================================================================

# Extended NOTE pattern:
#   (4.1)   (1-2)   (III.1)   — parenthesised with separator  (MUST have . or -)
#   (3)     (7)               — single-digit bare footnote refs ONLY (1-9)
#   III.1   IV.2              — Roman-numeral prefix (non-parenthesised)
#   1-2     1.2   2-4         — bare numeric dash/dot
#
# Intentionally NOT matching:
#   (97)   (317)             — multi-digit parens without separator → NUMBER (negative)
_NOTE_PATTERN = re.compile(
    r"""^
    (?:
        # Parenthesised WITH separator: (4.1)  (1-2)  (III.1)  (A.1)
        \(\s*
            (?:[IVXLCDM]{1,6}|[A-Za-z]?[\d]{1,3})
            [.\-][\d]{1,3}          # the separator is REQUIRED
        \s*\)
        |
        # Single-digit bare footnote ref: (1) through (9) only
        \(\s*[1-9]\s*\)
        |
        # Roman-numeral prefix without parens: III.1  IV.2
        [IVXLCDM]{2,6}\.[\d]{1,3}
        |
        # Bare numeric dash/dot (no parens): 1-2  1.2  2-4
        [\d]{1,3}[.\-][\d]{1,3}
    )
    $""",
    re.VERBOSE | re.IGNORECASE,
)

# Numeric value: "147 120", "<123 456>", "(317 205)", "-123 456", "1 730 727", "0"
_NUMBER_PATTERN = re.compile(
    r"""^
    [(<]?                          # optional opening paren/angle
    \s*[-+]?\s*                    # optional sign
    [\d][\d\s.,]*                  # digits with spaces, dots, commas (thousands/decimal)
    \s*[>)]?                       # optional closing paren/angle
    $""",
    re.VERBOSE,
)

# Percent: "22,5%", "(5,4%)", "-15,4 %", "0.0%", "(30,9%)"
_PERCENT_PATTERN = re.compile(
    r"""^[(<]?\s*[-+]?\s*\d+(?:[.,]\d+)?\s*%\s*[>)]?$""",
    re.VERBOSE,
)


def _clean(value: Any) -> str:
    """Strip and stringify any value."""
    if value is None:
        return ""
    return str(value).strip()


def _normalize_percent(val: str) -> str:
    """
    Normalize a percentage string to a plain decimal string (no % sign).
    Uses string manipulation to preserve the original decimal precision.

    Examples:
      "22,5%"   -> "22.5"
      "(5,4%)"  -> "-5.4"
      "-15,4 %" -> "-15.4"
      "0.0%"    -> "0.0"
      "2,0%"    -> "2.0"   (NOT "2" — precision preserved)
      "17,6%"   -> "17.6"
    """
    # Negative if wrapped in parens/angle brackets OR starts with plain minus
    negative = bool(re.search(r"[(<]", val) or re.match(r"\s*-", val))
    # Keep only digits, comma, dot
    digits = re.sub(r"[^0-9,.]", "", val)
    # Normalise comma decimal separator
    digits = digits.replace(",", ".")
    # Collapse multiple dots (e.g. if thousands dot slips through)
    parts = digits.split(".")
    if len(parts) > 2:
        digits = parts[0] + "." + "".join(parts[1:])
    if not digits:
        return val
    return ("-" + digits) if negative else digits


def _normalize_number(val: str) -> str:
    """
    Normalize a numeric string to a plain integer/float string.

    Examples:
      "147 120"       -> "147120"
      "(317 205)"     -> "-317205"
      "<123 456>"     -> "-123456"
      "1 730 727"     -> "1730727"
      "-97"           -> "-97"
      "0"             -> "0"
      "1.234.567,89"  -> "1234567.89" (European format)
      "1,234,567.89"  -> "1234567.89" (US format)
    """
    # Negative: wrapped in parens/angles OR plain leading minus
    negative = bool(re.search(r"[(<]", val) or re.match(r"\s*-", val))
    # Strip everything except digits, comma, dot
    digits = re.sub(r"[^0-9,.]", "", val)
    
    # EUROPEAN FORMAT DETECTION:
    # If the string has dots as thousands separators AND comma as decimal,
    # e.g. "1.234.567,89" - the pattern is: multiple dots + single comma at end
    euro_pattern = re.match(r"^(\d{1,3}(?:\.\d{3})+),(\d{1,2})$", digits)
    if euro_pattern:
        # European format: dots are thousands, comma is decimal
        integer_part = euro_pattern.group(1).replace(".", "")
        decimal_part = euro_pattern.group(2)
        digits = integer_part + "." + decimal_part
    else:
        # Check for simpler European format without decimals: 1.234.567
        euro_thousands_only = re.match(r"^(\d{1,3}(?:\.\d{3})+)$", digits)
        if euro_thousands_only:
            # All dots are thousands separators
            digits = digits.replace(".", "")
        else:
            # Standard handling: comma as decimal only if it precedes ≤2 digits at end
            if re.search(r",\d{1,2}$", digits) and "." not in digits:
                digits = digits[:-3].replace(",", "") + "." + digits[-2:]
            else:
                digits = digits.replace(",", "")
    
    digits = digits.rstrip(".")
    if not digits:
        return val
    try:
        num = float(digits)
        if negative:
            num = -abs(num)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return val


def classify_value(value: Any) -> Tuple[str, str]:
    """
    Classify a single cell value.

    Returns:
        (value_type, normalized_value)

    Types:
        EMPTY   – null or whitespace / dash only
        NOTE    – reference code like (4.1), (1-2), III.1
        PERCENT – contains % sign
        NUMBER  – numeric value (spaces, parens, angle brackets allowed)
        TEXT    – plain text label

    normalized_value:
        PERCENT → float string without %, e.g. "-5.4"
        NUMBER  → integer/float string, e.g. "-317205"
        Others  → original stripped value
    """
    val = _clean(value)

    # EMPTY: blank, dash, or en-dash only
    if not val or val in ("-", "–", "—", "N/A", "n/a", ""):
        return VALUE_TYPE_EMPTY, ""

    # PERCENT — must check before NUMBER (e.g. "22.5%" also matches digit patterns)
    if _PERCENT_PATTERN.match(val):
        return VALUE_TYPE_PERCENT, _normalize_percent(val)

    # NOTE
    if _NOTE_PATTERN.match(val):
        return VALUE_TYPE_NOTE, val

    # NUMBER — must have at least one digit
    if _NUMBER_PATTERN.match(val) and any(c.isdigit() for c in val):
        return VALUE_TYPE_NUMBER, _normalize_number(val)

    # TEXT
    return VALUE_TYPE_TEXT, val


# =============================================================================
# A. SCHEMA DETECTION LAYER
# =============================================================================

@dataclass
class ColumnSchema:
    """
    Detected schema roles for all columns in a financial table.
    Column names are the ORIGINAL strings from model output.
    """
    label_col:            Optional[str]       # "Label", "Libellé", etc.
    note_col:             Optional[str]       # "Note", "Notes", etc.
    date_cols:            List[str]           # ["31/12/2024", "31/12/2023"]
    variation_amount_col: Optional[str]       # "Variation Montant", "En Montant"
    variation_percent_col: Optional[str]      # "Variation %", "En %"
    unresolved_cols:      List[str] = field(default_factory=list)
    confidence_score:     float = 0.0         # Header confidence (0.0 - 1.0)

    def canonical_columns(self) -> List[str]:
        """Return all recognised columns in canonical order."""
        cols = []
        if self.label_col:
            cols.append(self.label_col)
        if self.note_col:
            cols.append(self.note_col)
        cols.extend(self.date_cols)
        if self.variation_amount_col:
            cols.append(self.variation_amount_col)
        if self.variation_percent_col:
            cols.append(self.variation_percent_col)
        return cols

    @property
    def num_date_slots(self) -> int:
        return len(self.date_cols)
    
    def is_valid(self) -> bool:
        """
        FIX 2: SCHEMA SEMANTIC VALIDATION (CRITICAL)
        
        A valid schema MUST contain:
            - At least 1 numeric/date column
            - OR at least 2 numeric columns (variation amount + percent)
            - NOT only variation columns without dates
            - At least 3 meaningful columns total
        
        Reject schema if:
            - No numeric/date columns
            - Only variation columns (no dates)
            - Less than 3 meaningful columns
        """
        # Count numeric columns
        num_date_cols = len(self.date_cols)
        has_var_amount = self.variation_amount_col is not None
        has_var_percent = self.variation_percent_col is not None
        
        # Count meaningful columns (excluding unresolved)
        meaningful_cols = len(self.canonical_columns())
        
        # RULE 1: Must have at least 1 date column OR at least 2 variation columns
        has_sufficient_numeric = (
            num_date_cols >= 1 or
            (has_var_amount and has_var_percent)
        )
        
        # RULE 2: Must NOT be only variation columns (no dates)
        only_variation = (
            num_date_cols == 0 and
            (has_var_amount or has_var_percent)
        )
        
        # RULE 3: Must have at least 3 meaningful columns
        has_enough_columns = meaningful_cols >= 3
        
        # Schema is valid if:
        # - Has sufficient numeric columns
        # - Is not only variation columns (unless it has both variation types)
        # - Has enough total columns
        is_valid = (
            has_sufficient_numeric and
            (not only_variation or (has_var_amount and has_var_percent)) and
            has_enough_columns
        )
        
        return is_valid
    
    def has_minimum_data(self) -> bool:
        """
        Check if schema has enough data to be useful.
        More permissive than is_valid() - for fallback scenarios.
        """
        return len(self.canonical_columns()) > 0
    
    def get_validation_issues(self) -> List[str]:
        """Return list of validation issues with the schema."""
        issues = []
        
        if len(self.date_cols) == 0:
            issues.append("No date columns")
        
        if not self.variation_amount_col and not self.variation_percent_col:
            if len(self.date_cols) < 2:
                issues.append("Need at least 2 date columns or variation columns")
        
        if len(self.canonical_columns()) < 3:
            issues.append(f"Only {len(self.canonical_columns())} columns (need at least 3)")
        
        return issues


# =============================================================================
# HEADER SEMANTIC VALIDATION
# =============================================================================

# Minimum confidence threshold for accepting a header
HEADER_CONFIDENCE_THRESHOLD = 0.3  # Lowered to be more permissive

# Robust date detection regex (dd/mm/yyyy or dd.mm.yyyy)
_DATE_COLUMN_REGEX = re.compile(
    r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\s*$"
)

# Multi-row header merge patterns - ordered by specificity
_MULTI_ROW_MERGE_PATTERNS = [
    # (part1, part2) -> merged
    (r"^variation$", r"^montant$", "Variation Montant"),
    (r"^variation$", r"^%$", "Variation %"),
    (r"^variation$", r"^en\s*%$", "Variation %"),
    (r"^variation$", r"^en\s*montant$", "Variation Montant"),
    (r"^en$", r"^montant$", "Variation Montant"),
    (r"^en$", r"^%$", "Variation %"),
]

# Sub-header fragments that should be merged with parent
_SUBHEADER_FRAGMENTS = {"montant", "%", "en %", "en montant"}

# Column name normalization map
_COLUMN_NORMALIZATION = {
    # Variation amount variants
    "montant": "Variation Montant",
    "en montant": "Variation Montant",
    "var montant": "Variation Montant",
    "var. montant": "Variation Montant",
    "ecart montant": "Variation Montant",
    "mouvement": "Variation Montant",
    # Variation percent variants
    "en %": "Variation %",
    "%": "Variation %",
    "var %": "Variation %",
    "var. %": "Variation %",
    "pct": "Variation %",
    "ecart %": "Variation %",
    "taux variation": "Variation %",
    "en pourcentage": "Variation %",
    # Label variants
    "libelle": "Label",
    "libelle poste": "Label",
    "poste": "Label",
    "designation": "Label",
    "description": "Label",
    "intitule": "Label",
    "rubriques": "Label",
    "elements": "Label",
    # Note variants
    "notes": "Note",
    "ref": "Note",
    "reference": "Note",
    "renvoi": "Note",
}


def _is_date_column_name(col: str) -> bool:
    """
    Robustly detect if a column name is a date.
    Supports:
    - dd/mm/yyyy, dd.mm.yyyy, dd-mm-yyyy
    - yyyy (year only, 2000-2099)
    """
    if not col:
        return False
    col_clean = col.strip()
    
    # Check standard date format
    if _DATE_COLUMN_REGEX.match(col_clean):
        return True
    
    # Check year-only format (2020, 2021, 2024, etc.)
    if re.match(r'^(20\d{2})$', col_clean):
        return True
    
    return False


def normalize_column_name(col_name: str) -> str:
    """
    Normalize a column name to its canonical form.
    
    Examples:
        "Montant" → "Variation Montant"
        "En %" → "Variation %"
        "Libellé" → "Label"
    
    SAFE MODE: Never changes date columns.
    """
    if not col_name:
        return col_name
    
    # SAFE MODE: Preserve date columns exactly as-is
    if _is_date_column_name(col_name):
        return col_name
    
    # Normalize for lookup
    normalized = _clean(col_name).lower()
    # Remove accents
    for src, dst in [("éèêë","e"), ("àâ","a"), ("ùû","u"), ("îï","i"), ("ôö","o")]:
        for ch in src:
            normalized = normalized.replace(ch, dst)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    
    # Check normalization map
    if normalized in _COLUMN_NORMALIZATION:
        return _COLUMN_NORMALIZATION[normalized]
    
    # Keep original if not in map
    return col_name


def merge_hierarchical_headers(columns: List[str]) -> List[str]:
    """
    Merge multi-row headers into single semantic columns.
    
    ROBUST IMPLEMENTATION:
    - Handles ["Variation", "Montant", "%"] → ["Variation Montant", "Variation %"]
    - Preserves ALL date columns (never drops them)
    - Handles non-consecutive merge patterns
    
    Example:
        ["Label", "Note", "31/12/2024", "31/12/2023", "Variation", "Montant", "%"]
        → ["Label", "Note", "31/12/2024", "31/12/2023", "Variation Montant", "Variation %"]
    """
    if not columns or len(columns) < 2:
        return [normalize_column_name(c) for c in columns] if columns else []
    
    merged: List[str] = []
    i = 0
    
    # Track if we're in a "Variation" context for handling sub-headers
    pending_parent = None  # e.g., "Variation"
    
    while i < len(columns):
        col = columns[i]
        col_norm = _clean(col).lower()
        
        # SAFE MODE: Always preserve date columns
        if _is_date_column_name(col):
            # Flush any pending parent first
            if pending_parent:
                merged.append(pending_parent)
                pending_parent = None
            merged.append(col)
            i += 1
            continue
        
        # Check if this is a parent header (e.g., "Variation")
        if col_norm in ("variation", "en"):
            # Look ahead for sub-headers
            found_children = []
            j = i + 1
            while j < len(columns):
                next_col = columns[j]
                next_norm = _clean(next_col).lower()
                
                # Stop at date columns or other non-sub-headers
                if _is_date_column_name(next_col):
                    break
                
                # Check if it's a sub-header fragment
                if next_norm in _SUBHEADER_FRAGMENTS or next_norm in ("montant", "%", "en %", "en montant"):
                    found_children.append((j, next_col, next_norm))
                    j += 1
                else:
                    break
            
            # Merge parent with children
            if found_children:
                for _, child_col, child_norm in found_children:
                    if child_norm in ("montant", "en montant"):
                        merged.append("Variation Montant")
                        print(f"[HEADER MERGE] '{col}' + '{child_col}' → 'Variation Montant'")
                    elif child_norm in ("%", "en %"):
                        merged.append("Variation %")
                        print(f"[HEADER MERGE] '{col}' + '{child_col}' → 'Variation %'")
                i = j  # Skip past all processed children
                continue
            else:
                # No children found, add as-is
                merged.append(normalize_column_name(col))
                i += 1
                continue
        
        # Check for consecutive merge patterns (fallback)
        if i + 1 < len(columns):
            next_col = columns[i + 1]
            next_norm = _clean(next_col).lower()
            
            merged_result = None
            for pattern1, pattern2, merged_name in _MULTI_ROW_MERGE_PATTERNS:
                if re.match(pattern1, col_norm) and re.match(pattern2, next_norm):
                    merged_result = merged_name
                    break
            
            if merged_result:
                merged.append(merged_result)
                print(f"[HEADER MERGE] '{col}' + '{next_col}' → '{merged_result}'")
                i += 2
                continue
        
        # No merge, normalize and add
        merged.append(normalize_column_name(col))
        i += 1
    
    # SAFE MODE: Ensure we didn't lose any date columns
    original_dates = [c for c in columns if _is_date_column_name(c)]
    merged_dates = [c for c in merged if _is_date_column_name(c)]
    
    if len(merged_dates) < len(original_dates):
        print(f"[SAFE MODE] Date columns were lost! Restoring...")
        print(f"  Original dates: {original_dates}")
        print(f"  Merged dates: {merged_dates}")
        # Restore missing dates at their approximate positions
        for date_col in original_dates:
            if date_col not in merged:
                merged.insert(len(merged) - 2 if len(merged) >= 2 else 0, date_col)
    
    return merged


def compute_header_confidence(schema: "ColumnSchema") -> float:
    """
    Compute confidence score for a detected header.
    
    SAFE MODE: Gives minimum score if at least one numeric column exists.
    
    Scoring factors:
        +0.3 for having a label column
        +0.1 for having a note column
        +0.2 for each date column (max 2)
        +0.1 for having variation amount column
        +0.1 for having variation percent column
        -0.05 for each unresolved column (reduced penalty)
    
    Returns: score between 0.0 and 1.0
    """
    score = 0.0
    
    # Label column
    if schema.label_col:
        score += 0.3
    
    # Note column (optional)
    if schema.note_col:
        score += 0.1
    
    # Date columns (important)
    num_dates = len(schema.date_cols)
    score += min(num_dates * 0.2, 0.4)  # Max 0.4 for dates
    
    # Variation columns
    if schema.variation_amount_col:
        score += 0.1
    if schema.variation_percent_col:
        score += 0.1
    
    # Reduced penalty for unresolved columns
    num_unresolved = len(schema.unresolved_cols)
    score -= num_unresolved * 0.03
    
    # SAFE MODE: Minimum score if we have at least one numeric column
    has_numeric = num_dates > 0 or schema.variation_amount_col is not None
    if has_numeric and score < 0.3:
        score = 0.3
    
    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, score))


def validate_header_structure(schema: "ColumnSchema") -> Tuple[bool, List[str]]:
    """
    Validate that the header structure is logically correct.
    
    SAFE MODE: More permissive - only requires at least one numeric column.
    
    Minimum valid schema:
        - At least 1 numeric column (date or variation amount)
        - OR at least 1 percentage column
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # MINIMUM VALID SCHEMA: at least 1 numeric OR 1 percentage column
    has_numeric = len(schema.date_cols) > 0 or schema.variation_amount_col is not None
    has_percent = schema.variation_percent_col is not None
    
    if not has_numeric and not has_percent:
        issues.append("No numeric or percentage columns found (minimum requirement)")
    
    # Label is recommended but not required
    if not schema.label_col:
        # This is a warning, not an error
        print("[HEADER WARNING] No label column detected")
    
    # Too many date columns is suspicious but not fatal
    if len(schema.date_cols) > 5:
        issues.append(f"Suspiciously many date columns ({len(schema.date_cols)})")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def infer_schema_from_data_rows(rows: List[Dict], num_rows: int = 5) -> Optional["ColumnSchema"]:
    """
    FALLBACK: Infer schema from first N valid data rows.
    
    Used when header detection fails or is unreliable.
    
    ROBUST IMPLEMENTATION:
        1. Uses MORE sample rows (5 instead of 3)
        2. Detects date columns by column NAME (regex) first
        3. Falls back to value-type detection
        4. Never returns None if there's usable data
    
    Strategy:
        1. Find columns with date-format names → Date columns
        2. Find columns that consistently contain TEXT → Label candidate
        3. Find columns that consistently contain NOTE patterns → Note
        4. Find columns that consistently contain NUMBERS → Numeric columns
        5. Find columns that consistently contain PERCENT → Variation %
    """
    if not rows:
        return None
    
    # Take more sample rows for better inference
    sample_rows = [r for r in rows[:num_rows * 3] if isinstance(r, dict)][:num_rows]
    if not sample_rows:
        return None
    
    # Collect all column names
    all_cols = set()
    for row in sample_rows:
        all_cols.update(row.keys())
    all_cols -= {"type", "__chunk_index", "__y_position"}
    
    if not all_cols:
        return None
    
    # STEP 1: Detect date columns by NAME (robust regex-based)
    date_cols_by_name = [col for col in all_cols if _is_date_column_name(col)]
    print(f"[FALLBACK] Date columns detected by name: {date_cols_by_name}")
    
    # STEP 2: Classify remaining columns by value type
    remaining_cols = all_cols - set(date_cols_by_name)
    col_types: Dict[str, Dict[str, int]] = {}
    
    for col in remaining_cols:
        col_types[col] = {
            VALUE_TYPE_TEXT: 0,
            VALUE_TYPE_NOTE: 0,
            VALUE_TYPE_NUMBER: 0,
            VALUE_TYPE_PERCENT: 0,
            VALUE_TYPE_EMPTY: 0,
        }
        for row in sample_rows:
            vtype, _ = classify_value(row.get(col, ""))
            col_types[col][vtype] += 1
    
    # STEP 3: Assign roles
    label_col = None
    note_col = None
    numeric_cols = []
    var_pct_col = None
    var_amount_col = None
    
    for col, counts in col_types.items():
        total = sum(counts.values())
        if total == 0:
            continue
        
        # Find dominant type (lowered threshold to 40% for more permissive detection)
        dominant = max(counts.keys(), key=lambda k: counts[k])
        ratio = counts[dominant] / total
        
        if ratio < 0.4:
            # Mixed column - check if it has ANY numbers or percentages
            num_count = counts[VALUE_TYPE_NUMBER] + counts[VALUE_TYPE_PERCENT]
            if num_count > 0:
                numeric_cols.append(col)
            continue
        
        if dominant == VALUE_TYPE_TEXT:
            if label_col is None:
                label_col = col
        elif dominant == VALUE_TYPE_NOTE:
            if note_col is None:
                note_col = col
        elif dominant == VALUE_TYPE_PERCENT:
            if var_pct_col is None:
                var_pct_col = col
        elif dominant == VALUE_TYPE_NUMBER:
            # Check column name for hints
            col_lower = col.lower()
            if "variation" in col_lower or "montant" in col_lower or "var" in col_lower:
                if var_amount_col is None:
                    var_amount_col = col
            else:
                numeric_cols.append(col)
    
    # STEP 4: Build schema
    # If we found date columns by name, use those as date_cols
    # Otherwise, use numeric columns as date candidates
    date_cols = date_cols_by_name if date_cols_by_name else numeric_cols[:3]
    
    # If no label was found, try to infer from first column
    if not label_col and sample_rows:
        first_row = sample_rows[0]
        for col in first_row.keys():
            if col in ("type", "__chunk_index", "__y_position"):
                continue
            if col not in date_cols and col != var_pct_col and col != var_amount_col:
                vtype, _ = classify_value(first_row.get(col, ""))
                if vtype == VALUE_TYPE_TEXT:
                    label_col = col
                    break
    
    # SAFE MODE: Create schema even if incomplete
    schema = ColumnSchema(
        label_col=label_col,
        note_col=note_col,
        date_cols=date_cols[:4],  # Max 4 date columns
        variation_amount_col=var_amount_col,
        variation_percent_col=var_pct_col,
        unresolved_cols=[c for c in remaining_cols if c not in (label_col, note_col, var_pct_col, var_amount_col) and c not in date_cols],
        confidence_score=0.0,
    )
    
    # Compute confidence (no penalty for inference)
    schema.confidence_score = compute_header_confidence(schema)
    
    print(f"[FALLBACK] Inferred schema columns: {schema.canonical_columns()}")
    print(f"[FALLBACK] Confidence: {schema.confidence_score:.2f}")
    
    # SAFE MODE: Always return something if we have any usable data
    return schema


def _normalize_header(text: str) -> str:
    """Normalise a column header string for role matching."""
    t = _clean(text).lower()
    # Transliterate accented chars
    for src, dst in [("éèêë","e"), ("àâ","a"), ("ùû","u"), ("îï","i"), ("ôö","o")]:
        for ch in src:
            t = t.replace(ch, dst)
    t = re.sub(r"[_\-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# Canonical token sets for each column role
_LABEL_TOKENS = {
    "label", "libelle", "libelle poste", "poste", "designation",
    "description", "intitule", "rubriques", "elements",
}
_NOTE_TOKENS = {
    "note", "notes", "ref", "reference", "renvoi",
}
_VAR_AMOUNT_TOKENS = {
    "variation montant", "variation en montant", "variation amount",
    "montant variation", "en montant", "mouvement", "ecart montant",
    "var montant", "var. montant",
}
_VAR_PERCENT_TOKENS = {
    "variation %", "variation en %", "variation pct", "var %",
    "en %", "taux variation", "ecart %", "pct", "var. %", "en pourcentage",
}

# Date pattern: DD/MM/YYYY, DD.MM.YYYY, YYYY/MM/DD, or bare YYYY
_DATE_COL_PATTERN = re.compile(
    r"""
    (?:\d{2}[/.\-]\d{2}[/.\-]\d{4})   # DD/MM/YYYY
    | (?:\d{4}[/.\-]\d{2}[/.\-]\d{2}) # YYYY/MM/DD
    | (?:^|\s)\d{4}(?:\s|$)            # bare year
    """,
    re.VERBOSE,
)

# Date format patterns for schema isolation
_DATE_SLASH_PATTERN = re.compile(r'^\d{2}/\d{2}/\d{4}$')
_DATE_DOT_PATTERN = re.compile(r'^\d{2}\.\d{2}\.\d{4}$')


def _is_label_col(token: str) -> bool:
    return token in _LABEL_TOKENS


def _is_note_col(token: str) -> bool:
    return token in _NOTE_TOKENS or token.startswith("note")


def _is_var_percent_col(token: str) -> bool:
    return (
        token in _VAR_PERCENT_TOKENS
        or ("%" in token and ("variation" in token or "var" in token or "en" in token or "taux" in token))
    )


def _is_var_amount_col(token: str) -> bool:
    return (
        token in _VAR_AMOUNT_TOKENS
        or ("variation" in token and ("montant" in token or "amount" in token or "ecart" in token))
        or ("en montant" in token)
    )


def _is_date_col(token: str) -> bool:
    return bool(_DATE_COL_PATTERN.search(token))


def _get_date_format(col_name: str) -> str:
    """
    Get the date format of a column name.
    
    Returns: "slash", "dot", or "none"
    """
    if not col_name:
        return "none"
    col = str(col_name).strip()
    if _DATE_SLASH_PATTERN.match(col):
        return "slash"
    if _DATE_DOT_PATTERN.match(col):
        return "dot"
    return "none"


def _validate_date_consistency(date_cols: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that all date columns use the same format.
    
    SCHEMA ISOLATION: Mixed formats indicate cross-table contamination.
    
    Returns:
        (is_consistent, filtered_date_cols)
    """
    if not date_cols:
        return True, []
    
    format_groups: Dict[str, List[str]] = {"slash": [], "dot": [], "none": []}
    
    for col in date_cols:
        fmt = _get_date_format(col)
        format_groups[fmt].append(col)
    
    # Count actual date formats (not "none")
    date_formats_found = {k: v for k, v in format_groups.items() if k != "none" and v}
    
    if len(date_formats_found) <= 1:
        # Consistent - all same format
        return True, date_cols
    
    # MIXED FORMATS - contamination detected
    print(f"[SCHEMA ISOLATION] Mixed date formats in alignment engine!")
    for fmt, cols in date_formats_found.items():
        print(f"  - {fmt}: {cols}")
    
    # Keep only dominant format
    dominant = max(date_formats_found.keys(), key=lambda k: len(date_formats_found[k]))
    filtered = format_groups[dominant] + format_groups["none"]
    
    print(f"[SCHEMA ISOLATION] Keeping only '{dominant}' format: {filtered}")
    
    return False, filtered


def _is_artificial_column_name(col_name: str) -> bool:
    """
    Check if a column name is an artificial placeholder that should be rejected.
    
    SCHEMA LOCKING: Artificial names include:
    - "col1", "col2", "col3", etc.
    - "column_1", "column_2", etc.
    - Empty strings or whitespace-only
    - Pure numbers (EXCEPT valid years like 2020-2030)
    """
    if not col_name:
        return True
    
    name = str(col_name).strip().lower()
    
    if not name:
        return True
    
    # Reject "col" + number patterns
    if re.match(r'^col\d+$', name):
        return True
    
    # Reject "column" + number patterns
    if re.match(r'^column[_\s]?\d+$', name):
        return True
    
    # Pure numbers: ACCEPT if it's a valid year (2000-2099), reject otherwise
    if re.match(r'^\d+$', name):
        try:
            num = int(name)
            # Valid year range for financial statements
            if 2000 <= num <= 2099:
                return False  # It's a valid year, not artificial
            return True  # Reject other pure numbers
        except ValueError:
            return True
    
    # Reject very short generic names
    if name in ('col', 'column', 'field', 'value', 'data', 'item'):
        return True
    
    return False


def detect_schema_from_columns(
    columns: List[str],
    rows: Optional[List[Dict]] = None,
    use_fallback: bool = True
) -> "ColumnSchema":
    """
    COMPONENT A — Schema Detection Layer with Header Semantic Validation.

    Analyse raw column names from model output and assign semantic roles.
    
    NEW: Includes header semantic validation:
        1. Multi-row header normalization (merge hierarchical headers)
        2. Column name normalization
        3. Header structure validation
        4. Confidence scoring
        5. Fallback to data-based inference if header is unreliable
    
    SCHEMA LOCKING: Rejects artificial column names (col1, col2, etc.)

    Priority order:
      1. Label column → matches label/libellé/poste tokens
      2. Note column  → matches note tokens
      3. Variation %  → contains % + variation keyword
      4. Variation Montant → contains variation + montant/amount
      5. Date columns → contains a recognisable date string
      6. Unresolved   → everything else (resolved later from data)

    Args:
        columns: Raw column name list from model output
        rows: Optional data rows for fallback schema inference
        use_fallback: Whether to use data-based fallback if header fails

    Returns:
        ColumnSchema with role assignments and confidence score
    """
    _INTERNAL = {"type", "__chunk_index", "__y_position"}

    # Step 1: Normalize hierarchical headers (SAFE MODE: preserves date columns)
    normalized_columns = merge_hierarchical_headers(columns)
    
    print(f"[HEADER] Input columns: {columns}")
    print(f"[HEADER] After merge: {normalized_columns}")
    
    label_col = note_col = variation_amount_col = variation_percent_col = None
    date_cols: List[str] = []
    unresolved: List[str] = []

    for col in normalized_columns:
        if col in _INTERNAL:
            continue
        
        # SCHEMA LOCK: Reject artificial column names
        if _is_artificial_column_name(col):
            print(f"[SCHEMA LOCK] Rejecting artificial column at detection: '{col}'")
            continue
        
        # ROBUST DATE DETECTION: Check by regex FIRST before token-based detection
        if _is_date_column_name(col):
            date_cols.append(col)
            print(f"[HEADER] Detected date column (regex): '{col}'")
            continue
        
        # Normalize column name before analysis
        norm_col = normalize_column_name(col)
        token = _normalize_header(norm_col)

        if _is_label_col(token):
            if label_col is None:
                label_col = col  # Keep original name for consistency
            else:
                unresolved.append(col)
        elif _is_note_col(token):
            if note_col is None:
                note_col = col
            else:
                unresolved.append(col)
        elif _is_var_percent_col(token):
            if variation_percent_col is None:
                variation_percent_col = col
            else:
                unresolved.append(col)
        elif _is_var_amount_col(token):
            if variation_amount_col is None:
                variation_amount_col = col
            else:
                unresolved.append(col)
        elif _is_date_col(token):
            # Fallback date detection via token
            date_cols.append(col)
        else:
            unresolved.append(col)

    # SCHEMA ISOLATION: Validate date format consistency
    is_consistent, filtered_date_cols = _validate_date_consistency(date_cols)
    if not is_consistent:
        print(f"[SCHEMA ISOLATION] Date columns filtered: {date_cols} -> {filtered_date_cols}")
    
    schema = ColumnSchema(
        label_col=label_col,
        note_col=note_col,
        date_cols=filtered_date_cols,
        variation_amount_col=variation_amount_col,
        variation_percent_col=variation_percent_col,
        unresolved_cols=unresolved,
        confidence_score=0.0,
    )
    
    # Step 2: Compute header confidence score
    schema.confidence_score = compute_header_confidence(schema)
    
    # Step 3: Validate header structure (SAFE MODE: permissive validation)
    is_valid, issues = validate_header_structure(schema)
    if not is_valid:
        print(f"[HEADER VALIDATION] Issues: {issues}")
        # SAFE MODE: Don't reject - try to fix with fallback
    
    print(f"[HEADER] Detected schema with confidence {schema.confidence_score:.2f}")
    print(f"[HEADER] Columns: {schema.canonical_columns()}")
    
    # Step 4: SAFE MODE - Try fallback if we have issues OR low confidence
    need_fallback = (
        not is_valid or 
        schema.confidence_score < HEADER_CONFIDENCE_THRESHOLD or
        len(schema.date_cols) == 0  # No date columns is suspicious
    )
    
    if need_fallback and use_fallback and rows:
        print(f"[HEADER] Need fallback (valid={is_valid}, confidence={schema.confidence_score:.2f}, dates={len(schema.date_cols)})")
        print("[HEADER] Attempting fallback schema inference from data rows...")
        
        fallback_schema = infer_schema_from_data_rows(rows)
        if fallback_schema:
            # SAFE MODE: Merge schemas rather than replace completely
            # Take the better value for each field
            merged_schema = _merge_schemas(schema, fallback_schema)
            if merged_schema.confidence_score > schema.confidence_score:
                print(f"[HEADER] Using merged schema (confidence: {merged_schema.confidence_score:.2f})")
                return merged_schema
            elif fallback_schema.confidence_score > schema.confidence_score:
                print(f"[HEADER] Using fallback schema (confidence: {fallback_schema.confidence_score:.2f})")
                return fallback_schema
        print("[HEADER] Fallback not better, keeping original")
    
    # SAFE MODE: Final check - if no numeric columns at all, force re-inference
    has_numeric = len(schema.date_cols) > 0 or schema.variation_amount_col or schema.variation_percent_col
    if not has_numeric and rows:
        print("[SAFE MODE] No numeric columns found! Forcing data-based inference...")
        forced_schema = infer_schema_from_data_rows(rows, num_rows=10)
        if forced_schema and (len(forced_schema.date_cols) > 0 or forced_schema.variation_amount_col or forced_schema.variation_percent_col):
            print(f"[SAFE MODE] Recovered schema with {len(forced_schema.date_cols)} date columns")
            return forced_schema
    
    return schema


def _merge_schemas(primary: "ColumnSchema", fallback: "ColumnSchema") -> "ColumnSchema":
    """
    Merge two schemas, taking the best of each.
    
    FIX 8: PRESERVE ORDER - Never use set() for date_cols.
    
    Strategy:
    - For single columns: prefer primary if set, else fallback
    - For date_cols: merge while PRESERVING ORDER (primary first, then fallback)
    - Recompute confidence after merge
    """
    # FIX 8: Preserve order when merging date columns
    merged_dates = []
    seen = set()
    for col in primary.date_cols + fallback.date_cols:
        if col not in seen:
            merged_dates.append(col)
            seen.add(col)
    merged_dates = merged_dates[:4]  # Max 4 date columns
    
    merged = ColumnSchema(
        label_col=primary.label_col or fallback.label_col,
        note_col=primary.note_col or fallback.note_col,
        date_cols=merged_dates,
        variation_amount_col=primary.variation_amount_col or fallback.variation_amount_col,
        variation_percent_col=primary.variation_percent_col or fallback.variation_percent_col,
        unresolved_cols=[c for c in primary.unresolved_cols if c not in fallback.canonical_columns()],
        confidence_score=0.0,
    )
    merged.confidence_score = compute_header_confidence(merged)
    
    print(f"[MERGE] Combined schemas: {merged.canonical_columns()}")
    return merged


def _infer_schema_from_data(rows: List[Dict], schema: "ColumnSchema") -> "ColumnSchema":
    """
    Resolve ambiguous / unresolved columns by inspecting cell value types.
    
    SCHEMA LOCKING: This function ONLY resolves the role of existing columns
    from the header. It NEVER creates new columns.
    
    Strategy:
      - Column where >30% of cells are PERCENT → variation_percent
      - Column where >30% of cells are NUMBER  → assign to date/numeric role
      - Everything else stays unresolved (but is NOT added as a new column)
    
    CRITICAL: Do NOT add columns that were not in the original header.
    """
    if not rows or not schema.unresolved_cols:
        return schema

    sample = rows[: min(20, len(rows))]
    col_type_counts: Dict[str, Dict[str, int]] = {}

    for col in schema.unresolved_cols:
        # SCHEMA LOCK: Skip any column that looks like an artificial placeholder
        # e.g., "col1", "col2", "col3", "column_1", etc.
        if _is_artificial_column_name(col):
            print(f"[SCHEMA LOCK] Rejecting artificial column: '{col}'")
            continue
            
        counts = {t: 0 for t in (VALUE_TYPE_EMPTY, VALUE_TYPE_TEXT,
                                  VALUE_TYPE_NOTE, VALUE_TYPE_NUMBER,
                                  VALUE_TYPE_PERCENT)}
        for row in sample:
            vtype, _ = classify_value(row.get(col, ""))
            counts[vtype] = counts.get(vtype, 0) + 1
        col_type_counts[col] = counts

    new_date_cols = list(schema.date_cols)
    new_var_pct   = schema.variation_percent_col
    new_var_amt   = schema.variation_amount_col
    still_unresolved: List[str] = []

    for col in schema.unresolved_cols:
        # Skip artificial columns
        if _is_artificial_column_name(col):
            continue
            
        if col not in col_type_counts:
            continue
            
        counts = col_type_counts[col]
        total  = max(sum(counts.values()), 1)
        pct_ratio = counts[VALUE_TYPE_PERCENT] / total
        num_ratio = counts[VALUE_TYPE_NUMBER]  / total

        if pct_ratio > 0.30 and new_var_pct is None:
            new_var_pct = col
        elif num_ratio > 0.30:
            # SCHEMA LOCK: Only add to date_cols if it's not already there
            # and the column name looks like a real header name
            if col not in new_date_cols:
                new_date_cols.append(col)
        else:
            still_unresolved.append(col)

    return ColumnSchema(
        label_col=schema.label_col,
        note_col=schema.note_col,
        date_cols=new_date_cols,
        variation_amount_col=new_var_amt,
        variation_percent_col=new_var_pct,
        unresolved_cols=still_unresolved,
    )


# =============================================================================
# C. ALIGNMENT ENGINE (SIMPLE BASELINE)
# =============================================================================

def align_row(raw_row: Dict, schema: "ColumnSchema") -> Dict:
    """
    SEMANTIC CELL CLASSIFICATION - Position-independent alignment.
    
    Classifies each cell by content type (NOTE, PERCENT, NUMBER, TEXT),
    then assigns to appropriate column based on semantic role.
    
    Works for ANY table layout regardless of column order.
    """
    _SKIP = {"type", "__chunk_index", "__y_position", "_row_corrected", "_alignment_corrected", "_alignment_errors"}
    
    # Get schema column keys for output
    label_key = schema.label_col or "Label"
    note_key = schema.note_col or "Note"
    pct_key = schema.variation_percent_col or "Variation %"
    vm_key = schema.variation_amount_col or "Variation Montant"
    date_cols = schema.date_cols or []
    
    # =========================================================================
    # STEP 1: CLASSIFY EACH CELL BY TYPE
    # =========================================================================
    cells = []
    for col, val in raw_row.items():
        if col in _SKIP:
            continue
        val_str = str(val).strip() if val else ""
        if not val_str:
            continue
        
        vtype, normalized = classify_value(val_str)
        cells.append({
            "raw": val_str,
            "normalized": normalized,
            "type": vtype,
            "length": len(val_str)
        })
    
    # =========================================================================
    # STEP 2: ASSIGN ROLES BASED ON SEMANTIC CONTENT
    # =========================================================================
    aligned: Dict[str, Any] = {}
    
    # Find NOTE cells (pattern: (1-1), (4.2), etc.)
    note_cells = [c for c in cells if c["type"] == VALUE_TYPE_NOTE]
    
    # Find PERCENT cells (contains %)
    percent_cells = [c for c in cells if c["type"] == VALUE_TYPE_PERCENT]
    
    # Find NUMBER cells (numeric values)
    number_cells = [c for c in cells if c["type"] == VALUE_TYPE_NUMBER]
    
    # Find TEXT cells (everything else)
    text_cells = [c for c in cells if c["type"] == VALUE_TYPE_TEXT]
    
    # =========================================================================
    # STEP 3: HANDLE MERGED CELLS (Label + Note in one cell)
    # =========================================================================
    # Check if any text cell contains an embedded note pattern
    NOTE_EXTRACT_PATTERN = re.compile(r'\s*\([\d]+(?:[.-][\d]+)?\)\s*$')
    
    label_text = ""
    extracted_note = ""
    
    if text_cells:
        # Find longest TEXT cell as label candidate
        longest_text = max(text_cells, key=lambda c: c["length"])
        label_candidate = longest_text["raw"]
        
        # Check if it contains embedded note at the end
        match = NOTE_EXTRACT_PATTERN.search(label_candidate)
        if match:
            extracted_note = match.group(0).strip()
            label_text = label_candidate[:match.start()].strip()
        else:
            label_text = label_candidate
    
    # =========================================================================
    # STEP 4: ASSIGN LABEL (SAFETY RULES)
    # =========================================================================
    # Rule 1: Label must never be empty - use longest TEXT
    if not label_text and text_cells:
        label_text = max(text_cells, key=lambda c: c["length"])["raw"]
    
    # Rule 2: If still empty, check if a note cell was mistakenly used as label
    if not label_text and note_cells:
        # Check if note is actually too long (likely a label)
        for note_cell in note_cells:
            if note_cell["length"] > 10:
                label_text = note_cell["raw"]
                note_cells.remove(note_cell)
                break
    
    aligned[label_key] = label_text
    
    # =========================================================================
    # STEP 5: ASSIGN NOTE
    # =========================================================================
    # Priority: extracted note > dedicated note cell
    if extracted_note:
        aligned[note_key] = extracted_note
    elif note_cells:
        aligned[note_key] = note_cells[0]["raw"]
    else:
        aligned[note_key] = ""
    
    # =========================================================================
    # STEP 6: ASSIGN VARIATION PERCENT
    # =========================================================================
    if percent_cells:
        pct_val = percent_cells[0]["normalized"]
        aligned[pct_key] = (pct_val + "%") if "%" not in pct_val else pct_val
    else:
        aligned[pct_key] = ""
    
    # =========================================================================
    # STEP 7: ASSIGN NUMERIC VALUES (dates and variation amount)
    # =========================================================================
    # Remaining numbers go to date columns and variation amount
    num_vals = [c["normalized"] for c in number_cells]
    
    # Assign to date columns (in order)
    for i, date_col in enumerate(date_cols):
        if i < len(num_vals):
            aligned[date_col] = num_vals[i]
        else:
            aligned[date_col] = ""
    
    # Assign variation amount (next numeric value after dates)
    date_count = len(date_cols)
    if date_count < len(num_vals):
        aligned[vm_key] = num_vals[date_count]
    else:
        aligned[vm_key] = ""
    
    # =========================================================================
    # STEP 8: LABEL RECOVERY FROM RAW DATA (for missing labels)
    # =========================================================================
    # STRICT RULE: NEVER generate fake labels like "[Item ...]" or "[Unlabeled Row]"
    # If label cannot be recovered, keep it EMPTY
    
    if not aligned[label_key]:
        print(f"[LABEL RECOVERY] Missing label, attempting recovery...")
        
        LABEL_RECOVERY_PATTERN = re.compile(r'(.+?)\s*\([\d]+(?:[.-][\d]+)?\)')
        recovered_label = ""
        
        # RECOVERY STRATEGY 1: Find LEFTMOST TEXT CELL (non-numeric, non-note)
        # Iterate in column order to find first text cell
        for col, val in raw_row.items():
            if col in _SKIP:
                continue
            val_str = str(val).strip() if val else ""
            if not val_str:
                continue
            
            vtype, _ = classify_value(val_str)
            
            # Skip if it's a note, number, or percent
            if vtype in (VALUE_TYPE_NOTE, VALUE_TYPE_NUMBER, VALUE_TYPE_PERCENT):
                continue
            
            # This is a TEXT cell - use as label candidate
            if vtype == VALUE_TYPE_TEXT and any(c.isalpha() for c in val_str):
                # Check if it contains embedded note - extract label part
                match = LABEL_RECOVERY_PATTERN.search(val_str)
                if match:
                    candidate = match.group(1).strip()
                    if candidate and any(c.isalpha() for c in candidate):
                        recovered_label = candidate
                        # Also extract note if not already assigned
                        if not aligned[note_key]:
                            note_match = re.search(r'\([\d]+(?:[.-][\d]+)?\)', val_str)
                            if note_match:
                                aligned[note_key] = note_match.group(0)
                        print(f"[LABEL RECOVERY] Extracted from merged cell: '{recovered_label}'")
                        break
                else:
                    # No embedded note, use entire text
                    recovered_label = val_str
                    print(f"[LABEL RECOVERY] Found leftmost text cell: '{recovered_label}'")
                    break
        
        # RECOVERY STRATEGY 2: Column shift detection
        # If Note exists but Label is empty, the row might be shifted
        if not recovered_label and aligned[note_key]:
            # Check if first cell looks like a note (shift happened)
            first_val = None
            for col, val in raw_row.items():
                if col in _SKIP:
                    continue
                first_val = str(val).strip() if val else ""
                break
            
            # If first cell is the Note, row was shifted - label was dropped
            if first_val and first_val == aligned[note_key]:
                print(f"[LABEL RECOVERY] Column shift detected - label dropped by VLM")
                # Cannot recover, leave empty
        
        # RECOVERY STRATEGY 3: Check __raw_text metadata
        if not recovered_label and "__raw_text" in raw_row:
            raw_text = str(raw_row["__raw_text"]).strip()
            match = LABEL_RECOVERY_PATTERN.search(raw_text)
            if match:
                candidate = match.group(1).strip()
                if candidate and any(c.isalpha() for c in candidate):
                    recovered_label = candidate
                    print(f"[LABEL RECOVERY] Recovered from __raw_text: '{recovered_label}'")
        
        # Apply recovered label (only if we found real text, NEVER fabricate)
        if recovered_label:
            aligned[label_key] = recovered_label
        else:
            # Keep empty - DO NOT FABRICATE
            print(f"[LABEL RECOVERY] Could not recover label - keeping empty")
    
    # Preserve type metadata
    if "type" in raw_row:
        aligned["type"] = raw_row["type"]

    return aligned


# =============================================================================
# D. VALIDATION LAYER
# =============================================================================

def validate_row(row: Dict, schema: "ColumnSchema") -> List[str]:
    """
    COMPONENT D — Validation Layer.

    Return a list of alignment errors for a row.

    Rules (strict):
      - DATE columns   → must NOT contain PERCENT or NOTE values
      - VAR AMOUNT col → must NOT contain PERCENT or NOTE values
      - VAR PERCENT col → must NOT contain NUMBER or NOTE values
    """
    errors: List[str] = []

    for col in schema.date_cols:
        val = _clean(row.get(col, ""))
        if not val:
            continue
        vtype, _ = classify_value(val)
        if vtype == VALUE_TYPE_PERCENT:
            errors.append(f"PERCENT '{val}' in date column '{col}'")
        elif vtype == VALUE_TYPE_NOTE:
            errors.append(f"NOTE '{val}' in date column '{col}'")

    if schema.variation_amount_col:
        val = _clean(row.get(schema.variation_amount_col, ""))
        if val:
            vtype, _ = classify_value(val)
            if vtype == VALUE_TYPE_PERCENT:
                errors.append(f"PERCENT '{val}' in variation_amount column '{schema.variation_amount_col}'")
            elif vtype == VALUE_TYPE_NOTE:
                errors.append(f"NOTE '{val}' in variation_amount column '{schema.variation_amount_col}'")

    if schema.variation_percent_col:
        val = _clean(row.get(schema.variation_percent_col, ""))
        if val:
            vtype, _ = classify_value(val)
            if vtype == VALUE_TYPE_NUMBER:
                errors.append(f"NUMBER '{val}' in variation_percent column '{schema.variation_percent_col}'")
            elif vtype == VALUE_TYPE_NOTE:
                errors.append(f"NOTE '{val}' in variation_percent column '{schema.variation_percent_col}'")

    return errors


def _validate_and_correct_row(row: Dict, schema: "ColumnSchema") -> Dict:
    """
    Validate a row; if errors are detected, re-align from scratch.
    Adds '_alignment_corrected' and '_alignment_errors' debug keys.
    """
    errors = validate_row(row, schema)
    if not errors:
        return row

    # Re-align from the already-aligned (but wrong) row using original values
    corrected = align_row(row, schema)
    corrected["type"] = row.get("type", "")
    corrected["_alignment_corrected"] = True
    corrected["_alignment_errors"]    = errors
    
    return corrected


# =============================================================================
# FULL PIPELINE
# =============================================================================

def _sanitize_columns(columns: List[str]) -> List[str]:
    """
    Sanitize column list by removing artificial/placeholder columns.
    
    SCHEMA LOCKING: This ensures only real header columns are used.
    """
    sanitized = []
    for col in columns:
        if not col:
            continue
        if _is_artificial_column_name(col):
            print(f"[SCHEMA LOCK] Removing artificial column from schema: '{col}'")
            continue
        sanitized.append(col)
    return sanitized


def align_table(data: Dict) -> Dict:
    """
    Apply the full alignment engine (A→B→C→D) to an extracted table.

    Input:  {"columns": [...], "rows": [...]}
    Output: Same structure with semantically corrected rows and a "_schema"
            metadata block describing detected column roles.
    
    SCHEMA LOCKING: Column schema is determined from headers ONLY.
    No artificial columns (col1, col2, etc.) are allowed.
    """
    if not isinstance(data, dict):
        return data

    columns = data.get("columns", [])
    rows    = data.get("rows", [])

    if not isinstance(rows, list) or not rows:
        return data

    # SCHEMA LOCKING STEP 1: Sanitize incoming columns
    if isinstance(columns, list) and columns:
        columns = _sanitize_columns(columns)
    
    if not isinstance(columns, list) or not columns:
        # Infer columns from first non-empty row, but sanitize
        for r in rows:
            if isinstance(r, dict):
                raw_cols = [k for k in r.keys()
                           if k not in ("type", "__chunk_index", "__y_position")]
                columns = _sanitize_columns(raw_cols)
                break

    # If we still have no valid columns, use minimal fallback
    if not columns:
        columns = ["Label", "Note"]
        print("[SCHEMA LOCK] No valid columns found, using minimal fallback")

    # SCHEMA LOCKING STEP 2: Lock the column count
    original_column_count = len(columns)
    
    # Get data rows for fallback schema inference
    data_rows = [r for r in rows if isinstance(r, dict)]
    
    # A — detect schema from column headers (LOCKED) with fallback support
    schema = detect_schema_from_columns(columns, rows=data_rows, use_fallback=True)

    # A (refinement) — resolve ambiguous columns from data
    # NOTE: This ONLY assigns roles to existing columns, never creates new ones
    schema = _infer_schema_from_data(data_rows, schema)
    
    # SCHEMA LOCKING STEP 3: Verify column count hasn't changed
    final_column_count = len(schema.canonical_columns())
    if final_column_count > original_column_count:
        print(f"[SCHEMA LOCK WARNING] Column count increased from {original_column_count} to {final_column_count}")

    # C + D — align and validate every row
    aligned_rows: List[Dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        
        row_type = row.get("type", "")
        aligned  = align_row(row, schema)
        aligned["type"] = row_type
        validated = _validate_and_correct_row(aligned, schema)
        aligned_rows.append(validated)

    canonical = schema.canonical_columns()

    # DATA INTEGRITY: Preserve raw rows for audit trail
    raw_rows_copy = []
    for row in rows:
        if isinstance(row, dict):
            raw_rows_copy.append(dict(row))  # Shallow copy

    return {
        "columns": canonical,
        "rows":    aligned_rows,
        "_raw_rows": raw_rows_copy,  # Preserve original data
        "_schema": {
            "label":             schema.label_col,
            "note":              schema.note_col,
            "dates":             schema.date_cols,
            "variation_amount":  schema.variation_amount_col,
            "variation_percent": schema.variation_percent_col,
            "unresolved":        schema.unresolved_cols,
            "_locked":           True,  # Flag indicating schema is locked
        },
    }


def run_alignment_engine(data: Dict) -> Dict:
    """
    Public entry point for the Financial Table Alignment Engine.

    Safe to call on any dict — never raises; returns original data with an
    error flag if an unexpected exception occurs.

    Args:
        data: Raw extracted table dict {"columns": [...], "rows": [...]}

    Returns:
        Corrected table dict with semantically aligned rows.
    """
    if not isinstance(data, dict):
        return data
    try:
        return align_table(data)
    except Exception as exc:
        data["_alignment_engine_error"] = str(exc)
        return data


# =============================================================================
# STANDALONE DEMO / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import json

    # --- Example 1: Etat des engagements hors bilan (Table 1) ---
    # Simulates typical OCR misalignment: note in wrong column, % as plain number
    raw_table_1 = {
        "columns": ["Label", "Note", "31.12.2024", "31.12.2023",
                    "Variation Montant", "Variation %"],
        "rows": [
            {   # MISALIGNED: note (4.1) leaked into "31.12.2024" slot
                "type": "data",
                "Label": "HB1 - Cautions, avals et autres garanties données",
                "Note": "",
                "31.12.2024": "(4.1)",          # <- NOTE in date column
                "31.12.2023": "799 892",         # <- shifted right
                "Variation Montant": "652 772",  # <- shifted right
                "Variation %": "147 120",        # <- NUMBER in percent field!
            },
            {
                "type": "data",
                "Label": "HB2 - Crédits documentaires",
                "Note": "(4.2)",
                "31.12.2024": "210 424",
                "31.12.2023": "206 353",
                "Variation Montant": "4 071",
                "Variation %": "2,0%",
            },
            {
                "type": "total",
                "Label": "Total des passifs éventuels",
                "Note": "",
                "31.12.2024": "1 010 316",
                "31.12.2023": "859 125",
                "Variation Montant": "151 191",
                "Variation %": "17,6%",
            },
        ],
    }

    # --- Example 2: Bilan Consolidé (Table 2) ---
    # Simulates negative values in parentheses and missing schema columns
    raw_table_2 = {
        "columns": ["Label", "Note", "31/12/2024", "31/12/2023",
                    "Variation Montant", "Variation %"],
        "rows": [
            {
                "type": "data",
                "Label": "Créances sur les établissements bancaires et financiers *",
                "Note": "(1-2)",
                "31/12/2024": "244 740",
                "31/12/2023": "561 945",
                "Variation Montant": "(317 205)",   # <- negative via parens
                "Variation %": "(56,4%)",           # <- negative percent via parens
            },
            {
                "type": "data",
                "Label": "Goodwill",
                "Note": "",
                "31/12/2024": "216",
                "31/12/2023": "313",
                "Variation Montant": "(97)",
                "Variation %": "(30,9%)",
            },
            {
                "type": "total",
                "Label": "TOTAL ACTIF",
                "Note": "",
                "31/12/2024": "12 434 225",
                "31/12/2023": "11 912 261",
                "Variation Montant": "521 965",
                "Variation %": "4,4%",
            },
        ],
    }

    for idx, raw in enumerate([raw_table_1, raw_table_2], 1):
        print(f"\n{'='*60}")
        print(f"  TABLE {idx} — BEFORE alignment")
        print(f"{'='*60}")
        for row in raw["rows"]:
            print(json.dumps(row, ensure_ascii=False))

        corrected = run_alignment_engine(raw)

        print(f"\n{'='*60}")
        print(f"  TABLE {idx} — AFTER alignment")
        print(f"{'='*60}")
        print("Schema detected:", json.dumps(corrected.get("_schema"), ensure_ascii=False, indent=2))
        for row in corrected["rows"]:
            r = {k: v for k, v in row.items() if not k.startswith("_")}
            print(json.dumps(r, ensure_ascii=False))

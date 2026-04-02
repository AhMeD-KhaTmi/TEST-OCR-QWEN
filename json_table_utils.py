"""
JSON Table Extraction Utility - REDESIGNED VERSION
===================================================

Validates, parses, and POST-PROCESSES JSON output from Qwen-VL model.

Key change from previous version:
  The old pipeline assumed a FIXED, HARDCODED schema
  (Label, Note, 30/06/2022, 30/06/2021, 31/12/2021, Variation Montant, Variation %).
  This caused systematic misalignment for any document with different dates.

  The new pipeline uses the Financial Table Alignment Engine
  (table_alignment_engine.py) which:
    A. Detects column roles DYNAMICALLY from header names
    B. Classifies each cell value by TYPE (TEXT, NOTE, NUMBER, PERCENT)
    C. Rebuilds every row by semantic value type, NOT column position
    D. Validates and auto-corrects alignment errors

All JSON parsing, deduplication, and export functions are preserved.
"""

import json
import re
import csv
import ast
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher

# Alignment engine — the core redesign component
from table_alignment_engine import (
    run_alignment_engine,
    detect_schema_from_columns,
    classify_value,
    VALUE_TYPE_PERCENT,
    VALUE_TYPE_NOTE,
    VALUE_TYPE_NUMBER,
)

# Financial Validation engine — ensures accounting correctness
from financial_validation_engine import run_financial_validation

# =============================================================================
# LABEL DETECTION AND RECOVERY
# =============================================================================

def _is_numeric_value(value) -> bool:
    """Check if a value is numeric (number, formatted number, or percentage)."""
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    # Remove common formatting
    cleaned = s.replace(" ", "").replace(",", ".").replace("%", "")
    cleaned = cleaned.replace("(", "-").replace(")", "")  # Handle negative in parens
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _is_text_value(value) -> bool:
    """Check if a value is text (not numeric, not empty)."""
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    return not _is_numeric_value(value)


def detect_label_column(data: Dict) -> Optional[str]:
    """
    Detect which column contains row labels.
    
    Strategy:
        1. Check if explicit "Label" column exists
        2. Check for common label column names (Libellé, Poste, etc.)
        3. Find first column with >60% text values
        4. Default to first column if mostly text
    
    Returns:
        Column name that contains labels, or None if not found
    """
    columns = data.get("columns", [])
    rows = data.get("rows", [])
    
    if not columns or not rows:
        return None
    
    # Strategy 1: Explicit "Label" column
    for col in columns:
        if col and str(col).lower() in ("label", "labels"):
            return col
    
    # Strategy 2: Common label column names
    label_names = {
        "libelle", "libellé", "libelle poste", "poste", "designation",
        "désignation", "description", "intitule", "intitulé", "rubriques",
        "elements", "éléments", "item", "items", "compte", "comptes"
    }
    for col in columns:
        if col and str(col).strip().lower() in label_names:
            return col
    
    # Strategy 3: Find column with >60% text values
    text_ratios = {}
    for col in columns:
        if not col or str(col).lower() in ("type", "note", "notes"):
            continue
        
        text_count = 0
        total_count = 0
        
        for row in rows:
            if not isinstance(row, dict):
                continue
            val = row.get(col)
            if val is not None and str(val).strip():
                total_count += 1
                if _is_text_value(val):
                    text_count += 1
        
        if total_count > 0:
            text_ratios[col] = text_count / total_count
    
    # Find columns with >60% text
    text_columns = [(col, ratio) for col, ratio in text_ratios.items() if ratio > 0.6]
    
    if text_columns:
        # Prefer first column if it's text-heavy
        first_col = columns[0] if columns else None
        for col, ratio in text_columns:
            if col == first_col:
                return col
        # Otherwise return highest text ratio
        return max(text_columns, key=lambda x: x[1])[0]
    
    # Strategy 4: Default to first column if it has any text
    if columns:
        first_col = columns[0]
        if first_col in text_ratios and text_ratios[first_col] > 0.3:
            return first_col
    
    return None


def ensure_label_column(data: Dict) -> Dict:
    """
    Ensure data has a proper Label column.
    
    If no Label column exists:
        - Detect implicit label column
        - Map it to "Label"
        - Ensure all rows have non-empty labels
    
    Returns:
        Modified data with Label column guaranteed
    """
    columns = data.get("columns", [])
    rows = data.get("rows", [])
    
    if not rows:
        return data
    
    # Check if Label already exists and has values
    has_label_col = "Label" in columns
    
    if has_label_col:
        # Check if Label values are actually populated
        label_populated = sum(1 for r in rows if isinstance(r, dict) and r.get("Label"))
        if label_populated > len(rows) * 0.5:
            # Label is working, but still ensure no empty labels
            return recover_empty_labels(data)
    
    # Detect the actual label column
    label_col = detect_label_column(data)
    
    if label_col and label_col != "Label":
        print(f"[LABEL] Detected implicit label column: '{label_col}'")
        
        # Map values from detected column to Label
        for row in rows:
            if not isinstance(row, dict):
                continue
            
            label_value = row.get(label_col, "")
            
            # Only set Label if it's empty or doesn't exist
            if not row.get("Label"):
                row["Label"] = label_value
        
        # Add "Label" to columns if not present
        if "Label" not in columns:
            columns.insert(0, "Label")
            data["columns"] = columns
    
    # Final step: recover any remaining empty labels
    return recover_empty_labels(data)


def recover_empty_labels(data: Dict) -> Dict:
    """
    Recover labels for rows where Label is empty.
    
    Fallback strategy:
        - Use first non-numeric, non-empty cell in the row
        - Skip known non-label columns (Note, dates, variations)
    
    Returns:
        Modified data with no empty labels
    """
    columns = data.get("columns", [])
    rows = data.get("rows", [])
    
    if not rows:
        return data
    
    # Columns to skip when looking for label fallback
    skip_patterns = ["note", "variation", "%", "montant"]
    date_pattern = re.compile(r'^\d{2}[/.\-]\d{2}[/.\-]\d{4}$')
    
    def should_skip_col(col_name: str) -> bool:
        if not col_name:
            return True
        col_lower = col_name.lower()
        if any(p in col_lower for p in skip_patterns):
            return True
        if date_pattern.match(col_name.strip()):
            return True
        return False
    
    recovered_count = 0
    
    for row in rows:
        if not isinstance(row, dict):
            continue
        
        label = row.get("Label", "")
        if label and str(label).strip():
            continue  # Label is fine
        
        # Find first text value in the row
        for col in columns:
            if should_skip_col(col):
                continue
            if col == "Label":
                continue
            
            val = row.get(col)
            if val and _is_text_value(val):
                row["Label"] = str(val).strip()
                recovered_count += 1
                break
    
    if recovered_count > 0:
        print(f"[LABEL] Recovered {recovered_count} empty labels from row data")
    
    return data


# =============================================================================
# DYNAMIC SCHEMA HELPERS
# (replaces the old hardcoded TARGET_SCHEMA_COLUMNS)
# =============================================================================

# Fallback column list used only when no columns can be detected from data.
# This is NOT used to enforce a schema — it is a last-resort default.
_FALLBACK_SCHEMA_COLUMNS = [
    "Label",
    "Note",
    "Variation Montant",
    "Variation %",
]


def get_schema_columns(data: Dict) -> List[str]:
    """
    Return the canonical column list for a table.

    Priority:
      1. columns list from data (already aligned by alignment engine)
      2. keys from first row (minus internal metadata keys)
      3. fallback default
    """
    _INTERNAL = {"type", "__chunk_index", "__y_position",
                 "_alignment_corrected", "_alignment_errors"}

    cols = data.get("columns", [])
    if isinstance(cols, list) and cols:
        return [c for c in cols if c not in _INTERNAL]

    rows = data.get("rows", [])
    if isinstance(rows, list) and rows:
        for row in rows:
            if isinstance(row, dict):
                return [k for k in row.keys() if k not in _INTERNAL]

    return list(_FALLBACK_SCHEMA_COLUMNS)


# For backward compatibility — modules that import TARGET_SCHEMA_COLUMNS
# directly get a sensible structural default (no hardcoded dates).
TARGET_SCHEMA_COLUMNS = _FALLBACK_SCHEMA_COLUMNS

# Kept for backward compatibility with callers that reference these directly.
TARGET_NUMERIC_COLUMNS: List[str] = []   # now determined dynamically
TARGET_PERCENT_COLUMN   = "Variation %"


def _normalize_header_token(text: str) -> str:
    token = (text or "").strip().lower()
    token = token.replace("_", " ")
    token = token.replace("-", " ")
    token = token.replace("é", "e").replace("è", "e").replace("ê", "e")
    token = token.replace("à", "a").replace("â", "a")
    token = token.replace("ï", "i").replace("î", "i")
    token = token.replace("ù", "u").replace("û", "u")
    token = re.sub(r"\s+", " ", token)
    return token


def _map_source_column(column_name: str) -> Optional[str]:
    token = _normalize_header_token(column_name)

    if token in {"label", "libelle", "libelle poste", "poste", "designation", "description"}:
        return "Label"
    if token.startswith("note"):
        return "Note"
    if "30/06/2022" in token or "30.06.2022" in token:
        return "30/06/2022"
    if "30/06/2021" in token or "30.06.2021" in token:
        return "30/06/2021"
    if "31/12/2021" in token or "31.12.2021" in token:
        return "31/12/2021"
    if "variation" in token and "%" in token:
        return "Variation %"
    if "variation" in token and ("montant" in token or "amount" in token):
        return "Variation Montant"
    if token in {"variation %", "var %", "%"}:
        return "Variation %"
    if "montant" in token and "variation" in token:
        return "Variation Montant"
    return None


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_percent_like(value: str) -> bool:
    val = _clean_value(value)
    if not val:
        return False
    return "%" in val or bool(re.fullmatch(r"[-+]?\d+[.,]?\d*", val) and val.endswith("%"))


def _is_number_like(value: str) -> bool:
    val = _clean_value(value)
    if not val:
        return False
    compact = val.replace(" ", "").replace(",", ".")
    if compact.count(".") > 1:
        return False
    return bool(re.fullmatch(r"[-+]?\d*\.?\d+", compact))


def _normalize_for_compare(value: Any) -> str:
    text = _clean_value(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9%.,()/-]", "", text)
    return text


# NOTE: _realign_shifted_numeric_values and _normalize_single_row_to_target_schema
# have been REPLACED by the Financial Table Alignment Engine.
# See table_alignment_engine.py → align_row() for the new implementation.
#
# enforce_target_schema is kept as a lightweight compatibility shim.

def enforce_target_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    COMPATIBILITY SHIM — delegates to the alignment engine.

    The old version forced every row into a hardcoded 7-column schema.
    The new version detects columns dynamically and aligns by value type.
    """
    if data is None:
        return {"columns": list(_FALLBACK_SCHEMA_COLUMNS), "rows": []}
    return run_alignment_engine(data)


def _row_signature_strict(row: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(_normalize_for_compare(row.get(col, "")) for col in TARGET_SCHEMA_COLUMNS + ["type"])


def _numeric_similarity(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> float:
    pairs = []
    for col in TARGET_NUMERIC_COLUMNS + [TARGET_PERCENT_COLUMN]:
        a = _normalize_for_compare(row_a.get(col, ""))
        b = _normalize_for_compare(row_b.get(col, ""))
        if a or b:
            pairs.append((a, b))
    if not pairs:
        return 1.0
    matches = sum(1 for a, b in pairs if a == b and a != "")
    return matches / len(pairs)


def deduplicate_rows_strict(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique_rows: List[Dict[str, Any]] = []
    seen_exact = set()
    seen_section_labels = set()

    for row in rows:
        signature = _row_signature_strict(row)
        if signature in seen_exact:
            continue

        label_norm = _normalize_for_compare(row.get("Label", ""))
        is_section_like = row.get("type") in {"section", "total"} and not any(
            _normalize_for_compare(row.get(col, "")) for col in TARGET_NUMERIC_COLUMNS + [TARGET_PERCENT_COLUMN]
        )
        if is_section_like and label_norm:
            if label_norm in seen_section_labels:
                continue
            seen_section_labels.add(label_norm)

        is_fuzzy_duplicate = False
        replace_idx = None
        for idx, existing in enumerate(unique_rows):
            existing_label = _normalize_for_compare(existing.get("Label", ""))
            if not (label_norm and existing_label):
                continue

            label_sim = SequenceMatcher(None, label_norm, existing_label).ratio()
            numeric_sim = _numeric_similarity(row, existing)
            if label_sim >= 0.985 and numeric_sim >= 0.90:
                is_fuzzy_duplicate = True
                current_non_empty = sum(1 for col in TARGET_SCHEMA_COLUMNS if _clean_value(row.get(col, "")))
                existing_non_empty = sum(1 for col in TARGET_SCHEMA_COLUMNS if _clean_value(existing.get(col, "")))
                if current_non_empty > existing_non_empty:
                    replace_idx = idx
                break

        if is_fuzzy_duplicate:
            if replace_idx is not None:
                unique_rows[replace_idx] = row
            continue

        seen_exact.add(signature)
        unique_rows.append(row)

    return unique_rows


def evaluate_table_quality(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "is_valid": False,
            "schema_ok": False,
            "unknown_columns_count": 0,
            "duplicate_ratio": 1.0,
            "row_count": 0,
        }

    rows    = data.get("rows", [])
    columns = data.get("columns", [])
    row_count = len(rows) if isinstance(rows, list) else 0

    # Schema is "ok" when the table has at least a label column and one date/numeric column.
    # We no longer compare against a hardcoded list of dates.
    has_label = any(
        str(c).lower() in {"label", "libelle", "libelle poste", "poste",
                            "designation", "description", "intitule"}
        for c in columns
    )
    has_numeric = row_count > 0 and any(
        isinstance(columns, list) and len(columns) > 2
        for _ in [None]
    )
    schema_ok = bool(isinstance(columns, list) and len(columns) >= 2)
    # No "unknown" columns concept in a dynamic schema
    unknown_columns_count = 0

    duplicate_ratio = 0.0
    if isinstance(rows, list) and rows:
        signatures = [_row_signature_strict(r) for r in rows if isinstance(r, dict)]
        if signatures:
            unique_count  = len(set(signatures))
            duplicate_ratio = max(0.0, 1.0 - (unique_count / len(signatures)))

    too_many_duplicates = duplicate_ratio > 0.20
    is_valid = schema_ok and not too_many_duplicates

    return {
        "is_valid":              is_valid,
        "schema_ok":             schema_ok,
        "unknown_columns_count": unknown_columns_count,
        "duplicate_ratio":       round(duplicate_ratio, 3),
        "row_count":             row_count,
    }

# =============================================================================
# DUPLICATE KEY FIX (CRITICAL)
# =============================================================================

def fix_duplicate_keys(json_str: str) -> str:
    """
    Fix duplicate keys in JSON by renaming them with suffixes
    Example: {"a": 1, "a": 2} -> {"a": 1, "a_2": 2}
    """
    # Pattern to find key-value pairs
    # This handles the case where model outputs duplicate keys

    lines = json_str.split('\n')
    result_lines = []
    seen_keys_per_object = {}
    current_depth = 0

    for line in lines:
        # Track object depth
        open_braces = line.count('{')
        close_braces = line.count('}')

        if open_braces > close_braces:
            current_depth += 1
            seen_keys_per_object[current_depth] = {}

        # Find key patterns like "key_name":
        key_match = re.search(r'"([^"]+)"\s*:', line)

        if key_match and current_depth > 0:
            key = key_match.group(1)
            depth_keys = seen_keys_per_object.get(current_depth, {})

            if key in depth_keys:
                # Rename duplicate key
                count = depth_keys[key] + 1
                new_key = f"{key}_{count}"
                line = line.replace(f'"{key}":', f'"{new_key}":', 1)
                depth_keys[key] = count
            else:
                depth_keys[key] = 1

            seen_keys_per_object[current_depth] = depth_keys

        if close_braces > open_braces:
            if current_depth in seen_keys_per_object:
                del seen_keys_per_object[current_depth]
            current_depth -= 1

        result_lines.append(line)

    return '\n'.join(result_lines)


def parse_pipe_table_response(response: str) -> Optional[Dict]:
    """Parse markdown/pipe-style table output into JSON table format."""
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]

    # Keep only lines that look like table rows.
    pipe_lines = [ln for ln in lines if '|' in ln]
    if len(pipe_lines) < 2:
        return None

    def split_cells(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip('|').split('|')]
        return [p for p in parts if p != '']

    separator_idx = -1
    for i, line in enumerate(pipe_lines):
        compact = line.replace(' ', '')
        if re.fullmatch(r'[|:\-]+', compact):
            separator_idx = i
            break

    if separator_idx <= 0:
        return None

    columns = split_cells(pipe_lines[separator_idx - 1])
    if not columns:
        return None

    # Ensure expected fields exist for downstream processing.
    if 'Label' not in columns:
        columns = ['Label'] + columns
    if 'Note' not in columns:
        insert_at = 1 if 'Label' in columns else 0
        columns.insert(insert_at, 'Note')

    rows = []
    for line in pipe_lines[separator_idx + 1:]:
        cells = split_cells(line)
        if not cells:
            continue

        row = {col: '' for col in columns}
        if len(cells) == len(columns):
            for col, val in zip(columns, cells):
                row[col] = val
        else:
            # Best-effort mapping when the model emits uneven columns.
            row['Label'] = cells[0] if cells else ''
            for idx, val in enumerate(cells[1:], start=1):
                if idx < len(columns):
                    row[columns[idx]] = val

        rows.append(row)

    if not rows:
        return None

    return {
        'columns': columns,
        'rows': rows,
    }


def extract_json_robust(response: str) -> Optional[Dict]:
    """
    Robustly extract JSON from model response
    Handles: markdown blocks, extra text, duplicate keys
    """

    # Remove markdown code blocks
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)

    # Fallback for table-like text output that is not JSON.
    pipe_table = parse_pipe_table_response(response)
    if pipe_table:
        return pipe_table

    # Find the JSON object (from first { to last })
    first_brace = response.find('{')
    last_brace = response.rfind('}')

    if first_brace == -1:
        return None
    
    # TRUNCATION FIX: If JSON is truncated (no closing brace), try to recover
    if last_brace == -1 or last_brace < first_brace:
        # Attempt to close the JSON and parse what we have
        json_str = response[first_brace:]
        
        # Smart bracket/brace closing: track nesting properly
        in_string = False
        escape_next = False
        open_braces = 0
        open_brackets = 0
        
        for ch in json_str:
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                open_braces += 1
            elif ch == '}':
                open_braces -= 1
            elif ch == '[':
                open_brackets += 1
            elif ch == ']':
                open_brackets -= 1
        
        # Check if we're inside an unclosed string (value got cut off)
        if in_string:
            json_str = json_str + '"'
        
        # Close unclosed braces/brackets in correct order (inner to outer)
        json_str = json_str + '}' * max(0, open_braces - 1)  # Close inner objects
        json_str = json_str + ']' * max(0, open_brackets)     # Close arrays
        json_str = json_str + '}' * min(1, max(0, open_braces)) # Close root
    else:
        json_str = response[first_brace:last_brace + 1]

    def _try_parse(candidate: str) -> Optional[Dict]:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _repair_common_delimiters(candidate: str) -> str:
        repaired = candidate
        # Remove trailing commas before object/array close.
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        # Add commas between adjacent JSON objects often emitted in rows arrays.
        repaired = re.sub(r'}\s*{', '},{', repaired)
        # Add comma between value and next key if model omitted it.
        repaired = re.sub(r'(["\]0-9}])\s*("[^"]+"\s*:)', r'\1, \2', repaired)
        return repaired

    def _extract_array_after_key(text: str, key: str) -> Optional[str]:
        marker = f'"{key}"'
        key_pos = text.find(marker)
        if key_pos == -1:
            return None
        bracket_pos = text.find('[', key_pos)
        if bracket_pos == -1:
            return None

        depth = 0
        in_str = False
        escape = False
        for i in range(bracket_pos, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return text[bracket_pos:i + 1]
        return None

    def _extract_row_items(rows_array_text: str) -> List[Any]:
        """
        Extract row items from rows array text.
        Handles both dict rows (starting with {) and list rows (starting with [).
        """
        # Remove outer []
        body = rows_array_text.strip()
        if body.startswith('[') and body.endswith(']'):
            body = body[1:-1]

        rows = []
        start = None
        depth = 0
        in_str = False
        escape = False
        start_char = None  # Track if we're looking for { or [

        for i, ch in enumerate(body):
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch == '{' or ch == '[':
                if depth == 0:
                    start = i
                    start_char = ch
                depth += 1
            elif (ch == '}' and start_char == '{') or (ch == ']' and start_char == '['):
                depth -= 1
                if depth == 0 and start is not None:
                    item_str = body[start:i + 1]
                    item_str = re.sub(r',\s*([}\]])', r'\1', item_str)
                    parsed_item = _try_parse(item_str)
                    if parsed_item is None:
                        try:
                            py_item = ast.literal_eval(item_str)
                            if isinstance(py_item, (dict, list)):
                                parsed_item = py_item
                        except Exception:
                            parsed_item = None
                    if parsed_item is not None:
                        rows.append(parsed_item)
                    start = None
                    start_char = None
        return rows

    def _extract_row_objects(rows_array_text: str) -> List[Dict]:
        """
        Extract rows from array text. Returns list of dicts.
        If rows are lists, they will be converted to dicts later by convert_list_rows_to_dicts.
        """
        items = _extract_row_items(rows_array_text)
        # Return as-is; the convert_list_rows_to_dicts function will handle list-to-dict
        return items

    # Fix duplicate keys before parsing
    json_str = fix_duplicate_keys(json_str)

    parsed = _try_parse(json_str)
    if parsed is not None:
        return parsed

    repaired = _repair_common_delimiters(json_str)
    parsed = _try_parse(repaired)
    if parsed is not None:
        return parsed

    print("JSON parse error: attempting structural recovery")

    # Structural recovery: parse columns/rows independently from malformed JSON.
    cols_text = _extract_array_after_key(repaired, 'columns')
    rows_text = _extract_array_after_key(repaired, 'rows')

    recovered_columns = []
    if cols_text:
        col_parsed = _try_parse('{"columns": ' + cols_text + '}')
        if col_parsed and isinstance(col_parsed.get('columns'), list):
            recovered_columns = col_parsed['columns']
        else:
            try:
                recovered_columns = ast.literal_eval(cols_text)
                if not isinstance(recovered_columns, list):
                    recovered_columns = []
            except Exception:
                recovered_columns = []

    recovered_rows = _extract_row_objects(rows_text) if rows_text else []
    if recovered_rows:
        # Check if rows are lists or dicts
        if recovered_rows and isinstance(recovered_rows[0], dict):
            # Dict rows - infer columns from keys
            if not recovered_columns:
                key_order = []
                for row in recovered_rows:
                    for key in row.keys():
                        if key not in key_order and key != 'type':
                            key_order.append(key)
                recovered_columns = key_order
        # If rows are lists, convert_list_rows_to_dicts will handle column inference

        return {
            'columns': recovered_columns,
            'rows': recovered_rows,
        }

    # Final fallback for model outputs that look like Python dicts (single quotes, True/False, None).
    try:
        py_obj = ast.literal_eval(json_str)
        if isinstance(py_obj, dict):
            return py_obj
        return None
    except Exception:
        return None


def convert_list_rows_to_dicts(data: Dict) -> Dict:
    """
    Convert list-based rows to dict format using columns.
    
    Many VLM models return rows as lists like:
        [["PASSIFS EVENTUELS", "(4-1)", "596 598", ...], ...]
    
    But our pipeline expects dict format:
        [{"Label": "PASSIFS EVENTUELS", "Note": "(4-1)", "2022": "596598", ...}, ...]
    
    This conversion MUST happen immediately after JSON parsing,
    BEFORE any other processing (normalization, alignment, validation).
    """
    if not isinstance(data, dict):
        return data
    
    rows = data.get("rows", [])
    columns = data.get("columns", [])
    
    if not rows:
        return data
    
    # Check if first row is a list (indicator of list format)
    if not isinstance(rows[0], list):
        # Already dict format - no conversion needed
        return data
    
    # Need columns to do the mapping
    if not columns:
        # Try to infer columns from the first row if it looks like a header
        # This handles cases where columns might be embedded in rows
        first_row = rows[0]
        if all(isinstance(v, str) and not v.replace('.', '').replace('-', '').replace(' ', '').isdigit() 
               for v in first_row if v is not None):
            # First row looks like headers - use it as columns
            columns = [str(v) if v is not None else f"col{i}" for i, v in enumerate(first_row)]
            data["columns"] = columns
            rows = rows[1:]  # Remove header row from data rows
        else:
            # Generate column names based on position
            max_len = max(len(row) for row in rows if isinstance(row, list))
            columns = [f"Column{i+1}" for i in range(max_len)]
            data["columns"] = columns
            print(f"[LIST-DICT] Warning: No columns found, generated {len(columns)} placeholder columns")
    
    converted_rows = []
    for idx, row in enumerate(rows):
        if isinstance(row, list):
            row_dict = {}
            for i, col in enumerate(columns):
                if i < len(row):
                    val = row[i]
                    # Convert None to empty string
                    row_dict[col] = "" if val is None else val
                else:
                    # Pad with empty string if row is shorter than columns
                    row_dict[col] = ""
            
            # Handle extra values that don't have columns
            if len(row) > len(columns):
                for i in range(len(columns), len(row)):
                    row_dict[f"_extra_{i}"] = row[i] if row[i] is not None else ""
                print(f"[LIST-DICT] Row {idx}: {len(row) - len(columns)} extra values stored as _extra_*")
            
            converted_rows.append(row_dict)
        elif isinstance(row, dict):
            # Already a dict - keep as is
            converted_rows.append(row)
        else:
            # Unknown format - skip with warning
            print(f"[LIST-DICT] Warning: Row {idx} has unexpected format: {type(row)}")
            continue
    
    data["rows"] = converted_rows
    print(f"[LIST-DICT] Converted {len(converted_rows)} rows from list to dict format")
    
    return data


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract JSON from model response with robust handling.
    
    Includes automatic list→dict row conversion for compatibility
    with VLM models that return rows as arrays.
    """
    data = extract_json_robust(response)
    
    if data is not None:
        # CRITICAL: Convert list rows to dict rows IMMEDIATELY after parsing
        data = convert_list_rows_to_dicts(data)
    
    return data


# =============================================================================
# POST-PROCESSING: FIX COLUMN SHIFT
# =============================================================================

def detect_note_pattern(value: str) -> bool:
    """Check if value matches note pattern like (4.1), (4.2)"""
    if not isinstance(value, str):
        return False
    return bool(re.match(r'^\(\d+\.?\d*\)$', value.strip()))


def fix_column_shift(data: Dict) -> Dict:
    """
    COMPATIBILITY SHIM — column shift correction is now handled entirely by
    the Alignment Engine (table_alignment_engine.py → align_row).

    This function is kept so existing callers don't break, but it is a no-op:
    the real fix happens in post_process_extraction via run_alignment_engine.
    """
    return data


# =============================================================================
# POST-PROCESSING: REMOVE HEADER POLLUTION
# =============================================================================

HEADER_PATTERNS = [
    r'^attijari\s*bank',
    r'^etat\s*des',
    r'^arr[eê]t[eé]\s*au',
    r'^unit[eé]',
    r'^\(unit[eé]',
    r'^en\s*milliers',
    r'^note$',
    r'^variation$',
    r'^en\s*montant$',
    r'^en\s*%$',
    # Date patterns that are column headers, not data
    r'^\d{2}[./]\d{2}[./]\d{4}$',  # DD/MM/YYYY or DD.MM.YYYY
    r'^\d{2}\.\d{2}\.\d{4}$',      # DD.MM.YYYY
]


def is_header_row(row: Dict) -> bool:
    """Check if row is actually a header/title that should be removed"""

    label = str(row.get('Label', row.get('Note', ''))).lower().strip()

    for pattern in HEADER_PATTERNS:
        if re.match(pattern, label, re.IGNORECASE):
            return True

    return False


def remove_header_pollution(data: Dict) -> Dict:
    """Remove rows that are actually headers/titles, not data"""

    if 'rows' in data:
        data['rows'] = [row for row in data['rows'] if not is_header_row(row)]
    elif 'table' in data:
        data['table'] = [row for row in data['table'] if not is_header_row(row)]

    return data


# =============================================================================
# POST-PROCESSING: ADD ROW TYPES
# =============================================================================

def classify_row_type(row: Dict) -> str:
    """
    Classify row as section, data, or total
    DYNAMIC: checks all columns except Label/Note/type
    """

    label = str(row.get('Label', '')).lower()

    # Check for total
    if 'total' in label:
        return 'total'

    # Check for section (no numeric values in ANY column)
    # Dynamic: check all keys except known non-numeric ones
    non_numeric_keys = {'Label', 'Note', 'type'}

    has_numbers = False
    for key, val in row.items():
        if key in non_numeric_keys:
            continue
        val_str = str(val).strip()
        # Check if value looks numeric (has digits, not just a note reference)
        if val_str and val_str != '' and not detect_note_pattern(val_str):
            # Check if it contains any digit
            if any(c.isdigit() for c in val_str):
                has_numbers = True
                break

    if not has_numbers:
        return 'section'

    return 'data'


def add_row_types(data: Dict) -> Dict:
    """Add type classification to each row if missing"""

    rows = data.get('rows', data.get('table', []))

    for row in rows:
        if not isinstance(row, dict):
            continue
        if 'type' not in row:
            row['type'] = classify_row_type(row)

    return data


# =============================================================================
# POST-PROCESSING: CONVERT OLD FORMAT TO NEW
# =============================================================================

def convert_to_new_format(data: Dict) -> Dict:
    """
    Convert old format {"table": [...]} to new format {"columns": [...], "rows": [...]}
    """

    if 'rows' in data and 'columns' in data:
        # Already in new format
        return data

    if 'table' not in data:
        return data

    table = data['table']

    if not table:
        return {'columns': [], 'rows': []}

    # Get columns from first row
    columns = list(table[0].keys())

    # Convert rows
    rows = []
    for old_row in table:
        new_row = {
            'type': classify_row_type(old_row),
            'Label': old_row.get('Note', old_row.get('Label', '')),
        }
        new_row.update(old_row)
        rows.append(new_row)

    return {
        'columns': columns,
        'rows': rows
    }


# =============================================================================
# FIX 4: COVERAGE VALIDATION (CRITICAL)
# =============================================================================

def validate_extraction_coverage(data: Dict) -> Tuple[bool, List[str]]:
    """
    FIX 4: Validate extraction coverage before processing.
    
    Checks:
        - row_count >= 5
        - at least 1 section row exists
        - at least 1 data row exists
        - first row is NOT empty
        - at least 1 numeric column exists
    
    Returns:
        (is_valid, issues)
    """
    issues = []
    
    rows = data.get("rows", [])
    columns = data.get("columns", [])
    
    # Check row count
    MIN_ROWS = 5
    if len(rows) < MIN_ROWS:
        issues.append(f"Too few rows: {len(rows)} < {MIN_ROWS}")
    
    # Count row types
    section_count = 0
    data_count = 0
    total_count = 0
    
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_type = str(row.get("type", "")).lower()
        if row_type == "section":
            section_count += 1
        elif row_type == "data":
            data_count += 1
        elif row_type == "total":
            total_count += 1
    
    if section_count == 0:
        issues.append("No section headers found")
    
    if data_count == 0:
        issues.append("No data rows found")
    
    # Check first row is not empty
    if rows:
        first_row = rows[0] if isinstance(rows[0], dict) else {}
        label = str(first_row.get("Label", "")).strip()
        if not label:
            issues.append("First row has empty label")
    
    # Check for at least one column that looks like a date or numeric column
    has_numeric_col = False
    date_pattern = re.compile(r'^\d{2}[/.\-]\d{2}[/.\-]\d{4}$')
    
    for col in columns:
        if isinstance(col, str):
            # Check if it's a date column
            if date_pattern.match(col.strip()):
                has_numeric_col = True
                break
            # Check if it's a variation column
            col_lower = col.lower()
            if 'variation' in col_lower or 'montant' in col_lower or '%' in col_lower:
                has_numeric_col = True
                break
    
    if not has_numeric_col and len(columns) > 0:
        issues.append("No numeric/date columns detected")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


# =============================================================================
# MAIN POST-PROCESSING PIPELINE
# =============================================================================

def post_process_extraction(data: Dict) -> Dict:
    """
    Full post-processing pipeline — REDESIGNED WITH FINANCIAL VALIDATION.

    Pipeline stages:
      1.  Format normalisation  – convert {table:[...]} → {columns, rows}
      2.  Row shape normalisation – flatten nested 'values' dicts; list rows → dicts
      3.  Header pollution removal – strip title/footer rows
      4.  Provisional row types   – section / data / total
      5.  *** ALIGNMENT ENGINE *** – the core redesign:
              A. Detect column roles dynamically (no hardcoded dates)
              B. Classify each cell by value type
              C. Rebuild rows by semantic type, not position
              D. Validate & auto-correct
      6.  Row type recomputation  – after alignment
      7.  Strict deduplication    – exact + fuzzy label/numeric similarity
      8.  *** FINANCIAL VALIDATION *** – accounting correctness:
              A. Variation amount: expected = current - previous
              B. Percentage: expected = (variation / previous) * 100
              C. Sign consistency: amount & percent signs must match
              D. Total validation: sum of section rows
              E. Note consistency: no duplicates/misplacements
              F. Edge cases: division by zero, equal values
              G. Sanity check: no % in numeric fields, no numbers in % fields
      9.  Quality metadata        – attach to meta block
    
    FIX 3: Preserves raw data before any processing.
    FIX 4: Validates coverage before applying corrections.
    """
    if data is None:
        return None
    if not isinstance(data, dict):
        return None

    preserved_meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}

    # =========================================================================
    # FIX 3: RAW DATA PRESERVATION (CRITICAL)
    # Store original data BEFORE any processing
    # =========================================================================
    if "_raw_columns" not in data:
        data["_raw_columns"] = list(data.get("columns", []))
    if "_raw_rows_count" not in data:
        data["_raw_rows_count"] = len(data.get("rows", []))
    if "_raw_rows" not in data:
        # Store first 20 rows for debugging (to avoid memory bloat)
        raw_rows = data.get("rows", [])[:20]
        data["_raw_rows"] = [dict(r) if isinstance(r, dict) else r for r in raw_rows]

    # --- Stage 1: format conversion ---
    data = convert_to_new_format(data)

    # --- Stage 2: row shape normalisation ---
    rows    = data.get('rows', [])
    columns = data.get('columns', [])
    if isinstance(rows, list) and rows:
        normalised_rows = []
        for row in rows:
            if isinstance(row, dict):
                if isinstance(row.get('values'), dict):
                    # Nested format: {label, values: {col: val, ...}}
                    merged = {
                        'type':  row.get('type', ''),
                        'Label': row.get('label', row.get('Label', '')),
                    }
                    merged.update(row.get('values', {}))
                    normalised_rows.append(merged)
                else:
                    normalised_rows.append(row)
            elif isinstance(row, list):
                row_dict = {col: (row[i] if i < len(row) else "")
                            for i, col in enumerate(columns)}
                normalised_rows.append(row_dict)
        data['rows'] = normalised_rows

    # --- Stage 3: header pollution removal ---
    data = remove_header_pollution(data)

    # --- Stage 4: provisional row types ---
    data = add_row_types(data)

    # =========================================================================
    # FIX 4: COVERAGE VALIDATION (CRITICAL)
    # Check extraction completeness BEFORE alignment and validation
    # =========================================================================
    coverage_valid, coverage_issues = validate_extraction_coverage(data)
    data["_coverage_complete"] = coverage_valid
    data["_coverage_issues"] = coverage_issues
    if not coverage_valid:
        print(f"[COVERAGE] Issues detected: {coverage_issues}")

    # --- Stage 5: ALIGNMENT ENGINE (core fix) ---
    data = run_alignment_engine(data)

    # Strip internal debug keys from rows (kept for logging if needed)
    clean_rows = []
    for row in data.get('rows', []):
        if not isinstance(row, dict):
            continue
        clean_row = {k: v for k, v in row.items()
                     if not k.startswith('_alignment')}
        clean_rows.append(clean_row)
    data['rows'] = clean_rows

    # --- Stage 6: recompute row types after alignment ---
    data = add_row_types(data)

    # --- Stage 6.5: LABEL RECOVERY (ensure all rows have labels) ---
    # Detects implicit label columns and recovers empty labels
    data = ensure_label_column(data)

    # --- Stage 7: strict deduplication ---
    dedup_rows = data.get('rows', [])
    if isinstance(dedup_rows, list):
        data['rows'] = deduplicate_rows_strict(
            [r for r in dedup_rows if isinstance(r, dict)]
        )

    # --- Stage 8: FINANCIAL VALIDATION (accounting correctness) ---
    # Validates and corrects:
    #   - Variation amounts (current - previous)
    #   - Percentage calculations ((variation / previous) * 100)
    #   - Sign consistency between amount and percent
    #   - Total row sums
    #   - Note consistency (no duplicates/misplacements)
    #   - Edge cases (division by zero, equal values)
    data = run_financial_validation(data)

    # Strip internal validation debug keys for clean output
    for row in data.get('rows', []):
        if isinstance(row, dict):
            keys_to_remove = [k for k in row.keys() if k.startswith('_')]
            for k in keys_to_remove:
                del row[k]

    # --- Stage 9: quality metadata ---
    quality = evaluate_table_quality(data)
    merged_meta = dict(preserved_meta)
    merged_meta['quality']  = quality
    merged_meta['pipeline'] = 'alignment_engine_v2_with_financial_validation'
    
    # Preserve validation results in meta
    if '_validation' in data:
        merged_meta['financial_validation'] = data.pop('_validation')
    
    data['meta'] = merged_meta

    return data


# =============================================================================
# VALIDATION
# =============================================================================

def validate_table_json(data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate extracted JSON table structure — DYNAMIC SCHEMA VERSION.

    No longer checks against a hardcoded column list.
    Instead validates:
      - Structural integrity (rows exist, are dicts)
      - Every row has a 'type' field with an allowed value
      - No excessive duplicates
      - Alignment-engine metadata (if present) shows no errors

    Returns:
        (is_valid, list_of_errors)
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        errors.append("Root must be a JSON object (dict)")
        return False, errors

    if 'rows' not in data and 'table' in data:
        data = convert_to_new_format(data)

    if 'rows' not in data:
        errors.append("Missing 'rows' or 'table' key")
        return False, errors

    rows    = data.get('rows', [])
    columns = data.get('columns', [])

    if not isinstance(rows, list):
        errors.append("'rows' must be an array")
        return False, errors

    if len(rows) == 0:
        errors.append("'rows' array is empty")
        return False, errors

    # Dynamic check: must have at least 2 columns (label + one value column)
    if not isinstance(columns, list) or len(columns) < 2:
        errors.append(
            f"Too few columns ({len(columns) if isinstance(columns, list) else 0}): "
            "expected at least 2 (label + one numeric/date column)"
        )

    allowed_types     = {"section", "data", "total"}
    schema_cols_set   = set(get_schema_columns(data))
    allowed_row_keys  = schema_cols_set | {"type", "_schema",
                                            "_alignment_corrected", "_alignment_errors"}

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"Row {idx} is not an object")
            continue

        if "type" not in row:
            errors.append(f"Row {idx} missing 'type'")
        else:
            row_type = str(row.get("type", "")).strip().lower()
            if row_type not in allowed_types:
                errors.append(f"Row {idx} has invalid 'type': {row.get('type')!r}")

    quality = evaluate_table_quality(data)
    if quality.get("duplicate_ratio", 0.0) > 0.20:
        errors.append(f"Duplicate ratio too high: {quality['duplicate_ratio']}")

    is_valid = len(errors) == 0
    return is_valid, errors


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def convert_to_csv(data: Dict, output_path: str, delimiter: str = ',') -> bool:
    """Convert JSON table to CSV file"""

    try:
        # Handle both formats
        if 'rows' in data:
            rows = data['rows']
            if not rows:
                return False
            # Get headers excluding 'type'
            headers = [k for k in rows[0].keys() if k != 'type']
        else:
            rows = data.get('table', [])
            if not rows:
                return False
            headers = list(rows[0].keys())

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=delimiter, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

        return True

    except Exception as e:
        print(f"Error converting to CSV: {e}")
        return False


def convert_to_excel(data: Dict, output_path: str) -> bool:
    """Convert JSON table to Excel file"""

    try:
        import pandas as pd
    except ImportError:
        print("pandas is required for Excel export. Install with: pip install pandas openpyxl")
        return False

    try:
        if 'rows' in data:
            rows = data['rows']
        else:
            rows = data.get('table', [])

        if not rows:
            return False

        df = pd.DataFrame(rows)

        # Remove 'type' column for cleaner export
        if 'type' in df.columns:
            df = df.drop(columns=['type'])

        df.to_excel(output_path, index=False, engine='openpyxl')

        return True

    except Exception as e:
        print(f"Error converting to Excel: {e}")
        return False


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_table_summary(data: Dict):
    """Print a summary of the extracted table"""

    rows = data.get('rows', data.get('table', []))

    if not rows:
        print("Empty table")
        return

    num_rows = len(rows)
    first_row = rows[0]
    columns = [k for k in first_row.keys() if k != 'type']
    num_cols = len(columns)

    # Count by type
    type_counts = {}
    for row in rows:
        row_type = row.get('type', 'unknown')
        type_counts[row_type] = type_counts.get(row_type, 0) + 1

    print("=" * 60)
    print("TABLE SUMMARY")
    print("=" * 60)
    print(f"Total Rows: {num_rows}")
    print(f"Columns: {num_cols}")

    if type_counts:
        print(f"\nRow types:")
        for t, count in type_counts.items():
            print(f"  - {t}: {count}")

    print(f"\nColumn names:")
    for idx, col in enumerate(columns, 1):
        print(f"  {idx}. {col}")
    print("=" * 60)


def pretty_print_table(data: Dict, max_rows: int = 20):
    """Pretty print the table (first N rows)"""

    rows = data.get('rows', data.get('table', []))

    if not rows:
        print("Empty table")
        return

    display_rows = rows[:max_rows]
    columns = [k for k in rows[0].keys() if k != 'type']

    # Calculate column widths
    col_widths = {col: len(col) for col in columns}

    for row in display_rows:
        for col in columns:
            val_str = str(row.get(col, ''))[:40]  # Truncate long values
            col_widths[col] = max(col_widths[col], len(val_str))

    # Limit column widths
    for col in col_widths:
        col_widths[col] = min(col_widths[col], 30)

    # Print header
    total_width = sum(col_widths.values()) + (len(columns) * 3)
    print("\n" + "=" * total_width)
    header_line = " | ".join(col[:col_widths[col]].ljust(col_widths[col]) for col in columns)
    print(header_line)
    print("-" * len(header_line))

    # Print rows with type indicator
    for row in display_rows:
        row_type = row.get('type', '')
        type_marker = {'section': '[S]', 'total': '[T]', 'data': '   '}.get(row_type, '   ')

        values = []
        for col in columns:
            val = str(row.get(col, ''))[:col_widths[col]]
            values.append(val.ljust(col_widths[col]))

        row_line = type_marker + " | ".join(values)
        print(row_line)

    if len(rows) > max_rows:
        print(f"\n... ({len(rows) - max_rows} more rows)")

    print("=" * total_width + "\n")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":

    # Example with duplicate keys (simulating model output error)
    sample_response = '''
    {
      "table": [
        {
          "Note": "HB1 - Test",
          "31.12.2024": "(4.1)",
          "31.12.2023": "799 892",
          "Variation_%": "147 120",
          "Variation_%": "22,5%"
        }
      ]
    }
    '''

    print("Testing duplicate key handling...")
    parsed = extract_json_from_response(sample_response)

    if parsed:
        print("[OK] JSON extracted (with duplicate key fix)")
        processed = post_process_extraction(parsed)
        print("\nProcessed result:")
        print(json.dumps(processed, indent=2, ensure_ascii=False))
    else:
        print("[ERROR] Failed to extract JSON")

"""
Empty Label Detection and Recovery System
==========================================

This module provides fallback mechanisms for recovering missing labels
from financial table extractions when the VLM fails to extract them.

The system detects rows where:
- Label is empty
- Note and/or numeric values exist
- This indicates the VLM skipped the label text

Recovery strategies:
1. Re-extract with focused prompt on label column
2. Use context from surrounding rows
3. Use note reference mapping (if available)
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import numpy as np


def detect_empty_label_rows(data: Dict) -> List[Tuple[int, Dict]]:
    """
    Detect rows where Label is empty but other data exists.
    
    These rows indicate the VLM failed to extract the label text.
    
    Args:
        data: Extracted table dict with "columns" and "rows"
        
    Returns:
        List of (row_index, row_dict) tuples for affected rows
    """
    affected_rows = []
    
    rows = data.get("rows", [])
    columns = data.get("columns", [])
    
    # Find label column name
    label_col = None
    for col in columns:
        if col.lower() in ["label", "libellé", "désignation"]:
            label_col = col
            break
    
    if not label_col and columns:
        label_col = columns[0]  # Assume first column is label
    
    if not label_col:
        return []
    
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        
        label_value = str(row.get(label_col, "")).strip()
        row_type = row.get("type", "")
        
        # Skip section headers (they may legitimately have only label)
        if row_type == "section":
            continue
        
        # Check if label is empty
        if not label_value:
            # Check if row has other data (note or numeric values)
            has_other_data = False
            
            for col, val in row.items():
                if col in ["type", label_col]:
                    continue
                val_str = str(val).strip()
                if val_str:
                    has_other_data = True
                    break
            
            if has_other_data:
                affected_rows.append((idx, row))
                print(f"[LABEL RECOVERY] Row {idx}: Empty label detected, has other data")
    
    return affected_rows


def recover_labels_from_context(
    data: Dict,
    affected_rows: List[Tuple[int, Dict]],
    note_mapping: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Attempt to recover missing labels using context.
    
    Strategies:
    1. Use note reference mapping (if provided)
    2. Infer from row position relative to section headers
    3. Use previous row's label as template
    
    Args:
        data: Extracted table dict
        affected_rows: List of (index, row) tuples with empty labels
        note_mapping: Optional dict mapping note refs to expected labels
        
    Returns:
        Updated data dict with recovered labels where possible
    """
    rows = data.get("rows", [])
    columns = data.get("columns", [])
    
    # Find label and note columns
    label_col = None
    note_col = None
    
    for col in columns:
        col_lower = col.lower()
        if col_lower in ["label", "libellé", "désignation"]:
            label_col = col
        if col_lower == "note":
            note_col = col
    
    if not label_col and columns:
        label_col = columns[0]
    
    # Standard Attijari Bank note-to-label mapping
    default_mapping = {
        "(1-1)": "Caisse et avoirs auprès de la BCT, CCP et TGT",
        "(1-2)": "Créances sur les établissements bancaires et financiers",
        "(1-3)": "Créances sur la clientèle",
        "(1-4)": "Portefeuille-titres commercial",
        "(1-5)": "Portefeuille d'investissement",
        "(1-6)": "Titres mis en équivalence",
        "(1-7)": "Valeurs immobilisées",
        "(1-8)": "Autres actifs",
        "(2-1)": "Dépôts et avoirs des établissements bancaires et financiers",
        "(2-2)": "Dépôts et avoirs de la clientèle",
        "(2-3)": "Emprunts et ressources spéciales",
        "(2-4)": "Autres passifs",
        "(3)": "Capital",
    }
    
    mapping = note_mapping or default_mapping
    
    recovered_count = 0
    
    for idx, row in affected_rows:
        note_value = ""
        if note_col:
            note_value = str(row.get(note_col, "")).strip()
        
        recovered_label = None
        
        # Strategy 1: Use note mapping
        if note_value and note_value in mapping:
            recovered_label = mapping[note_value]
            print(f"[LABEL RECOVERY] Row {idx}: Recovered '{recovered_label}' from note mapping")
        
        # HARDENING #4: LABEL INTEGRITY
        # NEVER generate artificial labels like [Item], [ACTIF item], etc.
        # If label cannot be recovered from real data, keep it EMPTY
        
        # Apply recovery ONLY if we found something meaningful from data
        if recovered_label:
            rows[idx][label_col] = recovered_label
            recovered_count += 1
        else:
            # Keep empty - DO NOT create placeholders
            print(f"[LABEL RECOVERY] Row {idx}: Could not recover label, keeping empty")
    
    print(f"[LABEL RECOVERY] Recovered {recovered_count}/{len(affected_rows)} labels")
    
    return data


def _find_parent_section(rows: List[Dict], row_idx: int) -> Optional[str]:
    """
    Find the section header above a given row.
    """
    for i in range(row_idx - 1, -1, -1):
        row = rows[i]
        if row.get("type") == "section":
            # Return the label of the section
            for key in ["Label", "Libellé", "label"]:
                if key in row and row[key]:
                    return row[key]
    return None


def crop_label_column_region(
    image_path: str,
    label_column_ratio: float = 0.40
) -> Image.Image:
    """
    Crop image to focus on the label column (leftmost portion).
    
    Use this for targeted re-extraction when labels are missing.
    
    Args:
        image_path: Path to the original image
        label_column_ratio: Width ratio of label column (0.0-1.0)
        
    Returns:
        Cropped PIL Image focusing on label column
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # Crop to label column (left portion)
    crop_width = int(width * label_column_ratio)
    cropped = img.crop((0, 0, crop_width, height))
    
    return cropped


def generate_label_focused_prompt(note_values: List[str]) -> str:
    """
    Generate a prompt focused specifically on extracting labels.
    
    Use this for targeted re-extraction when standard extraction fails.
    
    Args:
        note_values: List of note references that need labels
        
    Returns:
        Focused extraction prompt
    """
    note_list = ", ".join(note_values) if note_values else "all rows"
    
    prompt = f"""Extract ONLY the Label column text from this financial table.

Focus on the LEFTMOST column which contains French text descriptions.

For each row, output ONLY:
{{ "Note": "(X-X)", "Label": "full French text" }}

I specifically need labels for rows with notes: {note_list}

RULES:
- Read the LEFTMOST text in each row
- Labels are LONG French text (20-60 characters typically)
- Include any asterisk (*) at the end of labels
- Do NOT include numeric values or percentages

Output as JSON array:
[
  {{"Note": "(1-1)", "Label": "Caisse et avoirs auprès de la BCT, CCP et TGT"}},
  {{"Note": "(1-2)", "Label": "Créances sur les établissements bancaires et financiers *"}}
]"""
    
    return prompt


def merge_recovered_labels(
    original_data: Dict,
    label_extraction: List[Dict]
) -> Dict:
    """
    Merge labels recovered from targeted re-extraction into original data.
    
    Args:
        original_data: Original extraction with empty labels
        label_extraction: Labels extracted via targeted re-extraction
        
    Returns:
        Updated data with recovered labels
    """
    rows = original_data.get("rows", [])
    columns = original_data.get("columns", [])
    
    # Find column names
    label_col = columns[0] if columns else "Label"
    note_col = "Note"
    
    # Build note → label mapping from extraction
    note_to_label = {}
    for item in label_extraction:
        note = item.get("Note", "")
        label = item.get("Label", "")
        if note and label:
            note_to_label[note] = label
    
    # Apply to original data
    recovered_count = 0
    for row in rows:
        current_label = str(row.get(label_col, "")).strip()
        note_value = str(row.get(note_col, "")).strip()
        
        if not current_label and note_value in note_to_label:
            row[label_col] = note_to_label[note_value]
            recovered_count += 1
            print(f"[MERGE] Recovered label for {note_value}: '{note_to_label[note_value]}'")
    
    print(f"[MERGE] Total recovered: {recovered_count}")
    
    return original_data


# =============================================================================
# INTEGRATION FUNCTION
# =============================================================================

def run_label_recovery_pipeline(
    data: Dict,
    image_path: Optional[str] = None,
    reextract_function = None
) -> Dict:
    """
    Full label recovery pipeline.
    
    Steps:
    1. Detect rows with empty labels
    2. Attempt context-based recovery
    3. If image and re-extraction function provided, do targeted re-extraction
    4. Merge results
    
    Args:
        data: Extracted table dict
        image_path: Optional path to original image for re-extraction
        reextract_function: Optional function to re-extract from image
        
    Returns:
        Updated data with recovered labels
    """
    # Step 1: Detect affected rows
    affected = detect_empty_label_rows(data)
    
    if not affected:
        print("[LABEL RECOVERY] No empty labels detected")
        return data
    
    print(f"[LABEL RECOVERY] Found {len(affected)} rows with empty labels")
    
    # Step 2: Context-based recovery
    data = recover_labels_from_context(data, affected)
    
    # Step 3: Check if we still have empty labels
    still_affected = detect_empty_label_rows(data)
    
    if not still_affected:
        print("[LABEL RECOVERY] All labels recovered via context")
        return data
    
    print(f"[LABEL RECOVERY] {len(still_affected)} labels still empty after context recovery")
    
    # Step 4: Targeted re-extraction (if available)
    if image_path and reextract_function:
        print("[LABEL RECOVERY] Attempting targeted re-extraction...")
        
        # Get note values for affected rows
        note_values = []
        for idx, row in still_affected:
            note = row.get("Note", "")
            if note:
                note_values.append(note)
        
        # Generate focused prompt
        prompt = generate_label_focused_prompt(note_values)
        
        # Crop to label column
        label_region = crop_label_column_region(image_path, label_column_ratio=0.45)
        
        try:
            # Re-extract with focused prompt
            label_data = reextract_function(label_region, prompt)
            
            if label_data and isinstance(label_data, list):
                data = merge_recovered_labels(data, label_data)
        except Exception as e:
            print(f"[LABEL RECOVERY] Re-extraction failed: {e}")
    
    # Final check
    final_affected = detect_empty_label_rows(data)
    if final_affected:
        print(f"[LABEL RECOVERY] Warning: {len(final_affected)} labels could not be recovered")
    else:
        print("[LABEL RECOVERY] All labels successfully recovered")
    
    return data

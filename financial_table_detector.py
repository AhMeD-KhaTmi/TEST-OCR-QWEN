"""
Financial Statement Detector v3 - High Precision (99% Accuracy Goal)
====================================================================

A PRECISE detector with STRICT title matching and NEGATIVE filters:
1. EXACT title matching (not fuzzy contains)
2. NEGATIVE filters to reject false positives (e.g., "hors bilan")
3. Structure validation with STRONG keywords
4. Conservative expansion

Key Improvements over v2:
- "HORS BILAN" is REJECTED (not a balance sheet)
- "NOTES AUX ETATS" is REJECTED (notes section)
- Exact title position detection (must be primary heading)
- Better accent handling

Author: AI Financial Document Pipeline
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from rapidfuzz import fuzz


# =============================================================================
# CONFIGURATION - HIGH PRECISION
# =============================================================================

# EXACT title patterns - these must match as PRIMARY TITLES
# Order matters: more specific patterns first
TITLE_PATTERNS = {
    "balance_sheet": [
        # EXACT matches (high priority)
        "bilan arrete au",
        "bilan consolide",
        "bilan au",
        # These MUST be standalone "bilan" not part of other phrases
    ],
    "income_statement": [
        "etat de resultat",
        "compte de resultat",
        "etat des resultats",
        "income statement",
        "profit and loss",
    ],
    "cashflow": [
        "etat des flux de tresorerie",
        "etat de flux de tresorerie",
        "tableau des flux de tresorerie",
        "tableau de flux de tresorerie",
        "flux de tresorerie",
        "cash flow statement",
        "statement of cash flows",
    ],
}

# NEGATIVE patterns - if these appear, REJECT the page
NEGATIVE_PATTERNS = {
    "balance_sheet": [
        "hors bilan",           # Off-balance sheet - NOT the main balance sheet
        "engagements hors",     # Off-balance commitments
        "notes aux etats",      # Notes section
        "annexes",              # Annexes
        # Auditor report patterns - mentions "bilan" but is NOT the balance sheet
        "commissaire aux comptes",
        "rapport du commissaire",
        "examen limite",
        "nous avons effectue",
        "opinion",
        "audit",
    ],
    "income_statement": [
        "notes aux etats",
        "annexes",
        # Auditor report patterns
        "commissaire aux comptes",
        "rapport du commissaire",
        "examen limite",
        "nous avons effectue",
    ],
    "cashflow": [
        "notes aux etats",
        "annexes",
        "variation de l'actif net",  # SICAV variation - not cash flow
        "variation de lactif net",
    ],
}

# Structure validation keywords (need at least 2 matches)
STRUCTURE_KEYWORDS = {
    "balance_sheet": [
        "actif",
        "passif",
        "total actif",
        "total passif",
        "actif net",
        "capitaux propres",
        "capital souscrit",
        "sommes capitalisables",
        "sommes distribuables",
        "immobilisations",
        "creances",
        "dettes",
    ],
    "income_statement": [
        "revenus",
        "charges",
        "revenu net",
        "resultat net",
        "resultat d'exploitation",
        "resultat net de l'exercice",
        "produits",
        "produits d'exploitation",
        "charges d'exploitation",
        "charges de gestion",
        "marge",
        "benefice",
        "produit net bancaire",  # Bank-specific
    ],
    "cashflow": [
        "activites d'exploitation",
        "activites d'investissement",
        "activites de financement",
        "flux d'exploitation",
        "flux d'investissement",
        "flux de financement",
        "exploitation",
        "investissement",
        "financement",
        "variation de tresorerie",
        "tresorerie debut",
        "tresorerie fin",
    ],
}

# STRONG STRUCTURAL INDICATORS - for fallback detection when title is missing
# These are very distinctive patterns that ONLY appear in specific statement types
STRONG_STRUCTURE_PATTERNS = {
    "balance_sheet": [
        # Bank balance sheet specific line items (AC1, AC2, PA1, PA2 codes)
        "ac1-",
        "ac2-",
        "ac3-",
        "pa1-",
        "pa2-",
        # Common balance sheet totals
        "total de l'actif",
        "total actif",
        "total du passif",
        "total passif",
    ],
    "income_statement": [
        # Bank income statement specific line items (PR1, CH1 codes)
        "pr1-",
        "pr2-",
        "ch1-",
        "ch2-",
        # Key totals
        "produit net bancaire",
        "total produits d'exploitation",
        "total charges d'exploitation",
    ],
    "cashflow": [
        # Very distinctive cash flow patterns
        "tresorerie et equivalent",
        "tresorerie a la cloture",
        "variation nette de tresorerie",
    ],
}

# Fuzzy matching threshold (higher = stricter)
FUZZY_THRESHOLD = 80

# Maximum pages per statement
MAX_PAGES_PER_STATEMENT = 2

# Maximum gap between statements
MAX_GAP = 5


# =============================================================================
# TEXT UTILITIES
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove accents, collapse whitespace."""
    if not text:
        return ""
    
    # Remove accents
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Lowercase
    text = text.lower()
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def get_header(text: str) -> str:
    """Get first 20% of text (header area) - stricter than before."""
    if not text:
        return ""
    cutoff = max(len(text) // 5, 300)  # First 20%, at least 300 chars
    return text[:cutoff]


def get_first_lines(text: str, num_lines: int = 10) -> str:
    """Get first N lines of text - for precise title detection."""
    if not text:
        return ""
    lines = text.split('\n')
    return '\n'.join(lines[:num_lines])


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_text(pdf_path: str, use_ocr: bool = True) -> List[Tuple[int, str]]:
    """
    Extract text from all pages.
    
    If a page has no text and use_ocr=True, attempt OCR extraction.
    
    Returns:
        List of (page_number, text) tuples (1-indexed)
    """
    pdf_path = str(Path(pdf_path).resolve())
    pages = []
    
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text("text")
        
        # If no text extracted and OCR is enabled, try OCR
        if len(text.strip()) < 50 and use_ocr:
            ocr_text = extract_text_ocr(page)
            if ocr_text:
                text = ocr_text
        
        pages.append((i + 1, text))  # 1-indexed
    doc.close()
    
    return pages


def extract_text_ocr(page) -> Optional[str]:
    """
    Extract text from a PDF page using OCR.
    
    Requires pytesseract and Tesseract to be installed.
    Returns None if OCR fails or is not available.
    """
    try:
        import pytesseract
        from PIL import Image
        import io
        import os
        
        # Auto-detect Tesseract on Windows if not in PATH
        if os.name == 'nt':  # Windows
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
        # Render page to image at 300 DPI for good OCR quality
        mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Preprocess: convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Try French first, fallback to English only
        try:
            text = pytesseract.image_to_string(img, lang='fra+eng')
        except pytesseract.TesseractError:
            # French not available, try English only
            text = pytesseract.image_to_string(img, lang='eng')
        
        return text
        
    except ImportError:
        # pytesseract not installed
        return None
    except Exception as e:
        # OCR failed (Tesseract not installed, etc.)
        # Uncomment for debugging: print(f"OCR Error: {e}")
        return None


# =============================================================================
# TITLE DETECTION - HIGH PRECISION
# =============================================================================

def check_negative_patterns(text: str, statement_type: str) -> Tuple[bool, str]:
    """
    Check if text contains NEGATIVE patterns that disqualify it.
    
    IMPORTANT: We distinguish between:
    - HARD negatives: Always reject (e.g., "hors bilan", "notes aux etats")
    - SOFT negatives: Only reject if MULTIPLE appear (e.g., auditor report indicators)
    
    Returns:
        (is_rejected, rejection_reason)
    """
    normalized = normalize_text(text)
    
    # HARD negatives - always reject if found
    hard_negatives = {
        "balance_sheet": ["hors bilan", "engagements hors", "notes aux etats", "annexes"],
        "income_statement": ["notes aux etats", "annexes"],
        "cashflow": ["notes aux etats", "annexes", "variation de l'actif net", "variation de lactif net"],
    }
    
    # SOFT negatives - only reject if 2+ are found (indicates auditor report, not just mention)
    soft_negatives = {
        "balance_sheet": ["commissaire aux comptes", "rapport du commissaire", "examen limite", 
                         "nous avons effectue", "opinion", "audit"],
        "income_statement": ["commissaire aux comptes", "rapport du commissaire", "examen limite", 
                            "nous avons effectue"],
        "cashflow": [],
    }
    
    # Check hard negatives first
    for neg in hard_negatives.get(statement_type, []):
        if neg in normalized:
            return True, neg
    
    # Check soft negatives - need 2+ matches to reject
    soft_matches = []
    for neg in soft_negatives.get(statement_type, []):
        if neg in normalized:
            soft_matches.append(neg)
    
    if len(soft_matches) >= 2:
        return True, soft_matches[0]
    
    return False, ""


def detect_title(header: str, statement_type: str, full_text: str) -> Tuple[bool, int, str]:
    """
    Detect if header contains a title for the given statement type.
    
    This is a HIGH-PRECISION detection:
    1. Check NEGATIVE patterns first (reject if found)
    2. Check EXACT patterns in FIRST 5 LINES (primary title position)
    3. Check patterns in header area
    4. Use fuzzy matching only if exact fails
    
    Returns:
        (detected, confidence_score, matched_pattern)
    """
    # First check if page should be rejected
    rejected, reason = check_negative_patterns(header, statement_type)
    if rejected:
        return False, 0, f"REJECTED: {reason}"
    
    # Also check full text for negative patterns in the first half
    first_half = full_text[:len(full_text)//2]
    rejected, reason = check_negative_patterns(first_half, statement_type)
    if rejected:
        return False, 0, f"REJECTED: {reason}"
    
    normalized_header = normalize_text(header)
    
    # Get first 20 lines for PRIMARY title detection (covers intro + title)
    first_lines = normalize_text(get_first_lines(full_text, 20))
    
    patterns = TITLE_PATTERNS.get(statement_type, [])
    
    best_score = 0
    best_pattern = ""
    
    # HIGH PRIORITY: Check patterns in FIRST 5 LINES (true title position)
    for pattern in patterns:
        if pattern in first_lines:
            return True, 100, pattern
    
    # Special case for BILAN: must be in first 5 lines as standalone
    if statement_type == "balance_sheet":
        if re.search(r'\bbilan\b', first_lines):
            if "hors" not in first_lines:
                return True, 100, "bilan (primary)"
    
    # MEDIUM PRIORITY: Check full header for EXACT patterns
    for pattern in patterns:
        if pattern in normalized_header:
            # Lower confidence if not in first lines
            return True, 85, pattern + " (header)"
    
    # Special case for BILAN in header (not first lines)
    if statement_type == "balance_sheet":
        if re.search(r'\bbilan\b', normalized_header):
            if "hors" not in normalized_header:
                # Even lower confidence - might be a reference
                return True, 70, "bilan (mention)"
    
    # LOW PRIORITY: Fuzzy matching (only in first lines)
    for pattern in patterns:
        words = first_lines.split()
        for i in range(len(words)):
            for j in range(i + 1, min(i + 6, len(words) + 1)):
                chunk = ' '.join(words[i:j])
                score = fuzz.ratio(pattern, chunk)
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
    
    detected = best_score >= FUZZY_THRESHOLD
    return detected, best_score, best_pattern


# =============================================================================
# STRUCTURE VALIDATION
# =============================================================================

def validate_structure(text: str, statement_type: str) -> Tuple[bool, List[str]]:
    """
    Validate that page contains expected structure.
    Requires at least 2 keyword matches.
    
    Returns:
        (valid, matched_keywords)
    """
    normalized = normalize_text(text)
    keywords = STRUCTURE_KEYWORDS.get(statement_type, [])
    
    matched = []
    for kw in keywords:
        if kw in normalized:
            matched.append(kw)
    
    valid = len(matched) >= 2
    return valid, matched


def detect_strong_structure(text: str, statement_type: str) -> Tuple[bool, int, List[str]]:
    """
    Detect pages using STRONG structural indicators only.
    
    This is a FALLBACK for old PDFs that don't have clear titles but
    have distinctive structural patterns (like bank statements with AC1, PR1 codes).
    
    Returns:
        (detected, confidence_score, matched_patterns)
    """
    normalized = normalize_text(text)
    
    # Check negative patterns first
    rejected, reason = check_negative_patterns(text[:len(text)//2], statement_type)
    if rejected:
        return False, 0, []
    
    patterns = STRONG_STRUCTURE_PATTERNS.get(statement_type, [])
    matched = []
    
    for pattern in patterns:
        if pattern in normalized:
            matched.append(pattern)
    
    # Need at least 2 strong pattern matches for high confidence
    if len(matched) >= 2:
        # Also verify regular structure validation
        struct_valid, struct_kw = validate_structure(text, statement_type)
        if struct_valid:
            # Lower confidence than title-based detection (60 instead of 80+)
            return True, 60, matched
    
    return False, 0, matched


# =============================================================================
# ANCHOR SELECTION
# =============================================================================

@dataclass
class PageCandidate:
    """A candidate page for a statement type."""
    page_num: int
    statement_type: str
    title_score: int
    title_matched: str
    structure_valid: bool
    structure_keywords: List[str]
    
    @property
    def confidence(self) -> int:
        """Overall confidence score."""
        base = self.title_score
        if self.structure_valid:
            base += 20
        return base


def analyze_page(
    page_num: int, 
    text: str, 
    statement_type: str,
    verbose: bool = False,
    use_structure_fallback: bool = False
) -> Optional[PageCandidate]:
    """
    Analyze a single page for a specific statement type.
    
    HIGH-PRECISION analysis:
    1. Get header (first 20% + first 10 lines)
    2. Check negative patterns FIRST
    3. Detect title in header only
    4. Validate structure in full text
    5. (Optional) Fallback to structure-only detection
    
    Returns:
        PageCandidate if detected, None otherwise
    """
    header = get_header(text)
    first_lines = get_first_lines(text, 20)
    combined_header = header + " " + first_lines
    
    # Title detection with negative pattern checking
    detected, score, pattern = detect_title(combined_header, statement_type, text)
    
    if not detected:
        if verbose and "REJECTED" in pattern:
            print(f"  Page {page_num}: {pattern}")
        
        # FALLBACK: Try structure-only detection if enabled
        if use_structure_fallback:
            struct_detected, struct_score, struct_patterns = detect_strong_structure(text, statement_type)
            if struct_detected:
                struct_valid, struct_kw = validate_structure(text, statement_type)
                if verbose:
                    print(f"  Page {page_num}: STRUCTURE-ONLY score={struct_score} patterns={struct_patterns[:3]}")
                return PageCandidate(
                    page_num=page_num,
                    statement_type=statement_type,
                    title_score=struct_score,
                    title_matched=f"structure-only: {struct_patterns[0]}",
                    structure_valid=struct_valid,
                    structure_keywords=struct_kw
                )
        
        return None
    
    # Structure validation
    struct_valid, struct_kw = validate_structure(text, statement_type)
    
    if verbose:
        status = "[PASS]" if struct_valid else "[WEAK]"
        print(f"  Page {page_num}: title='{pattern}' score={score} struct={status} ({struct_kw[:3]})")
    
    return PageCandidate(
        page_num=page_num,
        statement_type=statement_type,
        title_score=score,
        title_matched=pattern,
        structure_valid=struct_valid,
        structure_keywords=struct_kw
    )


def select_anchor(candidates: List[PageCandidate], prefer_early: bool = False) -> Optional[PageCandidate]:
    """
    Select the BEST anchor from candidates.
    
    Selection criteria:
    1. Must have structure_valid = True (if any do)
    2. Highest confidence score (unless prefer_early=True)
    3. Earliest page (tie-breaker, or primary if prefer_early=True)
    
    When prefer_early=True:
    - Early pages (1-5) get strong preference
    - Structure-only matches on early pages are preferred over title matches on late pages
    """
    if not candidates:
        return None
    
    # Prefer candidates with valid structure
    valid_struct = [c for c in candidates if c.structure_valid]
    pool = valid_struct if valid_struct else candidates
    
    if prefer_early:
        # Give STRONG preference to early pages (1-5)
        early = [c for c in pool if c.page_num <= 5]
        if early:
            # Among early pages, prefer highest confidence
            early.sort(key=lambda c: (-c.confidence, c.page_num))
            return early[0]
        # If no early pages, fall back to normal selection
    
    # Sort by confidence (desc), then page number (asc)
    pool.sort(key=lambda c: (-c.confidence, c.page_num))
    
    return pool[0]


# =============================================================================
# LOCAL EXPANSION
# =============================================================================

def expand_anchor(anchor: PageCandidate, all_pages: List[Tuple[int, str]]) -> List[int]:
    """
    Expand anchor to include adjacent pages if they pass validation.
    
    CONSERVATIVE expansion:
    - Only expand to page + 1 (next page)
    - Next page must NOT contain NEGATIVE patterns
    - Next page must pass structure validation
    - Next page must NOT match a DIFFERENT statement type
    - Max 2 pages total
    """
    pages = [anchor.page_num]
    
    # Try to expand to next page
    next_page_num = anchor.page_num + 1
    next_page_text = None
    
    for pnum, text in all_pages:
        if pnum == next_page_num:
            next_page_text = text
            break
    
    if next_page_text:
        # FIRST: Check negative patterns
        header = get_header(next_page_text)
        rejected, reason = check_negative_patterns(header, anchor.statement_type)
        if rejected:
            # Don't expand - next page is disqualified
            return pages
        
        # Also check first half of text for negative patterns
        first_half = next_page_text[:len(next_page_text)//2]
        rejected, reason = check_negative_patterns(first_half, anchor.statement_type)
        if rejected:
            return pages
        
        # Check structure validation
        valid, _ = validate_structure(next_page_text, anchor.statement_type)
        
        if valid:
            # IMPORTANT: Check that next page doesn't look like a DIFFERENT statement
            first_lines = get_first_lines(next_page_text, 20)
            combined = header + " " + first_lines
            
            is_different_statement = False
            for other_type in ["balance_sheet", "income_statement", "cashflow"]:
                if other_type != anchor.statement_type:
                    detected, score, _ = detect_title(combined, other_type, next_page_text)
                    if detected and score > 80:
                        is_different_statement = True
                        break
            
            if not is_different_statement:
                pages.append(next_page_num)
    
    return pages[:MAX_PAGES_PER_STATEMENT]


# =============================================================================
# CONSISTENCY CONSTRAINTS
# =============================================================================

def apply_constraints(
    results: Dict[str, List[int]],
    verbose: bool = False
) -> Dict[str, List[int]]:
    """
    Apply consistency constraints:
    1. Distance: max gap between consecutive statements <= MAX_GAP
    
    NOTE: Order constraint (balance < income < cashflow) is REMOVED
    because some documents (especially SICAVs) don't follow this order.
    """
    # Get anchors (first page of each result)
    anchors = {}
    for stmt_type, pages in results.items():
        if pages:
            anchors[stmt_type] = min(pages)
    
    if len(anchors) < 2:
        return results  # Nothing to constrain
    
    # Check distance between statements
    sorted_anchors = sorted(anchors.items(), key=lambda x: x[1])
    
    # Find the main cluster of statements
    # Keep statements that are within MAX_GAP of each other
    if len(sorted_anchors) >= 2:
        # Build clusters
        clusters = []
        current_cluster = [sorted_anchors[0]]
        
        for i in range(1, len(sorted_anchors)):
            prev_type, prev_page = sorted_anchors[i - 1]
            curr_type, curr_page = sorted_anchors[i]
            
            if curr_page - prev_page <= MAX_GAP:
                current_cluster.append(sorted_anchors[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_anchors[i]]
        
        clusters.append(current_cluster)
        
        # Keep only the largest cluster
        largest_cluster = max(clusters, key=len)
        cluster_types = {t for t, p in largest_cluster}
        
        # Remove statements not in the largest cluster
        for stmt_type in list(results.keys()):
            if stmt_type not in cluster_types:
                if verbose:
                    print(f"  Removing {stmt_type} (outside main cluster)")
                results[stmt_type] = []
    
    return results


# =============================================================================
# MAIN DETECTOR
# =============================================================================

@dataclass
class DetectionResult:
    """Detection results container."""
    balance_sheet_pages: List[int]
    income_statement_pages: List[int]
    cashflow_pages: List[int]
    
    def to_dict(self) -> Dict:
        return {
            "balance_sheet_pages": self.balance_sheet_pages,
            "income_statement_pages": self.income_statement_pages,
            "cashflow_pages": self.cashflow_pages,
        }


def detect_financial_statements(
    pdf_path: str,
    verbose: bool = True
) -> DetectionResult:
    """
    Main detection function - HIGH PRECISION (v3).
    
    Algorithm:
    1. Extract text from all pages
    2. For each statement type:
       a. Check NEGATIVE patterns first (reject false positives)
       b. Detect title in header area only
       c. Validate structure
       d. Select best anchor
       e. Expand conservatively
    3. If nothing found, retry with structure-only fallback
    4. Apply consistency constraints
    5. Return results
    """
    pdf_path = str(Path(pdf_path).resolve())
    
    if verbose:
        print("=" * 60)
        print(f"FINANCIAL STATEMENT DETECTOR v3 (High Precision)")
        print(f"File: {Path(pdf_path).name}")
        print("=" * 60)
    
    # Step 1: Extract text
    pages = extract_text(pdf_path)
    
    if verbose:
        print(f"Total pages: {len(pages)}\n")
    
    # Step 2: Detect anchors for each statement type (title-based first)
    results = {}
    
    for statement_type in ["balance_sheet", "income_statement", "cashflow"]:
        if verbose:
            print(f"--- Detecting {statement_type.upper()} ---")
        
        # First pass: title-based detection
        candidates = []
        for page_num, text in pages:
            candidate = analyze_page(page_num, text, statement_type, verbose, use_structure_fallback=False)
            if candidate:
                candidates.append(candidate)
        
        # Select anchor
        anchor = select_anchor(candidates)
        
        if anchor:
            if verbose:
                print(f"  -> Anchor: page {anchor.page_num} (confidence={anchor.confidence})")
            
            # Expand locally
            expanded = expand_anchor(anchor, pages)
            results[statement_type] = expanded
            
            if verbose:
                print(f"  -> Final pages: {expanded}")
        else:
            results[statement_type] = []
            if verbose:
                print(f"  -> Not detected")
        
        if verbose:
            print()
    
    # Step 3: Check cluster coherence BEFORE applying constraints
    # If detected pages are scattered (will be removed by constraints), try structure fallback
    temp_results = apply_constraints(dict(results), verbose=False)
    
    # If constraints removed BS or IS, try structure-based fallback for early pages
    run_fallback = False
    for stmt_type in ["balance_sheet", "income_statement"]:
        if results.get(stmt_type) and not temp_results.get(stmt_type):
            run_fallback = True
            if verbose:
                print(f"--- {stmt_type.upper()} would be removed by constraints, trying structure fallback ---")
            break
    
    # Also run fallback if BS or IS is completely missing
    if not results.get("balance_sheet") or not results.get("income_statement"):
        run_fallback = True
    
    if run_fallback:
        if verbose:
            print("--- FALLBACK: Trying structure-only detection (pages 1-10) ---")
        
        for statement_type in ["balance_sheet", "income_statement"]:
            # Only try fallback if title-based detection failed OR would be removed
            if not results.get(statement_type) or not temp_results.get(statement_type):
                candidates = []
                # Only check first 10 pages for structure-based fallback
                early_pages = [(pn, txt) for pn, txt in pages if pn <= 10]
                for page_num, text in early_pages:
                    candidate = analyze_page(page_num, text, statement_type, verbose, use_structure_fallback=True)
                    if candidate:
                        candidates.append(candidate)
                
                # Use prefer_early=True to prioritize early pages even if lower score
                anchor = select_anchor(candidates, prefer_early=True)
                
                if anchor:
                    if verbose:
                        print(f"  -> FALLBACK Anchor: page {anchor.page_num}")
                    expanded = expand_anchor(anchor, pages)
                    results[statement_type] = expanded
                    if verbose:
                        print(f"  -> Final pages: {expanded}")
        
        if verbose:
            print()
    
    # Step 4: Apply constraints
    if verbose:
        print("--- Applying constraints ---")
    
    results = apply_constraints(results, verbose)
    
    # Build result
    result = DetectionResult(
        balance_sheet_pages=results.get("balance_sheet", []),
        income_statement_pages=results.get("income_statement", []),
        cashflow_pages=results.get("cashflow", []),
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"  Balance Sheet:    {result.balance_sheet_pages or '[]'}")
        print(f"  Income Statement: {result.income_statement_pages or '[]'}")
        print(f"  Cash Flow:        {result.cashflow_pages or '[]'}")
        print("=" * 60)
    
    return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def detect_financial_tables(pdf_path: str, verbose: bool = True, **kwargs) -> Dict:
    """Convenience function returning dict format."""
    result = detect_financial_statements(pdf_path, verbose=verbose)
    return result.to_dict()


def get_financial_pages(pdf_path: str) -> Dict[str, List[int]]:
    """
    Simple function for integration - returns just page numbers.
    
    Usage:
        pages = get_financial_pages("report.pdf")
        print(pages)
        # {'balance_sheet': [2, 3], 'income_statement': [4, 5], 'cashflow': [5]}
    
    Returns:
        Dict with keys: 'balance_sheet', 'income_statement', 'cashflow'
        Values are lists of page numbers (1-indexed)
    """
    result = detect_financial_statements(pdf_path, verbose=False)
    return {
        "balance_sheet": result.balance_sheet_pages,
        "income_statement": result.income_statement_pages,
        "cashflow": result.cashflow_pages,
    }


def check_ocr_status() -> Dict[str, any]:
    """
    Check if OCR is properly configured and return status.
    
    Returns:
        Dict with 'available', 'tesseract_path', 'languages', 'error' keys
    """
    import os
    result = {
        'available': False,
        'tesseract_path': None,
        'languages': [],
        'error': None
    }
    
    try:
        import pytesseract
        
        # Auto-detect Tesseract on Windows
        if os.name == 'nt':
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
        # Test if Tesseract works
        version = pytesseract.get_tesseract_version()
        result['available'] = True
        result['tesseract_path'] = pytesseract.pytesseract.tesseract_cmd
        
        # Get available languages
        try:
            langs = pytesseract.get_languages()
            result['languages'] = langs
        except:
            pass
            
    except ImportError:
        result['error'] = "pytesseract not installed. Run: pip install pytesseract"
    except Exception as e:
        result['error'] = str(e)
        
    return result


def print_ocr_instructions():
    """Print instructions for setting up OCR."""
    status = check_ocr_status()
    
    print("=" * 60)
    print("OCR STATUS CHECK")
    print("=" * 60)
    
    if status['available']:
        print(f"[OK] Tesseract OCR is available")
        print(f"     Path: {status['tesseract_path']}")
        print(f"     Languages: {', '.join(status['languages'][:5])}")
        if 'fra' not in status['languages']:
            print("\n[WARNING] French language pack not installed.")
            print("          For better results with French PDFs, install 'fra' language.")
    else:
        print(f"[ERROR] OCR not available: {status['error']}")
        print("\n" + "=" * 60)
        print("HOW TO INSTALL TESSERACT OCR (Windows)")
        print("=" * 60)
        print("""
1. Download Tesseract installer:
   https://github.com/UB-Mannheim/tesseract/wiki
   
2. Run the installer and:
   - Check "Add to PATH"
   - Select additional languages: French (fra)
   
3. Restart your terminal/IDE

4. Verify installation:
   tesseract --version
   
5. If not in PATH, the program will auto-detect from:
   C:\\Program Files\\Tesseract-OCR\\tesseract.exe
""")
    
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Financial Statement Detector v3")
    parser.add_argument("pdf_path", nargs='?', help="Path to PDF")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    parser.add_argument("--json", "-j", action="store_true", help="JSON output only")
    parser.add_argument("--check-ocr", action="store_true", help="Check OCR status")
    
    args = parser.parse_args()
    
    if args.check_ocr:
        print_ocr_instructions()
        return
    
    if not args.pdf_path:
        parser.print_help()
        return
    
    verbose = not args.quiet and not args.json
    result = detect_financial_statements(args.pdf_path, verbose=verbose)
    
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
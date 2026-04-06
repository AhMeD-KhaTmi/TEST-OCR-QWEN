# 🔍 COMPREHENSIVE TECHNICAL AUDIT REPORT
## Financial OCR Pipeline - Qwen-VL Based System

**Audit Date:** 2026-04-06  
**Auditor:** AI Systems Auditor  
**System Version:** Production candidate  
**Lines of Code Reviewed:** ~5,000+

---

## 1. EXECUTIVE SUMMARY

### System Classification: **FRAGILE PROTOTYPE** (Not Production-Ready)

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Architecture** | ⚠️ Adequate | Modular but tightly coupled |
| **Robustness** | ❌ Poor | Multiple single points of failure |
| **Correctness** | ⚠️ Partial | Validation logic has edge cases |
| **Maintainability** | ⚠️ Fair | Good documentation, but complex flows |
| **Reliability Score** | **52/100** | Not production-ready |

### Main Strengths
1. **Well-documented codebase** - Extensive docstrings and comments explaining logic
2. **Multi-layer validation** - Alignment engine + financial validation + sanity checks
3. **Defensive programming** - Multiple safeguards (schema locking, immutable columns)
4. **Dynamic schema detection** - Adapts to different table formats

### Main Risks
1. **VLM single point of failure** - No fallback if Qwen-VL fails or hallucinates
2. **Column misalignment can cascade** - Error in early stage corrupts all downstream
3. **Note vs. number disambiguation is fragile** - Magic thresholds prone to failure
4. **Confidence scores don't reflect actual correctness**
5. **JSON truncation handling is incomplete**

---

## 2. ARCHITECTURE DIAGRAM (TEXTUAL)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FLASK API (flask_app.py)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
            ┌───────▼───────┐                   ┌───────▼───────┐
            │   IMAGE PATH  │                   │   PDF BYTES   │
            └───────┬───────┘                   └───────┬───────┘
                    │                                   │
                    │                           ┌───────▼───────┐
                    │                           │ pdf_handler.py │
                    │                           │ - page render  │
                    │                           │ - DPI optimize │
                    │                           └───────┬───────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                              ┌───────▼───────┐
                              │ PREPROCESSING │
                              │ run_qwen_vl.py│
                              │ - auto crop   │
                              │ - resize      │
                              │ - VLM optimize│
                              └───────┬───────┘
                                      │
                              ┌───────▼───────┐
                              │   INFERENCE   │
                              │ Qwen3-VL-8B   │
                              │ (4-bit quant) │
                              │ → Raw JSON    │
                              └───────┬───────┘
                                      │
                              ┌───────▼───────────────┐
                              │ JSON EXTRACTION       │
                              │ extract_json_from_    │
                              │ response()            │
                              │ - regex extraction    │
                              │ - truncation recovery │
                              └───────┬───────────────┘
                                      │
                              ┌───────▼───────────────┐
                              │ SCHEMA DETECTION      │
                              │ table_alignment_      │
                              │ engine.py             │
                              │ - column role detect  │
                              │ - header merge        │
                              │ - schema lock         │
                              └───────┬───────────────┘
                                      │
                              ┌───────▼───────────────┐
                              │ ALIGNMENT ENGINE      │
                              │ - value classification│
                              │ - semantic alignment  │
                              │ - row validation      │
                              └───────┬───────────────┘
                                      │
                              ┌───────▼───────────────┐
                              │ COLUMN REALIGNMENT    │
                              │ column_realignment_   │
                              │ engine.py             │
                              │ - note detection      │
                              │ - shift correction    │
                              └───────┬───────────────┘
                                      │
                              ┌───────▼───────────────┐
                              │ FINANCIAL VALIDATION  │
                              │ financial_validation_ │
                              │ engine.py             │
                              │ - variation check     │
                              │ - total validation    │
                              │ - sign consistency    │
                              └───────┬───────────────┘
                                      │
                              ┌───────▼───────────────┐
                              │     OUTPUT JSON       │
                              │ + confidence score    │
                              │ + _schema metadata    │
                              │ + correction log      │
                              └───────────────────────┘
```

---

## 3. MODULE-BY-MODULE ANALYSIS

### 3.1 `run_qwen_vl.py` (1,650+ lines)

**Purpose:** Core VLM inference, image preprocessing, title detection

**Strengths:**
- GPU verification and VRAM monitoring
- Adaptive image resizing based on dimensions
- Smart cropping with safeguards (min 25% height preserved)
- Schema validation after inference
- Truncation detection

**Weaknesses:**
1. **Magic numbers throughout**: `MAX_DIMENSION = 1600`, `TARGET_DIMENSION = 1200`, `MIN_DIMENSION = 600` - no empirical justification
2. **Single retry on OOM**: Only halves tokens once, may still fail
3. **VLM optimization hardcoded**: Gamma 1.18, contrast 0.82 - not tunable per document
4. **Title detection is fragile**: Relies on keyword matching for table type classification

**Potential Bugs:**
- Line 1585: OOM retry doesn't verify success
- `_optimize_for_vlm()` may damage low-contrast images further

**Edge Cases Not Handled:**
- Documents in languages other than French/English
- Rotated or skewed tables
- Tables split across multiple images

---

### 3.2 `table_alignment_engine.py` (1,700+ lines)

**Purpose:** Schema detection, value classification, semantic alignment

**Strengths:**
- Comprehensive value type classification (TEXT, NOTE, NUMBER, PERCENT)
- Multi-row header normalization
- Schema locking prevents artificial columns
- Date format consistency validation
- Fallback schema inference from data

**Weaknesses:**
1. **NOTE pattern regex is complex and fragile** (lines 54-74):
   ```python
   _NOTE_PATTERN = re.compile(r"""^ ... $""", re.VERBOSE | re.IGNORECASE)
   ```
   - May miss unusual note formats: `[1]`, `*`, `†`
   
2. **European number handling has edge cases** (lines 151-168):
   - Pattern `1.234.567,89` works but `1,234.567` (mixed format) may break
   
3. **align_row() is position-dependent despite claims**:
   - Uses `max(text_cells, key=lambda c: c["length"])` - longest text wins
   - If two labels exist, shorter one is discarded

4. **Magic threshold for header confidence** (line 340):
   ```python
   HEADER_CONFIDENCE_THRESHOLD = 0.3  # Lowered to be more permissive
   ```
   - Too permissive may accept garbage headers

**Potential Bugs:**
- `_merge_schemas()` can lose date column order if both schemas have same dates
- `classify_value()` doesn't handle scientific notation (`1.5e6`)

---

### 3.3 `column_realignment_engine.py` (700+ lines)

**Purpose:** Detect and fix column shift corruptions

**Strengths:**
- Conservative approach: Only flags "VERY clearly notes"
- Column type detection by statistical analysis
- Shift left algorithm to fix note misplacement

**Weaknesses:**
1. **NOTE_VS_FINANCIAL_THRESHOLD = 50 is arbitrary**:
   - A value of 49 is treated as note, 51 as financial
   - No documentation on how this was determined

2. **MIN_MEDIAN_FOR_NOTE_DETECTION = 50,000**:
   - Tables with smaller magnitudes get NO note detection
   - Small company financials will fail

3. **Duplicate return statement** (line 584-585):
   ```python
       return realigned, corrections
       
       return realigned, corrections  # DEAD CODE
   ```

4. **`realign_row()` modifies Note column without checking schema**:
   - Hardcoded list of note column names (line 519-520)

**Potential Bugs:**
- `_is_small_integer()` can match negative numbers incorrectly
- Space-separated notes like "7 1" bypass numeric parsing

---

### 3.4 `financial_validation_engine.py` (1,100+ lines)

**Purpose:** Accounting correctness validation

**Strengths:**
- Date order detection (newest = current, oldest = previous)
- Immutable column protection (date columns never modified)
- Division by zero handling
- Sign consistency validation
- Total row validation with section detection

**Weaknesses:**
1. **VARIATION_TOLERANCE = 1.0 may be too strict**:
   - 1 unit tolerance on values in millions could fail due to rounding
   
2. **PERCENT_TOLERANCE = 0.5 may be too loose**:
   - 0.5% error on 100% = acceptable
   - 0.5% error on 0.1% = 500% relative error

3. **Total validation only works for simple structures**:
   - Nested sections (TOTAL ACTIF containing sub-totals) not handled
   - Sub-totals within sections will cause false positives

4. **Date order detection can fail** (lines 91-129):
   - Only tries 3 date formats
   - Falls back to column order if parsing fails

**Potential Bugs:**
- `handle_edge_cases()` uses `schema.date_cols[0]` directly without calling `identify_date_order()`
- `_is_total_row()` only checks for "total" in label - may miss "SOMME", "SUBTOTAL"

---

### 3.5 `json_table_utils.py` (1,500+ lines)

**Purpose:** JSON parsing, label recovery, schema enforcement

**Strengths:**
- Multiple JSON extraction strategies (regex, AST fallback)
- Label column detection with multiple strategies
- Deduplication with fuzzy matching
- Backward compatibility shims

**Weaknesses:**
1. **TARGET_SCHEMA_COLUMNS is still referenced** (line 306):
   - Supposed to be deprecated but still used in deduplication

2. **Label recovery is limited**:
   - Only recovers from first text cell in row
   - Cannot recover from OCR that completely missed the label

3. **`_map_source_column()` has hardcoded dates** (lines 332-335):
   ```python
   if "30/06/2022" in token or "30.06.2022" in token:
       return "30/06/2022"
   ```
   - Only recognizes 3 specific dates

---

### 3.6 `pdf_handler.py` (500+ lines)

**Purpose:** PDF to image conversion

**Strengths:**
- Auto-quality DPI optimization (200, 250, 300)
- Image quality metrics (contrast, blur)
- Table detection heuristics

**Weaknesses:**
1. **Table detection score threshold is extremely low**:
   ```python
   TABLE_DETECTION_MIN_SCORE = 0.02  # Very low threshold
   ```
   - Effectively disables table detection

2. **No support for PDF text extraction**:
   - Uses image rendering even for text-based PDFs
   - Loses precision that direct text extraction would provide

---

### 3.7 `flask_app.py` (1,000+ lines)

**Purpose:** REST API for extraction

**Strengths:**
- CORS enabled
- File validation
- PDF and image support
- VRAM monitoring

**Weaknesses:**
1. **No request rate limiting**
2. **No authentication**
3. **Synchronous processing** - single request blocks server
4. **Temp files may accumulate** if cleanup fails

---

### 3.8 `financial_table_detector.py` (900+ lines)

**Purpose:** Detect financial statement types in PDFs

**Strengths:**
- Comprehensive negative patterns (auditor reports, notes sections)
- Structure validation keywords
- Fuzzy matching with threshold

**Weaknesses:**
1. **French-centric**: English keywords present but limited
2. **No support for Arabic, Spanish, German financials**

---

### 3.9 `label_recovery.py` (400 lines)

**Purpose:** Recover missing labels post-extraction

**Strengths:**
- Uses note-to-label mapping
- NEVER generates artificial labels

**Weaknesses:**
1. **Hardcoded Attijari Bank mapping** (lines 126-139):
   - Only works for this specific bank's documents
2. **Targeted re-extraction not implemented**:
   - `reextract_function` parameter exists but no implementation

---

## 4. CRITICAL ISSUES (RANKED)

### 🔴 CRITICAL (System-Breaking)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | **VLM hallucination has no detection** | `run_qwen_vl.py` | Can output completely fabricated data with high confidence |
| 2 | **JSON truncation recovery is incomplete** | `json_table_utils.py` | Large tables may lose 50%+ of rows silently |
| 3 | **Note/number disambiguation uses magic thresholds** | `column_realignment_engine.py` | Tables with small numbers get wrong column alignment |
| 4 | **No cross-validation between extracted and original** | Entire pipeline | No way to verify extraction correctness |

### 🟠 HIGH (Data Corruption Risk)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 5 | **Schema contamination from multi-table PDFs** | `table_alignment_engine.py` | Dates from different tables can merge |
| 6 | **Column shift detection can miss shifts** | `column_realignment_engine.py` | Values end up in wrong columns |
| 7 | **Date order fallback uses column position** | `financial_validation_engine.py` | Variation calculated backwards |
| 8 | **Total validation assumes simple structure** | `financial_validation_engine.py` | Nested totals trigger false corrections |

### 🟡 MEDIUM (Reliability Issues)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 9 | **European number format edge cases** | `table_alignment_engine.py` | Numbers parsed incorrectly |
| 10 | **Hardcoded VLM optimization parameters** | `run_qwen_vl.py` | Low-contrast images get worse |
| 11 | **OOM handling only retries once** | `run_qwen_vl.py` | Large tables may fail extraction |
| 12 | **Confidence score is structural, not semantic** | `table_alignment_engine.py` | High confidence ≠ correct data |

### 🟢 LOW (Quality Issues)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 13 | **Dead code** | `column_realignment_engine.py:584-585` | Maintainability |
| 14 | **French-centric detection** | `financial_table_detector.py` | Limited international support |
| 15 | **No request authentication** | `flask_app.py` | Security risk |

---

## 5. ROOT CAUSE ANALYSIS

### Root Cause 1: **VLM Output Treated as Ground Truth**
The system implicitly trusts VLM output. All post-processing tries to "fix" minor issues but cannot detect or correct fundamental hallucinations.

**Evidence:**
- No comparison between extracted values and image content
- Confidence is based on schema completeness, not data accuracy

### Root Cause 2: **Magic Thresholds Without Empirical Validation**
Multiple critical thresholds were set by intuition:
- `NOTE_VS_FINANCIAL_THRESHOLD = 50`
- `MIN_MEDIAN_FOR_NOTE_DETECTION = 50000`
- `HEADER_CONFIDENCE_THRESHOLD = 0.3`

**Evidence:**
- No test suite validates these thresholds
- Comments say "CONSERVATIVE" but no data supports this

### Root Cause 3: **Position-Based Logic Disguised as Semantic**
Despite claims of "semantic alignment", the system still relies on:
- Longest text cell = label
- First numeric values = date columns
- Column order for unresolved cases

**Evidence:**
- `align_row()` uses `max(text_cells, key=lambda c: c["length"])`
- Date columns assigned to `num_vals[i]` by position

### Root Cause 4: **Insufficient Test Coverage**
Test files exist but are minimal:
- `test_alignment_engine.py`
- `test_financial_validation.py`
- `test_header_validation.py`

**Evidence:**
- No integration tests
- No fuzzing
- No regression test suite for known failures

---

## 6. FAILURE SCENARIOS

### Scenario 1: Multi-Year Table with Similar Structure
**Input:** Balance sheet with columns: `Label, Note, 2024, 2023, 2022, Variation 24-23, Variation 23-22`

**Failure Mode:**
1. Schema detection finds 3 date columns
2. Only first 2 are used for variation calculation
3. `2022` column values get silently ignored
4. Variation calculations only use 2024-2023

**Impact:** Half the variation data is wrong

---

### Scenario 2: Table with Small Values
**Input:** Startup financials with values in thousands (5, 12, 87, 150)

**Failure Mode:**
1. Median magnitude = 63 (below 50,000 threshold)
2. Note detection is COMPLETELY DISABLED
3. Note "(1-2)" in Note column gets classified as NUMBER
4. Entire row shifts left

**Impact:** Catastrophic column misalignment

---

### Scenario 3: Truncated Large Table
**Input:** 80-row income statement

**Failure Mode:**
1. VLM hits token limit at row 45
2. JSON ends abruptly: `{"Label": "Operating exp..`
3. `extract_json_from_response()` tries recovery
4. Recovery adds missing `}` but loses 35 rows
5. Output shows 45 rows with "complete" status

**Impact:** Half the table is missing, user unaware

---

### Scenario 4: PDF with Multiple Tables
**Input:** Annual report PDF with balance sheet (page 12) and income statement (page 15)

**Failure Mode:**
1. User selects pages [12, 15]
2. Both pages processed
3. Results MERGED into single table
4. Balance sheet ACTIF rows followed by income PRODUITS rows
5. Schema contaminated with dates from both tables

**Impact:** Nonsensical merged output

---

### Scenario 5: OCR Drift on Scanned Document
**Input:** Low-quality scan with slight skew

**Failure Mode:**
1. OCR reads "1 234 567" as "12 34 567" (space drift)
2. Value parsed as 12 (small integer)
3. Detected as NOTE
4. Shift correction moves it to Note column
5. Row completely corrupted

**Impact:** Random data corruption

---

## 7. RELIABILITY SCORE

### Scoring Breakdown (100 points total)

| Category | Max Points | Score | Reasoning |
|----------|------------|-------|-----------|
| **Correctness** | 25 | 12 | Validation logic has edge cases, no ground truth comparison |
| **Robustness** | 20 | 8 | Single points of failure, magic thresholds |
| **Error Handling** | 15 | 10 | Good try/except coverage but errors not always surfaced |
| **Test Coverage** | 15 | 5 | Minimal tests, no integration tests |
| **Documentation** | 10 | 8 | Good inline docs, but no API docs |
| **Maintainability** | 10 | 6 | Complex flows, tight coupling |
| **Security** | 5 | 3 | No auth, temp file risks |

### **TOTAL: 52/100**

---

## 8. FINAL VERDICT

### Is the System Production-Ready?

# ❌ NO

### What Blocks Production Deployment?

1. **No correctness verification**: System cannot distinguish between correct extraction and hallucination

2. **Magic thresholds**: Critical logic depends on arbitrary numbers that will fail on edge cases

3. **Truncation handling**: Large tables may silently lose data

4. **Multi-table contamination**: PDF pages processed together will corrupt each other

5. **Confidence is misleading**: High confidence does not indicate correct data

### Minimum Requirements for Production

1. **Ground truth comparison**: Implement OCR cross-validation or human-in-the-loop verification

2. **Threshold calibration**: Run systematic tests to calibrate thresholds with real data

3. **Table isolation**: Process each table independently with strict schema isolation

4. **Truncation detection**: Add explicit row count validation and retry with more tokens

5. **Confidence redesign**: Base confidence on mathematical verification, not schema completeness

6. **Integration tests**: Build test suite with 100+ real document samples

7. **Monitoring**: Add logging for all corrections and confidence drops

---

## APPENDIX A: Files Reviewed

| File | Lines | Status |
|------|-------|--------|
| `run_qwen_vl.py` | 1,650+ | Reviewed |
| `table_alignment_engine.py` | 1,700+ | Reviewed |
| `column_realignment_engine.py` | 700+ | Reviewed |
| `financial_validation_engine.py` | 1,100+ | Reviewed |
| `json_table_utils.py` | 1,500+ | Reviewed |
| `pdf_handler.py` | 500+ | Reviewed |
| `flask_app.py` | 1,000+ | Reviewed |
| `financial_table_detector.py` | 900+ | Reviewed |
| `label_recovery.py` | 400 | Reviewed |
| `extract_json.py` | 263 | Reviewed |
| `extract_pdf.py` | - | Skipped (not found) |

---

## APPENDIX B: Data Flow Trace (Single Row)

### Stage 1: Raw OCR Output
```json
{
  "Label": "",
  "Note": "(4.1)",
  "31.12.2024": "799 892",
  "31.12.2023": "652 772",
  "Variation Montant": "147 120",
  "Variation %": "22,5%"
}
```
**Problem:** Label is empty, Note leaked into first position

### Stage 2: After Schema Detection
```
Schema detected:
  label_col: "Label"
  note_col: "Note"
  date_cols: ["31.12.2024", "31.12.2023"]
  variation_amount_col: "Variation Montant"
  variation_percent_col: "Variation %"
```
**Status:** Schema correct, but row data still wrong

### Stage 3: After Alignment Engine
```json
{
  "Label": "",
  "Note": "(4.1)",
  "31.12.2024": "799892",
  "31.12.2023": "652772",
  "Variation Montant": "147120",
  "Variation %": "22.5%"
}
```
**Problem:** Label still empty, numbers normalized but no correction

### Stage 4: After Column Realignment
```json
{
  "Label": "",
  "Note": "(4.1)",
  "31.12.2024": "799892",
  "31.12.2023": "652772",
  "Variation Montant": "147120",
  "Variation %": "22.5%"
}
```
**Problem:** No correction applied because median magnitude check passed

### Stage 5: After Financial Validation
```json
{
  "Label": "",
  "Note": "(4.1)",
  "31.12.2024": "799892",
  "31.12.2023": "652772",
  "Variation Montant": "147120",
  "Variation %": "22.5%"
}
```
**Final Output:** Label is STILL empty. System outputs this with high confidence.

### **Corruption Point Identified:** 
The VLM failed to extract the label. No subsequent stage can recover it because:
1. Label recovery only works if text exists in another column
2. Note mapping only works for known note codes
3. No re-extraction mechanism implemented

---

*End of Audit Report*

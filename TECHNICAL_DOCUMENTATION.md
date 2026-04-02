# Financial Statement OCR System - Complete Technical Documentation

## 0. Scope, Method, and Audit Boundaries

### Scope of this document
- This document is a static code audit of the codebase located at:
  - `c:\Users\THOURAYA\test qwen`
- The analysis covers:
  - Architecture
  - Workflow/data flow
  - File-by-file documentation
  - Function-level behavior
  - Dependencies
  - Issues, risks, and concrete fixes
  - Setup and execution guidance
  - Handoff notes for another AI agent (Claude)

### Static audit method
- No runtime execution was required.
- The audit was performed by reading source/config files and metadata of binary assets.
- Output behavior is inferred from code.

### Boundary clarification (important)
- Workspace contains many third-party/generated files:
  - `venv/**` (Python environment)
  - `frontend/node_modules/**` (NPM dependencies)
  - `__pycache__/**`
- These are external/vendor artifacts, not first-party project logic.
- They are not documented line-by-line; their internal behavior is treated as external/unknown.

---

## 1. Project Overview

### Purpose
This project extracts structured financial tables from:
- Image files (PNG/JPG/JPEG/TIFF)
- Selected pages from PDF files

using a vision-language model (`Qwen/Qwen3-VL-8B-Instruct`) with 4-bit quantization for GPU-constrained inference (target RTX 4060 8GB).

### Main functionality
- Backend API (Flask):
  - Upload image or PDF
  - OCR + table extraction via Qwen-VL
  - JSON parsing/post-processing/validation
  - Return structured JSON
- Frontend (React + Vite + Tailwind):
  - Drag-and-drop file upload
  - PDF page selection with thumbnails
  - Results view in table and JSON tabs
- CLI utility:
  - Standalone extraction script for local files
  - Exports JSON/CSV/Excel

### Global workflow (input -> processing -> output)

1. Input
- User uploads image or PDF (frontend) or provides image path (CLI).

2. Processing
- Preprocess image (crop/resize/chunking)
- Run model inference
- Extract JSON from model text output
- Post-process rows/columns (header cleanup, type classification, shift correction)
- Validate schema

3. Output
- API JSON response and UI display
- Optional file exports (`.json`, `.csv`, `.xlsx`, raw text) via CLI

---

## 2. Architecture

## 2.1 High-level components

1. Inference core
- `run_qwen_vl.py`
- Owns model loading, VRAM logic, image preprocessing, chunk inference, merge/dedup, rescue passes.

2. Parsing/post-processing core
- `json_table_utils.py`
- Converts model text to structured JSON and applies normalization/validation/export utilities.

3. API layer
- `flask_app.py`
- Defines REST endpoints for image and PDF extraction.

4. CLI layer
- `extract_json.py`
- Scriptable entry point for one-shot extraction with export formats.

5. Frontend
- `frontend/src/App.jsx` and related files
- Upload UX, page selection, extraction triggering, result rendering.

## 2.2 Component interaction map

1. Frontend -> Flask
- `POST /extract` for images
- `POST /pdf/info` to inspect PDFs
- `POST /pdf/extract` for selected pages

2. Flask -> Inference/Parser
- `run_qwen_vl.extract_table_from_image()`
- `json_table_utils.extract_json_from_response()`
- `json_table_utils.post_process_extraction()`
- `json_table_utils.validate_table_json()`

3. CLI -> Inference/Parser
- `extract_json.py` uses the same core modules as API.

## 2.3 Architectural patterns
- Pipeline pattern:
  - preprocess -> infer -> parse -> post-process -> validate
- Shared singleton pattern for model in API:
  - global `model`, `processor`, `model_loaded`
- Fallback strategy pattern:
  - strict prompt retry
  - chunked extraction
  - full-image rescue
  - top/bottom sectional rescue

---

## 3. Project Structure (first-party and runtime-managed files)

```text
test qwen/
  README.md
  requirements.txt
  run_qwen_vl.py
  json_table_utils.py
  flask_app.py
  extract_json.py
  test.png
  attijari_statement.png
  uploads/
    1774792590_test.png
    1774793157_test.png
    1774793486_test.png
    1774800199_test.png
    1774800586_image_2026-03-29_170944013.png
    1774808792_Capture_decran_2026-03-29_172321.png
    1774815205_at.png
    1774815510_at.png
  frontend/
    .gitignore
    package.json
    package-lock.json
    vite.config.js
    tailwind.config.js
    postcss.config.js
    eslint.config.js
    index.html
    src/
      main.jsx
      App.jsx
      index.css
```

Additional folders present:
- `offload/` (empty at audit time; purpose unknown)
- `venv/` (external environment)
- `frontend/node_modules/` (external dependencies)
- `__pycache__/` (generated)

---

## 4. File-by-File Analysis (all first-party/runtime files found)

## 4.1 Backend and core Python files

### `run_qwen_vl.py`
- Role:
  - Core model orchestration and extraction pipeline.
- Key functions:
  - `verify_gpu()`
  - `load_model()`
  - `preprocess_image()`
  - `split_image()`
  - `process_chunks()`
  - `run_inference()`
  - `extract_table_from_image()`
- Inputs:
  - Local image path
  - Optional custom prompt
  - inference params (`max_new_tokens`, `max_image_size`, crop flags)
- Outputs:
  - Raw model output string (often JSON text)
  - In chunked mode, can synthesize merged JSON string
- Dependencies:
  - `torch`, `transformers`, `bitsandbytes`, `Pillow`, `numpy`, `qwen_vl_utils`, `json_table_utils`

### `json_table_utils.py`
- Role:
  - Robust parser and post-processing for model outputs.
- Key functions:
  - `extract_json_from_response()`
  - `post_process_extraction()`
  - `validate_table_json()`
  - `convert_to_csv()`, `convert_to_excel()`
  - debug display helpers
- Inputs:
  - Model response string or parsed dict
- Outputs:
  - Normalized table dict (`columns`, `rows`)
  - Validation result/errors
  - CSV/Excel files
- Dependencies:
  - `json`, `re`, `csv`, `ast`, optional `pandas`

### `flask_app.py`
- Role:
  - Production API for image and PDF workflows.
- Key functions/routes:
  - Helpers for PDF rendering/quality/table detection:
    - `_parse_selected_pages()`, `_render_page_to_pil()`, `_image_quality_metrics()`, `detect_table_page()`, `pdf_to_images()`, `process_pages()`
  - Routes:
    - `/health`, `/extract`, `/extract-batch`, `/prompts`, `/pdf/info`, `/pdf/extract`, `/pdf/page-image/<page>`
- Inputs:
  - Multipart upload payloads
- Outputs:
  - JSON responses with extraction metadata and structured table data
- Dependencies:
  - `flask`, `flask-cors`, `werkzeug`, `fitz` (PyMuPDF), `Pillow`, `numpy`, `torch`

### `extract_json.py`
- Role:
  - CLI utility for extraction + export.
- Key functions:
  - `extract_and_parse_table()`
  - `main()`
- Inputs:
  - Image path via CLI or interactive input
- Outputs:
  - Saved files in `extracted_tables/`
  - terminal summary + execution log
- Dependencies:
  - Core project modules + Python stdlib

### `requirements.txt`
- Role:
  - Python dependency manifest.
- Notes:
  - Contains model/runtime/API/PDF packages.
  - `pandas` and `openpyxl` are not listed but Excel export defaults to enabled in CLI.

### `README.md`
- Role:
  - User-facing setup and usage guide.
- Notes:
  - Mostly aligned with current architecture.
  - One model naming inconsistency exists (see issues section).

## 4.2 Frontend files

### `frontend/src/App.jsx`
- Role:
  - Main React UI containing all app screens and most components.
- Key components/functions:
  - Presentational components:
    - `LoadingSpinner`, `ErrorAlert`, `FileUpload`, `PdfPageSelector`, `TableView`, `JsonView`, `StatsBar`, `PageResult`
  - Main stateful app:
    - `App` with handlers:
      - `handleFileSelect`, `handlePageToggle`, `handleExtractImage`, `handleExtractPdf`, `handleAxiosError`, `handleReset`
- Inputs:
  - User file interactions and API responses
- Outputs:
  - Rendered extraction UI
- Dependencies:
  - `react`, `axios`

### `frontend/src/main.jsx`
- Role:
  - React bootstrap entry file.
- Behavior:
  - Renders `<App />` into `#root` under `StrictMode`.

### `frontend/src/index.css`
- Role:
  - Tailwind import and base custom styles.
- Behavior:
  - Defines background and custom horizontal scrollbar style for table container.

### `frontend/index.html`
- Role:
  - Vite HTML entry template.

### `frontend/vite.config.js`
- Role:
  - Vite config and dev server proxy mapping `/api` -> `http://localhost:5000`.
- Note:
  - Current frontend code does not use `/api` proxy path.

### `frontend/tailwind.config.js`
- Role:
  - Tailwind content scanning configuration.

### `frontend/postcss.config.js`
- Role:
  - PostCSS plugin wiring for Tailwind and Autoprefixer.

### `frontend/eslint.config.js`
- Role:
  - ESLint flat config for JS/JSX with React hooks/reload rules.

### `frontend/package.json`
- Role:
  - Frontend package and script manifest.
- Notes:
  - Includes `pdfjs-dist`, but code currently relies on backend thumbnails instead.

### `frontend/package-lock.json`
- Role:
  - Exact dependency lockfile (generated).
- Notes:
  - No custom logic; should stay synchronized with `package.json`.

### `frontend/.gitignore`
- Role:
  - Frontend-only ignore rules.

## 4.3 Image assets and runtime upload artifacts

These files are binary images (no callable code). They are likely test inputs or persisted uploads.

### Root images
- `attijari_statement.png`
  - Role: sample financial statement image
  - Size: 58,689 bytes
  - Resolution: 693x515
- `test.png`
  - Role: test/sample input
  - Size: 376,776 bytes
  - Resolution: 1201x1699

### Uploads folder images
- `uploads/1774792590_test.png` - 376,776 bytes - 1201x1699
- `uploads/1774793157_test.png` - 376,776 bytes - 1201x1699
- `uploads/1774793486_test.png` - 376,776 bytes - 1201x1699
- `uploads/1774800199_test.png` - 376,776 bytes - 1201x1699
- `uploads/1774800586_image_2026-03-29_170944013.png` - 75,399 bytes - 559x809
- `uploads/1774808792_Capture_decran_2026-03-29_172321.png` - 155,058 bytes - 631x593
- `uploads/1774815205_at.png` - 117,527 bytes - 771x680
- `uploads/1774815510_at.png` - 117,527 bytes - 771x680

---

## 5. Function-Level Explanation (important functions)

## 5.1 `run_qwen_vl.py`

### `verify_gpu()`
- Purpose:
  - Hard-fail if CUDA is unavailable.
- Parameters:
  - None
- Returns:
  - `True` on success
- Logic:
  1. Check `torch.cuda.is_available()`
  2. Print GPU/device metadata
  3. Raise runtime error with reinstall instructions if CUDA missing

### `load_model()`
- Purpose:
  - Load quantized Qwen3-VL model + processor with GPU-first placement.
- Parameters:
  - None
- Returns:
  - `(model, processor)`
- Logic:
  1. Clear CUDA cache
  2. Build 4-bit `BitsAndBytesConfig`
  3. `from_pretrained()` with `device_map="auto"` and `max_memory={0: "7.8GB"}`
  4. Reject load if any module maps to CPU
  5. Load processor

### `crop_table_region(img, header_crop_ratio=0.18, margin_ratio=0.04)`
- Purpose:
  - Remove likely non-table margins/header.
- Parameters:
  - `img` (`PIL.Image`)
  - crop/margin ratios
- Returns:
  - Cropped `PIL.Image`
- Logic:
  1. Geometric crop
  2. Convert to grayscale
  3. Compute row ink density
  4. Refine top boundary to avoid cutting header line

### `preprocess_image(image_path, max_image_size=1200, enable_crop=True)`
- Purpose:
  - Prepare image for fast OCR with quality-preserving downscale.
- Parameters:
  - `image_path`, max width, crop toggle
- Returns:
  - `(temp_preprocessed_png_path, stats_dict)`
- Logic:
  1. Open RGB image
  2. Optional table-focused crop
  3. Downscale width and max height constraints
  4. Save temp PNG and return sizes

### `split_image(image, overlap=120, chunk_height=520)`
- Purpose:
  - Split tall images into overlapping vertical chunks.
- Parameters:
  - image + chunk config
- Returns:
  - List of chunk dicts with `image`, `y_start`, `y_end`, `header_attached`
- Logic:
  1. If short image, return single chunk
  2. For multi-chunk images, attach top header strip to non-first chunks
  3. Use overlap to reduce missed row boundaries

### `remove_duplicate_rows(rows, columns)`
- Purpose:
  - Remove chunk overlap duplicates.
- Parameters:
  - merged rows + column set
- Returns:
  - deduped rows list
- Logic:
  1. Build row signature
  2. Remove exact signature duplicates
  3. Secondary fuzzy label match (`SequenceMatcher`) to catch near-duplicates

### `merge_results(results)`
- Purpose:
  - Merge parsed JSON outputs from chunks/rescue runs into one table.
- Parameters:
  - chunk result list
- Returns:
  - `{columns, rows}` or `None`
- Logic:
  1. Keep successful parsed chunks
  2. Union columns and inferred fallback keys
  3. Normalize row objects to merged schema
  4. Deduplicate rows

### `process_chunks(chunks, model, processor, prompt, max_new_tokens=768)`
- Purpose:
  - Run inference chunk-by-chunk with retry/repair prompts.
- Parameters:
  - chunk list + model objects + prompt config
- Returns:
  - per-chunk result records with parse status/errors
- Logic:
  1. Save each chunk to temp file
  2. Infer with chunk prompt
  3. If parse fails, retry with strict JSON repair prompt
  4. Special crop retry for first chunk if needed

### `run_inference(model, processor, image_path, prompt, max_new_tokens=768)`
- Purpose:
  - Execute one deterministic generation pass.
- Parameters:
  - model, processor, image path, prompt, token limit
- Returns:
  - Decoded model text
- Logic:
  1. Build multimodal chat message
  2. Prepare tensors via `AutoProcessor`
  3. Move tensors to CUDA
  4. `model.generate(...)` with `do_sample=False`, `num_beams=1`
  5. Strip prompt tokens and decode output

### `extract_table_from_image(...)`
- Purpose:
  - End-to-end extraction orchestrator.
- Parameters:
  - model/processor/image path plus tuning params
- Returns:
  - Raw output text (often JSON string)
- Logic:
  1. Preprocess image
  2. Single-pass inference if short image
  3. For large images:
    - chunk process
    - merge and rescue attempts
  4. Return merged JSON string if available, else fallback output

## 5.2 `json_table_utils.py`

### `extract_json_robust(response)`
- Purpose:
  - Best-effort parsing of imperfect model responses.
- Parameters:
  - raw text response
- Returns:
  - parsed dict or `None`
- Logic:
  1. Remove markdown fences
  2. Try pipe-table parser
  3. Slice likely JSON region
  4. Apply duplicate key fix and delimiter repairs
  5. Structural recovery for `columns`/`rows`
  6. Final Python-literal fallback via `ast.literal_eval`

### `post_process_extraction(data)`
- Purpose:
  - Normalize and clean extracted tables.
- Parameters:
  - parsed dict
- Returns:
  - transformed dict
- Logic:
  1. Convert old `table` format to new `columns + rows`
  2. Normalize row shape
  3. Remove header/title pollution rows
  4. Fix column shift (note misalignment)
  5. Add missing row `type`

### `validate_table_json(data)`
- Purpose:
  - Structural validation and warning collection.
- Parameters:
  - parsed table dict
- Returns:
  - `(is_valid: bool, errors: list[str])`

### `convert_to_csv(data, output_path, delimiter=',')`
- Purpose:
  - Export table to CSV.
- Parameters:
  - table dict, target path, delimiter
- Returns:
  - `True/False`

### `convert_to_excel(data, output_path)`
- Purpose:
  - Export table to Excel using pandas/openpyxl.
- Parameters:
  - table dict, target path
- Returns:
  - `True/False`

## 5.3 `flask_app.py`

### `_parse_selected_pages(pages_str)`
- Purpose:
  - Parse selected pages from JSON array string or CSV string.
- Returns:
  - sorted unique page number list

### `_image_quality_metrics(image)`
- Purpose:
  - Heuristic quality scoring for PDF render optimization.
- Returns:
  - dict with blur/contrast/resolution metrics and `low_quality` flag

### `detect_table_page(page, image)`
- Purpose:
  - Skip likely non-table PDF pages before expensive OCR.
- Returns:
  - `(is_table: bool, metrics: dict)`

### `pdf_to_images(pdf_bytes, selected_pages)`
- Purpose:
  - Render selected pages to PNG (raw + OCR-enhanced), with adaptive DPI.
- Returns:
  - dict with `page_count` and per-page artifact metadata

### `process_pages(pdf_bytes, selected_pages)`
- Purpose:
  - Run page extraction sequentially with fallback prompts/images.
- Returns:
  - dict with extraction results per page

### Route: `GET /health`
- Returns:
  - service/model/GPU state and VRAM metrics

### Route: `POST /extract`
- Input:
  - multipart `image` file + optional `prompt`
- Output:
  - extraction result with parsed JSON when available

### Route: `POST /extract-batch`
- Input:
  - multipart `images[]`
- Output:
  - per-file raw extraction results

### Route: `GET /prompts`
- Output:
  - predefined prompt templates

### Route: `POST /pdf/info`
- Input:
  - multipart `pdf`
- Output:
  - page count + base64 thumbnails

### Route: `POST /pdf/extract`
- Input:
  - multipart `pdf` + selected `pages`
- Output:
  - structured extraction per requested page

### Route: `POST /pdf/page-image/<int:page_num>`
- Input:
  - multipart `pdf`
- Output:
  - rendered PNG for one page

## 5.4 `extract_json.py`

### `extract_and_parse_table(model, processor, image_path, output_dir, save_formats)`
- Purpose:
  - One-call pipeline for CLI use.
- Returns:
  - result dict with success flag, parsed JSON, errors, output files, VRAM usage
- Logic:
  1. call core extractor
  2. parse JSON
  3. post-process + validate
  4. print preview/summary
  5. export raw/json/csv/excel

### `main()`
- Purpose:
  - CLI entrypoint with argument/interactive path handling and logging.

## 5.5 `frontend/src/App.jsx`

### Presentational components
- `LoadingSpinner`, `ErrorAlert`, `FileUpload`, `PdfPageSelector`, `TableView`, `JsonView`, `StatsBar`, `PageResult`
- Purpose:
  - UI rendering and interaction shell.

### Stateful main component `App`
- Purpose:
  - Owns app state and API calls.
- Important handlers:
  - `handleFileSelect(file)`
  - `handlePageToggle(pageNum)`
  - `handleExtractImage()`
  - `handleExtractPdf()`
  - `handleAxiosError(err)`
  - `handleReset()`

## 5.6 Complete function inventory by file

### `run_qwen_vl.py` (all functions)
- `verify_gpu()` -> validates CUDA availability; no params; returns `True` or raises.
- `get_vram_usage()` -> reads allocated/reserved VRAM; no params; returns tuple `(allocated_gb, reserved_gb)`.
- `print_vram_status(stage="")` -> logs VRAM usage label; param `stage: str`; returns `None`.
- `load_model()` -> loads quantized model+processor; no params; returns `(model, processor)`.
- `crop_table_region(img, header_crop_ratio, margin_ratio)` -> geometric+ink-density crop; params image and float ratios; returns cropped image.
- `preprocess_image(image_path, max_image_size, enable_crop)` -> reads and resizes image; returns `(temp_path, stats_dict)`.
- `split_image(image, overlap, chunk_height)` -> chunking strategy for tall pages; returns list of chunk dicts.
- `_normalize_text(value)` -> lowercase/clean helper; param any value; returns normalized `str`.
- `_row_signature(row, columns)` -> deterministic signature for dedup; returns `str`.
- `remove_duplicate_rows(rows, columns)` -> dedup rows using exact and fuzzy checks; returns list of row dicts.
- `merge_results(results)` -> merges chunk parsed payloads into one schema; returns dict or `None`.
- `_contains_label(rows, token)` -> checks if any row label contains token; returns `bool`.
- `_needs_sectional_rescue(merged)` -> heuristic trigger for additional rescue; returns `bool`.
- `_run_top_bottom_rescue(preprocessed_path, model, processor, prompt, max_new_tokens)` -> extracts top/bottom slices; returns chunk-like results list.
- `process_chunks(chunks, model, processor, prompt, max_new_tokens)` -> chunk infer + repair retries; returns result list with status/error.
- `run_inference(model, processor, image_path, prompt, max_new_tokens)` -> one generation pass; returns output `str`.
- `extract_table_from_image(model, processor, image_path, prompt, max_new_tokens, max_image_size, enable_crop)` -> full orchestration; returns output `str`.
- `main()` -> interactive script mode; no params; returns `None`.

### `json_table_utils.py` (all functions)
- `fix_duplicate_keys(json_str)` -> renames duplicate JSON keys by suffix; returns corrected JSON string.
- `parse_pipe_table_response(response)` -> parse markdown pipe table text; returns table dict or `None`.
- `extract_json_robust(response)` -> multi-strategy parser+repair; returns dict or `None`.
- `extract_json_from_response(response)` -> wrapper to robust parser; returns dict or `None`.
- `detect_note_pattern(value)` -> checks note token pattern like `(4.1)`; returns `bool`.
- `fix_column_shift(data)` -> shifts misplaced note values and columns; returns modified dict.
- `is_header_row(row)` -> detects title/header pollution rows; returns `bool`.
- `remove_header_pollution(data)` -> removes non-data rows; returns modified dict.
- `classify_row_type(row)` -> infers `section`/`data`/`total`; returns type string.
- `add_row_types(data)` -> fills missing row type values; returns modified dict.
- `convert_to_new_format(data)` -> converts legacy `{table:[...]}` to `{columns,rows}`; returns dict.
- `post_process_extraction(data)` -> executes full normalization pipeline; returns dict or `None`.
- `validate_table_json(data)` -> schema validation; returns `(is_valid, errors)`.
- `convert_to_csv(data, output_path, delimiter)` -> writes CSV; returns `bool`.
- `convert_to_excel(data, output_path)` -> writes Excel via pandas/openpyxl; returns `bool`.
- `print_table_summary(data)` -> prints human-readable table metrics; returns `None`.
- `pretty_print_table(data, max_rows)` -> prints formatted table preview; returns `None`.

### `flask_app.py` (all functions/routes)
- `allowed_file(filename)` -> extension whitelist check; returns `bool`.
- `_parse_selected_pages(pages_str)` -> parses page selection payload; returns sorted `list[int]`.
- `_render_page_to_pil(page, dpi)` -> renders PDF page with given DPI; returns `PIL.Image`.
- `_laplacian_variance(gray_array)` -> blur metric helper; returns `float`.
- `_image_quality_metrics(image)` -> image quality scoring; returns metrics dict.
- `enhance_image(image, apply_threshold=False)` -> grayscale/contrast/noise cleanup; returns enhanced RGB image.
- `detect_table_page(page, image)` -> table-likelihood heuristic; returns `(bool, metrics_dict)`.
- `pdf_to_images(pdf_bytes, selected_pages)` -> page rendering, enhancement, temp artifacts; returns conversion dict.
- `process_pages(pdf_bytes, selected_pages)` -> per-page OCR + fallback + parse; returns results dict.
- `health()` (`GET /health`) -> service status endpoint; returns JSON.
- `extract()` (`POST /extract`) -> single image extraction endpoint; returns JSON and status code.
- `extract_batch()` (`POST /extract-batch`) -> multi-image extraction endpoint; returns JSON.
- `get_prompts()` (`GET /prompts`) -> static prompt catalog endpoint; returns JSON.
- `pdf_info()` (`POST /pdf/info`) -> PDF metadata + thumbnails; returns JSON.
- `pdf_extract()` (`POST /pdf/extract`) -> selected-page extraction endpoint; returns JSON.
- `get_page_image(page_num)` (`POST /pdf/page-image/<int:page_num>`) -> rendered page image stream.
- `load_model_on_startup()` -> initializes global model objects; returns `None`.

### `extract_json.py` (all functions)
- `extract_and_parse_table(model, processor, image_path, output_dir=OUTPUT_DIR, save_formats=None)` -> CLI pipeline call; returns structured result dict.
- `main()` -> CLI argument handling + orchestration + log export; returns `None`.

### `frontend/src/App.jsx` (all top-level components/functions)
- `LoadingSpinner({message, subMessage})` -> loading UI block.
- `ErrorAlert({message, onClose})` -> dismissible error panel.
- `FileUpload({onFileSelect, selectedFile, filePreview, fileType})` -> drag-drop/upload UI and preview.
- `PdfPageSelector({thumbnails, selectedPages, onPageToggle, onSelectAll, onDeselectAll})` -> page selection grid.
- `TableView({data})` -> dynamic table renderer using `columns`/`rows`.
- `JsonView({data})` -> formatted JSON renderer with copy action.
- `StatsBar({stats})` -> extraction metric badges.
- `PageResult({pageData, pageNum})` -> PDF page card with table/json tabs.
- `App()` -> root container for all state, API calls, and page composition.

---

## 6. Data Flow and Transformations

## 6.1 Image extraction path

1. Frontend
- User uploads image -> `POST /extract`

2. Backend `/extract`
- Save temporary file
- `extract_table_from_image()`
  - preprocess image
  - optional chunking + merge + rescue
  - model output text
- `extract_json_from_response()`
- `post_process_extraction()`
- `validate_table_json()`
- Return JSON payload

3. Frontend display
- `TableView` renders `columns` and `rows`
- `JsonView` shows full structured payload

## 6.2 PDF extraction path

1. Frontend
- Upload PDF -> `POST /pdf/info`
- Receive thumbnails -> user selects page numbers
- Submit selected pages -> `POST /pdf/extract`

2. Backend `/pdf/extract`
- Parse selected pages
- `pdf_to_images()`:
  - adaptive DPI rendering
  - quality metrics
  - OCR-focused enhancement
  - table-likelihood scoring
- `process_pages()`:
  - per-page extraction (enhanced image first, raw fallback)
  - JSON parsing/post-processing/validation
- Return per-page result array

3. Frontend display
- Page cards (`PageResult`) with table/json tabs and timing status

## 6.3 Key transformations

- Image preprocessing:
  - geometric crop, grayscale-based top refinement, bounded resize
- Chunking:
  - vertical overlap + header strip injection for schema continuity
- Model text -> JSON:
  - markdown strip, delimiter repair, duplicate key repair, structural recovery
- Post-processing:
  - remove title/header pollution rows
  - infer `type` (`section`, `data`, `total`)
  - correct column shifts when note token appears in numeric column

---

## 7. Dependencies

## 7.1 Python dependencies

- `transformers`
  - Qwen model and processor APIs
- `accelerate`
  - model loading/runtime support
- `bitsandbytes`
  - 4-bit quantization for reduced VRAM
- `qwen-vl-utils`
  - multimodal input preparation (`process_vision_info`)
- `torch`
  - model execution and GPU management
- `pillow`
  - image I/O and preprocessing
- `numpy`
  - array operations and quality metrics
- `flask`, `flask-cors`, `werkzeug`
  - REST service and upload handling
- `PyMuPDF (fitz)`
  - PDF page rendering and text extraction
- `pynvml`
  - listed in requirements but not used in code (current state)
- Optional but currently needed by default CLI behavior:
  - `pandas`, `openpyxl` for Excel export

## 7.2 Frontend dependencies

- `react`, `react-dom`
  - UI rendering
- `axios`
  - HTTP client for backend calls
- `vite`
  - dev server and build tooling
- `tailwindcss`, `@tailwindcss/postcss`, `postcss`, `autoprefixer`
  - styling pipeline
- `eslint` plugins/tooling
  - linting
- `pdfjs-dist`
  - declared but unused in current source code

---

## 8. Issues and Remarks (priority list with fixes)

## 8.1 High severity

### 1) Upload size limit is defined but not enforced
- Location:
  - `flask_app.py:42`
- Problem:
  - `MAX_FILE_SIZE` constant is never applied (`app.config["MAX_CONTENT_LENGTH"]` missing).
- Why it matters:
  - Large uploads can exhaust memory and increase DoS risk.
- Concrete fix:
  1. Set `app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE`
  2. Add Flask handler for `RequestEntityTooLarge` returning 413 JSON.

### 2) Shared GPU model with threaded Flask can cause concurrency instability
- Location:
  - `flask_app.py:877` (`threaded=True`)
- Problem:
  - Multiple requests may simultaneously call `model.generate()` on a shared model without locking.
- Why it matters:
  - Can cause CUDA OOM, degraded latency, or nondeterministic failures.
- Concrete fix:
  1. Add an inference queue or global lock around extraction calls.
  2. Prefer process-based workers with one GPU worker and job queue for production.

### 3) `/extract` accepts PDFs because extension gate is global
- Location:
  - `flask_app.py:93`, `flask_app.py:434`
- Problem:
  - `allowed_file()` includes `pdf`, but `/extract` is image-only logic.
- Why it matters:
  - Wrong endpoint/file-type mismatch returns avoidable runtime errors.
- Concrete fix:
  - Use endpoint-specific validators:
    - `/extract`: image extensions only
    - `/pdf/*`: pdf only

### 4) Broad unauthenticated CORS exposure
- Location:
  - `flask_app.py:38`
- Problem:
  - `CORS(app)` enables all origins by default.
- Why it matters:
  - Increases abuse surface if service is exposed publicly.
- Concrete fix:
  - Restrict `origins` to trusted frontend domains.
  - Add auth/rate limiting for non-local deployment.

## 8.2 Medium severity

### 5) Frontend hardcodes backend URL and bypasses Vite proxy
- Location:
  - `frontend/src/App.jsx:5`
  - `frontend/vite.config.js` already defines `/api` proxy
- Problem:
  - Hardcoded `http://localhost:5000` breaks deploy flexibility and env portability.
- Why it matters:
  - Requires code edits per environment and can fail under reverse proxy.
- Concrete fix:
  - Use env var (`import.meta.env.VITE_API_BASE`) or relative `/api` path.

### 6) Object URL memory leak for uploaded images
- Location:
  - `frontend/src/App.jsx:462`
- Problem:
  - `URL.createObjectURL(file)` is never revoked.
- Why it matters:
  - Repeated uploads can leak browser memory.
- Concrete fix:
  - Store created URL and call `URL.revokeObjectURL(previousUrl)` on reset/new selection/unmount.

### 7) Metrics hidden when value is `0`
- Location:
  - `frontend/src/App.jsx:303`, `frontend/src/App.jsx:309`
- Problem:
  - Rendering checks use truthiness (`if (stats.inferenceTime)`), hiding zero values.
- Why it matters:
  - UI can display incomplete telemetry.
- Concrete fix:
  - Check with `!== undefined` / `!= null` instead of truthiness.

### 8) Inconsistent API contract between single and batch extraction
- Location:
  - `flask_app.py:566` (`extract_batch`)
- Problem:
  - Batch endpoint returns raw text only; no structured parsing like `/extract`.
- Why it matters:
  - Clients need different code paths for nearly same feature.
- Concrete fix:
  - Reuse parse/post-process/validate pipeline in batch responses too.

### 9) Duplicate-row logic compares only last value column in fuzzy branch
- Location:
  - `run_qwen_vl.py:362`
- Problem:
  - `signature.split("|")[-1]` compares only last segment, not full numeric vector.
- Why it matters:
  - False positive/negative dedup for multi-column rows.
- Concrete fix:
  - Compare full normalized value tuple/string, not just the last token.

### 10) Optional Excel dependencies are not in `requirements.txt`
- Location:
  - `extract_json.py:31`, `requirements.txt`
- Problem:
  - CLI defaults `SAVE_EXCEL=True`, but `pandas/openpyxl` are not declared.
- Why it matters:
  - Default workflow raises warning or partial failure on clean installs.
- Concrete fix:
  - Either add optional extra requirements or set `SAVE_EXCEL=False` by default.

## 8.3 Low severity / maintenance quality

### 11) Unused imports/dependencies
- Location:
  - `flask_app.py:7` (`send_from_directory` unused)
  - `requirements.txt:18` (`pynvml` not used)
  - `frontend/package.json:14` (`pdfjs-dist` not used)
- Why it matters:
  - Noise, larger install footprint, audit confusion.
- Fix:
  - Remove or implement actual usage.

### 12) Runtime artifacts are kept in repository workspace
- Location:
  - `uploads/*.png`
  - `venv/`, `frontend/node_modules/` present in workspace
- Why it matters:
  - Repository bloat and noisy diffs if versioned.
- Fix:
  - Add root `.gitignore` covering `uploads/`, `venv/`, `node_modules/`, `extracted_tables/`, cache/temp files.

### 13) README model reference inconsistency
- Location:
  - `README.md:158`
- Problem:
  - States "Qwen3-VL-8B (Qwen/Qwen2.5-VL-7B-Instruct)".
- Why it matters:
  - Confuses model identity/versioning.
- Fix:
  - Align README with actual `MODEL_ID` in `run_qwen_vl.py`.

### 14) JSON repair heuristics are aggressive
- Location:
  - `json_table_utils.py` (`extract_json_robust`)
- Problem:
  - Regex-based delimiter patching can accidentally alter legitimate text patterns.
- Why it matters:
  - Potential silent corruption of extracted values.
- Fix:
  - Prefer strict parser + targeted JSON5 fallback or constrained grammar parsing.

### 15) Frontend is monolithic in a single large component file
- Location:
  - `frontend/src/App.jsx`
- Problem:
  - UI and data orchestration are tightly coupled in one file.
- Why it matters:
  - Harder testing and maintenance.
- Fix:
  - Split into `components/`, `hooks/`, `services/api.js`, and feature modules.

---

## 9. Improvement Recommendations

## 9.1 Architecture improvements

1. Introduce an explicit service layer
- Move extraction orchestration from routes into service classes/functions.

2. Add request serialization for GPU
- Single-consumer queue or lock-based gate around inference.

3. Add stable contracts and schemas
- Use pydantic/dataclasses for response schema definitions and strict validation.

4. Modularize frontend
- Separate API client, state hooks, and presentational components.

## 9.2 OCR/model pipeline improvements

1. Improve table detection before OCR
- Combine text-layout score with connected-component/table-line detection.

2. Add deterministic post-merge ordering
- Preserve source row order with chunk coordinates and tie-break strategies.

3. Add confidence and provenance metadata
- Return per-row confidence and source chunk/page for auditability.

4. Reduce repeated full-image retries
- Retry only when parse confidence is low; cap rescue loops explicitly.

5. GPU optimization
- Benchmark `max_new_tokens`, chunk size, overlap, and image width per hardware profile.
- Consider adaptive token budget by estimated table complexity.

## 9.3 Reliability improvements

1. Add tests
- Unit tests:
  - JSON extraction and repair
  - post-processing and row classification
  - page parsing and validation logic
- Integration tests:
  - API endpoints with fixture images/PDFs

2. Add structured logging
- Include request IDs, per-step timing, and failure taxonomy.

3. Add idempotent temp-file management
- Dedicated temp namespace and periodic cleanup strategy.

---

## 10. Setup and Execution Guide

## 10.1 Hardware/software requirements

- GPU:
  - CUDA-capable NVIDIA GPU, target: RTX 4060 8GB+
- RAM:
  - 16GB minimum recommended
- Python:
  - 3.10+
- Node.js:
  - 18+
- Disk:
  - Additional space for model downloads (several GB)

## 10.2 Backend setup

```bash
cd "c:\Users\THOURAYA\test qwen"

# (optional) create virtualenv if needed
python -m venv venv
venv\Scripts\activate

# install torch with matching CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# install project dependencies
pip install -r requirements.txt

# optional (needed for Excel export path in CLI)
pip install pandas openpyxl
```

## 10.3 Frontend setup

```bash
cd "c:\Users\THOURAYA\test qwen\frontend"
npm install
```

## 10.4 Run backend + frontend

Terminal 1:
```bash
cd "c:\Users\THOURAYA\test qwen"
venv\Scripts\activate
python flask_app.py
```

Terminal 2:
```bash
cd "c:\Users\THOURAYA\test qwen\frontend"
npm run dev
```

Open:
- `http://localhost:5173`

## 10.5 CLI usage

```bash
cd "c:\Users\THOURAYA\test qwen"
venv\Scripts\activate
python extract_json.py path\to\image.png
```

## 10.6 API examples

Single image:
```bash
curl -X POST -F "image=@statement.png" http://localhost:5000/extract
```

PDF selected pages:
```bash
curl -X POST -F "pdf=@report.pdf" -F "pages=1,2,3" http://localhost:5000/pdf/extract
```

---

## 11. Summary for Claude (handoff-critical)

## 11.1 Key points to understand before edits

1. Core extraction contract
- The system is built around this normalized output:
  - `{ "columns": [...], "rows": [...] }`
- Many downstream UI and validation steps assume this contract.

2. GPU-first constraints
- Model loading and inference are tuned around 8GB VRAM with 4-bit quantization.
- Small parameter changes can break latency/stability.

3. Fallback-heavy pipeline
- Robustness is achieved through layered retries (chunk repair, full-image rescue, sectional rescue).
- Changes to prompting/chunking can significantly affect extraction quality.

4. PDF path differs from image path
- PDF uses pre-render quality optimization and table detection before OCR.
- Keep this distinction clear when refactoring.

## 11.2 Critical files

- `run_qwen_vl.py`
  - Inference control, chunking, merge, rescue behavior
- `json_table_utils.py`
  - JSON extraction robustness and schema normalization
- `flask_app.py`
  - API entry points, PDF flow, global model lifecycle
- `frontend/src/App.jsx`
  - Full UI + API interaction logic

## 11.3 Highest-risk areas for regressions

1. JSON parsing/recovery heuristics (`extract_json_robust`)
2. Row dedup/merge behavior in chunked extraction
3. Concurrency and GPU memory behavior in API service mode
4. Frontend-backend contract assumptions (`parsed_json`, timing fields)

## 11.4 What should not be changed casually

1. Output schema (`columns`, `rows`, row `type`)
2. Strict JSON-only prompt constraints without replacement guardrails
3. VRAM-protective defaults (`4-bit quantization`, bounded image sizes) without benchmarking
4. PDF preprocessing + table-page filtering sequence

## 11.5 Unknowns / assumptions to preserve

- `offload/` directory intention is unknown (empty at audit time).
- No explicit test suite is present, so behavior preservation must rely on fixture-based manual validation.
- Deployment target (local-only vs public) is not fully specified; security hardening should match intended exposure.

---

## 12. Quick Action Plan (recommended next development steps)

1. Stabilize API safety
- enforce max upload size
- add inference lock/queue
- restrict CORS + file-type validators

2. Clean contracts and deps
- unify batch/single extraction response schema
- remove unused deps/imports
- align README with actual model

3. Improve maintainability
- split frontend monolith
- add tests for parser/post-processing and endpoint fixtures

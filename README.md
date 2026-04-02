# Financial Statement OCR

Extract tables from financial statement images and PDFs using Qwen3-VL-8B vision-language model.

## Features

- **Image Extraction**: Upload PNG, JPG, or JPEG images
- **PDF Support**: Select specific pages to extract from multi-page PDFs
- **Dynamic Schema**: Automatically detects columns from the document (no hardcoded format)
- **Structured Output**: JSON with typed rows (section/data/total)
- **Web Interface**: Modern React frontend with drag & drop
- **REST API**: Flask backend for programmatic access

## System Requirements

- **GPU**: NVIDIA RTX 4060 8GB or higher (CUDA compatible)
- **RAM**: 16GB minimum
- **Python**: 3.10+
- **Node.js**: 18+ (for frontend)

## Project Structure

```
test qwen/
├── flask_app.py          # Flask REST API (main backend)
├── run_qwen_vl.py        # Qwen3-VL model loading & inference
├── json_table_utils.py   # JSON parsing & validation
├── extract_json.py       # CLI tool for standalone extraction
├── requirements.txt      # Python dependencies
├── attijari_statement.png # Sample test image
├── uploads/              # Temporary upload folder
├── venv/                 # Python virtual environment
└── frontend/             # React web application
    ├── src/
    │   ├── App.jsx       # Main React component
    │   ├── index.css     # TailwindCSS styles
    │   └── main.jsx      # React entry point
    ├── package.json      # Node dependencies
    └── vite.config.js    # Vite configuration
```

## Installation

### 1. Python Backend

```bash
# Activate virtual environment
cd "test qwen"
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
pip install flask flask-cors PyMuPDF
```

### 2. React Frontend

```bash
cd frontend
npm install
```

## Usage

### Option 1: Web Interface (Recommended)

**Terminal 1 - Start Backend:**
```bash
cd "test qwen"
venv\Scripts\activate
python flask_app.py
```
Wait for "MODEL LOADED SUCCESSFULLY!" message (30-60 seconds first run).

**Terminal 2 - Start Frontend:**
```bash
cd "test qwen/frontend"
npm run dev
```

**Open:** http://localhost:5173

1. Drag & drop an image or PDF
2. For PDFs: select pages to extract
3. Click "Extract Table"
4. View results in Table or JSON format

### Option 2: CLI Tool

```bash
python extract_json.py path/to/image.png
```

Outputs JSON, CSV, and raw text to `extracted_tables/` folder.

### Option 3: REST API

```bash
# Single image
curl -X POST -F "image=@statement.png" http://localhost:5000/extract

# PDF pages
curl -X POST -F "pdf=@report.pdf" -F "pages=1,2,3" http://localhost:5000/pdf/extract
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API status & GPU info |
| POST | `/extract` | Extract from single image |
| POST | `/extract-batch` | Extract from multiple images |
| GET | `/prompts` | Predefined extraction prompts |
| POST | `/pdf/info` | Get PDF page count & thumbnails |
| POST | `/pdf/extract` | Extract from selected PDF pages |

## Output Format

```json
{
  "columns": ["Label", "Note", "30/06/2022", "31/12/2021"],
  "rows": [
    {"type": "section", "Label": "ACTIFS"},
    {"type": "data", "Label": "Caisse", "Note": "1", "30/06/2022": "100,000", "31/12/2021": "90,000"},
    {"type": "total", "Label": "Total Actifs", "30/06/2022": "500,000", "31/12/2021": "450,000"}
  ]
}
```

**Row Types:**
- `section`: Category header (no numeric values)
- `data`: Regular data row
- `total`: Summary/total row

## Troubleshooting

### Model Loading Slow
First load downloads ~5GB model. Subsequent loads use cached files.

### CUDA Out of Memory
- Close other GPU applications
- Model uses ~6GB VRAM with 4-bit quantization

### TailwindCSS Not Working
Ensure you have TailwindCSS v4 packages:
```bash
cd frontend
npm install -D tailwindcss @tailwindcss/postcss autoprefixer
```

## Technology Stack

- **Backend**: Flask, PyTorch, Transformers, BitsAndBytes (4-bit quantization)
- **Model**: Qwen3-VL-8B (Qwen/Qwen2.5-VL-7B-Instruct)
- **Frontend**: React, Vite, TailwindCSS, Axios
- **PDF Processing**: PyMuPDF (fitz)

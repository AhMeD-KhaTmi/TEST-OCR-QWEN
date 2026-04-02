import { useState, useCallback, useRef } from 'react'
import axios from 'axios'

// API Configuration
const API_URL = 'http://localhost:5000'

// =============================================================================
// LOADING SPINNER COMPONENT
// =============================================================================
function LoadingSpinner({ message = "Extracting table...", subMessage = "This may take 30-90 seconds" }) {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent"></div>
      <p className="mt-4 text-gray-600 font-medium">{message}</p>
      <p className="text-sm text-gray-400">{subMessage}</p>
    </div>
  )
}

// =============================================================================
// ERROR ALERT COMPONENT
// =============================================================================
function ErrorAlert({ message, onClose }) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
      <div className="flex items-start">
        <svg className="w-5 h-5 text-red-500 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
        </svg>
        <div className="ml-3 flex-1">
          <p className="text-sm font-medium text-red-800">Error</p>
          <p className="text-sm text-red-600 mt-1">{message}</p>
        </div>
        <button onClick={onClose} className="text-red-400 hover:text-red-600">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </button>
      </div>
    </div>
  )
}

// =============================================================================
// FILE UPLOAD COMPONENT (IMAGES + PDF)
// =============================================================================
function FileUpload({ onFileSelect, selectedFile, filePreview, fileType }) {
  const [isDragOver, setIsDragOver] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) {
      onFileSelect(file)
    }
  }, [onFileSelect])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      onFileSelect(file)
    }
  }

  return (
    <div className="space-y-4">
      {/* Drag & Drop Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
          transition-all duration-200
          ${isDragOver
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }
        `}
      >
        <input
          type="file"
          accept="image/*,.pdf"
          onChange={handleFileChange}
          className="hidden"
          id="file-upload"
        />
        <label htmlFor="file-upload" className="cursor-pointer">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 48 48">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M24 8v24m-12-12h24" />
            <rect x="6" y="6" width="36" height="36" rx="8" strokeWidth={2} />
          </svg>
          <p className="mt-4 text-lg font-medium text-gray-700">
            {isDragOver ? 'Drop file here' : 'Drag & drop an image or PDF'}
          </p>
          <p className="mt-1 text-sm text-gray-500">or click to browse</p>
          <p className="mt-2 text-xs text-gray-400">PNG, JPG, JPEG, PDF up to 10MB</p>
        </label>
      </div>

      {/* File Preview (Image only) */}
      {filePreview && fileType === 'image' && (
        <div className="border rounded-xl p-4 bg-white">
          <p className="text-sm font-medium text-gray-700 mb-2">Selected Image:</p>
          <div className="flex items-center gap-4">
            <img
              src={filePreview}
              alt="Preview"
              className="max-h-48 rounded-lg border shadow-sm"
            />
            <div className="text-left">
              <p className="text-sm font-medium text-gray-800">{selectedFile?.name}</p>
              <p className="text-xs text-gray-500">
                {selectedFile && (selectedFile.size / 1024).toFixed(1)} KB
              </p>
            </div>
          </div>
        </div>
      )}

      {/* PDF Selected Indicator */}
      {selectedFile && fileType === 'pdf' && (
        <div className="border rounded-xl p-4 bg-blue-50 border-blue-200">
          <div className="flex items-center gap-3">
            <svg className="w-10 h-10 text-red-600" fill="currentColor" viewBox="0 0 24 24">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm-1 2l5 5h-5V4zM8.5 13h1v4h-1v-4zm2 0h1.5c.83 0 1.5.67 1.5 1.5s-.67 1.5-1.5 1.5H11.5v1h-1v-4zm1 2h.5c.28 0 .5-.22.5-.5s-.22-.5-.5-.5h-.5v1zm3-2h1.5c.83 0 1.5.67 1.5 1.5v1c0 .83-.67 1.5-1.5 1.5H14.5v-4zm1 3h.5c.28 0 .5-.22.5-.5v-1c0-.28-.22-.5-.5-.5h-.5v2z"/>
            </svg>
            <div>
              <p className="font-medium text-gray-800">{selectedFile.name}</p>
              <p className="text-sm text-gray-500">{(selectedFile.size / 1024).toFixed(1)} KB - Loading pages...</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// =============================================================================
// PDF PAGE SELECTOR COMPONENT
// =============================================================================
function PdfPageSelector({ thumbnails, selectedPages, onPageToggle, onSelectAll, onDeselectAll }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-gray-800">
          Select Pages to Extract ({selectedPages.length} selected)
        </h3>
        <div className="flex gap-2">
          <button
            onClick={onSelectAll}
            className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
          >
            Select All
          </button>
          <button
            onClick={onDeselectAll}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
          >
            Deselect All
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 max-h-96 overflow-y-auto p-2">
        {thumbnails.map((thumb) => (
          <div
            key={thumb.page}
            onClick={() => onPageToggle(thumb.page)}
            className={`
              relative cursor-pointer rounded-lg overflow-hidden border-2 transition-all
              ${selectedPages.includes(thumb.page)
                ? 'border-blue-500 ring-2 ring-blue-200'
                : 'border-gray-200 hover:border-gray-400'
              }
            `}
          >
            <img
              src={thumb.thumbnail}
              alt={`Page ${thumb.page}`}
              className="w-full h-auto"
            />
            <div className={`
              absolute top-2 right-2 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
              ${selectedPages.includes(thumb.page)
                ? 'bg-blue-500 text-white'
                : 'bg-white text-gray-600 border'
              }
            `}>
              {selectedPages.includes(thumb.page) ? '✓' : thumb.page}
            </div>
            <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-center text-xs py-1">
              Page {thumb.page}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// =============================================================================
// TABLE VIEW COMPONENT (DYNAMIC COLUMNS)
// =============================================================================
function TableView({ data }) {
  if (!data || !data.columns || !data.rows) {
    return <p className="text-gray-500 text-center py-8">No table data available</p>
  }

  const { columns, rows } = data

  const getRowStyle = (type) => {
    switch (type) {
      case 'section':
        return 'bg-blue-50 font-semibold text-blue-800'
      case 'total':
        return 'bg-gray-100 font-bold border-t-2 border-gray-300'
      default:
        return 'bg-white hover:bg-gray-50'
    }
  }

  return (
    <div className="table-container overflow-x-auto rounded-lg border border-gray-200">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-800">
          <tr>
            {columns.map((col, idx) => (
              <th
                key={idx}
                className="px-4 py-3 text-left text-xs font-semibold text-white uppercase tracking-wider whitespace-nowrap"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {rows.map((row, rowIdx) => (
            <tr key={rowIdx} className={getRowStyle(row.type)}>
              {columns.map((col, colIdx) => (
                <td
                  key={colIdx}
                  className="px-4 py-3 text-sm whitespace-nowrap"
                >
                  {row[col] || row.values?.[col] || ''}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// =============================================================================
// JSON VIEW COMPONENT
// =============================================================================
function JsonView({ data }) {
  const [copied, setCopied] = useState(false)

  const jsonString = JSON.stringify(data, null, 2)

  const handleCopy = () => {
    navigator.clipboard.writeText(jsonString)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="relative">
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 px-3 py-1 text-xs bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
      >
        {copied ? 'Copied!' : 'Copy'}
      </button>
      <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm max-h-96">
        {jsonString}
      </pre>
    </div>
  )
}

// =============================================================================
// STATS BAR COMPONENT
// =============================================================================
function StatsBar({ stats }) {
  return (
    <div className="flex flex-wrap gap-4 bg-gray-100 rounded-lg p-4 mb-4">
      {stats.inferenceTime && (
        <div className="flex items-center gap-2">
          <span className="text-gray-500 text-sm">Inference:</span>
          <span className="font-semibold text-gray-800">{stats.inferenceTime.toFixed(2)}s</span>
        </div>
      )}
      {stats.vramUsed && (
        <div className="flex items-center gap-2">
          <span className="text-gray-500 text-sm">VRAM:</span>
          <span className="font-semibold text-gray-800">{stats.vramUsed.toFixed(2)} GB</span>
        </div>
      )}
      {stats.rowCount !== undefined && (
        <div className="flex items-center gap-2">
          <span className="text-gray-500 text-sm">Rows:</span>
          <span className="font-semibold text-gray-800">{stats.rowCount}</span>
        </div>
      )}
      {stats.columnCount !== undefined && (
        <div className="flex items-center gap-2">
          <span className="text-gray-500 text-sm">Columns:</span>
          <span className="font-semibold text-gray-800">{stats.columnCount}</span>
        </div>
      )}
      {stats.pagesProcessed && (
        <div className="flex items-center gap-2">
          <span className="text-gray-500 text-sm">Pages:</span>
          <span className="font-semibold text-gray-800">{stats.pagesProcessed}</span>
        </div>
      )}
    </div>
  )
}

// =============================================================================
// PAGE RESULT COMPONENT (FOR PDF MULTI-PAGE RESULTS)
// =============================================================================
function PageResult({ pageData, pageNum }) {
  const [activeTab, setActiveTab] = useState('table')

  return (
    <div className="border rounded-lg overflow-hidden">
      <div className="bg-gray-100 px-4 py-2 flex items-center justify-between">
        <span className="font-semibold text-gray-800">Page {pageNum}</span>
        {pageData.success ? (
          <span className="text-green-600 text-sm flex items-center gap-1">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            {pageData.inference_time_seconds?.toFixed(1)}s
          </span>
        ) : (
          <span className="text-red-600 text-sm">Failed</span>
        )}
      </div>

      {pageData.success && pageData.parsed_json ? (
        <div className="p-4">
          {/* Mini Tabs */}
          <div className="flex gap-2 mb-3 border-b">
            <button
              onClick={() => setActiveTab('table')}
              className={`px-3 py-1 text-sm font-medium ${
                activeTab === 'table'
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-500'
              }`}
            >
              Table
            </button>
            <button
              onClick={() => setActiveTab('json')}
              className={`px-3 py-1 text-sm font-medium ${
                activeTab === 'json'
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-500'
              }`}
            >
              JSON
            </button>
          </div>

          {activeTab === 'table' ? (
            <TableView data={pageData.parsed_json} />
          ) : (
            <JsonView data={pageData.parsed_json} />
          )}
        </div>
      ) : (
        <div className="p-4 text-red-600">
          {pageData.error || 'Failed to extract table from this page'}
        </div>
      )}
    </div>
  )
}

// =============================================================================
// MAIN APP COMPONENT
// =============================================================================
function App() {
  // File state
  const [selectedFile, setSelectedFile] = useState(null)
  const [filePreview, setFilePreview] = useState(null)
  const [fileType, setFileType] = useState(null) // 'image' or 'pdf'

  // PDF state
  const [pdfThumbnails, setPdfThumbnails] = useState([])
  const [selectedPages, setSelectedPages] = useState([])
  const [pdfLoading, setPdfLoading] = useState(false)

  // Processing state
  const [loading, setLoading] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState('')
  const [error, setError] = useState(null)

  // Results
  const [result, setResult] = useState(null)
  const [pdfResults, setPdfResults] = useState(null)
  const [activeTab, setActiveTab] = useState('table')
  const activeRequestRef = useRef(0)

  // Handle file selection
  const handleFileSelect = async (file) => {
    // Invalidate any in-flight extraction result to avoid stale UI updates.
    activeRequestRef.current += 1

    setSelectedFile(file)
    setResult(null)
    setPdfResults(null)
    setError(null)
    setPdfThumbnails([])
    setSelectedPages([])

    if (file.type === 'application/pdf') {
      setFileType('pdf')
      setFilePreview(null)
      setPdfLoading(true)

      // Get PDF info from backend
      try {
        const formData = new FormData()
        formData.append('pdf', file)

        const response = await axios.post(`${API_URL}/pdf/info`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 60000, // 1 minute for PDF info
        })

        if (response.data.success) {
          setPdfThumbnails(response.data.thumbnails)
          // Auto-select first page
          setSelectedPages([1])
        } else {
          setError(response.data.error || 'Failed to load PDF')
        }
      } catch (err) {
        setError('Failed to load PDF: ' + (err.response?.data?.error || err.message))
      } finally {
        setPdfLoading(false)
      }
    } else {
      setFileType('image')
      setFilePreview(URL.createObjectURL(file))
    }
  }

  // Toggle page selection
  const handlePageToggle = (pageNum) => {
    setSelectedPages(prev =>
      prev.includes(pageNum)
        ? prev.filter(p => p !== pageNum)
        : [...prev, pageNum].sort((a, b) => a - b)
    )
  }

  // Select all pages
  const handleSelectAll = () => {
    setSelectedPages(pdfThumbnails.map(t => t.page))
  }

  // Deselect all pages
  const handleDeselectAll = () => {
    setSelectedPages([])
  }

  // Extract from image
  const handleExtractImage = async () => {
    if (!selectedFile) {
      setError('Please select an image first')
      return
    }

    const requestToken = activeRequestRef.current + 1
    activeRequestRef.current = requestToken

    setLoading(true)
    setLoadingMessage('Extracting table from image...')
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('image', selectedFile)

      const response = await axios.post(`${API_URL}/extract`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 1200000, // 20 min for long OCR/VLM extraction
      })

      if (requestToken !== activeRequestRef.current) {
        return
      }

      if (response.data.success) {
        setResult({
          parsedJson: response.data.parsed_json,
          inferenceTime: response.data.inference_time_seconds,
          vramUsed: response.data.vram_used_gb,
          requestId: response.data.request_id,
          filename: response.data.filename,
        })
      } else {
        setError(response.data.error || 'Extraction failed')
      }
    } catch (err) {
      if (requestToken !== activeRequestRef.current) {
        return
      }
      handleAxiosError(err)
    } finally {
      if (requestToken === activeRequestRef.current) {
        setLoading(false)
      }
    }
  }

  // Extract from PDF pages
  const handleExtractPdf = async () => {
    if (selectedPages.length === 0) {
      setError('Please select at least one page')
      return
    }

    const requestToken = activeRequestRef.current + 1
    activeRequestRef.current = requestToken

    setLoading(true)
    setLoadingMessage(`Extracting tables from ${selectedPages.length} page(s)...`)
    setError(null)
    setPdfResults(null)

    try {
      const formData = new FormData()
      formData.append('pdf', selectedFile)
      formData.append('pages', JSON.stringify(selectedPages))

      const response = await axios.post(`${API_URL}/pdf/extract`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 1200000, // 20 min for PDF extraction
      })

      if (requestToken !== activeRequestRef.current) {
        return
      }

      if (response.data.success) {
        setPdfResults({
          pages: response.data.pages,
          totalTime: response.data.total_time_seconds,
          vramUsed: response.data.vram_used_gb,
          pagesProcessed: response.data.pages_processed,
        })
      } else {
        setError(response.data.error || 'Extraction failed')
      }
    } catch (err) {
      if (requestToken !== activeRequestRef.current) {
        return
      }
      handleAxiosError(err)
    } finally {
      if (requestToken === activeRequestRef.current) {
        setLoading(false)
      }
    }
  }

  // Handle axios errors
  const handleAxiosError = (err) => {
    console.error('Error:', err)
    if (err.code === 'ECONNABORTED') {
      setError('Request timed out. The model may still be processing.')
    } else if (err.response) {
      setError(err.response.data?.error || `Server error: ${err.response.status}`)
    } else if (err.request) {
      setError('Cannot connect to server. Make sure Flask backend is running on port 5000.')
    } else {
      setError(err.message)
    }
  }

  // Reset everything
  const handleReset = () => {
    activeRequestRef.current += 1
    setSelectedFile(null)
    setFilePreview(null)
    setFileType(null)
    setPdfThumbnails([])
    setSelectedPages([])
    setResult(null)
    setPdfResults(null)
    setError(null)
  }

  const hasResults = result || pdfResults

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-blue-600 text-white p-2 rounded-lg">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-800">OCR Table Extractor</h1>
                <p className="text-xs text-gray-500">Powered by Qwen3-VL-8B | Images & PDF</p>
              </div>
            </div>
            {hasResults && (
              <button
                onClick={handleReset}
                className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
              >
                New Extraction
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Error Alert */}
        {error && <ErrorAlert message={error} onClose={() => setError(null)} />}

        {/* Upload Section */}
        {!hasResults && !loading && (
          <div className="max-w-3xl mx-auto">
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-2 text-center">
                Extract Tables from Images or PDF
              </h2>
              <p className="text-gray-500 text-center mb-8">
                Upload a financial statement to extract structured table data
              </p>

              <FileUpload
                onFileSelect={handleFileSelect}
                selectedFile={selectedFile}
                filePreview={filePreview}
                fileType={fileType}
              />

              {/* PDF Loading */}
              {pdfLoading && (
                <div className="mt-6 text-center">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-2 border-blue-500 border-t-transparent"></div>
                  <p className="mt-2 text-gray-500">Loading PDF pages...</p>
                </div>
              )}

              {/* PDF Page Selector */}
              {pdfThumbnails.length > 0 && (
                <div className="mt-6">
                  <PdfPageSelector
                    thumbnails={pdfThumbnails}
                    selectedPages={selectedPages}
                    onPageToggle={handlePageToggle}
                    onSelectAll={handleSelectAll}
                    onDeselectAll={handleDeselectAll}
                  />
                </div>
              )}

              {/* Extract Button - Image */}
              {fileType === 'image' && selectedFile && (
                <button
                  onClick={handleExtractImage}
                  className="mt-6 w-full bg-blue-600 text-white py-3 px-6 rounded-xl font-semibold hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  Extract Table
                </button>
              )}

              {/* Extract Button - PDF */}
              {fileType === 'pdf' && selectedPages.length > 0 && (
                <button
                  onClick={handleExtractPdf}
                  className="mt-6 w-full bg-blue-600 text-white py-3 px-6 rounded-xl font-semibold hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  Extract from {selectedPages.length} Page{selectedPages.length > 1 ? 's' : ''}
                </button>
              )}
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <LoadingSpinner
                message={loadingMessage}
                subMessage={fileType === 'pdf' ? `Processing ${selectedPages.length} page(s) sequentially` : 'This may take 2-10 minutes for large tables'}
              />
            </div>
          </div>
        )}

        {/* Image Results */}
        {result && (
          <div className="space-y-6">
            <StatsBar
              stats={{
                inferenceTime: result.inferenceTime,
                vramUsed: result.vramUsed,
                rowCount: result.parsedJson?.rows?.length || 0,
                columnCount: result.parsedJson?.columns?.length || 0,
              }}
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="bg-white rounded-xl shadow-lg p-4">
                <h3 className="font-semibold text-gray-800 mb-3">Source Image</h3>
                <img src={filePreview} alt="Source" className="w-full rounded-lg border" />
              </div>

              <div className="lg:col-span-2 bg-white rounded-xl shadow-lg p-4">
                <div className="flex gap-2 mb-4 border-b">
                  <button
                    onClick={() => setActiveTab('table')}
                    className={`px-4 py-2 font-medium transition-colors ${
                      activeTab === 'table'
                        ? 'text-blue-600 border-b-2 border-blue-600'
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    Table View
                  </button>
                  <button
                    onClick={() => setActiveTab('json')}
                    className={`px-4 py-2 font-medium transition-colors ${
                      activeTab === 'json'
                        ? 'text-blue-600 border-b-2 border-blue-600'
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    JSON View
                  </button>
                </div>

                {activeTab === 'table' ? (
                  <TableView data={result.parsedJson} />
                ) : (
                  <JsonView data={result.parsedJson} />
                )}
              </div>
            </div>
          </div>
        )}

        {/* PDF Results */}
        {pdfResults && (
          <div className="space-y-6">
            <StatsBar
              stats={{
                inferenceTime: pdfResults.totalTime,
                vramUsed: pdfResults.vramUsed,
                pagesProcessed: pdfResults.pagesProcessed,
              }}
            />

            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="font-semibold text-gray-800 mb-4">
                Extraction Results ({pdfResults.pages.length} pages)
              </h3>

              <div className="space-y-6">
                {pdfResults.pages.map((pageData) => (
                  <PageResult
                    key={pageData.page}
                    pageData={pageData}
                    pageNum={pageData.page}
                  />
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-auto py-4 text-center text-sm text-gray-400">
        OCR Table Extractor - Using Qwen3-VL-8B with 4-bit quantization
      </footer>
    </div>
  )
}

export default App

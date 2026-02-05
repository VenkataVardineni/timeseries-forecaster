import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import './DataUpload.css'

function DataUpload() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState(null)
  const [files, setFiles] = useState([])
  const navigate = useNavigate()

  useEffect(() => {
    loadFiles()
  }, [])

  const loadFiles = async () => {
    try {
      const response = await api.listFiles()
      setFiles(response.data.files || [])
    } catch (error) {
      console.error('Error loading files:', error)
    }
  }

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
    setMessage(null)
  }

  const handleUpload = async () => {
    if (!file) {
      setMessage({ type: 'error', text: 'Please select a file' })
      return
    }

    setUploading(true)
    setMessage(null)

    try {
      const response = await api.uploadFile(file)
      setMessage({ type: 'success', text: response.data.message })
      setFile(null)
      loadFiles()
      // Reset file input
      document.getElementById('file-input').value = ''
    } catch (error) {
      setMessage({
        type: 'error',
        text: error.response?.data?.error || 'Upload failed',
      })
    } finally {
      setUploading(false)
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
  }

  return (
    <div className="card">
      <h2>📤 Upload Time Series Data</h2>

      {message && (
        <div className={message.type === 'error' ? 'error' : 'success'}>
          {message.text}
        </div>
      )}

      <div className="upload-section">
        <div className="file-input-wrapper">
          <input
            id="file-input"
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="file-input"
          />
          <label htmlFor="file-input" className="file-label">
            {file ? file.name : 'Choose CSV file'}
          </label>
        </div>

        {file && (
          <div className="file-info">
            <p>
              <strong>Selected:</strong> {file.name} ({formatFileSize(file.size)})
            </p>
          </div>
        )}

        <button
          className="btn"
          onClick={handleUpload}
          disabled={!file || uploading}
        >
          {uploading ? 'Uploading...' : 'Upload File'}
        </button>
      </div>

      {files.length > 0 && (
        <div className="files-list">
          <h3>Uploaded Files</h3>
          <table>
            <thead>
              <tr>
                <th>Filename</th>
                <th>Size</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {files.map((f) => (
                <tr key={f.filename}>
                  <td>{f.filename}</td>
                  <td>{formatFileSize(f.size)}</td>
                  <td>
                    <button
                      className="btn btn-secondary"
                      onClick={() => navigate('/train', { state: { csvPath: f.path } })}
                    >
                      Use for Training
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="info-box">
        <h4>📋 File Format Requirements</h4>
        <ul>
          <li>CSV format with header row</li>
          <li>Must include a <code>timestamp</code> column (parsable dates)</li>
          <li>Must include a target column (default: <code>y</code>)</li>
          <li>Optional: Additional feature columns</li>
        </ul>
      </div>
    </div>
  )
}

export default DataUpload


import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { api } from '../api'
import './ModelConfig.css'

function ModelConfig() {
  const location = useLocation()
  const navigate = useNavigate()
  const [files, setFiles] = useState([])
  const [config, setConfig] = useState({
    csv_path: location.state?.csvPath || '',
    model_type: 'arima',
    timestamp_col: 'timestamp',
    target_col: 'y',
    freq: 'D',
    context_length: 60,
    horizon: 30,
    n_folds: 5,
    // ARIMA specific
    order: [2, 1, 2],
    seasonal_order: [0, 0, 0, 0],
    // Seq2Seq specific
    hidden_size: 64,
    num_layers: 2,
    dropout: 0.1,
    batch_size: 64,
    max_epochs: 50,
    learning_rate: 0.001,
  })
  const [training, setTraining] = useState(false)
  const [message, setMessage] = useState(null)

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

  const handleSubmit = async (e) => {
    e.preventDefault()
    setTraining(true)
    setMessage(null)

    try {
      const trainingConfig = {
        model_type: config.model_type,
        csv_path: config.csv_path,
        config: {
          timestamp_col: config.timestamp_col,
          target_col: config.target_col,
          freq: config.freq,
          context_length: parseInt(config.context_length),
          horizon: parseInt(config.horizon),
          n_folds: parseInt(config.n_folds),
        },
      }

      if (config.model_type === 'arima') {
        trainingConfig.config.order = config.order
        trainingConfig.config.seasonal_order = config.seasonal_order
      } else {
        trainingConfig.config.hidden_size = parseInt(config.hidden_size)
        trainingConfig.config.num_layers = parseInt(config.num_layers)
        trainingConfig.config.dropout = parseFloat(config.dropout)
        trainingConfig.config.batch_size = parseInt(config.batch_size)
        trainingConfig.config.max_epochs = parseInt(config.max_epochs)
        trainingConfig.config.learning_rate = parseFloat(config.learning_rate)
      }

      const response = await api.startTraining(trainingConfig)
      navigate(`/training/${response.data.job_id}`)
    } catch (error) {
      setMessage({
        type: 'error',
        text: error.response?.data?.error || 'Failed to start training',
      })
      setTraining(false)
    }
  }

  return (
    <div className="card">
      <h2>⚙️ Configure Model Training</h2>

      {message && (
        <div className={message.type === 'error' ? 'error' : 'success'}>
          {message.text}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>CSV File</label>
          <select
            value={config.csv_path}
            onChange={(e) => setConfig({ ...config, csv_path: e.target.value })}
            required
          >
            <option value="">Select a file...</option>
            {files.map((f) => (
              <option key={f.filename} value={f.path}>
                {f.filename}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Model Type</label>
          <select
            value={config.model_type}
            onChange={(e) => setConfig({ ...config, model_type: e.target.value })}
          >
            <option value="arima">ARIMA</option>
            <option value="seq2seq_attention">Seq2Seq LSTM with Attention</option>
          </select>
        </div>

        <div className="config-grid">
          <div className="form-group">
            <label>Timestamp Column</label>
            <input
              type="text"
              value={config.timestamp_col}
              onChange={(e) => setConfig({ ...config, timestamp_col: e.target.value })}
              required
            />
          </div>

          <div className="form-group">
            <label>Target Column</label>
            <input
              type="text"
              value={config.target_col}
              onChange={(e) => setConfig({ ...config, target_col: e.target.value })}
              required
            />
          </div>

          <div className="form-group">
            <label>Frequency</label>
            <select
              value={config.freq}
              onChange={(e) => setConfig({ ...config, freq: e.target.value })}
            >
              <option value="D">Daily</option>
              <option value="H">Hourly</option>
              <option value="W">Weekly</option>
              <option value="M">Monthly</option>
            </select>
          </div>

          <div className="form-group">
            <label>Context Length</label>
            <input
              type="number"
              value={config.context_length}
              onChange={(e) => setConfig({ ...config, context_length: e.target.value })}
              min="1"
              required
            />
          </div>

          <div className="form-group">
            <label>Forecast Horizon</label>
            <input
              type="number"
              value={config.horizon}
              onChange={(e) => setConfig({ ...config, horizon: e.target.value })}
              min="1"
              required
            />
          </div>

          <div className="form-group">
            <label>Walk-Forward Folds</label>
            <input
              type="number"
              value={config.n_folds}
              onChange={(e) => setConfig({ ...config, n_folds: e.target.value })}
              min="1"
              max="10"
              required
            />
          </div>
        </div>

        {config.model_type === 'arima' && (
          <div className="model-specific-config">
            <h3>ARIMA Parameters</h3>
            <div className="config-grid">
              <div className="form-group">
                <label>Order (p, d, q)</label>
                <input
                  type="text"
                  value={config.order.join(', ')}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      order: e.target.value.split(',').map((x) => parseInt(x.trim())),
                    })
                  }
                  placeholder="2, 1, 2"
                />
              </div>
              <div className="form-group">
                <label>Seasonal Order (P, D, Q, s)</label>
                <input
                  type="text"
                  value={config.seasonal_order.join(', ')}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      seasonal_order: e.target.value.split(',').map((x) => parseInt(x.trim())),
                    })
                  }
                  placeholder="0, 0, 0, 0"
                />
              </div>
            </div>
          </div>
        )}

        {config.model_type === 'seq2seq_attention' && (
          <div className="model-specific-config">
            <h3>Seq2Seq Parameters</h3>
            <div className="config-grid">
              <div className="form-group">
                <label>Hidden Size</label>
                <input
                  type="number"
                  value={config.hidden_size}
                  onChange={(e) => setConfig({ ...config, hidden_size: e.target.value })}
                  min="16"
                  max="512"
                />
              </div>
              <div className="form-group">
                <label>Number of Layers</label>
                <input
                  type="number"
                  value={config.num_layers}
                  onChange={(e) => setConfig({ ...config, num_layers: e.target.value })}
                  min="1"
                  max="5"
                />
              </div>
              <div className="form-group">
                <label>Dropout</label>
                <input
                  type="number"
                  value={config.dropout}
                  onChange={(e) => setConfig({ ...config, dropout: e.target.value })}
                  min="0"
                  max="0.5"
                  step="0.1"
                />
              </div>
              <div className="form-group">
                <label>Batch Size</label>
                <input
                  type="number"
                  value={config.batch_size}
                  onChange={(e) => setConfig({ ...config, batch_size: e.target.value })}
                  min="8"
                  max="256"
                />
              </div>
              <div className="form-group">
                <label>Max Epochs</label>
                <input
                  type="number"
                  value={config.max_epochs}
                  onChange={(e) => setConfig({ ...config, max_epochs: e.target.value })}
                  min="1"
                  max="200"
                />
              </div>
              <div className="form-group">
                <label>Learning Rate</label>
                <input
                  type="number"
                  value={config.learning_rate}
                  onChange={(e) => setConfig({ ...config, learning_rate: e.target.value })}
                  min="0.0001"
                  max="0.1"
                  step="0.0001"
                />
              </div>
            </div>
          </div>
        )}

        <button type="submit" className="btn" disabled={training || !config.csv_path}>
          {training ? 'Starting Training...' : 'Start Training'}
        </button>
      </form>
    </div>
  )
}

export default ModelConfig


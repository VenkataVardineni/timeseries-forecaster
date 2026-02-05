import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import './TrainingStatus.css'

function TrainingStatus() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [polling, setPolling] = useState(true)

  const { data: job, isLoading } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => api.getJobStatus(jobId).then((res) => res.data),
    refetchInterval: polling ? 2000 : false,
    enabled: !!jobId,
  })

  useEffect(() => {
    if (job?.status === 'completed' && job?.run_id) {
      setPolling(false)
      setTimeout(() => {
        navigate(`/results/${job.run_id}`)
      }, 2000)
    } else if (job?.status === 'failed') {
      setPolling(false)
    }
  }, [job, navigate])

  if (isLoading) {
    return (
      <div className="card">
        <div className="loading">Loading job status...</div>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="card">
        <div className="error">Job not found</div>
      </div>
    )
  }

  return (
    <div className="card">
      <h2>🔄 Training Status</h2>

      <div className="status-section">
        <div className="status-header">
          <span className={`status-badge status-${job.status}`}>
            {job.status.toUpperCase()}
          </span>
          <span className="job-id">Job ID: {jobId}</span>
        </div>

        {job.status === 'running' && (
          <div className="progress-section">
            <div className="spinner"></div>
            <p>Training in progress... This may take several minutes.</p>
          </div>
        )}

        {job.status === 'completed' && (
          <div className="success-section">
            <p className="success-message">✅ Training completed successfully!</p>
            {job.run_id && (
              <p>
                Redirecting to results...
                <br />
                <button
                  className="btn"
                  onClick={() => navigate(`/results/${job.run_id}`)}
                >
                  View Results Now
                </button>
              </p>
            )}
          </div>
        )}

        {job.status === 'failed' && (
          <div className="error-section">
            <p className="error-message">❌ Training failed</p>
            {job.error && (
              <pre className="error-details">{job.error}</pre>
            )}
            <button className="btn" onClick={() => navigate('/train')}>
              Try Again
            </button>
          </div>
        )}

        {job.config && (
          <div className="config-display">
            <h3>Configuration</h3>
            <pre>{JSON.stringify(job.config, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  )
}

export default TrainingStatus


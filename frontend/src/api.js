import axios from 'axios'

const API_BASE = '/api'

export const api = {
  // File operations
  uploadFile: (file) => {
    const formData = new FormData()
    formData.append('file', file)
    return axios.post(`${API_BASE}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },

  listFiles: () => axios.get(`${API_BASE}/files`),

  // Training operations
  startTraining: (config) => axios.post(`${API_BASE}/train`, config),

  getJobStatus: (jobId) => axios.get(`${API_BASE}/jobs/${jobId}`),

  listJobs: () => axios.get(`${API_BASE}/jobs`),

  // Results operations
  listResults: () => axios.get(`${API_BASE}/results`),

  getRunInfo: (runId) => axios.get(`${API_BASE}/results/${runId}/info`),

  getMetrics: (runId) => axios.get(`${API_BASE}/results/${runId}/metrics`),

  getPredictions: (runId) => axios.get(`${API_BASE}/results/${runId}/predictions`),

  getPlotUrl: (runId, plotName) => `${API_BASE}/results/${runId}/plots/${plotName}`,
}


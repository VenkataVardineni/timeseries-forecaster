import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './ResultsView.css'

function ResultsView() {
  const { runId } = useParams()
  const [selectedRun, setSelectedRun] = useState(runId || null)

  const { data: runs } = useQuery({
    queryKey: ['results'],
    queryFn: () => api.listResults().then((res) => res.data.runs),
  })

  const { data: runInfo } = useQuery({
    queryKey: ['runInfo', selectedRun],
    queryFn: () => api.getRunInfo(selectedRun).then((res) => res.data),
    enabled: !!selectedRun,
  })

  const { data: metrics } = useQuery({
    queryKey: ['metrics', selectedRun],
    queryFn: () => api.getMetrics(selectedRun).then((res) => res.data.metrics),
    enabled: !!selectedRun,
  })

  useEffect(() => {
    if (runId && !selectedRun) {
      setSelectedRun(runId)
    } else if (!runId && runs && runs.length > 0 && !selectedRun) {
      setSelectedRun(runs[0].run_id)
    }
  }, [runId, runs, selectedRun])

  if (!selectedRun) {
    return (
      <div className="card">
        <div className="loading">No runs available. Start training to see results.</div>
      </div>
    )
  }

  return (
    <div>
      {runs && runs.length > 0 && (
        <div className="card">
          <h2>📊 Results</h2>
          <div className="runs-selector">
            <label>Select Run:</label>
            <select
              value={selectedRun}
              onChange={(e) => setSelectedRun(e.target.value)}
            >
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {runInfo && (
        <>
          {metrics && metrics.length > 0 && (
            <div className="card">
              <h2>📈 Metrics</h2>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="horizon_step" label={{ value: 'Horizon Step', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Error', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="mae" stroke="#8884d8" name="MAE" />
                  <Line type="monotone" dataKey="rmse" stroke="#82ca9d" name="RMSE" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {runInfo.plots && runInfo.plots.length > 0 && (
            <div className="card">
              <h2>🖼️ Plots</h2>
              <div className="plots-grid">
                {runInfo.plots.map((plotName) => (
                  <div key={plotName} className="plot-item">
                    <h3>{plotName.replace('.png', '').replace(/_/g, ' ')}</h3>
                    <img
                      src={api.getPlotUrl(selectedRun, plotName)}
                      alt={plotName}
                      className="plot-image"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {runInfo.config && (
            <div className="card">
              <h2>⚙️ Configuration</h2>
              <pre className="config-display">{JSON.stringify(runInfo.config, null, 2)}</pre>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default ResultsView


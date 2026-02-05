import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import DataUpload from './components/DataUpload'
import ModelConfig from './components/ModelConfig'
import TrainingStatus from './components/TrainingStatus'
import ResultsView from './components/ResultsView'
import './App.css'

function App() {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="nav-container">
            <h1 className="nav-title">📊 TimeSeries Forecaster</h1>
            <div className="nav-links">
              <Link to="/">Upload</Link>
              <Link to="/train">Train</Link>
              <Link to="/results">Results</Link>
            </div>
          </div>
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<DataUpload />} />
            <Route path="/train" element={<ModelConfig />} />
            <Route path="/training/:jobId" element={<TrainingStatus />} />
            <Route path="/results" element={<ResultsView />} />
            <Route path="/results/:runId" element={<ResultsView />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App


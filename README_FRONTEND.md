# Frontend Setup Guide

This project includes a React-based web interface for interacting with the TimeSeries Forecaster toolkit.

## Prerequisites

- Node.js 18+ and npm
- Python 3.10+ (for backend API)
- All Python dependencies installed (see main README)

## Quick Start

### 1. Install Backend Dependencies

```bash
pip install -r api/requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Start the Backend API

In one terminal:

```bash
python api/app.py
```

The API will run on `http://127.0.0.1:5001` (port 5000 is blocked by macOS AirPlay)

### 4. Start the Frontend

In another terminal:

```bash
cd frontend
npm run dev
```

The frontend will run on `http://localhost:3000`

## Usage

1. **Upload Data**: Go to the Upload page and upload your CSV file
2. **Configure Training**: Go to the Train page, select your file and configure model parameters
3. **Monitor Training**: Watch training progress in real-time
4. **View Results**: See metrics, plots, and predictions in the Results page

## Features

- 📤 **Data Upload**: Upload CSV files with drag-and-drop interface
- ⚙️ **Model Configuration**: Configure ARIMA or Seq2Seq models with a user-friendly form
- 🔄 **Real-time Training Status**: Monitor training progress with live updates
- 📊 **Interactive Results**: View metrics charts and forecast plots
- 🖼️ **Visualizations**: See forecast plots, horizon error plots, and residuals

## API Endpoints

The backend API provides the following endpoints:

- `GET /api/health` - Health check
- `POST /api/upload` - Upload CSV file
- `GET /api/files` - List uploaded files
- `POST /api/train` - Start training job
- `GET /api/jobs/<job_id>` - Get job status
- `GET /api/results` - List all runs
- `GET /api/results/<run_id>/info` - Get run information
- `GET /api/results/<run_id>/metrics` - Get metrics
- `GET /api/results/<run_id>/plots/<plot_name>` - Get plot image

## Development

### Backend

The Flask API server (`api/app.py`) handles:
- File uploads
- Training job management
- Results retrieval
- Plot serving

### Frontend

The React app (`frontend/`) uses:
- **Vite** for fast development
- **React Router** for navigation
- **React Query** for data fetching
- **Recharts** for data visualization
- **Axios** for API calls

## Production Build

To build for production:

```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/`. You can serve them with any static file server or integrate with the Flask backend.


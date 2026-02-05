# Setup Instructions

Complete guide to set up and run the TimeSeries Forecaster application.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** (check with `python --version`)
- **Node.js 18+** and npm (check with `node --version` and `npm --version`)
- **Git** (for cloning the repository)

## Step 1: Clone the Repository

```bash
git clone https://github.com/VenkataVardineni/timeseries-forecaster.git
cd timeseries-forecaster
```

## Step 2: Python Environment Setup

### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Option B: Using conda

```bash
conda create -n timeseries-forecaster python=3.10
conda activate timeseries-forecaster
```

## Step 3: Install Python Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install API dependencies
pip install -r api/requirements.txt
```

## Step 4: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

## Step 5: Generate Sample Data (Optional)

If you want to test with sample data:

```bash
python scripts/generate_sample_data.py
```

This creates `data/raw/example_daily.csv` with synthetic time series data.

## Step 6: Start the Application

### Option A: Quick Start Script

```bash
./start_web.sh
```

This script starts both backend and frontend automatically.

### Option B: Manual Start (Two Terminals)

**Terminal 1 - Backend API:**
```bash
python api/app.py
```

The API will start on `http://127.0.0.1:5001`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:3000`

## Step 7: Access the Application

Open your web browser and navigate to:

```
http://localhost:3000
```

## Verification

To verify everything is working:

1. **Backend Health Check:**
   ```bash
   curl http://127.0.0.1:5001/api/health
   ```
   Should return: `{"status": "ok", "message": "TimeSeries Forecaster API"}`

2. **Frontend:**
   - Open `http://localhost:3000` in your browser
   - You should see the TimeSeries Forecaster interface

## Directory Structure

After setup, your directory structure should look like:

```
timeseries-forecaster/
├── api/                    # Flask API backend
│   ├── app.py             # Main API server
│   └── requirements.txt   # API dependencies
├── frontend/               # React frontend
│   ├── src/               # React source code
│   ├── package.json       # Frontend dependencies
│   └── vite.config.js     # Vite configuration
├── src/                    # Core Python modules
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Forecasting models
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation and metrics
│   └── utils/             # Utility functions
├── configs/                # Configuration files
├── data/                   # Data directories
│   ├── raw/               # Input CSV files
│   └── processed/         # Processed features
├── reports/                # Generated reports
├── scripts/                # Utility scripts
├── requirements.txt        # Main Python dependencies
└── SETUP.md               # This file
```

## Port Configuration

- **Backend API:** Port 5001 (port 5000 is blocked by macOS AirPlay Receiver)
- **Frontend:** Port 3000

If you need to change ports:

1. **Backend:** Edit `api/app.py` - change port in the last line
2. **Frontend:** Edit `frontend/vite.config.js` - update proxy target and server port

## Troubleshooting

### Python Dependencies Issues

If you encounter dependency conflicts:

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Node Modules Issues

If frontend dependencies fail:

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Port Already in Use

If port 5001 or 3000 is already in use:

1. Find the process: `lsof -i :5001` or `lsof -i :3000`
2. Kill the process: `kill -9 <PID>`
3. Or change ports in configuration files

### Permission Issues

On macOS/Linux, if scripts aren't executable:

```bash
chmod +x start_web.sh
chmod +x scripts/quickstart.sh
```

## Production Deployment

For production deployment:

### Backend

1. Use a production WSGI server (e.g., Gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 127.0.0.1:5001 api.app:app
   ```

2. Set environment variables:
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=False
   ```

### Frontend

1. Build for production:
   ```bash
   cd frontend
   npm run build
   ```

2. Serve the `dist/` folder with a web server (nginx, Apache, etc.)

## Docker Deployment (Optional)

Build and run with Docker:

```bash
# Build image
docker build -t timeseries-forecaster .

# Run container
docker run -p 5001:5001 -v $(pwd)/data:/app/data -v $(pwd)/reports:/app/reports timeseries-forecaster
```

## Next Steps

After successful setup:

1. Read the main [README.md](README.md) for application overview
2. Check [README_FRONTEND.md](README_FRONTEND.md) for frontend details
3. Upload your CSV data through the web interface
4. Configure and train models
5. View results and forecasts

## Support

For issues or questions:
- Check the main README.md for feature documentation
- Review code comments in source files
- Check GitHub issues (if repository is public)


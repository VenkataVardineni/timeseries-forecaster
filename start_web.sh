#!/bin/bash
# Start script for web interface

set -e

echo "=== Starting TimeSeries Forecaster Web Interface ==="
echo ""

# Check if backend dependencies are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing backend dependencies..."
    pip install -q -r api/requirements.txt
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

echo ""
echo "Starting backend API on http://127.0.0.1:5001"
echo "Starting frontend on http://localhost:3000"
echo ""
echo "Note: Port 5000 is blocked by macOS AirPlay, using 5001 instead"
echo "Press Ctrl+C to stop both services"
echo ""

# Start backend in background
python api/app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend
cd frontend
npm run dev &
FRONTEND_PID=$!

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

wait


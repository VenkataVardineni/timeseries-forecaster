#!/bin/bash
# Quickstart script for TimeSeries Forecaster

set -e

echo "=== TimeSeries Forecaster Quickstart ==="
echo ""

# Generate sample data if it doesn't exist
if [ ! -f "data/raw/example_daily.csv" ]; then
    echo "Generating sample dataset..."
    python scripts/generate_sample_data.py
    echo ""
fi

# Install dependencies if needed
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
    echo ""
fi

echo "Running ARIMA baseline..."
python -m src.training.train --config configs/exp_arima.yaml

echo ""
echo "=== Training complete! ==="
echo "Check reports/ directory for results."
echo ""
echo "To train Seq2Seq model, run:"
echo "  python -m src.training.train --config configs/exp_seq2seq_attention.yaml"


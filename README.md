# TimeSeries Forecaster

A production-style forecasting toolkit for **multi-horizon time-series forecasting** with uncertainty quantification, walk-forward validation, and comprehensive evaluation reports.

## Features

- **Multiple Models**: 
  - Classical **ARIMA** baseline with automatic order selection
  - Deep **Seq2Seq LSTM with Bahdanau attention** for complex patterns
- **Multi-Horizon Forecasting**: Predict multiple steps ahead (e.g., t+1, t+7, t+30) simultaneously
- **Probabilistic Forecasts**: Quantile predictions (p10/p50/p90) trained with pinball loss
- **Walk-Forward Validation**: Realistic evaluation with rolling-origin folds and leakage detection
- **Forecast Lab Reports**: Automated generation of metrics, plots, and diagnostic artifacts

## Problem Statement

Traditional time-series forecasting often suffers from:
- **Data leakage** from future information in training
- **Overly optimistic metrics** from single train/test splits
- **Lack of uncertainty quantification** (point forecasts only)
- **Limited horizon analysis** (only aggregate metrics)

This toolkit addresses these issues by:
1. Implementing strict **walk-forward validation** with leakage guards
2. Providing **per-horizon metrics** to understand degradation over time
3. Training models to output **quantile forecasts** with proper probabilistic loss functions
4. Generating comprehensive **Forecast Lab reports** for model comparison and selection

## What This Toolkit Outputs

Each training run generates a complete **Forecast Lab** report under `reports/<run_id>/`:

### Metrics
- **Per-horizon metrics** (`metrics_*.csv`): MAE and RMSE for each forecast step (t+1, t+2, ..., t+H)
- **Probabilistic metrics**: Pinball loss (p10/p50/p90) and interval coverage statistics
- **Summary metrics** (`metrics_summary.csv`): Aggregated performance across folds

### Predictions
- **Detailed predictions** (`predictions.csv`): All forecasts with quantiles for each sample, fold, and horizon step
- **Fold metadata** (`folds.json`): Train/test boundaries, leakage checks, and model diagnostics

### Visualizations
- **Forecast plots** (`forecast_plot_fold_*.png`): Actual vs predicted (p50) with shaded p10–p90 uncertainty bands
- **Horizon error plots** (`horizon_error_*.png`): Error degradation as forecast horizon increases
- **Residual plots** (`residuals_plot.png`): Diagnostic plots for model validation

### Model Artifacts
- **Checkpoints** (`checkpoints/`): Saved model weights for Seq2Seq models
- **Configuration** (`config.json`): Complete experiment configuration for reproducibility

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd timeseries-forecaster
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

Prepare your dataset as a CSV file in `data/raw/` with:
- A **timestamp column** (e.g., `timestamp`) that pandas can parse
- A **target column** (e.g., `y`) containing the time series values
- Optional: Additional feature columns (will be used as covariates)

Example CSV structure:
```csv
timestamp,y,feature1,feature2
2020-01-01,100.5,10,20
2020-01-02,102.3,11,21
...
```

Update the `data` section in your config files (`configs/exp_*.yaml`) to point to your CSV and specify column names.

## Usage

### Training Models

Train models with walk-forward evaluation:

**ARIMA Baseline:**
```bash
python -m src.training.train --config configs/exp_arima.yaml
```

**Seq2Seq LSTM with Attention:**
```bash
python -m src.training.train --config configs/exp_seq2seq_attention.yaml
```

These commands will:
1. Load and feature-engineer the dataset (calendar features, lags, rolling stats)
2. Build supervised windows (context + horizon)
3. Create walk-forward folds with leakage checks
4. Train/evaluate the model on each fold
5. Generate comprehensive Forecast Lab reports

### Standalone Walk-Forward Evaluation

You can also run walk-forward data preparation separately:

```bash
python -m src.evaluation.walk_forward --config configs/exp_arima.yaml
```

This creates fold boundaries and performs leakage checks without training models.

### Configuration

Edit `configs/exp_*.yaml` to customize:
- **Data paths** and column names
- **Feature engineering** (lags, rolling windows, calendar features)
- **Window sizes** (context_length, horizon)
- **Model hyperparameters** (ARIMA orders, LSTM hidden size, etc.)
- **Training settings** (batch size, learning rate, early stopping)
- **Walk-forward** (number of folds)

## Walk-Forward Validation: Methodology

**Walk-forward validation** (also called rolling-origin evaluation) is the gold standard for time-series model evaluation because it simulates real-world deployment:

### How It Works

1. **Multiple Folds**: Instead of a single train/test split, we create multiple folds where:
   - Fold 1: Train on data [0..T₁], test on [T₁+1..T₁+H]
   - Fold 2: Train on data [0..T₂], test on [T₂+1..T₂+H]
   - Fold N: Train on data [0..Tₙ], test on [Tₙ+1..Tₙ+H]

2. **Rolling Origin**: The training window expands with each fold, mimicking how models are retrained in production as new data arrives.

3. **Leakage Prevention**: Strict checks ensure no future timestamps appear in training windows, preventing data leakage that would inflate performance metrics.

### Why It Matters

- **Realistic Evaluation**: Metrics reflect how the model performs when deployed
- **Temporal Integrity**: Respects the temporal ordering of time-series data
- **Robust Estimates**: Multiple folds provide more reliable performance estimates
- **Production-Ready**: Mirrors the actual retraining and forecasting workflow

Fold boundaries and leakage validation results are stored in `reports/<run_id>/folds.json`.

## Project Structure

```
timeseries-forecaster/
├── configs/                 # Experiment configurations
│   ├── exp_arima.yaml
│   └── exp_seq2seq_attention.yaml
├── data/
│   ├── raw/                 # Input CSV files
│   └── processed/           # Feature-engineered data
├── src/
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Forecasting models
│   ├── training/            # Training loops and losses
│   ├── evaluation/          # Metrics, plots, walk-forward
│   └── utils/               # Configuration and I/O utilities
├── reports/                 # Generated Forecast Lab reports
│   └── <run_id>/           # Per-run outputs
├── notebooks/              # Optional Jupyter notebooks
├── Dockerfile              # Containerization
└── README.md
```

## Example Outputs

After training, inspect `reports/<run_id>/` for:

- **`forecast_plot_fold_*.png`**: Visual comparison of actual vs predicted with uncertainty bands
- **`horizon_error_*.png`**: Error degradation analysis across forecast horizons
- **`metrics_*.csv`**: Detailed per-horizon performance metrics
- **`predictions.csv`**: All forecasts with quantiles for further analysis

These artifacts can be used in portfolios, presentations, or model comparison studies.

## Docker Usage

### Build Image

```bash
docker build -t timeseries-forecaster .
```

### Run Training

Mount your local data and reports directories:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  timeseries-forecaster \
  python -m src.training.train --config configs/exp_seq2seq_attention.yaml
```

This containerization enables:
- **Reproducible environments** across different machines
- **Easy deployment** to cloud platforms or orchestration systems
- **Isolation** from host Python environments

## Model Selection Guide

Use the Forecast Lab reports to select the best model:

1. **Compare per-horizon metrics**: Which model has lower MAE/RMSE at your target horizon?
2. **Check coverage**: Do p10–p90 intervals contain ~80% of actual values?
3. **Examine horizon degradation**: How quickly does error increase with forecast distance?
4. **Review residuals**: Are there systematic patterns indicating model misspecification?

## Contributing

This is a portfolio project demonstrating production-style ML engineering practices:
- Modular, maintainable code structure
- Comprehensive evaluation methodology
- Reproducible experiments via configs
- Containerized deployment

## License

[Specify your license here]

## Acknowledgments

Built with:
- PyTorch for deep learning
- statsmodels for ARIMA
- pandas/numpy for data processing
- matplotlib/seaborn for visualization



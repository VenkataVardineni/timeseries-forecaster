## TimeSeries Forecaster

TimeSeries Forecaster is a small production-style toolkit for **multi-horizon time-series forecasting** with:

- **Multiple models**: classical ARIMA and a deep **Seq2Seq LSTM with attention**
- **Multi-horizon forecasts** (e.g., t+1, t+7, t+30)
- **Probabilistic forecasts** via quantile (pinball) loss (p10/p50/p90)
- **Walk-forward validation** with leakage checks
- A local **“Forecast Lab”** report under `reports/<run_id>/` with metrics and plots.

The initial reference setup assumes a **daily univariate target** (e.g., energy usage, weather, or prices) plus optional covariates loaded from CSV.

### What this toolkit outputs

- **Trained models** (ARIMA and/or Seq2Seq LSTM with attention)
- **Per-horizon metrics** (MAE/RMSE for t+1, t+2, … t+H)
- **Probabilistic metrics** (pinball loss and interval coverage between p10–p90)
- **Walk-forward fold artifacts**:
  - Train/test boundaries and leakage checks
  - Fold-wise predictions for each horizon and quantile
- **Plots**:
  - Actual vs p50 with shaded p10–p90 interval
  - Error vs horizon step
  - Basic residual plots for ARIMA

Everything for one run is saved under:

- `reports/<run_id>/metrics_*.csv`
- `reports/<run_id>/folds.json`
- `reports/<run_id>/plots/*.png`

### Quickstart

Create and activate a Python 3.10+ environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Prepare your dataset as a CSV in `data/raw/your_dataset.csv` with at least:

- A **timestamp column** (e.g., `timestamp`) parsable by pandas
- A **target column** (e.g., `y`)

Update the `data` section of the config files to point to your CSV and column names.

### Run training

Train models (ARIMA or Seq2Seq) and run walk-forward evaluation in one go using:

```bash
python -m src.training.train --config configs/exp_arima.yaml
```

or

```bash
python -m src.training.train --config configs/exp_seq2seq_attention.yaml
```

These commands will:

- Load and feature-engineer the dataset
- Build context/horizon windows
- Run walk-forward validation
- Train/evaluate the chosen model on each fold
- Write a **Forecast Lab** report under `reports/<run_id>/`.

### Standalone walk-forward evaluation

You can also explicitly run walk-forward evaluation (e.g. for ARIMA) via:

```bash
python -m src.evaluation.walk_forward --config configs/exp_arima.yaml
```

This assumes the config describes the data pipeline and model settings for ARIMA.

### Walk-forward validation: what and why

Instead of a single train/test split, **walk-forward validation** (also called rolling-origin evaluation) simulates how the model would be used in production:

- We define multiple **folds** where each fold trains on all data up to a cutoff time and tests on the **next horizon window**.
- Then the origin “walks forward”, expanding the training window and forecasting the next horizon again.
- This mimics repeatedly deploying the model on new data and ensures **no future information leaks** into training, which would otherwise give overly optimistic metrics.

Fold boundaries and leakage checks are stored in `reports/<run_id>/folds.json`.

### Reports and screenshots

After a run, you will find:

- `reports/<run_id>/forecast_plot.png` – sample actual vs p50 forecast with p10–p90 band
- `reports/<run_id>/metrics.csv` – per-horizon metrics summary

You can use these as **screenshots** in a portfolio or documentation.

### Docker usage

Build the image:

```bash
docker build -t timeseries-forecaster .
```

Run training inside Docker (mounting your local data and reports):

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  timeseries-forecaster \
  python -m src.training.train --config configs/exp_seq2seq_attention.yaml
```

This makes it easy to plug the project into a larger orchestration system later (e.g., a FlowForge-style “Run ML Node”).



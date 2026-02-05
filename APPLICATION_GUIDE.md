# TimeSeries Forecaster - Complete Application Guide

## Overview

TimeSeries Forecaster is a production-ready toolkit for multi-horizon time-series forecasting with uncertainty quantification. It combines classical statistical methods (ARIMA) with modern deep learning (Seq2Seq LSTM with attention) to provide accurate, probabilistic forecasts with comprehensive evaluation.

## Key Features

### 1. Multiple Forecasting Models

**ARIMA (AutoRegressive Integrated Moving Average)**
- Classical statistical forecasting model
- Handles trend and seasonality
- Configurable order parameters (p, d, q)
- Seasonal ARIMA support
- Fast training and inference

**Seq2Seq LSTM with Attention**
- Deep learning model for complex patterns
- Encoder-decoder architecture with Bahdanau attention
- Multi-layer LSTM support
- Teacher forcing during training
- Handles non-linear relationships

### 2. Multi-Horizon Forecasting

- Predict multiple future steps simultaneously (e.g., t+1, t+7, t+30)
- Per-horizon metrics to understand error degradation
- Efficient batch prediction for all horizons

### 3. Probabilistic Forecasting

- Quantile predictions (p10, p50, p90)
- Pinball loss for quantile training
- Uncertainty intervals (80% prediction intervals)
- Coverage metrics to validate interval quality

### 4. Walk-Forward Validation

- Rolling-origin evaluation methodology
- Multiple folds for robust performance estimates
- Automatic leakage detection and prevention
- Realistic production simulation

### 5. Comprehensive Reporting

- Per-horizon metrics (MAE, RMSE)
- Forecast plots with uncertainty bands
- Horizon error degradation analysis
- Residual diagnostics
- Model comparison capabilities

### 6. Web Interface

- Modern React-based UI
- Drag-and-drop file upload
- Interactive model configuration
- Real-time training monitoring
- Visual results exploration

## Application Architecture

### Backend (Flask API)

**Purpose:** Handles all ML operations and serves data to frontend

**Key Components:**
- File upload and management
- Training job orchestration
- Results retrieval and serving
- Plot image serving

**API Endpoints:**
- `GET /api/health` - Health check
- `POST /api/upload` - Upload CSV files
- `GET /api/files` - List uploaded files
- `POST /api/train` - Start training job
- `GET /api/jobs/<job_id>` - Get job status
- `GET /api/results` - List all runs
- `GET /api/results/<run_id>/info` - Get run information
- `GET /api/results/<run_id>/metrics` - Get metrics
- `GET /api/results/<run_id>/predictions` - Get predictions
- `GET /api/results/<run_id>/plots/<plot_name>` - Get plot images

### Frontend (React + Vite)

**Purpose:** User interface for interacting with the forecasting system

**Key Components:**
- Data upload interface
- Model configuration forms
- Training status monitoring
- Results visualization

**Technologies:**
- React 18 for UI components
- React Router for navigation
- React Query for data fetching
- Recharts for data visualization
- Axios for API communication

### Core ML Pipeline

**Data Processing:**
- CSV loading with timestamp parsing
- Missing value handling (forward-fill)
- Feature engineering (calendar, lags, rolling stats)
- Normalization (train-only statistics)
- Windowed dataset creation

**Model Training:**
- Walk-forward fold creation
- Model-specific training loops
- Early stopping and checkpointing
- Gradient clipping for deep models

**Evaluation:**
- Per-horizon metric computation
- Probabilistic metric calculation
- Plot generation
- Report artifact creation

## Workflow

### 1. Data Preparation

**Input Requirements:**
- CSV file with timestamp column
- Target column (time series values)
- Optional: Feature columns (covariates)

**Supported Formats:**
- Daily, hourly, weekly, monthly frequencies
- Any pandas-parsable timestamp format

**Data Processing:**
- Automatic timestamp parsing and sorting
- Missing timestamp interpolation
- Forward-fill for missing values
- Feature engineering pipeline

### 2. Model Configuration

**ARIMA Configuration:**
- Order parameters (p, d, q)
- Seasonal order (P, D, Q, s)
- Context length (training window)
- Forecast horizon

**Seq2Seq Configuration:**
- Hidden size (LSTM units)
- Number of layers
- Dropout rate
- Batch size
- Learning rate
- Maximum epochs
- Early stopping patience

**Common Settings:**
- Walk-forward folds count
- Feature engineering options
- Data frequency

### 3. Training Process

**Walk-Forward Validation:**
1. Create multiple folds with rolling origin
2. For each fold:
   - Train on historical data
   - Test on next horizon window
   - Ensure no data leakage
3. Aggregate metrics across folds

**Training Features:**
- Background job processing
- Real-time status updates
- Automatic checkpointing
- Error handling and reporting

### 4. Results Analysis

**Generated Artifacts:**
- Metrics CSV files (per-horizon and summary)
- Predictions CSV (all forecasts with quantiles)
- Forecast plots (actual vs predicted with intervals)
- Horizon error plots (degradation analysis)
- Residual plots (diagnostics)
- Configuration JSON (reproducibility)

**Metrics Provided:**
- MAE (Mean Absolute Error) per horizon
- RMSE (Root Mean Squared Error) per horizon
- Pinball loss (p10, p50, p90)
- Interval coverage (p10-p90)

## Use Cases

### 1. Energy Demand Forecasting
- Predict daily/hourly energy consumption
- Plan capacity and resource allocation
- Uncertainty quantification for risk management

### 2. Sales Forecasting
- Multi-horizon sales predictions
- Inventory planning
- Revenue projections with confidence intervals

### 3. Financial Time Series
- Stock price forecasting
- Volatility prediction
- Risk assessment with quantiles

### 4. Weather Forecasting
- Temperature/precipitation predictions
- Multi-day forecasts
- Uncertainty in weather patterns

### 5. Resource Planning
- Capacity planning
- Demand forecasting
- Supply chain optimization

## Model Selection Guide

### When to Use ARIMA

- **Small datasets** (< 1000 points)
- **Linear trends** and seasonality
- **Fast inference** requirements
- **Interpretable** forecasts needed
- **Limited computational** resources

### When to Use Seq2Seq LSTM

- **Large datasets** (> 1000 points)
- **Complex non-linear** patterns
- **Multiple covariates** available
- **Long-term dependencies**
- **Computational resources** available

## Best Practices

### Data Preparation

1. **Ensure data quality:**
   - Check for outliers
   - Handle missing values appropriately
   - Verify timestamp consistency

2. **Feature engineering:**
   - Include relevant calendar features
   - Add domain-specific lags
   - Consider rolling statistics

3. **Train/test split:**
   - Use walk-forward validation (automatic)
   - Reserve sufficient data for testing
   - Maintain temporal ordering

### Model Configuration

1. **Start with defaults:**
   - Use provided default configurations
   - Adjust based on results

2. **Hyperparameter tuning:**
   - Adjust context length based on data frequency
   - Set horizon based on use case
   - Balance model complexity vs. overfitting

3. **Validation:**
   - Use multiple folds for robust estimates
   - Check for data leakage
   - Monitor training convergence

### Evaluation

1. **Compare models:**
   - Use per-horizon metrics
   - Consider probabilistic metrics
   - Check coverage rates

2. **Visual inspection:**
   - Review forecast plots
   - Check residual patterns
   - Analyze horizon degradation

3. **Production deployment:**
   - Select best model per horizon
   - Monitor performance over time
   - Retrain periodically

## Output Interpretation

### Metrics

**MAE (Mean Absolute Error):**
- Average absolute difference between predicted and actual
- Lower is better
- Same units as target variable

**RMSE (Root Mean Squared Error):**
- Penalizes larger errors more
- Lower is better
- Same units as target variable

**Pinball Loss:**
- Measures quantile prediction quality
- Lower is better
- Different for each quantile

**Coverage:**
- Percentage of actual values within prediction interval
- Should be close to intended coverage (e.g., 80% for p10-p90)
- Too high: intervals too wide (conservative)
- Too low: intervals too narrow (overconfident)

### Plots

**Forecast Plots:**
- Show actual vs predicted (p50)
- Shaded area represents uncertainty (p10-p90)
- Good forecasts: actual values within shaded area
- Check for systematic biases

**Horizon Error Plots:**
- Show how error increases with forecast distance
- Expected: gradual increase
- Warning: sudden jumps or erratic behavior

**Residual Plots:**
- Should show random scatter around zero
- Patterns indicate model misspecification
- Check for trends, seasonality, or heteroscedasticity

## Performance Considerations

### Training Time

- **ARIMA:** Seconds to minutes (depends on data size)
- **Seq2Seq:** Minutes to hours (depends on epochs and data size)

### Inference Time

- **ARIMA:** Milliseconds per forecast
- **Seq2Seq:** Milliseconds to seconds per forecast

### Memory Usage

- Scales with:
  - Data size
  - Context length
  - Batch size (for Seq2Seq)
  - Number of features

## Limitations

1. **Univariate focus:** Primary target variable, covariates supported
2. **Fixed frequency:** Requires consistent time intervals
3. **Stationarity:** ARIMA assumes stationarity (handled via differencing)
4. **Computational:** Seq2Seq requires GPU for large datasets
5. **Data requirements:** Minimum data needed for reliable forecasts

## Extensibility

The application is designed for extension:

- **New models:** Add to `src/models/`
- **New metrics:** Add to `src/evaluation/metrics.py`
- **New features:** Extend `src/data/features.py`
- **API endpoints:** Add routes to `api/app.py`
- **UI components:** Add to `frontend/src/components/`

## Security Considerations

- File upload validation (CSV only)
- Path traversal prevention
- Input sanitization
- CORS configuration for API
- Error message sanitization

## Future Enhancements

Potential additions:
- More forecasting models (Prophet, Transformer)
- Automated hyperparameter tuning
- Model ensemble capabilities
- Real-time streaming forecasts
- Advanced visualization options
- Export capabilities (PDF reports)
- Model versioning and management


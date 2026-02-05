"""
Flask API server for TimeSeries Forecaster frontend.
"""

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports"
UPLOAD_DIR = DATA_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Store training jobs
training_jobs: Dict[str, Dict] = {}


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "TimeSeries Forecaster API"})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Upload CSV file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "File must be a CSV"}), 400

    filename = file.filename
    filepath = UPLOAD_DIR / filename
    file.save(str(filepath))

    return jsonify({
        "message": "File uploaded successfully",
        "filename": filename,
        "path": str(filepath.relative_to(BASE_DIR))
    })


@app.route("/api/files", methods=["GET"])
def list_files():
    """List uploaded CSV files."""
    files = []
    for filepath in DATA_DIR.glob("*.csv"):
        files.append({
            "filename": filepath.name,
            "size": filepath.stat().st_size,
            "path": str(filepath.relative_to(BASE_DIR))
        })
    return jsonify({"files": files})


@app.route("/api/train", methods=["POST"])
def start_training():
    """Start a training job."""
    data = request.json
    model_type = data.get("model_type", "arima")
    csv_path = data.get("csv_path")
    config_overrides = data.get("config", {})

    if not csv_path:
        return jsonify({"error": "csv_path is required"}), 400

    # Create temporary config file
    import tempfile
    import yaml
    
    config_template = {
        "run": {"name": f"{model_type}_interactive"},
        "data": {
            "csv_path": csv_path,
            "timestamp_col": config_overrides.get("timestamp_col", "timestamp"),
            "target_col": config_overrides.get("target_col", "y"),
            "freq": config_overrides.get("freq", "D"),
            "train_ratio": config_overrides.get("train_ratio", 0.8),
        },
        "features": {
            "calendar": config_overrides.get("calendar", True),
            "lags": config_overrides.get("lags", [1, 7, 14]),
            "rolling": config_overrides.get("rolling", [{"window": 7, "stats": ["mean", "std"]}]),
        },
        "windows": {
            "context_length": config_overrides.get("context_length", 60),
            "horizon": config_overrides.get("horizon", 30),
        },
        "model": {
            "type": model_type,
        },
        "walk_forward": {
            "n_folds": config_overrides.get("n_folds", 5),
        },
        "reports": {
            "base_dir": "reports",
        },
    }

    # Add model-specific config
    if model_type == "arima":
        config_template["model"]["order"] = config_overrides.get("order", [2, 1, 2])
        config_template["model"]["seasonal_order"] = config_overrides.get("seasonal_order", [0, 0, 0, 0])
    elif model_type == "seq2seq_attention":
        config_template["model"]["hidden_size"] = config_overrides.get("hidden_size", 64)
        config_template["model"]["num_layers"] = config_overrides.get("num_layers", 2)
        config_template["model"]["dropout"] = config_overrides.get("dropout", 0.1)
        config_template["model"]["quantiles"] = config_overrides.get("quantiles", [0.1, 0.5, 0.9])
        config_template["training"] = {
            "batch_size": config_overrides.get("batch_size", 64),
            "max_epochs": config_overrides.get("max_epochs", 50),
            "learning_rate": config_overrides.get("learning_rate", 1e-3),
            "gradient_clip_val": config_overrides.get("gradient_clip_val", 1.0),
            "early_stopping_patience": config_overrides.get("early_stopping_patience", 5),
        }

    # Save config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_template, f)
        config_path = f.name

    # Generate job ID
    import uuid
    job_id = str(uuid.uuid4())

    # Start training in background thread
    def run_training():
        try:
            training_jobs[job_id]["status"] = "running"
            result = subprocess.run(
                ["python", "-m", "src.training.train", "--config", config_path],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                training_jobs[job_id]["status"] = "completed"
                # Extract run_id from output
                output_lines = result.stdout.split("\n")
                for line in output_lines:
                    if "Results saved to" in line:
                        run_id = line.split("reports/")[-1].strip()
                        training_jobs[job_id]["run_id"] = run_id
            else:
                training_jobs[job_id]["status"] = "failed"
                training_jobs[job_id]["error"] = result.stderr
        except Exception as e:
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["error"] = str(e)
        finally:
            # Clean up temp config
            os.unlink(config_path)

    training_jobs[job_id] = {
        "status": "queued",
        "model_type": model_type,
        "config": config_template,
    }

    thread = threading.Thread(target=run_training)
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "queued",
        "message": "Training job started"
    })


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """Get training job status."""
    if job_id not in training_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = training_jobs[job_id]
    return jsonify(job)


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    """List all training jobs."""
    return jsonify({"jobs": list(training_jobs.values())})


@app.route("/api/results/<run_id>/metrics", methods=["GET"])
def get_metrics(run_id: str):
    """Get metrics for a run."""
    run_dir = REPORTS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "Run not found"}), 404

    metrics_file = run_dir / "metrics_summary.csv"
    if not metrics_file.exists():
        return jsonify({"error": "Metrics file not found"}), 404

    import pandas as pd
    df = pd.read_csv(metrics_file)
    return jsonify({"metrics": df.to_dict("records")})


@app.route("/api/results/<run_id>/predictions", methods=["GET"])
def get_predictions(run_id: str):
    """Get predictions for a run."""
    run_dir = REPORTS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "Run not found"}), 404

    predictions_file = run_dir / "predictions.csv"
    if not predictions_file.exists():
        return jsonify({"error": "Predictions file not found"}), 404

    import pandas as pd
    df = pd.read_csv(predictions_file)
    # Limit to first 1000 rows for performance
    df = df.head(1000)
    return jsonify({"predictions": df.to_dict("records")})


@app.route("/api/results/<run_id>/plots/<plot_name>", methods=["GET"])
def get_plot(run_id: str, plot_name: str):
    """Get a plot image."""
    run_dir = REPORTS_DIR / run_id
    plot_path = run_dir / plot_name

    if not plot_path.exists():
        return jsonify({"error": "Plot not found"}), 404

    return send_file(str(plot_path), mimetype="image/png")


@app.route("/api/results/<run_id>/info", methods=["GET"])
def get_run_info(run_id: str):
    """Get run information."""
    run_dir = REPORTS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "Run not found"}), 404

    info = {
        "run_id": run_id,
        "plots": [],
        "has_metrics": False,
        "has_predictions": False,
    }

    # List available plots
    for plot_file in run_dir.glob("*.png"):
        info["plots"].append(plot_file.name)

    # Check for metrics and predictions
    if (run_dir / "metrics_summary.csv").exists():
        info["has_metrics"] = True
    if (run_dir / "predictions.csv").exists():
        info["has_predictions"] = True

    # Load config if available
    config_file = run_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            info["config"] = json.load(f)

    return jsonify(info)


@app.route("/api/results", methods=["GET"])
def list_results():
    """List all available runs."""
    runs = []
    for run_dir in REPORTS_DIR.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith("."):
            runs.append({
                "run_id": run_dir.name,
                "created": run_dir.stat().st_mtime,
            })
    # Sort by creation time, newest first
    runs.sort(key=lambda x: x["created"], reverse=True)
    return jsonify({"runs": runs})


if __name__ == "__main__":
    app.run(debug=True, port=5000)


from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def create_run_id(name: str | None = None) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if name:
        return f"{name}_{ts}"
    return ts


def get_run_dir(base_dir: str | Path, run_id: str) -> Path:
    base = Path(base_dir)
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    return run_dir


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)



import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)



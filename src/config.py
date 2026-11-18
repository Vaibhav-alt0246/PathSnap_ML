"""Central YAML configuration loading."""
from pathlib import Path
from typing import Any
import yaml

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load settings from the repository configuration file."""
    config_path = Path(path) if path else Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    with config_path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}

from pathlib import Path
from typing import Any
import yaml
import sys


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the configuration file."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    except Exception:
        sys.exit(1)

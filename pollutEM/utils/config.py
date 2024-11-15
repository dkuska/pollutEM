from typing import Any
import yaml


def load_config(path: str) -> Any:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

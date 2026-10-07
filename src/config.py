"""Loads project configuration from config.yaml.

Keeping this as a single, tiny module means every other module gets paths
and parameters the same way, instead of each script guessing relative paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and return the project config as a plain dict.

    Parameters
    ----------
    path : Path
        Path to the YAML config file. Defaults to config.yaml at the repo root.

    Returns
    -------
    dict[str, Any]
        Parsed configuration.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml relative to the project root.

    Every path in config.yaml is written relative to the repo root so the
    project works the same regardless of the current working directory it's
    invoked from.
    """
    return PROJECT_ROOT / relative_path

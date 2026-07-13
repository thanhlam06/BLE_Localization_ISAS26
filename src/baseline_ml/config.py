"""Configuration loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the repository root based on the installed source location."""
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (root or project_root()) / path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the baseline JSON config."""
    root = project_root()
    config_path = resolve_path(path or "configs/baseline.json", root)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(root)
    return config

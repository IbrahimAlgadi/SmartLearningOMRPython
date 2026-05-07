"""Repository paths shared by the webapp and OMR detectors (single source of truth)."""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
BUBBLE_COORDS_DIR = DATA_DIR / "bubble_coords"


def model_path(name: str) -> pathlib.Path:
    """CNN weights (ONNX / TorchScript) are stored at the repository root."""
    return REPO_ROOT / name

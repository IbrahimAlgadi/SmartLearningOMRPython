"""
webapp/services/omr_service.py
──────────────────────────────
Service layer: save uploads, resolve templates, run the OMR pipeline,
and return JSON-serialisable result data.
"""

from __future__ import annotations

import dataclasses
import pathlib
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from omr_detector_enhanced_v3 import OMRPipelineV3, FinalResultContract
from omr_templates import get_template, infer_template

# ─────────────────────────────────────────────────────────────────────────────
#  Runs root — all per-run folders live here.
#  Resolved relative to the project root (where uvicorn is started from).
# ─────────────────────────────────────────────────────────────────────────────

RUNS_ROOT = pathlib.Path("web_runs")


# One shared pipeline instance — avoid repeated model-file loading.
_pipeline: Optional[OMRPipelineV3] = None


def _get_pipeline() -> OMRPipelineV3:
    global _pipeline
    if _pipeline is None:
        _pipeline = OMRPipelineV3()
    return _pipeline


# ─────────────────────────────────────────────────────────────────────────────
#  Template helpers
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_CHOICES = [
    ("Q20_5ch",  "Q20  — 20 questions, 5 choices"),
    ("Q50_5ch",  "Q50  — 50 questions, 5 choices"),
    ("Q100_5ch", "Q100 — 100 questions, 5 choices"),
]


def resolve_template(template_id: str):
    """Return TemplateSpec or raise ValueError for an unknown id."""
    return get_template(template_id)


# ─────────────────────────────────────────────────────────────────────────────
#  Run directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_run_dir() -> tuple[str, pathlib.Path]:
    """Create and return a unique run_id and its directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{uuid.uuid4().hex[:8]}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def save_upload(image_bytes: bytes, original_filename: str) -> tuple[str, pathlib.Path, pathlib.Path]:
    """
    Save the uploaded bytes to a new run directory.
    Returns (run_id, run_dir, image_path).
    """
    suffix = pathlib.Path(original_filename).suffix.lower() or ".jpg"
    run_id, run_dir = _make_run_dir()
    image_path = run_dir / f"upload{suffix}"
    image_path.write_bytes(image_bytes)
    return run_id, run_dir, image_path


# ─────────────────────────────────────────────────────────────────────────────
#  Result serialisation
# ─────────────────────────────────────────────────────────────────────────────

DEBUG_IMAGES = [
    ("00_original.jpg",   "Original"),
    ("03_warped.jpg",     "Warped"),
    ("04_layout.jpg",     "Layout"),
    ("06_grid.jpg",       "Grid"),
    ("08_classified.jpg", "Classified"),
    ("10_answered.jpg",   "Answered"),
]

MARKED_IMAGE = "10_answered.jpg"


def _contract_to_dict(result: FinalResultContract) -> Dict[str, Any]:
    """Convert the dataclass result to a plain dict suitable for JSON."""
    d = dataclasses.asdict(result)
    # The debug_dir is an absolute path string — keep it as-is for reference.
    return d


def available_debug_images(run_id: str) -> list[dict]:
    """Return a list of {name, label, url} dicts for images that exist."""
    run_dir = RUNS_ROOT / run_id
    out = []
    for filename, label in DEBUG_IMAGES:
        if (run_dir / filename).exists():
            out.append({
                "name":  filename,
                "label": label,
                "url":   f"/artifacts/{run_id}/{filename}",
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_omr(
    image_bytes: bytes,
    original_filename: str,
    template_id: str,
    white_balance: bool = True,
    denoise: bool = True,
) -> dict:
    """
    Full pipeline call.
    Returns a dict with keys:
      run_id, status, result (FinalResultContract as dict on success),
      error (on failure), debug_images, marked_image_url.
    """
    run_id, run_dir, image_path = save_upload(image_bytes, original_filename)

    try:
        template = resolve_template(template_id)
    except (KeyError, ValueError) as exc:
        return {"run_id": run_id, "status": "ERROR", "error": str(exc)}

    pipeline = _get_pipeline()

    try:
        result: FinalResultContract = pipeline.run(
            str(image_path),
            template=template,
            debug_dir=run_dir,
            white_balance=white_balance,
            denoise=denoise,
        )
    except Exception as exc:
        return {
            "run_id": run_id,
            "status": "ERROR",
            "error":  str(exc),
            "debug_images": available_debug_images(run_id),
        }

    result_dict = _contract_to_dict(result)

    return {
        "run_id":           run_id,
        "status":           "OK",
        "result":           result_dict,
        "debug_images":     available_debug_images(run_id),
        "marked_image_url": f"/artifacts/{run_id}/{MARKED_IMAGE}",
    }

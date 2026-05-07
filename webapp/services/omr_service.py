"""
webapp/services/omr_service.py
──────────────────────────────
Service layer: save uploads, resolve templates, run the OMR pipeline,
and return JSON-serialisable result data.

Changes from v1:
  • Uses OMRPipelineV3Optimized (3× faster, shared scratch buffers).
  • RUNS_ROOT is now configurable via settings.runs_dir.
  • run_omr accepts an injected pipeline (from app.state) and a
    debug_artifacts flag; when False no debug JPEGs are written.
  • Thread-safe lazy singleton fallback (_get_pipeline) for CLI / tests.
  • Error returns carry a stable error code; raw str(exc) never leaves the
    service layer (full traceback is logged via logger.exception).
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import threading
import traceback
import uuid
from datetime import datetime
from typing import Any, Optional

from omr_detector_enhanced_v3_optimized_1 import (
    FinalResultContract,
    OMRPipelineV3Optimized,
)
from omr_templates import get_template

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Template helpers
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_CHOICES = [
    ("Q10_5ch",  "Q10  — 10 questions, 5 choices"),
    ("Q15_5ch",  "Q15  — 15 questions, 5 choices"),
    ("Q18_5ch",  "Q18  — 18 questions, 5 choices"),
    ("Q20_5ch",  "Q20  — 20 questions, 5 choices"),
    ("Q30_5ch",  "Q30  — 30 questions, 5 choices"),
    ("Q40_5ch",  "Q40  — 40 questions, 5 choices"),
    ("Q50_5ch",  "Q50  — 50 questions, 5 choices"),
    ("Q60_5ch",  "Q60  — 60 questions, 5 choices"),
    ("Q80_5ch",  "Q80  — 80 questions, 5 choices"),
    ("Q100_5ch", "Q100 — 100 questions, 5 choices"),
    # ── 4 choices ─────────────────────────────────────────────────────────────
    ("Q10_4ch",  "Q10  — 10 questions, 4 choices"),
    ("Q15_4ch",  "Q15  — 15 questions, 4 choices"),
    ("Q18_4ch",  "Q18  — 18 questions, 4 choices"),
    ("Q20_4ch",  "Q20  — 20 questions, 4 choices"),
    ("Q30_4ch",  "Q30  — 30 questions, 4 choices"),
    ("Q40_4ch",  "Q40  — 40 questions, 4 choices"),
    ("Q50_4ch",  "Q50  — 50 questions, 4 choices"),
    ("Q60_4ch",  "Q60  — 60 questions, 4 choices"),
    ("Q80_4ch",  "Q80  — 80 questions, 4 choices"),
    ("Q100_4ch", "Q100 — 100 questions, 4 choices"),
    # ── 3 choices ─────────────────────────────────────────────────────────────
    ("Q10_3ch",  "Q10  — 10 questions, 3 choices"),
    ("Q15_3ch",  "Q15  — 15 questions, 3 choices"),
    ("Q18_3ch",  "Q18  — 18 questions, 3 choices"),
    ("Q20_3ch",  "Q20  — 20 questions, 3 choices"),
    ("Q30_3ch",  "Q30  — 30 questions, 3 choices"),
    ("Q40_3ch",  "Q40  — 40 questions, 3 choices"),
    ("Q50_3ch",  "Q50  — 50 questions, 3 choices"),
    ("Q60_3ch",  "Q60  — 60 questions, 3 choices"),
    ("Q80_3ch",  "Q80  — 80 questions, 3 choices"),
    ("Q100_3ch", "Q100 — 100 questions, 3 choices"),
    # ── 2 choices ─────────────────────────────────────────────────────────────
    ("Q10_2ch",  "Q10  — 10 questions, 2 choices"),
    ("Q15_2ch",  "Q15  — 15 questions, 2 choices"),
    ("Q18_2ch",  "Q18  — 18 questions, 2 choices"),
    ("Q20_2ch",  "Q20  — 20 questions, 2 choices"),
    ("Q30_2ch",  "Q30  — 30 questions, 2 choices"),
    ("Q40_2ch",  "Q40  — 40 questions, 2 choices"),
    ("Q50_2ch",  "Q50  — 50 questions, 2 choices"),
    ("Q60_2ch",  "Q60  — 60 questions, 2 choices"),
    ("Q80_2ch",  "Q80  — 80 questions, 2 choices"),
    ("Q100_2ch", "Q100 — 100 questions, 2 choices"),
]


def resolve_template(template_id: str):
    return get_template(template_id)


# ─────────────────────────────────────────────────────────────────────────────
#  Artifact helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEBUG_IMAGES = [
    ("00_original.jpg",   "Original"),
    ("03_warped.jpg",     "Warped"),
    ("04_layout.jpg",     "Layout"),
    ("06_grid.jpg",       "Grid"),
    ("08_classified.jpg", "Classified"),
    ("10_answered.jpg",   "Answered"),
]

# Whitelist of filenames that may be served via /artifacts/.
# Kept in sync with _DEBUG_IMAGES + the result JSON.
ALLOWED_ARTIFACTS: frozenset[str] = frozenset(
    [fname for fname, _ in _DEBUG_IMAGES] + ["result_v3.json", "result_v3_error.json"]
)


def available_debug_images(run_id: str, runs_dir: pathlib.Path) -> list[dict]:
    run_dir = runs_dir / run_id
    out = []
    for filename, label in _DEBUG_IMAGES:
        if (run_dir / filename).exists():
            out.append({
                "name":  filename,
                "label": label,
                "url":   f"/artifacts/{run_id}/{filename}",
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Thread-safe singleton fallback (used by CLI / tests only).
#  In production the pipeline is injected from app.state by the lifespan.
# ─────────────────────────────────────────────────────────────────────────────

_pipeline_lock: threading.Lock = threading.Lock()
_pipeline_singleton: Optional[OMRPipelineV3Optimized] = None


def _get_pipeline() -> OMRPipelineV3Optimized:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        with _pipeline_lock:
            if _pipeline_singleton is None:       # double-check after lock
                _pipeline_singleton = OMRPipelineV3Optimized()
    return _pipeline_singleton


# ─────────────────────────────────────────────────────────────────────────────
#  Run directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_run_dir(runs_dir: pathlib.Path) -> tuple[str, pathlib.Path]:
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{uuid.uuid4().hex[:8]}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def save_upload(
    image_bytes: bytes,
    original_filename: str,
    runs_dir: pathlib.Path,
) -> tuple[str, pathlib.Path, pathlib.Path]:
    suffix     = pathlib.Path(original_filename).suffix.lower() or ".jpg"
    run_id, run_dir = _make_run_dir(runs_dir)
    image_path = run_dir / f"upload{suffix}"
    image_path.write_bytes(image_bytes)
    return run_id, run_dir, image_path


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point — called from run_in_threadpool in main.py
# ─────────────────────────────────────────────────────────────────────────────

def run_omr(
    image_bytes: bytes,
    original_filename: str,
    template_id: str,
    white_balance: bool = True,
    denoise: bool = True,
    pipeline: Optional[OMRPipelineV3Optimized] = None,
    debug_artifacts: bool = False,
    runs_dir: Optional[pathlib.Path] = None,
) -> dict:
    """
    Full OMR pipeline call.  Designed to run in a thread (not async).

    Parameters
    ----------
    pipeline        : Injected OMRPipelineV3Optimized from app.state.
                      Falls back to the module-level singleton when None
                      (CLI / test usage).
    debug_artifacts : When True, write intermediate debug JPEGs.
                      When False (production default), only result_v3.json
                      is written, saving ~4 MB per run.
    runs_dir        : Base directory for per-run folders.  Defaults to
                      Path("web_runs") for backwards compatibility; the
                      lifespan passes settings.runs_dir explicitly.
    """
    if runs_dir is None:
        runs_dir = pathlib.Path("web_runs")

    run_id, run_dir, image_path = save_upload(image_bytes, original_filename, runs_dir)

    try:
        template = resolve_template(template_id)
    except (KeyError, ValueError) as exc:
        logger.warning("Unknown template_id=%r  run_id=%s", template_id, run_id)
        return {
            "run_id": run_id,
            "status": "ERROR",
            "error":  {"code": "UNKNOWN_TEMPLATE",
                       "message": "The requested template ID is not recognised."},
        }

    used_pipeline = pipeline or _get_pipeline()

    # Always write debug output into the run_dir so available_debug_images()
    # can find them.  The pipeline falls back to a CWD-relative directory
    # when debug_dir=None, which would scatter files outside web_runs/.
    # When debug_artifacts=False we delete the images after the run, keeping
    # only result_v3.json.
    debug_dir: pathlib.Path = run_dir

    try:
        result: FinalResultContract = used_pipeline.run(
            str(image_path),
            template=template,
            debug_dir=debug_dir,
            white_balance=white_balance,
            denoise=denoise,
        )
    except Exception:
        logger.exception("Pipeline failed  run_id=%s", run_id)
        # Write a minimal error JSON so /api/runs/{id} returns 422 correctly.
        import json as _json
        err_path = run_dir / "result_v3_error.json"
        err_path.write_text(
            _json.dumps({
                "run_id": run_id,
                "status": "ERROR",
                "error": {"code": "PIPELINE_ERROR",
                          "message": "OMR processing failed. See server logs for details."},
            }),
            encoding="utf-8",
        )
        return {
            "run_id": run_id,
            "status": "ERROR",
            "error":  {"code": "PIPELINE_ERROR",
                       "message": "OMR processing failed."},
            "debug_images": available_debug_images(run_id, runs_dir),
        }

    # The pipeline writes result_v3.json to debug_dir (= run_dir).
    # If debug_artifacts is disabled, remove the intermediate images but keep
    # the JSON so the result page and API still work.
    if not debug_artifacts:
        for fname, _ in _DEBUG_IMAGES:
            p = run_dir / fname
            if p.exists():
                p.unlink(missing_ok=True)

    return {
        "run_id":           run_id,
        "status":           "OK",
        "result":           dataclasses.asdict(result),
        "debug_images":     available_debug_images(run_id, runs_dir),
        "marked_image_url": f"/artifacts/{run_id}/10_answered.jpg",
    }

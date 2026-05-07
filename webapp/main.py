"""
webapp/main.py
──────────────
Production-hardened FastAPI application for the OMR auto-marking service.

Changes from the original (v1):
  • lifespan context manager — eager pipeline/ONNX load at startup, sweeper
    task, clean shutdown.
  • Non-blocking /scan — CPU work offloaded to a threadpool via
    run_in_threadpool; a bounded asyncio.Semaphore limits concurrency per
    worker and returns 503 + Retry-After when the queue is full.
  • Upload validation — Content-Length / streaming size cap (MaxBodySizeMiddleware),
    magic-byte check (JPEG/PNG only), registry validation for template_id.
  • Auth — X-API-Key on all state-changing / data-returning endpoints.
  • Rate limiting — slowapi per-IP limiter on /scan and /api/runs/*.
  • Error responses — stable {error:{code,message}} envelope; no stack
    traces to clients.
  • Hardened artifact serving — path.resolve().is_relative_to() + filename
    whitelist; no directory traversal possible.
  • Health probes split: /healthz (liveness, no auth) and /readyz (readiness).

Start:
    uvicorn webapp.main:app --host 0.0.0.0 --port 8000          # dev
    gunicorn -k uvicorn.workers.UvicornWorker -w 2 webapp.main:app  # prod
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import pathlib
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    _SLOWAPI = True
except ImportError:
    _SLOWAPI = False

from omr_detector_enhanced_v3_optimized_1 import OMRPipelineV3Optimized

from webapp.config import get_settings
from webapp.errors import add_exception_handlers
from webapp.middleware import MaxBodySizeMiddleware, RequestIdMiddleware
from webapp.security import verify_api_key
from webapp.services.omr_service import (
    ALLOWED_ARTIFACTS,
    TEMPLATE_CHOICES,
    available_debug_images,
    run_omr,
)
from webapp.sweeper import sweeper_loop
from omr_templates import REGISTRY as _TEMPLATE_REGISTRY

logger = logging.getLogger(__name__)

# Magic bytes for JPEG and PNG — no extra dependency needed.
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC  = b"\x89PNG\r\n\x1a\n"

# ─────────────────────────────────────────────────────────────────────────────
#  Rate limiter (optional; disabled gracefully if slowapi not installed)
# ─────────────────────────────────────────────────────────────────────────────

if _SLOWAPI:
    _limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{get_settings().rate_limit_per_minute}/minute"],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Lifespan — startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Resolve runs_dir to absolute so it works regardless of cwd.
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    app.state.settings = settings

    logger.info("Loading OMR pipeline (ONNX model) …")
    app.state.pipeline = OMRPipelineV3Optimized()
    app.state.pipeline_ready = True
    logger.info("Pipeline ready.")

    app.state.job_semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)

    app.state.sweeper_task = asyncio.create_task(
        sweeper_loop(
            settings.runs_dir,
            settings.run_ttl_hours,
            interval_s=settings.sweeper_interval_s,
        ),
        name="disk-sweeper",
    )

    try:
        yield
    finally:
        # Cancel sweeper first (fast).
        app.state.sweeper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.sweeper_task

        # Mark not ready so /readyz returns 503 during drain.
        app.state.pipeline_ready = False
        logger.info("Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
#  App factory
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OMR Auto-Marking",
    version="2.0",
    lifespan=lifespan,
    # Disable auto-generated /docs and /redoc in production if desired.
    # docs_url=None, redoc_url=None,
)

# ── Middlewares (order matters: outermost added last) ────────────────────────
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    MaxBodySizeMiddleware,
    max_bytes=get_settings().max_upload_bytes,
)

if _SLOWAPI:
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

add_exception_handlers(app)

_WEBAPP_DIR    = pathlib.Path(__file__).parent
_STATIC_DIR    = _WEBAPP_DIR / "static"
_TEMPLATES_DIR = _WEBAPP_DIR / "templates"

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ─────────────────────────────────────────────────────────────────────────────
#  Health probes  (NO auth — used by load-balancers and Docker HEALTHCHECK)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/healthz", tags=["ops"])
async def healthz():
    """Liveness probe — always 200 while the process is running."""
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
async def readyz(request: Request):
    """
    Readiness probe — 200 only when:
      • Pipeline/ONNX is loaded
      • RUNS_ROOT is writable
      • Sweeper task is alive
    """
    settings: get_settings().__class__ = request.app.state.settings
    checks: dict[str, str] = {}

    # Pipeline loaded?
    if getattr(request.app.state, "pipeline_ready", False):
        checks["pipeline"] = "ok"
    else:
        checks["pipeline"] = "not_ready"

    # Runs dir writable?
    try:
        probe = settings.runs_dir / ".write_probe"
        probe.touch()
        probe.unlink()
        checks["storage"] = "ok"
    except OSError:
        checks["storage"] = "error"

    # Sweeper alive?
    sweeper: asyncio.Task | None = getattr(request.app.state, "sweeper_task", None)
    checks["sweeper"] = "ok" if (sweeper and not sweeper.done()) else "not_running"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "not_ready", "checks": checks},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  UI routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def upload_page(
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"template_choices": TEMPLATE_CHOICES},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /scan — main workhorse endpoint
# ─────────────────────────────────────────────────────────────────────────────

if _SLOWAPI:
    # Decorate /scan with a per-IP rate limit.  Must be done AFTER the
    # route function is defined; we apply it immediately below.
    def _rate_limit_scan(fn):
        return _limiter.limit(f"{get_settings().rate_limit_per_minute}/minute")(fn)
else:
    def _rate_limit_scan(fn):  # type: ignore[misc]
        return fn


@app.post("/scan")
@_rate_limit_scan
async def scan(
    request: Request,
    image: UploadFile = File(...),
    template_id: Annotated[
        str,
        Form(),
    ] = "Q20_5ch",
    white_balance: bool = Form(True),
    denoise: bool = Form(True),
    _auth: None = Depends(verify_api_key),
):
    # ── Read and validate upload ─────────────────────────────────────────────
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload — no bytes received.")

    if not (image_bytes.startswith(_JPEG_MAGIC) or image_bytes.startswith(_PNG_MAGIC)):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a JPEG or PNG image.",
        )

    if template_id not in _TEMPLATE_REGISTRY:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown template '{template_id}'. "
                   f"Valid options: {sorted(_TEMPLATE_REGISTRY)}",
        )

    # ── Bounded concurrency — offload blocking work to threadpool ────────────
    sem: asyncio.Semaphore = request.app.state.job_semaphore
    pipeline               = request.app.state.pipeline
    settings               = request.app.state.settings

    try:
        await asyncio.wait_for(sem.acquire(), timeout=settings.queue_wait_s)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Service busy. Please retry shortly.",
            headers={"Retry-After": "5"},
        )

    try:
        outcome = await run_in_threadpool(
            run_omr,
            image_bytes=image_bytes,
            original_filename=image.filename or "upload.jpg",
            template_id=template_id,
            white_balance=white_balance,
            denoise=denoise,
            pipeline=pipeline,
            debug_artifacts=settings.debug_artifacts,
        )
    finally:
        sem.release()

    return RedirectResponse(url=f"/runs/{outcome['run_id']}", status_code=303)


# ─────────────────────────────────────────────────────────────────────────────
#  UI result page
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def result_page(
    request: Request,
    run_id: str,
    _auth: None = Depends(verify_api_key),
):
    settings  = request.app.state.settings
    run_dir   = settings.runs_dir / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found.")

    result_path = run_dir / "result_v3.json"
    error_path  = run_dir / "result_v3_error.json"

    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        status = "OK"
        error  = None
    elif error_path.exists():
        result = None
        error  = json.loads(error_path.read_text(encoding="utf-8"))
        status = "ERROR"
    else:
        result = None
        error  = {"error": {"code": "PROCESSING_INCOMPLETE",
                            "message": "Processing did not complete."}}
        status = "ERROR"

    debug_images = available_debug_images(run_id, settings.runs_dir)

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "run_id":           run_id,
            "status":           status,
            "result":           result,
            "error":            error,
            "debug_images":     debug_images,
            "marked_image_url": f"/artifacts/{run_id}/10_answered.jpg"
                if any(d["name"] == "10_answered.jpg" for d in debug_images)
                else None,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  REST API result endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/runs/{run_id}")
async def api_result(
    request: Request,
    run_id: str,
    _auth: None = Depends(verify_api_key),
):
    settings  = request.app.state.settings
    run_dir   = settings.runs_dir / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found.")

    result_path = run_dir / "result_v3.json"
    error_path  = run_dir / "result_v3_error.json"

    if result_path.exists():
        return JSONResponse(json.loads(result_path.read_text(encoding="utf-8")))
    if error_path.exists():
        return JSONResponse(
            json.loads(error_path.read_text(encoding="utf-8")),
            status_code=422,
        )
    raise HTTPException(status_code=404, detail="Result not ready.")


# ─────────────────────────────────────────────────────────────────────────────
#  Artifact serving — hardened path traversal protection
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/artifacts/{run_id}/{filename}")
async def serve_artifact(
    request: Request,
    run_id: str,
    filename: str,
    _auth: None = Depends(verify_api_key),
):
    settings  = request.app.state.settings
    runs_root = settings.runs_dir.resolve()

    # Filename whitelist — only allow known output files.
    if filename not in ALLOWED_ARTIFACTS:
        raise HTTPException(
            status_code=404,
            detail="File not found.",
        )

    file_path = (settings.runs_dir / run_id / filename).resolve()

    # Strict path containment check — prevents any form of directory traversal.
    if not file_path.is_relative_to(runs_root):
        raise HTTPException(status_code=400, detail="Invalid path.")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(str(file_path))

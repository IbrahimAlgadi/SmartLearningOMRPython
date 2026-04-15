"""
webapp/main.py
──────────────
FastAPI application — LAN-accessible OMR auto-marking service.

Start from the project root:
    uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import pathlib

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.services.omr_service import (
    RUNS_ROOT,
    TEMPLATE_CHOICES,
    available_debug_images,
    run_omr,
)

# ─────────────────────────────────────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="OMR Auto-Marking", version="1.0")

_WEBAPP_DIR = pathlib.Path(__file__).parent
_STATIC_DIR = _WEBAPP_DIR / "static"
_TEMPLATES_DIR = _WEBAPP_DIR / "templates"

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Ensure runs root exists at startup
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"template_choices": TEMPLATE_CHOICES},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/scan")
async def scan(
    request: Request,
    image: UploadFile = File(...),
    template_id: str = Form("Q100_5ch"),
    white_balance: bool = Form(True),
    denoise: bool = Form(True),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    outcome = run_omr(
        image_bytes=image_bytes,
        original_filename=image.filename or "upload.jpg",
        template_id=template_id,
        white_balance=white_balance,
        denoise=denoise,
    )

    return RedirectResponse(url=f"/runs/{outcome['run_id']}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def result_page(request: Request, run_id: str):
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

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
        error  = {"error": "Processing did not complete."}
        status = "ERROR"

    debug_images = available_debug_images(run_id)

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
                                if any(d["name"] == "10_answered.jpg" for d in debug_images) else None,
        },
    )


@app.get("/api/runs/{run_id}")
async def api_result(run_id: str):
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    result_path = run_dir / "result_v3.json"
    error_path  = run_dir / "result_v3_error.json"

    if result_path.exists():
        return JSONResponse(json.loads(result_path.read_text(encoding="utf-8")))
    if error_path.exists():
        return JSONResponse(
            json.loads(error_path.read_text(encoding="utf-8")),
            status_code=422,
        )
    raise HTTPException(status_code=404, detail="Result not ready")


@app.get("/artifacts/{run_id}/{filename}")
async def serve_artifact(run_id: str, filename: str):
    # Safety: no directory traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = RUNS_ROOT / run_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))

# OMR auto-marking service

FastAPI web app and REST API for scanning filled OMR sheets. The pipeline lives in `omr_detector_enhanced_v3_optimized_1.py`; HTTP routes and UI are in `webapp/main.py`.

## Requirements

- **Python 3.11+** (matches the Docker image).
- Install dependencies from the repo root:

  ```cmd
  pip install -r requirements.txt
  ```

  PyTorch is optional for inference if you use ONNX; the `requirements.txt` header describes a **CPU-only** PyTorch install if you want a smaller footprint.

- **Windows development:** use **Uvicorn** or `run_app.py`. **Gunicorn** in `requirements.txt` is aimed at Linux/macOS production; on Windows, prefer the commands below.

## Runtime assets

Before scans work end-to-end, you need:

1. **Classifier weights** at the repository root (as configured in `omr_detector_enhanced_v3_optimized_1.py`):
   - `bubble_classifier_v3.onnx` — primary path with **onnxruntime**
   - `bubble_classifier_v3.pt` — optional TorchScript fallback  
   If neither file is present, the detector can fall back to fill-ratio heuristics (lower accuracy).

2. **Bubble geometry JSON** under `data/bubble_coords/` (see `omr_paths.BUBBLE_COORDS_DIR`). Templates must match what the app exposes as `template_id` choices.

## Configuration

1. Copy `.env.example` to `.env` in the project root.
2. Variables use the **`OMR_`** prefix and are loaded by `webapp/config.py` (including from `.env`).

Notable settings (see `.env.example` for the full list):

| Variable | Role |
| -------- | ---- |
| `OMR_API_KEY` | If set, protected endpoints require header **`X-API-Key`**. Empty = open dev mode (a warning is logged). |
| `OMR_RUNS_DIR` | Directory for per-run output (default `web_runs`). |
| `OMR_MAX_UPLOAD_MB` | Max upload size. |
| `OMR_RATE_LIMIT_PER_MINUTE` | Per-IP rate limit on `/scan` (when slowapi is installed). |
| `OMR_MAX_CONCURRENT_JOBS` / `OMR_QUEUE_WAIT_S` | Concurrency and queue wait before 503. |
| `OMR_RUN_TTL_HOURS` / `OMR_SWEEPER_INTERVAL_S` | Background cleanup of old runs. |
| `OMR_DEBUG_ARTIFACTS` | Write step JPEGs per run (`1`) or only JSON (`0`). |
| `OMR_LOG_LEVEL` | Logging level. |

## Run locally (Windows cmd)

From the repository root:

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Start the server (ASGI app is **`app`** in module **`webapp.main`** — not `webapp:main`):

```cmd
uvicorn webapp.main:app --host 0.0.0.0 --port 8000
```

Optional auto-reload:

```cmd
uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload
```

**Convenience launcher** (see header comments in `run_app.py`):

```cmd
python run_app.py
python run_app.py --port 5000 --reload
```

Open **http://127.0.0.1:8000/** for the HTML UI.

## Run with Docker

Build (from repo root):

```bash
docker build -t omr-api:latest .
```

Run — **Linux/macOS** (same shape as the top of `Dockerfile`; mount ONNX separately if it is not in the image context):

```bash
docker run --rm -p 8000:8000 \
  -e OMR_API_KEY=changeme \
  -e OMR_DEBUG_ARTIFACTS=0 \
  -v "$(pwd)/web_runs:/app/web_runs" \
  -v "$(pwd)/bubble_classifier_v3.onnx:/app/bubble_classifier_v3.onnx:ro" \
  omr-api:latest
```

**Windows cmd** (line continuation is `^`):

```cmd
docker run --rm -p 8000:8000 ^
  -e OMR_API_KEY=changeme ^
  -e OMR_DEBUG_ARTIFACTS=0 ^
  -v "%cd%\web_runs:/app/web_runs" ^
  -v "%cd%\bubble_classifier_v3.onnx:/app/bubble_classifier_v3.onnx:ro" ^
  omr-api:latest
```

The image `HEALTHCHECK` calls **`GET /healthz`**. More detail is in the comments at the top of `Dockerfile`.

Production-style command inside the image: **gunicorn** with `uvicorn.workers.UvicornWorker` and module **`webapp.main:app`** (see `Dockerfile` `CMD`).

## API and documentation

FastAPI serves interactive docs at:

- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **OpenAPI JSON:** `/openapi.json`

There is no standard **`/swagger`** path; use **`/docs`** instead.

### Main HTTP routes

| Method | Path | Notes |
| ------ | ---- | ----- |
| GET | `/` | Web UI |
| POST | `/scan` | Upload + process (API key when configured) |
| GET | `/runs/{run_id}` | HTML result page |
| GET | `/api/runs/{run_id}` | JSON for a run |
| GET | `/artifacts/{run_id}/{filename}` | Served artifacts (validated paths) |
| GET | `/healthz` | Liveness (no auth) |
| GET | `/readyz` | Readiness |

When `OMR_API_KEY` is set, state-changing and data-returning endpoints expect **`X-API-Key`** (see `webapp/security.py`).

## Tests

From the repo root:

```cmd
pytest
```

Configuration is in `pytest.ini` (`testpaths = tests`).

## Further reading

- `docs/` — design and extra notes  
- `scripts/` — training, sheet generation, and dev utilities  

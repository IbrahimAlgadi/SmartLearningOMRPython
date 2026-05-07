#!/usr/bin/env python3
"""
run_app.py
──────────
DEV-MODE convenience launcher for the FastAPI OMR app.

Layout (repository root):
  webapp/              — FastAPI app
  scripts/             — training, sheet generators, one-off dev tools
  sheets/              — printable answer-sheet HTML (generated / archived)
  docs/                — design notes
  data/bubble_coords/  — per-template bubble geometry JSON (runtime)
  tests/manual/        — ad-hoc experimental scripts (not pytest tests)

⚠  NOT FOR PRODUCTION USE.
   In production, run via gunicorn with UvicornWorker (see Dockerfile).

Usage (from the project root):
    python run_app.py               # default: 0.0.0.0:8000, 1 worker
    python run_app.py --port 5000   # custom port
    python run_app.py --reload      # auto-reload on code changes

Production equivalent:
    gunicorn -k uvicorn.workers.UvicornWorker \\
             -w 2 -b 0.0.0.0:8000 \\
             --timeout 60 --graceful-timeout 30 \\
             --access-logfile - \\
             webapp.main:app

Then open  http://<your-LAN-IP>:8000  on a device on the same network.
Find your LAN IP with:  ipconfig  (look for "IPv4 Address" under Wi-Fi)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

_DEV_BANNER = """
+----------------------------------------------------------+
|  OMR Auto-Marking -- DEV MODE                            |
|  Do NOT use this launcher in production.                 |
|  Use gunicorn + UvicornWorker (see Dockerfile).          |
+----------------------------------------------------------+
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Start the OMR FastAPI server (dev mode)")
    ap.add_argument("--host",    default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    ap.add_argument("--port",    type=int, default=8000, help="Port (default: 8000)")
    ap.add_argument("--reload",  action="store_true", help="Auto-reload on code changes")
    ap.add_argument("--workers", type=int, default=1,
                    help="Worker processes (ignored when --reload is set)")
    args = ap.parse_args()

    print(_DEV_BANNER)

    # Verify model file without changing cwd.
    onnx_path = _PROJECT_ROOT / "bubble_classifier_v3.onnx"
    pt_path   = _PROJECT_ROOT / "bubble_classifier_v3.pt"
    if not onnx_path.exists() and not pt_path.exists():
        print(
            "[WARN] Neither bubble_classifier_v3.onnx nor bubble_classifier_v3.pt\n"
            "       found in the project root. The detector will fall back to\n"
            "       fill-ratio classification (lower accuracy).\n"
            f"       Expected location: {_PROJECT_ROOT}\n"
        )
    elif onnx_path.exists():
        print(f"[OK]  ONNX model found: {onnx_path}")
    else:
        print(f"[OK]  TorchScript model found: {pt_path}")

    print(
        f"\n  Starting on  http://{args.host}:{args.port}\n"
        f"  Phone access: http://<your-LAN-IP>:{args.port}\n"
        f"  (Run 'ipconfig' to find your Wi-Fi IPv4 address)\n"
    )

    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn not installed.\n  pip install uvicorn[standard]")
        sys.exit(1)

    # Run uvicorn with an absolute app path so it resolves imports correctly
    # regardless of where the user runs the script from.
    uvicorn.run(
        "webapp.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,
        # Let uvicorn find the project root via sys.path rather than cwd.
        app_dir=str(_PROJECT_ROOT),
        log_level="debug",
    )


if __name__ == "__main__":
    main()

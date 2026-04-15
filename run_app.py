#!/usr/bin/env python3
"""
run_app.py
──────────
Startup helper for the FastAPI OMR app.

Run from the project root (the directory that contains omr_detector_enhanced_v3.py
and bubble_classifier_v3.onnx):

    python run_app.py               # default: 0.0.0.0:8000
    python run_app.py --port 5000   # custom port
    python run_app.py --reload      # auto-reload on code changes (dev mode)

Then open  http://<your-LAN-IP>:8000  on a phone connected to the same network.
Find your LAN IP with:  ipconfig  (look for "IPv4 Address" under your Wi-Fi adapter)
"""

import argparse
import os
import pathlib
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Start the OMR FastAPI server")
    ap.add_argument("--host",   default="0.0.0.0",      help="Bind host (default: 0.0.0.0)")
    ap.add_argument("--port",   type=int, default=8000,  help="Port (default: 8000)")
    ap.add_argument("--reload", action="store_true",     help="Auto-reload on changes")
    ap.add_argument("--workers", type=int, default=1,    help="Worker processes (default: 1)")
    args = ap.parse_args()

    # ── Ensure we are running from the project root ──────────────────────────
    project_root = pathlib.Path(__file__).parent.resolve()
    os.chdir(project_root)

    # ── Verify the model file is reachable ───────────────────────────────────
    onnx_path = project_root / "bubble_classifier_v3.onnx"
    pt_path   = project_root / "bubble_classifier_v3.pt"
    if not onnx_path.exists() and not pt_path.exists():
        print(
            "[WARN] Neither bubble_classifier_v3.onnx nor bubble_classifier_v3.pt\n"
            "       found in the project root. The detector will fall back to fill-ratio\n"
            "       classification (lower accuracy). Place the model file here:\n"
            f"       {project_root}\n"
        )
    elif onnx_path.exists():
        print(f"[OK]  ONNX model found: {onnx_path}")
    else:
        print(f"[OK]  TorchScript model found: {pt_path}")

    # ── Print LAN access hint ─────────────────────────────────────────────────
    print(
        f"\n  OMR app starting on  http://{args.host}:{args.port}\n"
        f"  Open on your phone:  http://<your-LAN-IP>:{args.port}\n"
        f"  (Run 'ipconfig' to find your Wi-Fi IPv4 address)\n"
    )

    # ── Launch uvicorn ────────────────────────────────────────────────────────
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn not installed. Run:\n  pip install uvicorn[standard]")
        sys.exit(1)

    uvicorn.run(
        "webapp.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,
    )


if __name__ == "__main__":
    main()

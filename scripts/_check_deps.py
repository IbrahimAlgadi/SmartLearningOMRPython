# ── repo root on sys.path (script lives in scripts/) ──────────────────────────
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
print("Python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
except ImportError as e:
    print("torch: NOT FOUND -", e)
try:
    import onnxruntime
    print("onnxruntime:", onnxruntime.__version__)
except ImportError as e:
    print("onnxruntime: NOT FOUND -", e)
try:
    import onnx
    print("onnx:", onnx.__version__)
except ImportError as e:
    print("onnx: NOT FOUND -", e)
try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except ImportError as e:
    print("torchvision: NOT FOUND -", e)

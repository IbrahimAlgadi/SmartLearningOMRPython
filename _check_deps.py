import sys
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

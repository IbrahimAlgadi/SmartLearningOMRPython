import sys
print("step 1", flush=True)
import numpy as np
print("step 2 numpy ok", flush=True)
import cv2
print("step 3 cv2 ok", flush=True)
import torch
print("step 4 torch ok", torch.__version__, flush=True)
print("CUDA:", torch.cuda.is_available(), flush=True)
import pathlib
npz = pathlib.Path(r'd:\Mine\e_learning\Grade_OMR\omr_python\dataset_v3\crops_train.npz')
print("step 5 loading npz", npz, flush=True)
data = np.load(str(npz))
print("step 6 npz keys:", data.files, flush=True)
imgs = data["images"]
print("step 7 images shape:", imgs.shape, flush=True)
lbls = data["labels"]
print("step 8 labels shape:", lbls.shape, flush=True)
print("DONE", flush=True)

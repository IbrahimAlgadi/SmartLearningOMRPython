"""Re-export bubble_classifier_v3.pt -> bubble_classifier_v3.onnx"""
import pathlib, torch, numpy as np

# Load the TorchScript model
pt_path   = pathlib.Path("bubble_classifier_v3.pt")
onnx_path = pathlib.Path("bubble_classifier_v3.onnx")

model = torch.jit.load(str(pt_path), map_location="cpu")
model.eval()

dummy = torch.zeros(1, 1, 32, 32)

torch.onnx.export(
    model, dummy, str(onnx_path),
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
    dynamo=False,
)
print(f"Saved ONNX: {onnx_path}  ({onnx_path.stat().st_size // 1024} KB)")

# Verify
import onnxruntime as ort
sess = ort.InferenceSession(str(onnx_path))
out  = sess.run(None, {"input": dummy.numpy()})[0]
print(f"ONNX sanity check: shape={out.shape}  logits={out[0]}")
print("Done.")

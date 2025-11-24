import time
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "simcse_onnx/model.onnx"
model_int8 = "simcse_onnx/model_int8.onnx"

print("🔄 Converting SimCSE ONNX → INT8 ...")

# Fake progress bar (vì quantize chạy rất nhanh)
for i in range(0, 101, 5):
    print(f"⏳ Progress: {i}%", end="\r")
    time.sleep(0.05)

quantize_dynamic(
    model_input=model_fp32,
    model_output=model_int8,
    weight_type=QuantType.QInt8
)

print("\n✅ DONE — Saved:", model_int8)

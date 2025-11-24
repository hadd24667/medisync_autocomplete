from optimum.exporters.onnx import main_export
import os

MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
ONNX_DIR = "simcse_onnx"

os.makedirs(ONNX_DIR, exist_ok=True)

main_export(
    model_name_or_path=MODEL_NAME,
    output=ONNX_DIR,
    task="feature-extraction",
    opset=14,
    device="cpu",
    no_post_process=True,
    framework="pt",
)

print("\n✓ Export SimCSE ONNX thành công → simcse_onnx/\n")

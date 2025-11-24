import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from unidecode import unidecode

SESSION = ort.InferenceSession("simcse_onnx/model.onnx")
tokenizer = AutoTokenizer.from_pretrained("simcse_onnx")

# Lấy danh sách input hợp lệ từ model ONNX
VALID_INPUTS = {i.name for i in SESSION.get_inputs()}
print("ONNX inputs:", VALID_INPUTS)


def embed(text):
    text = unidecode(text.lower().strip())

    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=128)

    # Bỏ token_type_ids vì PhoBERT không dùng
    ort_inputs = {k: v for k, v in inputs.items() if k in VALID_INPUTS}

    outputs = SESSION.run(None, ort_inputs)

    # ONNX output 0 = last_hidden_state
    last_hidden = outputs[0]      # (1, seq_len, 768)

    # Mean pooling
    emb = last_hidden.mean(axis=1)

    return emb.squeeze().astype("float32")


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Test nhanh
a = embed("paracetamol")
b = embed("acetaminophen")

print("Cosine =", cosine(a, b))

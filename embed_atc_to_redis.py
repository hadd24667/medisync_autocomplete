import json
import redis
import numpy as np
from unidecode import unidecode
from transformers import AutoTokenizer
import onnxruntime as ort

# 1) Load metadata
META_FILE = "atc_synonyms_meta.json"

# 2) Load SimCSE
ONNX_PATH = "simcse_onnx/model.onnx"
SESSION = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained("simcse_onnx")

# 3) Redis
r = redis.Redis(host="localhost", port=6379, decode_responses=False)

# ---- EMBEDDING FUNCTION ----
def embed_simcse(text):
    text = unidecode(text.lower().strip())
    tokens = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=64,
        padding="max_length",
    )
    out = SESSION.run(
        None,
        {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
    )

    # out[0] shape: (1, 4, 768)
    hidden = out[0][0]          # (4, 768)

    # Mean pooling over 4 layers → (768,)
    vec = hidden.mean(axis=0).astype("float32")

    # Normalize
    return vec / (np.linalg.norm(vec) + 1e-8)


# ---- MAIN ----
def main():
    print("Loading metadata...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    total = len(meta)
    print(f"Total ATC codes: {total}")

    for idx, (code, item) in enumerate(meta.items()):
        # Embed bằng INN là đủ (INN là chuẩn nhất)
        inn_text = item["inn"]

        if not inn_text:
            print(f"[WARN] {code} không có INN, skip")
            continue

        emb = embed_simcse(inn_text)

        # Save to Redis as hex
        r.set(
            f"atc:vec:{code}".encode("utf-8"),
            emb.tobytes().hex()
        )
        
        if idx % 100 == 0:
            print(f"{idx}/{total} embedded...")

    print("✓ DONE — tất cả vector đã lưu vào Redis!")

if __name__ == "__main__":
    main()

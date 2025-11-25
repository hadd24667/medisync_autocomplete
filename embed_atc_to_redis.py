import json
import redis
import numpy as np
from unidecode import unidecode
from transformers import AutoTokenizer
import onnxruntime as ort


META_FILE = "atc_synonyms_meta_cleaned.json"

# --- SimCSE INT8 ---
ONNX_PATH = "simcse_onnx/model_int8.onnx"
SESSION = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained("simcse_onnx")

# --- Redis ---
r = redis.Redis(host="localhost", port=6379, decode_responses=False)


def embed_simcse(text):
    text = unidecode(text.lower().strip())

    tokens = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=64,
        padding="max_length",
    )

    out = SESSION.run(None, {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    })

    hidden = out[0]                        # (1, seq_len, 768)
    mask = tokens["attention_mask"]        # (1, seq_len)

    masked = hidden * mask[:, :, None]     # apply mask
    vec = masked.sum(axis=1) / mask.sum(axis=1)[:, None]
    vec = vec[0].astype("float32")

    return vec / (np.linalg.norm(vec) + 1e-8)



def main():
    print("Loading metadata...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    total = len(meta)
    print(f"Total ATC codes: {total}")

    for idx, (code, item) in enumerate(meta.items()):
        inn_text = item.get("inn", "")
        if not inn_text:
            print(f"[WARN] {code} thiếu INN, skip")
            continue

        emb = embed_simcse(inn_text)
        r.set(f"atc:vec:{code}", emb.tobytes().hex())

        if idx % 100 == 0:
            print(f"{idx}/{total} embedded...")

    print("✓ DONE — tất cả vector đã được embed bằng INT8 và lưu vào Redis!")


if __name__ == "__main__":
    main()

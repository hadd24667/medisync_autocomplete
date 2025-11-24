# ======================================================
# SMARTEMR — TRAIN FEATURE EXTRACTOR (Final v5 + tqdm)
# ======================================================

import json
import redis
import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer
from unidecode import unidecode
from tqdm import tqdm   # <── progress bar

# ----------------------------------------
# Load ONNX SimCSE
# ----------------------------------------
ONNX = "simcse_onnx/model_int8.onnx"
SESS = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained("simcse_onnx")


def embed(text):
    text = unidecode(text.lower().strip())
    toks = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=64,
        padding="max_length",
    )

    out = SESS.run(
        None,
        {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }
    )

    hidden = out[0]              # shape (1, seq, 768)
    mask = toks["attention_mask"]

    masked = hidden * mask[:, :, None]
    vec = masked.sum(axis=1) / mask.sum(axis=1)[:, None]

    vec = vec[0]
    return vec / (np.linalg.norm(vec) + 1e-8)


def cosine(a, b):
    return float(np.dot(a, b))


# ----------------------------------------
# Redis: metadata + vec
# ----------------------------------------
r = redis.Redis(host="localhost", port=6379, decode_responses=False)


def get_meta(code):
    raw = r.get(f"atc:meta:{code}")
    return json.loads(raw) if raw else {}


def get_vec(code):
    raw = r.get(f"atc:vec:{code}")
    arr = np.frombuffer(bytes.fromhex(raw.decode()), dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-8)


# ----------------------------------------
# Extract 7 features
# ----------------------------------------
def extract_feats(query, code):
    q = unidecode(query.lower())
    meta = get_meta(code)

    # semantic
    q_vec = embed(q)
    d_vec = get_vec(code)
    f_sem = cosine(q_vec, d_vec)

    # fields
    inn = unidecode(meta.get("inn", "").lower())
    brands = [unidecode(x.lower()) for x in meta.get("brand", [])]
    syns = [unidecode(x.lower()) for x in meta.get("synonyms", [])]

    f_inn_sub = 1.0 if q in inn else 0.0
    f_brand_sub = 1.0 if any(q in b for b in brands) else 0.0
    f_syn_sub = 1.0 if any(q in s for s in syns) else 0.0

    f_len_ratio = min(len(q) / (len(inn) + 1e-6), 1.0)

    overlap = len(set(q) & set(inn))
    f_overlap = overlap / max(len(q), 1)

    f_edit = 1 if inn and q[:4] == inn[:4] else 0

    return [
        f_sem,
        f_inn_sub,
        f_brand_sub,
        f_syn_sub,
        f_len_ratio,
        f_overlap,
        f_edit
    ]


# ----------------------------------------
# Build Train Feature CSV (with TQDM)
# ----------------------------------------
def main():
    df = pd.read_csv("train_pairs.csv")

    rows = []
    print(f"🔍 Extracting {len(df)} feature rows...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), ncols=90):
        feats = extract_feats(row["query"], row["atc_code"])

        rows.append([
            row["group_id"],
            row["query"],
            row["atc_code"],
            *feats,
            row["label"]
        ])

    cols = [
        "group_id", "query", "atc_code",
        "f_semantic",
        "f_inn_sub",
        "f_brand_sub",
        "f_syn_sub",
        "f_len_ratio",
        "f_char_overlap",
        "f_edit_prefix",
        "label"
    ]

    pd.DataFrame(rows, columns=cols).to_csv("train_features.csv", index=False)
    print("\n✓ DONE — train_features.csv generated!")


if __name__ == "__main__":
    main()

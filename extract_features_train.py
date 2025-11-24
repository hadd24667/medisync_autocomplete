# ======================================================
# SMARTEMR — TRAIN FEATURE EXTRACTOR (v2)
# ======================================================

import json
import redis
import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer
from unidecode import unidecode

# -------- ONNX SimCSE Loader ----------
ONNX = "simcse_onnx/model.onnx"
SESS = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained("simcse_onnx")


def embed(text):
    text = unidecode(text.lower().strip())
    inp = tokenizer(text, return_tensors="np", truncation=True, max_length=64)
    out = SESS.run(None, {"input_ids": inp["input_ids"],
                          "attention_mask": inp["attention_mask"]})
    vec = out[0][0]
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.astype("float32")


def cosine(a, b):
    return float(np.dot(a, b))


# -------- Redis ----------
r = redis.Redis(host="localhost", port=6379)


def load_meta(code):
    raw = r.get(f"atc:meta:{code}")
    return json.loads(raw) if raw else {}


def load_vec(code):
    raw = r.get(f"atc:vec:{code}")
    arr = np.frombuffer(bytes.fromhex(raw.decode()), dtype=np.float32)
    arr = arr / (np.linalg.norm(arr) + 1e-8)
    return arr


# -------- Feature Engineering ----------
def extract_feats(query, code):
    q = unidecode(query.lower())
    meta = load_meta(code)
    d_vec = load_vec(code)
    q_vec = embed(q)

    f_semantic = cosine(q_vec, d_vec)

    inn = unidecode(meta.get("inn", "").lower())
    brands = [unidecode(b.lower()) for b in meta.get("brand", [])]
    synonyms = [unidecode(s.lower()) for s in meta.get("synonyms", [])]
    forms = [unidecode(f.lower()) for f in meta.get("forms", [])]
    routes = [unidecode(r.lower()) for r in meta.get("routes", [])]

    # exact / substring
    f_inn_exact = 1 if q == inn else 0
    f_brand_exact = 1 if any(q == b for b in brands) else 0
    f_syn_exact = 1 if any(q == s for s in synonyms) else 0

    f_inn_sub = 1 if q in inn else 0
    f_brand_sub = 1 if any(q in b for b in brands) else 0
    f_syn_sub = 1 if any(q in s for s in synonyms) else 0

    # length ratio
    f_len_ratio = min(len(q) / (len(inn) + 1e-6), 1)

    # char overlap
    f_char_overlap = len(set(q) & set(inn)) / max(len(q), 1)

    # form / route match
    f_form = 1 if any(ff in q for ff in forms) else 0
    f_route = 1 if any(rr in q for rr in routes) else 0

    return [
        f_semantic,
        f_inn_exact,
        f_brand_exact,
        f_syn_exact,
        f_inn_sub,
        f_brand_sub,
        f_syn_sub,
        f_len_ratio,
        f_char_overlap,
        f_form,
        f_route,
        # no f_class/ped/contra in v2 → có thể thêm sau
    ]


# -------- Build Train Table ----------
def main():
    df = pd.read_csv("train_pairs.csv")
    rows = []

    for _, row in df.iterrows():
        feats = extract_feats(row["query"], row["atc_code"])
        rows.append([row["group_id"], row["query"], row["atc_code"]] + feats + [row["label"]])

    cols = ["group_id", "query", "atc_code",
            "f_semantic",
            "f_inn_exact",
            "f_brand_exact",
            "f_syn_exact",
            "f_inn_sub",
            "f_brand_sub",
            "f_syn_sub",
            "f_len_ratio",
            "f_char_overlap",
            "f_form",
            "f_route",
            "label"]

    pd.DataFrame(rows, columns=cols).to_csv("train_features.csv", index=False)
    print("✓ DONE — train_features.csv generated.")


if __name__ == "__main__":
    main()

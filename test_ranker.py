import json
import redis
import numpy as np
from unidecode import unidecode
from onnxruntime import InferenceSession
from transformers import AutoTokenizer
import lightgbm as lgb
import time


# ======================================================
# CONFIG
# ======================================================

REDIS_HOST = "localhost"
REDIS_PORT = 6379

ATC_EMB_PREFIX = "atc:vec:"     
ATC_META_PREFIX = "atc:meta:"   
MODEL_FILE = "lgbm_atc_ranker.txt"

SIMCSE_ONNX_PATH = "simcse_onnx/model.onnx"   
SIMCSE_TOKENIZER_PATH = "simcse_onnx"


# ======================================================
# INIT
# ======================================================

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

lgbm_model = lgb.Booster(model_file=MODEL_FILE)

onnx_session = InferenceSession(SIMCSE_ONNX_PATH, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(SIMCSE_TOKENIZER_PATH)

ATC_EMB_CACHE = {}
ATC_META_CACHE = {}


# ======================================================
# UTILS
# ======================================================

def normalize(x: str) -> str:
    return unidecode(x.lower().strip()) if x else ""


def embed_query(text: str) -> np.ndarray:
    """
    Embed query 1 lần bằng ONNX + ép shape về (768,).
    """
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="np",
    )

    ort_inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

    out = onnx_session.run(None, ort_inputs)
    emb = np.array(out[0]).reshape(-1)  # ép về vector 1D

    norm = np.linalg.norm(emb)
    return emb if norm == 0 else emb / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine safe-check: ép về vector 1D, fallback nếu shape lệch.
    """
    a = np.array(a).reshape(-1)
    b = np.array(b).reshape(-1)

    if a.shape != b.shape:
        return 0.0

    return float(np.dot(a, b))


def get_atc_emb(code: str, dim: int) -> np.ndarray:
    """
    Lấy embedding ATC từ Redis + reshape + normalize.
    """
    if code in ATC_EMB_CACHE:
        return ATC_EMB_CACHE[code]

    key = (ATC_EMB_PREFIX + code).encode("utf-8")
    v = r.get(key)

    if v is None:
        arr = np.zeros(dim, dtype=np.float32)
        ATC_EMB_CACHE[code] = arr
        return arr

    try:
        raw = bytes.fromhex(v.decode("ascii"))
        arr = np.frombuffer(raw, dtype=np.float32)

        # ép về (dim,)
        if arr.shape[0] != dim:
            arr = np.zeros(dim, dtype=np.float32)
        arr = arr.reshape(-1)

        # normalize
        n = np.linalg.norm(arr)
        if n > 0:
            arr = arr / n

        ATC_EMB_CACHE[code] = arr
        return arr

    except:
        arr = np.zeros(dim, dtype=np.float32)
        ATC_EMB_CACHE[code] = arr
        return arr


def get_atc_meta(code: str) -> dict:
    if code in ATC_META_CACHE:
        return ATC_META_CACHE[code]

    key = (ATC_META_PREFIX + code).encode("utf-8")
    v = r.get(key)

    if v is None:
        ATC_META_CACHE[code] = {}
        return {}

    try:
        meta = json.loads(v.decode("utf-8"))
    except:
        meta = {}

    ATC_META_CACHE[code] = meta
    return meta


# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_features_from_emb(query, q_emb, atc_code):
    nq = normalize(query)
    meta = get_atc_meta(atc_code)

    # ====== semantic ======
    atc_emb = get_atc_emb(atc_code, q_emb.shape[0])
    f_semantic = cosine(q_emb, atc_emb)

    # ====== lexical lists ======
    inn_list = []
    if isinstance(meta.get("inn"), str):
        inn_list = [normalize(x) for x in meta["inn"].split(",")]

    brand_list = [normalize(x) for x in meta.get("brand_names", [])]
    synonym_list = [normalize(x) for x in meta.get("synonyms", [])]
    alias_list = [normalize(x) for x in meta.get("aliases", [])]

    specialties = [normalize(x) for x in meta.get("specialties", [])]
    risk_tags  = [normalize(x) for x in meta.get("risk_tags", [])]

    # ====== feature 1: inn exact ======
    f_inn_exact = 1 if nq in inn_list else 0

    # ====== feature 2: brand exact ======
    f_brand = 1 if nq in brand_list else 0

    # ====== feature 3: synonym exact ======
    f_syn = 1 if nq in synonym_list else 0

    # ====== feature 4: alias exact ======
    f_alias = 1 if nq in alias_list else 0

    # ====== feature 5: substring in inn ======
    f_inn_sub = 1 if any(nq in inn for inn in inn_list) else 0

    # ====== feature 6: substring in brand ======
    f_brand_sub = 1 if any(nq in b for b in brand_list) else 0

    # ====== feature 7: specialty match ======
    f_spec = 1 if nq in specialties else 0

    # ====== feature 8: risk tag match ======
    f_risk = 1 if nq in risk_tags else 0

    # ====== feature 9: length ratio ======
    inn_joined = " ".join(inn_list)
    if len(inn_joined) > 0:
        f_len_ratio = min(len(nq) / (len(inn_joined) + 1e-6), 1.0)
    else:
        f_len_ratio = 0.0

    # ====== feature 10: char overlap ======
    if inn_joined:
        overlap = len(set(nq) & set(inn_joined))
        f_char_overlap = overlap / max(len(nq), 1)
    else:
        f_char_overlap = 0.0

    # ====== feature 11: edit-like (compare first 4 chars) ======
    if inn_list:
        f_edit_sim = 1 if nq[:4] == inn_list[0][:4] else 0
    else:
        f_edit_sim = 0

    # ====== assemble feature vector ======
    return np.array([
        f_semantic,
        f_inn_exact,
        f_brand,
        f_syn,
        f_alias,
        f_inn_sub,
        f_brand_sub,
        f_spec,
        f_risk,
        f_len_ratio,
        f_char_overlap,
        f_edit_sim,
    ], dtype=np.float32)



# ======================================================
# MAIN RANK FUNCTION
# ======================================================

def rank_candidates(query, codes, top_k=10):
    if not codes:
        return []

    q_emb = embed_query(query)

    feat_list = []
    valid_codes = []

    for c in codes:
        feats = extract_features_from_emb(query, q_emb, c)
        feat_list.append(feats)
        valid_codes.append(c)

    X = np.stack(feat_list, axis=0)
    preds = lgbm_model.predict(X)

    pairs = list(zip(valid_codes, preds))
    pairs.sort(key=lambda x: x[1], reverse=True)

    return pairs[:top_k]


# ======================================================
# DEMO
# ======================================================

if __name__ == "__main__":
    test_queries = [
        "para 500",
        "creon",
        "vit b5 5mg",
        "nicorette",
        "siro ho",
    ]

    sample = [
        "N02BE01",   # paracetamol
        "N02BE51",
        "N02BE03",   # paracetamol + caffeine
        "R05DA04",   # codeine + antihistamine
        "A11HA04",   # vitamin b5
    ]

    for q in test_queries:
        print("\n=========================")
        print("🔍 Query:", q)

        t0 = time.perf_counter()
        out = rank_candidates(q, sample, top_k=5)
        t1 = time.perf_counter()

        print(f"⏱ Time: {(t1 - t0)*1000:.2f} ms")
        for code, score in out:
            print(f"{code:10s} → {score:.4f}")

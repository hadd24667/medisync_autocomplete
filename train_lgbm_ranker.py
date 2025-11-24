import json
import redis
import numpy as np
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
from sklearn.metrics import ndcg_score
import lightgbm as lgb
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

# ======================================================
# CONFIG
# ======================================================

REDIS_HOST = "localhost"
REDIS_PORT = 6379

ATC_META_PREFIX = "atc:meta:"        # chứa metadata ATC (JSON)
ATC_EMB_PREFIX  = "atc:vec:"         # chứa vector embedding ATC (hex float32)

TRAIN_FILE = "train_pairs.csv"
MODEL_FILE = "lgbm_atc_ranker.txt"

SIMCSE_ONNX_PATH = "simcse_onnx/model.onnx"   # sửa lại path nếu khác
SIMCSE_TOKENIZER_PATH = "simcse_onnx"         # hoặc "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"


# ======================================================
# LOAD REDIS + SimCSE ONNX
# ======================================================

print("🔌 Connecting Redis...")
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

print("🤖 Loading SimCSE PhoBERT ONNX model...")
session = InferenceSession(SIMCSE_ONNX_PATH)
tokenizer = AutoTokenizer.from_pretrained(SIMCSE_TOKENIZER_PATH)

# lấy tên input của ONNX (thường là input_ids, attention_mask)
INPUT_NAMES = {inp.name for inp in session.get_inputs()}


def normalize(s: str) -> str:
    return unidecode(str(s).lower().strip())


def embed_text(text: str) -> np.ndarray:
    """Embed 1 câu bằng SimCSE PhoBERT ONNX → vector float32."""
    text_norm = normalize(text)
    encoded = tokenizer(
        text_norm,
        return_tensors="np",
        truncation=True,
        max_length=128,
    )
    # chỉ giữ các key mà ONNX thực sự có (tránh lỗi token_type_ids)
    ort_inputs = {k: v for k, v in encoded.items() if k in INPUT_NAMES}

    outputs = session.run(None, ort_inputs)
    last_hidden = outputs[0]          # (1, seq_len, hidden_size)
    emb = last_hidden.mean(axis=1)    # mean pooling → (1, hidden)
    return emb.squeeze().astype("float32")


def load_atc_embedding(code: str, like: np.ndarray) -> np.ndarray:
    """Load embedding ATC từ Redis, decode từ hex (hoặc JSON fallback)."""
    key = ATC_EMB_PREFIX + code
    v = r.get(key)
    if v is None:
        return np.zeros_like(like)

    # thử decode hex trước
    try:
        raw = bytes.fromhex(v)
        arr = np.frombuffer(raw, dtype=np.float32)
        # nếu kích thước khác với query vector thì fallback
        if arr.shape != like.shape:
            raise ValueError("dim mismatch")
        return arr
    except Exception:
        # fallback: assume JSON list
        try:
            arr = np.array(json.loads(v), dtype=np.float32)
            if arr.shape != like.shape:
                return np.zeros_like(like)
            return arr
        except Exception:
            return np.zeros_like(like)


def cosine(a, b) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ======================================================
# FEATURE EXTRACTION
# ======================================================

def extract_features(query: str, atc_code: str):
    """
    Sinh 12 feature cho LightGBM Ranker.
    Chỉ dùng metadata sạch: inn, brand, synonyms, aliases, specialties, risk_tags.
    Không dùng trực tiếp form/route/drug_class làm query để tránh nhiễm.
    """

    nq = normalize(query)

    # 1) Query embedding (SimCSE)
    q_emb = embed_text(query)

    # 2) ATC embedding (từ Redis)
    atc_emb = load_atc_embedding(atc_code, like=q_emb)

    # 3) Metadata ATC
    key_meta = ATC_META_PREFIX + atc_code
    raw = r.get(key_meta)
    if raw is None:
        # fallback nếu thiếu metadata
        inn_list = []
        brand_list = []
        synonym_list = []
        specialties = []
        risk_tags = []
        alias_list = []
    else:
        meta = json.loads(raw)

        inn_raw = meta.get("inn", "") or ""
        inn_list = [normalize(x) for x in inn_raw.split(",")] if inn_raw else []

        brand_list = [normalize(x) for x in meta.get("brand", [])]
        synonym_list = [normalize(x) for x in meta.get("synonyms", [])]
        specialties = [normalize(x) for x in meta.get("specialties", [])]
        risk_tags = [normalize(x) for x in meta.get("risk_tags", [])]
        alias_list = [normalize(x) for x in meta.get("aliases", [])]

    # -----------------------
    # FEATURE SET
    # -----------------------

    # 1. semantic / embedding similarity
    f_semantic = cosine(q_emb, atc_emb)

    # 2. inn exact match
    f_inn_exact = 1 if nq in inn_list else 0

    # 3. brand exact match
    f_brand = 1 if nq in brand_list else 0

    # 4. synonym exact match
    f_syn = 1 if nq in synonym_list else 0

    # 5. alias exact match
    f_alias = 1 if nq in alias_list else 0

    # 6. partial substring match với INN (para ⊂ paracetamol)
    f_inn_sub = 1 if any(nq in inn for inn in inn_list) else 0

    # 7. partial substring match brand (muci ⊂ mucinex)
    f_brand_sub = 1 if any(nq in b for b in brand_list) else 0

    # 8. specialty match (sau này nếu query = "respiratory", "neurology"...)
    f_spec = 1 if nq in specialties else 0

    # 9. risk tag match (query kiểu: "suy gan", "hepatic"... có thể map sau)
    f_risk = 1 if nq in risk_tags else 0

    # 10. length ratio giữa query và full INN string
    inn_joined = " ".join(inn_list)
    if len(inn_joined) > 0:
        f_len_ratio = min(len(nq) / (len(inn_joined) + 1e-6), 1.0)
    else:
        f_len_ratio = 0.0

    # 11. character overlap giữa query và INN (thô nhưng rẻ)
    if inn_joined:
        overlap = len(set(nq) & set(inn_joined))
        f_char_overlap = overlap / max(len(nq), 1)
    else:
        f_char_overlap = 0.0

    # 12. "edit-like" feature: so 4 ký tự đầu với INN đầu tiên
    if inn_list:
        f_edit_sim = 1 if nq[:4] == inn_list[0][:4] else 0
    else:
        f_edit_sim = 0

    return [
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
    ]


# ======================================================
# LOAD TRAINING DATA
# ======================================================

print("📥 Loading training pairs...")
df = pd.read_csv(TRAIN_FILE)

group_sizes = df.groupby("group_id").size().values

print(f"→ {len(df)} rows")
print(f"→ {len(group_sizes)} groups")


# ======================================================
# BUILD FEATURE MATRIX
# ======================================================

features = []
labels = []

print("🔨 Extracting features...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    f = extract_features(row["query"], row["atc_code"])
    features.append(f)
    labels.append(row["label"])

X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.int32)


# ======================================================
# TRAIN LIGHTGBM RANKER
# ======================================================

print("🚀 Training LightGBM Ranker (LambdaRank)...")

train_data = lgb.Dataset(X, y, group=group_sizes)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_pre_filter": False,
}

model = lgb.train(
    params=params,
    train_set=train_data,
    num_boost_round=300,
)

model.save_model(MODEL_FILE)
print(f"🎉 Saved model → {MODEL_FILE}")


# ======================================================
# LOCAL EVALUATION (NDCG)
# ======================================================

def compute_ndcg(df_eval, pred):
    df_eval = df_eval.copy()
    df_eval["pred"] = pred

    ndcg5 = []
    ndcg10 = []

    for _, g in df_eval.groupby("group_id"):

        #  Skip group chỉ có 1 sample
        if len(g) < 2:
            continue

        y_true = g["label"].values.reshape(1, -1)
        y_score = g["pred"].values.reshape(1, -1)

        ndcg5.append(ndcg_score(y_true, y_score, k=5))
        ndcg10.append(ndcg_score(y_true, y_score, k=10))

    return float(np.mean(ndcg5)), float(np.mean(ndcg10))



print("📈 Running evaluation on training set (for sanity check)...")
pred = model.predict(X)
ndcg5, ndcg10 = compute_ndcg(df, pred)

print(f"✔ NDCG@5  = {ndcg5:.4f}")
print(f"✔ NDCG@10 = {ndcg10:.4f}")


# ======================================================
# DEMO INFERENCE
# ======================================================

def rank_candidates(query, atc_list):
    feats = [extract_features(query, c) for c in atc_list]
    preds = model.predict(np.array(feats, dtype=np.float32))
    result = sorted(zip(atc_list, preds), key=lambda x: x[1], reverse=True)
    return result


if __name__ == "__main__":
    print("\n🔍 DEMO: Ranking for query 'para'")
    sample_atc = ["N02BE01", "A09AA02", "N07BA01", "R05CA03"]
    print(rank_candidates("para", sample_atc))

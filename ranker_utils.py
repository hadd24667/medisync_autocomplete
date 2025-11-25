# ranker_utils.py
# Shared utilities for ATC ranker and autocomplete

import json
import redis
import numpy as np
from unidecode import unidecode
from transformers import AutoTokenizer
from onnxruntime import InferenceSession

# -------------------------------
# Global config
# -------------------------------
REDIS_HOST = "localhost"
REDIS_PORT = 6379
ONNX_PATH = "simcse_onnx/model_int8.onnx"
TOKENIZER_PATH = "simcse_onnx"

# -------------------------------
# Init shared resources
# -------------------------------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

sess = InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

ZERO_VEC = np.zeros(768, dtype=np.float32)

_query_emb_cache = {}


def embed_query(text: str) -> np.ndarray:
    """
    Embed query 1 lần, có cache theo query_norm
    """
    text_norm = unidecode(text.lower().strip())
    if text_norm in _query_emb_cache:
        return _query_emb_cache[text_norm]

    toks = tokenizer(
        text_norm,
        return_tensors="np",
        truncation=True,
        max_length=64,
        padding="max_length",
    )

    out = sess.run(
        None,
        {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        },
    )

    hidden = out[0]  # (1, seq, 768)
    mask = toks["attention_mask"]  # (1, seq)

    masked = hidden * mask[:, :, None]
    vec = masked.sum(axis=1) / mask.sum(axis=1)[:, None]
    v = vec[0]
    v = v / (np.linalg.norm(v) + 1e-8)

    _query_emb_cache[text_norm] = v
    return v


def get_vec(code: str, dim: int = 768) -> np.ndarray:
    """
    Lấy embedding thuốc từ Redis
    """
    key = f"atc:vec:{code}".encode("utf-8")
    raw = r.get(key)
    if raw is None:
        return ZERO_VEC.copy()
    try:
        v = np.frombuffer(bytes.fromhex(raw.decode()), dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        return v
    except Exception:
        return ZERO_VEC.copy()


def get_meta(code: str) -> dict:
    """
    Lấy metadata ATC từ Redis
    """
    key = f"atc:meta:{code}".encode("utf-8")
    raw = r.get(key)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}

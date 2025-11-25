# ===========================================================
# Medisync — Tier-2 ATC Ranker Training (LightGBM Ranker)
# Version: v8 (rich feature pipeline)
# ===========================================================

import math
import re
from collections import defaultdict
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from unidecode import unidecode

from synonym_profile import (
    normalize_text,
    build_synonym_profiles,
    pick_matched_syn_from_query,
)


# dùng lại utils sẵn có (SimCSE PhoBERT INT8 + Redis)
from ranker_utils import get_meta, get_vec, embed_query


# -----------------------------------------------------------
# 0. FEATURE LIST (PHẢI GIỐNG HỆ INFER)
# -----------------------------------------------------------
FEATURE_COLS = [
    "f_semantic",
    "f_semantic_sq",
    "f_inn_sub",
    "f_brand_sub",
    "f_syn_sub",
    "f_len_ratio",
    "f_char_overlap",
    "f_has_number",
    "f_match_brand",
    "f_match_inn",
    "f_match_alias",
    "f_match_exact",
    "f_syn_ambiguity",
    "f_dose_match",
    "f_dose_diff_min",
    "f_route_match",
    "f_is_combo_drug",
    "f_is_pediatric_form",
    "f_risk_hepatic",
]


# -----------------------------------------------------------
# 1. HELPERS CHUNG (NORMALIZE, DOSE, ROUTE)
# -----------------------------------------------------------

def parse_numbers_from_text(text: str):
    if not text:
        return []
    nums = re.findall(r'\d+(?:\.\d+)?', text)
    out = []
    for x in nums:
        try:
            out.append(float(x))
        except Exception:
            continue
    return out


def compute_dose_features(q_norm: str, meta: dict):
    """
    f_dose_match, f_dose_diff_min
    - Dùng trực tiếp meta["doses"] đã được chuẩn hoá sẵn (ví dụ: ["80", "80mg", "500", "500mg"])
    """
    q_nums = parse_numbers_from_text(q_norm)

    # Lấy trực tiếp dose từ metadata
    dose_tokens = [str(d) for d in (meta.get("doses") or [])]
    dose_nums = parse_numbers_from_text(" ".join(dose_tokens))

    if not q_nums or not dose_nums:
        return 0.0, 0.0

    # 1) Có số nào trùng hẳn không?
    match = any(abs(q - d) < 1e-3 for q in q_nums for d in dose_nums)
    f_dose_match = 1.0 if match else 0.0

    # 2) Độ lệch nhỏ nhất giữa query và các dose
    min_diff = min(abs(q - d) for q in q_nums for d in dose_nums)
    # scale: càng gần càng ~1
    f_dose_diff_min = math.exp(-min_diff / 200.0)

    return f_dose_match, f_dose_diff_min



ROUTE_KEYWORDS = {
    "oral": ["uong"],
    "injection": ["tiem", "chich", "iv", "im"],
    "inhalation": ["xit", "hit", "hit mui"],
    "ophthalmic": ["nho mat"],
    "topical": ["boi", "xoa", "kem"],
}


def compute_route_match(q_norm: str, routes):
    if not routes:
        return 0.0
    routes = [r.lower() for r in routes]
    text = q_norm

    def has_any(keywords):
        return any(k in text for k in keywords)

    score = 0.0
    if any(r in ("oral", "po") for r in routes):
        if has_any(ROUTE_KEYWORDS["oral"]):
            score = 1.0
    if any(r in ("injection", "iv", "im", "sc") for r in routes):
        if has_any(ROUTE_KEYWORDS["injection"]):
            score = 1.0
    if any(r in ("inhalation",) for r in routes):
        if has_any(ROUTE_KEYWORDS["inhalation"]):
            score = 1.0
    if any(r in ("ophthalmic",) for r in routes):
        if has_any(ROUTE_KEYWORDS["ophthalmic"]):
            score = 1.0
    if any(r in ("topical", "dermal") for r in routes):
        if has_any(ROUTE_KEYWORDS["topical"]):
            score = 1.0

    return score


def is_combo_drug(inn_raw: str) -> float:
    if not inn_raw:
        return 0.0
    # tách theo / + , ; (thô nhưng đủ xài)
    parts = re.split(r"[\/+;,]", inn_raw)
    cnt = sum(1 for p in parts if p.strip())
    return 1.0 if cnt > 1 else 0.0


def compute_synonym_features(
    q_norm: str,
    meta: dict,
    brand_index_for_group: dict,
    matched_syn: str | None = None,
):
    """
    Trả về:
      f_inn_sub, f_brand_sub, f_syn_sub,
      f_len_ratio, f_char_overlap,
      f_match_brand, f_match_inn, f_match_alias,
      f_match_exact, f_syn_ambiguity

    - q_norm: query đã normalize (unidecode + lower + strip)
    - brand_index_for_group: dict {syn_norm -> set(atc_code)} phục vụ ambiguity
    - matched_syn: nếu Tier-1 đã xác định được token (vd: 'para', 'tylenol'),
                   bạn có thể truyền xuống; nếu None, hàm sẽ tự pick.
    """
    inn_raw = meta.get("inn") or ""
    brands = meta.get("brand") or []
    syns = meta.get("synonyms") or []

    # -------- substring features dựa trên toàn query --------
    inn_norm = normalize_text(inn_raw)
    brands_norm = [normalize_text(b) for b in brands]
    syns_norm = [normalize_text(s) for s in syns]

    f_inn_sub = 1.0 if inn_norm and q_norm in inn_norm else 0.0
    f_brand_sub = 1.0 if brands_norm and any(q_norm in b for b in brands_norm) else 0.0
    f_syn_sub = 1.0 if syns_norm and any(q_norm in s for s in syns_norm) else 0.0

    # -------- length ratio + char overlap (INN vs query) --------
    if inn_norm:
        f_len_ratio = min(len(q_norm) / (len(inn_norm) + 1e-6), 1.0)
        overlap = len(set(q_norm) & set(inn_norm))
        f_char_overlap = overlap / max(len(q_norm), 1)
    else:
        f_len_ratio = 0.0
        f_char_overlap = 0.0

    # -------- synonym_profile: detect nguồn từ synonym --------
    code = meta.get("atc_code") or meta.get("code") or ""
    profiles = build_synonym_profiles(code, meta)

    # matched_syn:
    #   - nếu được truyền từ ngoài → normalize + check trong profile
    #   - nếu không → tự pick từ query
    if matched_syn:
        matched_norm = normalize_text(matched_syn)
        if matched_norm not in profiles:
            matched_norm = None
    else:
        matched_norm = pick_matched_syn_from_query(q_norm, profiles)

    f_match_brand = f_match_inn = f_match_alias = 0.0

    if matched_norm and matched_norm in profiles:
        src = profiles[matched_norm]["source"]
        if src == "brand":
            f_match_brand = 1.0
        elif src == "inn":
            f_match_inn = 1.0
        else:
            # "code" hoặc các loại khác → coi như alias/viết tắt
            f_match_alias = 1.0

    f_match_exact = 1.0 if (f_match_brand or f_match_inn or f_match_alias) else 0.0

    # -------- ambiguity: chỉ meaningful nếu match brand --------
    if f_match_brand and matched_norm:
        # brand_index_for_group nên được build với key là syn_norm
        codes = brand_index_for_group.get(matched_norm, set())
        n = len(codes) if codes else 1
        f_syn_ambiguity = 1.0 / float(n)
    else:
        f_syn_ambiguity = 1.0

    return (
        f_inn_sub,
        f_brand_sub,
        f_syn_sub,
        f_len_ratio,
        f_char_overlap,
        f_match_brand,
        f_match_inn,
        f_match_alias,
        f_match_exact,
        f_syn_ambiguity,
    )



def build_brand_index_for_group(codes):
    """
    Map brand_norm -> set(code) cho chính group này.
    """
    brand_index = defaultdict(set)
    for code in codes:
        meta = get_meta(code)
        if not meta:
            continue
        for b in meta.get("brand", []) or []:
            b_norm = normalize_text(b)
            if b_norm:
                brand_index[b_norm].add(code)
    return brand_index


def extract_features_for_pair(q_vec: np.ndarray, q_norm: str, code: str, brand_index_for_group: dict):
    """
    Tính full FEATURE_COLS cho (query, code).
    """
    meta = get_meta(code)
    if meta is None:
        return None

    # --- semantic cos(q_vec, d_vec)
    d_vec = get_vec(code)
    if d_vec is None:
        return None

    q = np.asarray(q_vec, dtype=np.float32)
    d = np.asarray(d_vec, dtype=np.float32)
    nq = float(np.linalg.norm(q))
    nd = float(np.linalg.norm(d))
    if nq == 0.0 or nd == 0.0:
        f_sem = 0.0
    else:
        f_sem = float(np.dot(q, d) / (nq * nd))
    f_sem_sq = f_sem * f_sem

    # --- lexical + synonym-aware
    (
        f_inn_sub,
        f_brand_sub,
        f_syn_sub,
        f_len_ratio,
        f_char_overlap,
        f_match_brand,
        f_match_inn,
        f_match_alias,
        f_match_exact,
        f_syn_ambiguity,
    ) = compute_synonym_features(q_norm, meta, brand_index_for_group)

    # --- dose features
    f_dose_match, f_dose_diff_min = compute_dose_features(q_norm, meta)

    # --- route match
    f_route_match = compute_route_match(q_norm, meta.get("routes") or [])

    # --- combo / pediatric / risk
    f_is_combo_drug = is_combo_drug(meta.get("inn") or "")
    f_is_ped = 1.0 if meta.get("is_pediatric_form") else 0.0
    risk_tags = meta.get("risk_tags", []) or []
    f_risk_hepatic = 1.0 if "hepatic_toxic" in risk_tags else 0.0

    # --- has number
    f_has_number = 1.0 if any(ch.isdigit() for ch in q_norm) else 0.0

    return [
        f_sem,
        f_sem_sq,
        f_inn_sub,
        f_brand_sub,
        f_syn_sub,
        f_len_ratio,
        f_char_overlap,
        f_has_number,
        f_match_brand,
        f_match_inn,
        f_match_alias,
        f_match_exact,
        f_syn_ambiguity,
        f_dose_match,
        f_dose_diff_min,
        f_route_match,
        f_is_combo_drug,
        f_is_ped,
        f_risk_hepatic,
    ]


# -----------------------------------------------------------
# 2. MAIN: TẠO train_features_v8.csv + TRAIN LIGHTGBM
# -----------------------------------------------------------
def main():
    T0 = time.time()
    print("=======================================")
    print("⚡ Extracting TRAIN FEATURES v8 ...")
    print("=======================================")

    # ------------------------------------
    # 1) LOAD TRAIN PAIRS
    # ------------------------------------
    t_load = time.time()
    df = pd.read_csv("benchmark_pairs.csv")
    print(f"✓ Loaded train_pairs.csv: {len(df):,} rows ({time.time() - t_load:.2f}s)")

    # Kiểm tra cột
    required_cols = {"group_id", "query", "atc_code", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"train_pairs_v8.csv thiếu cột: {missing}")

    # ------------------------------------
    # 2) BUILD brand_index cho từng group
    # ------------------------------------
    t_brand = time.time()
    group_codes = df.groupby("group_id")["atc_code"].unique().to_dict()
    brand_index_by_group = {
        gid: build_brand_index_for_group(codes)
        for gid, codes in group_codes.items()
    }
    print(f"✓ Built brand_index_for_group for {len(brand_index_by_group)} groups ({time.time() - t_brand:.2f}s)")

    # ------------------------------------
    # 3) INIT OUTPUT STORAGE
    # ------------------------------------
    feats = []
    t_feat = time.time()
    print("🚀 Generating features ...")

    # Cache embedding cho query để tránh embed lại nhiều lần
    embed_cache = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=80, desc="Extract"):
        q = row["query"]
        code = row["atc_code"]
        y = float(row["label"])
        group = row["group_id"]

        # Lấy embed + q_norm từ cache
        if q in embed_cache:
            q_vec, q_norm = embed_cache[q]
        else:
            q_norm = normalize_text(q)
            q_vec = embed_query(q)  # SimCSE PhoBERT INT8
            embed_cache[q] = (q_vec, q_norm)

        brand_index_for_group = brand_index_by_group.get(group, {})

        fvec = extract_features_for_pair(q_vec, q_norm, code, brand_index_for_group)
        if fvec is None:
            # meta hoặc vector thiếu, bỏ qua dòng này
            continue

        feats.append(
            [group, q, code, y] + fvec
        )

    print(f"✓ Feature extraction completed ({time.time() - t_feat:.2f}s)")
    print(f"  - Valid rows with features: {len(feats):,}")

    # ------------------------------------
    # 4) BUILD FINAL DF
    # ------------------------------------
    t_build = time.time()
    colnames = ["group_id", "query", "code", "label"] + FEATURE_COLS
    df_out = pd.DataFrame(feats, columns=colnames)
    print(f"✓ Built DataFrame ({time.time() - t_build:.2f}s)")
    print(f"  - Final rows: {len(df_out):,}")
    print(f"  - Total features per row: {len(FEATURE_COLS)}")

    # ------------------------------------
    # 5) SAVE CSV
    # ------------------------------------
    t_save = time.time()
    df_out.to_csv("benchmark_features_v8.csv", index=False)
    print(f"✓ Saved: train_features_v8.csv ({time.time() - t_save:.2f}s)")

    # ------------------------------------
    # 6) SUMMARY
    # ------------------------------------
    print("=======================================")
    print(f"⏱ TOTAL TIME: {time.time() - T0:.2f}s")
    print("=======================================")


if __name__ == "__main__":
    main()

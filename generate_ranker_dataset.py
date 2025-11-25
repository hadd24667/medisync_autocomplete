# ======================================================
# SMARTEMR — RANKER TRAIN PAIRS GENERATOR (v6 CLEAN)
# ------------------------------------------------------
# Quy tắc sinh query (đã được làm sạch hoàn toàn):
#
#   ✔ synonyms (GIỮ)
#   ✔ brand_full (GIỮ)
#   ✔ inn_full (GIỮ)
#   ✔ full_brand + dose (GIỮ)
#   ✔ full_inn + dose (GIỮ)
#
#    KHÔNG sinh prefix từ synonyms
#    KHÔNG sinh prefix từ brand
#    KHÔNG sinh dose-alone
#    KHÔNG sinh prefix + dose
#
# Design này sạch, không nhiễu, không conflict:
#   - "para" chỉ xuất hiện ở N02BE01
#   - "para 500" không nằm trong train — Tier-1 xử lý
#   - Ranker học đúng semantic/logic
# ======================================================

import json
import random
import pandas as pd
from unidecode import unidecode
from collections import Counter

ATC_FILE = "atc_synonyms_meta_cleaned.json"
OUT_FILE = "train_pairs.csv"

QUERIES_PER_ATC = 10
NEG_PER_QUERY   = 8


def norm(x):
    return unidecode(x.lower().strip()) if isinstance(x, str) else ""


def is_clean_synonym(s):
    # Không nhận synonym có số — tránh nhiễu
    return not any(ch.isdigit() for ch in s)


def extract_queries(meta, dose_freq):
    qs = set()

    # ============================================
    # 1) SYNONYMS (CHỈ GIỮ ĐẦY ĐỦ, KHÔNG PREFIX)
    # ============================================
    for s in meta.get("synonyms", []):
        s = norm(s)
        if not is_clean_synonym(s):
            continue
        if len(s) >= 3:
            qs.add(s)  # full synonym ONLY

    # ============================================
    # 2) BRAND (KHÔNG PREFIX)
    # ============================================
    brands = [norm(b) for b in meta.get("brand", [])]
    for b in brands:
        if len(b) >= 3:
            qs.add(b)  # brand full ONLY

    # ============================================
    # 3) INN FULL
    # ============================================
    inn_parts = []
    if isinstance(meta.get("inn"), str):
        for p in meta["inn"].split(","):
            p = norm(p)
            if len(p) >= 3:
                qs.add(p)
            inn_parts.append(p)

    # ============================================
    # 4) BRAND + DOSE, INN + DOSE (FULL FORM)
    # ============================================
    doses = meta.get("doses", [])

    for d in doses:
        # Loại dose quá phổ biến → gây nhiễu
        if dose_freq[d] > 8:
            continue

        # --- Brand + Dose (FULL ONLY) ---
        for b in brands:
            qs.add(f"{b} {d}")

        # --- INN + Dose (FULL ONLY) ---
        for inn in inn_parts:
            qs.add(f"{inn} {d}")

    qs = list(qs)
    random.shuffle(qs)
    return qs[:QUERIES_PER_ATC]


def pick_neg(all_codes, true_code, k):
    cand = [c for c in all_codes if c != true_code]
    return random.sample(cand, k) if len(cand) >= k else random.choices(cand, k)


def generate_pairs():
    with open(ATC_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = list(data.keys())

    # Count dose frequency to filter dose noise
    dose_freq = Counter()
    for code, meta in data.items():
        dose_freq.update(meta.get("doses", []))

    rows = []
    gid = 0

    for code, meta in data.items():
        queries = extract_queries(meta, dose_freq)

        for q in queries:
            # Positive sample
            rows.append(dict(group_id=gid, query=q, atc_code=code, label=1))

            # Negative samples
            for neg in pick_neg(codes, code, NEG_PER_QUERY):
                rows.append(dict(group_id=gid, query=q, atc_code=neg, label=0))

            gid += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print("✓ DONE — train_pairs.csv generated (v6 CLEAN)!")
    print("Total samples:", len(df))
    print("Total groups:", df['group_id'].nunique())


if __name__ == "__main__":
    generate_pairs()

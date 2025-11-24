# ======================================================
# SMARTEMR — RANKER TRAIN PAIRS GENERATOR (v5)
# ------------------------------------------------------
# Chỉ sinh:
#   - synonyms
#   - brand
#   - inn
#   - (brand + dose)
#   - (inn + dose)
#
# KHÔNG sinh:
#   - dose alone
#   - form / route
#   - form+dose / route+dose
# ======================================================

import json
import random
import pandas as pd
from unidecode import unidecode
from collections import Counter

ATC_FILE = "atc_synonyms_meta.json"
OUT_FILE = "train_pairs.csv"

QUERIES_PER_ATC = 10
NEG_PER_QUERY   = 8


def norm(x):
    return unidecode(x.lower().strip()) if isinstance(x, str) else ""

def is_clean_synonym(s):
    # loại nếu chứa số (dose, hoàn toàn không dùng trong synonyms)
    return not any(ch.isdigit() for ch in s)


def extract_queries(meta, dose_freq):
    qs = set()

    # --------------------------------------------------
    # 1) synonyms
    # --------------------------------------------------
    for s in meta.get("synonyms", []):
        s = norm(s)
        if not is_clean_synonym(s):
            continue
        if len(s) >= 3:
            qs.add(s)
        if len(s) >= 4:
            qs.add(s[:4])

    # --------------------------------------------------
    # 2) brand
    # --------------------------------------------------
    brands = [norm(b) for b in meta.get("brand", [])]
    for b in brands:
        if len(b) >= 3:
            qs.add(b)
        if len(b) >= 4:
            qs.add(b[:4])

    # --------------------------------------------------
    # 3) INN parts
    # --------------------------------------------------
    inn_parts = []
    if isinstance(meta.get("inn"), str):
        for p in meta["inn"].split(","):
            p = norm(p)
            inn_parts.append(p)

            if len(p) >= 4:
                qs.add(p)
                qs.add(p[:4])

    # --------------------------------------------------
    # 4) brand + dose, inn + dose
    # --------------------------------------------------
    doses = meta.get("doses", [])

    for d in doses:
        # Loại dose xuất hiện quá nhiều → gây nhiễu
        if dose_freq[d] > 8:
            continue

        # --- Brand + Dose ---
        for b in brands:
            qs.add(f"{b} {d}")
            if len(b) >= 4:
                qs.add(f"{b[:4]} {d}")

        # --- INN + Dose ---
        for inn in inn_parts:
            qs.add(f"{inn} {d}")
            if len(inn) >= 4:
                qs.add(f"{inn[:4]} {d}")

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
        doses = meta.get("doses", [])
        dose_freq.update(doses)

    rows = []
    gid = 0

    for code, meta in data.items():
        queries = extract_queries(meta, dose_freq)

        for q in queries:
            rows.append(dict(group_id=gid, query=q, atc_code=code, label=1))

            for neg in pick_neg(codes, code, NEG_PER_QUERY):
                rows.append(dict(group_id=gid, query=q, atc_code=neg, label=0))

            gid += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print("✓ DONE — train_pairs.csv generated!")
    print("Total samples:", len(df))
    print("Total groups:", df["group_id"].nunique())


if __name__ == "__main__":
    generate_pairs()

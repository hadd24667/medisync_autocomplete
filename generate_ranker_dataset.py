# ======================================================
# SMARTEMR — RANKER TRAIN PAIRS GENERATOR (v3)
# ------------------------------------------------------
# Chống nhiễu dose (500mg), ưu tiên query dạng thực tế:
# "para 500", "paracetamol 500mg", "hapacol 500"
# ======================================================

import json
import random
import pandas as pd
from unidecode import unidecode
from collections import Counter

ATC_FILE = "atc_synonyms_meta.json"
OUT_FILE = "train_pairs.csv"

# Số query / ATC
QUERIES_PER_ATC = 10
NEG_PER_QUERY = 8

def norm(x):
    return unidecode(x.lower().strip()) if isinstance(x, str) else ""

def extract_doses(forms):
    doses = []
    for f in forms:
        f = norm(f)
        parts = f.split()
        for p in parts:
            if p.endswith("mg") or p.endswith("g"):
                doses.append(p)
                # “500mg” → “500”
                if p[:-2].isdigit():
                    doses.append(p[:-2])
    return doses

def extract_queries(meta, dose_freq):
    """
    Sinh query theo thứ tự ưu tiên:
    1. synonyms
    2. brand
    3. inn
    4. dose kết hợp INN/brand (không sinh đơn độc)
    """
    qs = set()

    # 1. synonyms
    for s in meta.get("synonyms", []):
        s2 = norm(s)
        if len(s2) >= 3:
            qs.add(s2)

    # 2. brand
    for b in meta.get("brand", []):
        b2 = norm(b)
        if len(b2) >= 3:
            qs.add(b2)

    # 3. INN expansions
    if isinstance(meta.get("inn"), str):
        for part in meta["inn"].split(","):
            p = norm(part)
            if len(p) >= 4:
                qs.add(p)

    # 4. Dose combination
    doses = extract_doses(meta.get("forms", []))
    inn_parts = []
    if isinstance(meta.get("inn"), str):
        inn_parts = [norm(x) for x in meta["inn"].split(",")]

    brands = [norm(x) for x in meta.get("brand", [])]

    for d in doses:
        # Bỏ dose phổ biến gây nhiễu (xuất hiện trong nhiều ATC)
        if dose_freq[d] > 8:
            continue

        # Chỉ sinh dạng kết hợp, không sinh "500", "500mg"
        for inn in inn_parts:
            qs.add(f"{inn} {d}")
            # dạng bác sĩ hay gõ “para 500”
            if len(inn) >= 4:
                qs.add(f"{inn[:4]} {d}")

        for brand in brands:
            qs.add(f"{brand} {d}")

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

    # Đếm tần suất dose để loại dose gây nhiễu
    dose_counter = Counter()
    for code, meta in data.items():
        ds = extract_doses(meta.get("forms", []))
        dose_counter.update(ds)

    rows = []
    gid = 0

    for code, meta in data.items():
        queries = extract_queries(meta, dose_counter)

        for q in queries:
            rows.append(dict(group_id=gid, query=q, atc_code=code, label=1))

            for neg in pick_neg(codes, code, NEG_PER_QUERY):
                rows.append(dict(group_id=gid, query=q, atc_code=neg, label=0))

            gid += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"✓ DONE — {len(df)} samples → {OUT_FILE}")
    print(f"Total groups: {df['group_id'].nunique()}")


if __name__ == "__main__":
    generate_pairs()

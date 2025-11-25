import json
import random
import pandas as pd
from unidecode import unidecode
from collections import Counter

ATC_FILE = "atc_synonyms_meta_cleaned.json"
OUT_FILE = "benchmark_pairs.csv"

# số query benchmark mỗi ATC
QUERIES_PER_ATC = 6
NEG_PER_QUERY   = 10

def norm(x):
    return unidecode(x.lower().strip()) if isinstance(x, str) else ""

def pick_neg(all_codes, true_code, k):
    cand = [c for c in all_codes if c != true_code]
    if len(cand) >= k:
        return random.sample(cand, k)
    return random.choices(cand, k)

def generate_benchmark_queries(meta):
    """
    Tạo truy vấn đẹp, khác train, dành riêng benchmark
    Ưu tiên:
      - inn + route
      - brand + route
      - synonym + dose
      - dạng viết tắt bác sĩ hay gõ
    """
    qs = set()

    inn = meta.get("inn") or ""
    inn = norm(inn)

    brands = [norm(b) for b in (meta.get("brand") or [])]
    syns   = [norm(s) for s in (meta.get("synonyms") or [])]
    doses  = meta.get("doses") or []
    routes = meta.get("routes") or []

    # ----- INN + route -----
    for r in routes:
        r = r.lower()
        token = "uong" if r == "oral" else r
        if inn:
            qs.add(f"{inn} {token}")

    # ----- BRAND + route -----
    for b in brands:
        for r in routes:
            token = "uong" if r == "oral" else r
            qs.add(f"{b} {token}")

    # ----- synonym + dose -----
    for s in syns:
        for d in doses:
            if not any(ch.isdigit() for ch in s):
                qs.add(f"{s} {d}")

    # ----- viết tắt style bác sĩ -----
    if inn:
        parts = inn.split()
        if len(parts) >= 2:
            abbr = "".join(w[:3] for w in parts)
            qs.add(abbr)

    qs = list(qs)
    random.shuffle(qs)
    return qs[:QUERIES_PER_ATC]


def generate():
    with open(ATC_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = list(data.keys())

    rows = []
    gid = 0

    for code, meta in data.items():
        qs = generate_benchmark_queries(meta)

        for q in qs:
            # POS
            rows.append(dict(group_id=gid, query=q, atc_code=code, label=1))

            # NEG
            for neg in pick_neg(codes, code, NEG_PER_QUERY):
                rows.append(dict(group_id=gid, query=q, atc_code=neg, label=0))

            gid += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print("✓ benchmark_pairs.csv created")
    print("  groups:", df["group_id"].nunique())
    print("  rows:", len(df))


if __name__ == "__main__":
    generate()

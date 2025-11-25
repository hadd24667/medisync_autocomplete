import json
from collections import defaultdict

import redis
from unidecode import unidecode


# ==============================
#  CONFIG
# ==============================

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

ATC_META_FILE = "atc_synonyms_meta_cleaned.json"

KEY_ATC_META_PREFIX = "atc:meta:"              # atc:meta:N02BE01 -> full metadata
KEY_BRAND_INDEX_PREFIX = "atc:index:brand:"    # atc:index:brand:tylenol -> ["N02BE01", "N02AJ06", ...]
KEY_INN_INDEX_PREFIX = "atc:index:inn:"        # atc:index:inn:paracetamol -> [...]
KEY_SYN_INDEX_PREFIX = "atc:index:syn:"        # atc:index:syn:para -> [...]


# ==============================
#  HELPERS
# ==============================

def normalize_text(s: str) -> str:
    """Chuẩn hoá: lowercase + bỏ dấu + strip."""
    if s is None:
        return ""
    return unidecode(str(s).lower().strip())


def ensure_list(v):
    """Đảm bảo luôn trả về list (cho brand, synonyms...)."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [v]
    # fallback: cast iterable khác về list
    try:
        return list(v)
    except Exception:
        return [str(v)]


# ==============================
#  MAIN
# ==============================

def main():
    # 1) Kết nối Redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

    # 2) Load file JSON ATC metadata
    with open(ATC_META_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Index ngược trong RAM (sẽ flush xuống Redis sau)
    brand_index = defaultdict(set)
    inn_index = defaultdict(set)
    syn_index = defaultdict(set)

    pipe = r.pipeline()
    n_ok = 0
    n_err = 0

    for atc_code, meta in data.items():
        code = atc_code.strip()
        if not code:
            n_err += 1
            continue

        # --- Đảm bảo một số field có type chuẩn ---
        meta = dict(meta)  # copy để chắc chắn mutable
        meta.setdefault("inn", "")
        meta.setdefault("brand", [])
        meta.setdefault("forms", [])
        meta.setdefault("routes", [])
        meta.setdefault("doses", [])
        meta.setdefault("risk_tags", [])
        meta.setdefault("synonyms", [])
        meta.setdefault("aliases", [])
        meta.setdefault("specialties", [])

        # --- Ghi metadata chính vào Redis ---
        key_meta = KEY_ATC_META_PREFIX + code
        pipe.set(key_meta, json.dumps(meta, ensure_ascii=False))

        # --- Build index brand -> codes ---
        for b in ensure_list(meta.get("brand")):
            b_norm = normalize_text(b)
            if b_norm:
                brand_index[b_norm].add(code)

        # --- Build index inn -> codes ---
        inn_raw = meta.get("inn") or ""
        # inn có thể chứa nhiều hoạt chất, phân tách bởi dấu phẩy
        for part in inn_raw.split(","):
            part_norm = normalize_text(part)
            if part_norm:
                inn_index[part_norm].add(code)

        # --- Build index synonym -> codes ---
        for s in ensure_list(meta.get("synonyms")):
            s_norm = normalize_text(s)
            if s_norm:
                syn_index[s_norm].add(code)

        n_ok += 1

    # Flush metadata
    pipe.execute()

    # 3) Ghi index ngược xuống Redis
    pipe = r.pipeline()

    for b_norm, codes in brand_index.items():
        key = KEY_BRAND_INDEX_PREFIX + b_norm
        pipe.set(key, json.dumps(sorted(codes), ensure_ascii=False))

    for inn_norm, codes in inn_index.items():
        key = KEY_INN_INDEX_PREFIX + inn_norm
        pipe.set(key, json.dumps(sorted(codes), ensure_ascii=False))

    for syn_norm, codes in syn_index.items():
        key = KEY_SYN_INDEX_PREFIX + syn_norm
        pipe.set(key, json.dumps(sorted(codes), ensure_ascii=False))

    pipe.execute()

    # 4) Summary
    print("✓ Đã import ATC metadata vào Redis")
    print(f"  - Số ATC code ghi thành công : {n_ok}")
    print(f"  - Số record lỗi               : {n_err}")
    print(f"  - Số brand index khác nhau    : {len(brand_index)}")
    print(f"  - Số INN index khác nhau      : {len(inn_index)}")
    print(f"  - Số synonym index khác nhau  : {len(syn_index)}")


if __name__ == "__main__":
    main()

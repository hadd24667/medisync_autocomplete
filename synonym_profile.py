# synonym_profile.py
# Shared module for ATC synonym handling (Tier-1 & Tier-2)

import re
from unidecode import unidecode


def normalize_text(s: str) -> str:
    """Lowercase + remove accents + strip."""
    if s is None:
        return ""
    return unidecode(str(s).lower().strip())


def ensure_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [v]
    try:
        return list(v)
    except Exception:
        return [str(v)]


def build_synonym_profiles(code: str, meta: dict):
    """
    Sinh synonym_profiles cho 1 ATC code, dựa trên:
      - inn (chuỗi, có thể chứa nhiều inn ngăn bởi dấu phẩy)
      - brand (list)
      - synonyms (list phẳng)
    Trả về dict: { normalized_syn: {"source": ..., "value": ...}, ... }
    """
    profiles = {}

    raw_inn = meta.get("inn") or ""
    inn_list = [s.strip() for s in raw_inn.split(",") if s.strip()]
    brand_list = meta.get("brand") or []
    synonyms = meta.get("synonyms") or []

    def norm(s: str) -> str:
        return unidecode(s.lower().strip())

    # map brand_norm -> brand_canonical
    brand_map = {norm(b): b for b in brand_list}

    # map inn_norm -> inn_canonical (bỏ ngoặc cho gọn)
    inn_map = {}
    for inn in inn_list:
        inn_norm = norm(inn)
        base = re.sub(r"\(.*?\)", "", inn).strip()
        inn_canon = base if base else inn
        inn_map[inn_norm] = inn_canon

    # fallback canonical nếu không match được gì
    if inn_list:
        default_canonical = min(
            [re.sub(r"\(.*?\)", "", i).strip() or i for i in inn_list],
            key=len,
        )
    elif brand_list:
        default_canonical = brand_list[0]
    else:
        default_canonical = code

    for syn in synonyms:
        s_norm = norm(syn)
        if not s_norm:
            continue

        # (1) thử match brand
        matched = False
        for b_norm, b_canon in brand_map.items():
            if b_norm.startswith(s_norm) or s_norm.startswith(b_norm):
                profiles[s_norm] = {
                    "source": "brand",
                    "value": b_canon,
                }
                matched = True
                break
        if matched:
            continue

        # (2) thử match inn
        for i_norm, i_canon in inn_map.items():
            if i_norm.startswith(s_norm) or s_norm.startswith(i_norm):
                profiles[s_norm] = {
                    "source": "inn",
                    "value": i_canon,
                }
                matched = True
                break
        if matched:
            continue

        # (3) fallback: gắn theo code / canonical default
        profiles[s_norm] = {
            "source": "code",
            "value": default_canonical,
        }
    
    return profiles


def tokenize_query_norm(q_norm: str) -> list[str]:
    """
    Token hóa đơn giản trên query đã normalize:
    - tách theo whitespace + ký tự không phải chữ/số
    """
    if not q_norm:
        return []
    tokens = re.split(r"[^0-9a-zA-Z]+", q_norm)
    return [t for t in tokens if t]


def pick_matched_syn_from_query(query: str, profile: dict) -> str | None:
    """
    Chọn 1 synonym từ query:
    - Normalize query
    - Tách token
    - Ưu tiên token dài nhất xuất hiện trong profile
    """
    q_norm = normalize_text(query)
    tokens = tokenize_query_norm(q_norm)
    if not tokens:
        return None

    candidates = [t for t in tokens if t in profile]
    if not candidates:
        return None

    # Ưu tiên token dài nhất (para, tylenol, prctm,...)
    candidates.sort(key=len, reverse=True)
    return candidates[0]

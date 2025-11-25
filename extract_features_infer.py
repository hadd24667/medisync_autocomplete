# extract_features_infer.py
# Inference feature extractor for ATC ranker (v8)

import math
import re
from collections import defaultdict
from unidecode import unidecode

import numpy as np
from synonym_profile import (
    normalize_text,
    build_synonym_profiles,
    pick_matched_syn_from_query,
)

from ranker_utils import get_meta, get_vec


# FEATURE_COLS phải giống bên train
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


# ----------------- HELPERS (giống bên train) -----------------

def normalize_text(s: str) -> str:
    if not s:
        return ""
    return unidecode(s.lower().strip())


def extract_doses_from_forms(forms):
    doses = set()
    if not forms:
        return []
    for f in forms:
        f = normalize_text(f)

        mg = re.findall(r'(\d+(?:\.\d+)?)\s*mg', f)
        for x in mg:
            doses.add(f"{x}mg")
            doses.add(x)
            try:
                mg_int = str(int(float(x)))
                doses.add(f"{mg_int}mg")
                doses.add(mg_int)
            except Exception:
                pass

        g = re.findall(r'(\d+(?:\.\d+)?)\s*g', f)
        for x in g:
            try:
                mg_val = int(float(x) * 1000)
                doses.add(f"{mg_val}mg")
                doses.add(str(mg_val))
                doses.add(x)
            except Exception:
                pass

        mcg = re.findall(r'(\d+(?:\.\d+)?)\s*mcg', f)
        for x in mcg:
            doses.add(f"{x}mcg")
            doses.add(x)

        percent = re.findall(r'(\d+(?:\.\d+)?)\s*%', f)
        for x in percent:
            doses.add(f"{x}%")
            doses.add(x)

        iu = re.findall(r'(\d+(?:\.\d+)?)\s*(?:iu|ui)', f)
        for x in iu:
            doses.add(f"{x}iu")
            doses.add(x)

        per_dose = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mcg|mg)/lieu', f)
        for x in per_dose:
            doses.add(x)

        standalone_num = re.findall(r'\b(\d+(?:\.\d+)?)\b', f)
        for x in standalone_num:
            if "." in x or len(x) <= 4:
                doses.add(x)

    return sorted(doses)


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
    q_nums = parse_numbers_from_text(q_norm)

    dose_tokens = []
    for d in meta.get("doses", []) or []:
        dose_tokens.append(str(d))
    dose_tokens.extend(extract_doses_from_forms(meta.get("forms", [])))

    dose_nums = parse_numbers_from_text(" ".join(dose_tokens))

    if not q_nums or not dose_nums:
        return 0.0, 0.0

    match = any(abs(q - d) < 1e-3 for q in q_nums for d in dose_nums)
    f_dose_match = 1.0 if match else 0.0

    min_diff = min(abs(q - d) for q in q_nums for d in dose_nums)
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


# ----------------- API DÙNG CHO RANKER -----------------

def extract_feats_from_vec(
    q_vec: np.ndarray,
    query: str,
    code: str,
    codes_in_group=None,
    matched_syn: str | None = None,
):
    q_norm = normalize_text(query)
    meta = get_meta(code)
    if meta is None:
        return None

    # 1) semantic (giữ nguyên)
    c_vec = get_vec(code)
    if c_vec is None:
        f_sem = 0.0
    else:
        # cos similarity
        denom = (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-8)
        f_sem = float(np.dot(q_vec, c_vec) / denom)
    f_sem_sq = f_sem * f_sem

    # 2) build brand_index_for_group
    if codes_in_group:
        brand_index_for_group = build_brand_index_for_group(codes_in_group)
    else:
        brand_index_for_group = build_brand_index_for_group([code])

    # 3) synonym-based features (dùng profile + matched_syn)
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
    ) = compute_synonym_features(
        q_norm=q_norm,
        meta=meta,
        brand_index_for_group=brand_index_for_group,
        matched_syn=matched_syn,   # 🔥 Tier-1 truyền xuống ở đây
    )

    # 4) dose / route / flags (giữ logic v8 hiện tại của bạn)
    f_dose_match, f_dose_diff_min = compute_dose_features(q_norm, meta)
    f_route_match = compute_route_match(q_norm, meta.get("routes") or [])
    f_is_combo = 1.0 if (" + " in (meta.get("inn") or "")) else 0.0
    f_is_pediatric = 1.0 if meta.get("is_pediatric_form") else 0.0
    f_risk_hepatic = 1.0 if "hepatic_toxic" in (meta.get("risk_tags") or []) else 0.0
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
        f_is_combo,
        f_is_pediatric,
        f_risk_hepatic,
    ]


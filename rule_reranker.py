import math
from unidecode import unidecode
import numpy as np


# ================================
# NORMALIZE
# ================================

def norm(x):
    if not x:
        return ""
    return unidecode(x.lower().strip())


# ================================
# SEMANTIC BOOST (optional)
# ================================

def semantic_boost(query_emb, atc_emb):
    """Cosine similarity (0 → 1)."""
    if query_emb is None or atc_emb is None:
        return 0.0

    a = query_emb
    b = atc_emb

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # scale [-1..1] → [0..1]
    return max(0.0, min(1.0, (cosine + 1) / 2))


# ================================
# CONTEXT RULES
# ================================

def apply_context_rules(meta, context):
    """
    meta: metadata của ATC (inn, brand, forms, routes, specialties...)
    context: {'age', 'gender', 'specialty', 'conditions'}
    """
    score = 0

    age = context.get("age")
    spec = norm(context.get("specialty", ""))
    conds = [norm(c) for c in context.get("conditions", [])]

    # ---- 1. Pediatric logic ----
    if age is not None and age < 12:
        if meta.get("is_pediatric_form"):
            score += 4   # ưu tiên mạnh

        # tránh viên nén to
        form_text = " ".join(meta.get("forms", []))
        if "vien nen" in form_text or "viên nén" in form_text:
            score -= 3

    # ---- 2. Specialty match ----
    if spec:
        for sp in meta.get("specialties", []):
            if norm(sp) == spec:
                score += 2

    # ---- 3. Contraindication removal ----
    for c in conds:
        for cc in meta.get("contraindications", []):
            if c in norm(cc):
                score -= 999  # loại hẳn

    # ---- 4. Risk boosting (vd: disease risk) ----
    for c in conds:
        for rt in meta.get("risk_tags", []):
            if c in norm(rt):
                score += 1

    return score


# ================================
# COMBINE FINAL SCORE
# ================================

def combine_score(prefix_score, semantic_score, context_boost):
    """
    final score = 3 * prefix + 2 * semantic + context
    """
    return 3 * prefix_score + 2 * semantic_score + context_boost


# ================================
# MAIN RERANK FUNCTION
# ================================

def rerank(query, candidates, get_meta_fn, embed_fn=None, context={}):
    """
    query: str
    candidates: [(atc_code, prefix_score)]
    get_meta_fn: fn(atc_code) -> metadata dict
    embed_fn: optional semantic embed
    context: dict (age, specialty, conditions)

    return: sorted list of (atc_code, final_score)
    """

    # semantic query vector nếu có embed_fn
    q_emb = embed_fn(query) if embed_fn else None

    results = []

    for code, prefix_score in candidates:
        meta = get_meta_fn(code)

        # semantic
        atc_emb = None
        if embed_fn and "embedding" in meta:
            atc_emb = np.array(meta["embedding"], dtype=np.float32)

        semantic_score = semantic_boost(q_emb, atc_emb) if embed_fn else 0.0

        # context boost
        ctx_boost = apply_context_rules(meta, context)

        # final score
        final = combine_score(prefix_score, semantic_score, ctx_boost)

        results.append((code, final))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# tier2_ranker.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import re
from unidecode import unidecode


PEDIATRIC_AGE = 12  # < 12 coi như trẻ em


# ============================
# 1. PATIENT CONTEXT
# ============================

@dataclass
class PatientContext:
    age: Optional[int] = None
    sex: Optional[str] = None              # "M" / "F" / None
    specialty: Optional[str] = None        # "pediatrics", "cardiology", ...
    active_icd: List[str] = field(default_factory=list)
    active_atc: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)


# ============================
# 2. UTILITIES
# ============================

def normalize_text(text: str) -> str:
    return unidecode(text.lower().strip())


def tokenize(text: str) -> List[str]:
    """
    Tách token đơn giản: chữ + số
    """
    t = normalize_text(text)
    return re.findall(r"[a-z0-9]+", t)


def has_any(text: str, keywords: List[str]) -> bool:
    if not text:
        return False
    t = normalize_text(text)
    return any(kw in t for kw in keywords)


def safe_get(meta: Dict[str, Any], key: str, default=None):
    v = meta.get(key, default)
    return v if v is not None else default


# ============================
# 3. QUERY FEATURES
# ============================

@dataclass
class QueryFeatures:
    tokens: List[str]
    numbers: List[str]
    has_oral: bool
    has_inject: bool
    has_syrup: bool
    has_topical: bool
    has_pediatric_hint: bool


def extract_query_features(query: str) -> QueryFeatures:
    tokens = tokenize(query)
    numbers = [t for t in tokens if t.isdigit()]

    has_oral = any(t in ["uong", "po", "oral"] for t in tokens)
    has_inject = any(t in ["tiem", "tiemtm", "tm", "iv", "im", "sc"] for t in tokens)
    has_syrup = any(t in ["siro", "sir", "syr", "syrup", "gom", "goi"] for t in tokens)
    has_topical = any(t in ["boi", "cream", "gel", "ointment"] for t in tokens)

    has_pediatric_hint = any(t in ["tre", "be", "nhi", "child", "kid"] for t in tokens)

    return QueryFeatures(
        tokens=tokens,
        numbers=numbers,
        has_oral=has_oral,
        has_inject=has_inject,
        has_syrup=has_syrup,
        has_topical=has_topical,
        has_pediatric_hint=has_pediatric_hint,
    )


# ============================
# 4. SCORING COMPONENTS
# ============================

# ---- 4.1 Query → metadata match ----

def score_query_match(qf: QueryFeatures, meta: Dict[str, Any]) -> float:
    """
    Match query với INN, brand, synonyms.
    """
    score = 0.0

    inn = safe_get(meta, "inn", "") or ""
    brands = safe_get(meta, "brand", []) or []
    synonyms = safe_get(meta, "synonyms", []) or []

    full_text = " ".join([inn] + brands + synonyms)
    full_text_norm = normalize_text(full_text)

    # Match từng token
    for tok in qf.tokens:
        if len(tok) <= 1:
            continue
        if tok in full_text_norm:
            score += 0.4  # mỗi token match +0.4

    # Nếu query bắt đầu bằng alias của brand/inn
    first_tok = qf.tokens[0] if qf.tokens else ""
    if first_tok:
        if any(normalize_text(first_tok) == normalize_text(b.split()[0]) for b in brands):
            score += 0.6
        if normalize_text(first_tok) in normalize_text(inn):
            score += 0.6

    # Giới hạn
    return min(score, 2.0)  # tối đa 2.0 điểm


# ---- 4.2 Form / dosage match ----

def score_dosage(qf: QueryFeatures, meta: Dict[str, Any]) -> float:
    """
    Ưu tiên hàm lượng khớp số mg trong query.
    """
    forms = safe_get(meta, "forms", []) or []
    if not forms or not qf.numbers:
        return 0.0

    forms_text = normalize_text(" ".join(forms))
    score = 0.0

    for n in qf.numbers:
        if n in forms_text:
            score += 1.0  # số mg khớp rất mạnh

    return min(score, 2.0)


def score_form_route(qf: QueryFeatures, meta: Dict[str, Any]) -> float:
    """
    Dựa trên forms + routes, xem có khớp intent query không.
    """
    forms = safe_get(meta, "forms", []) or []
    routes = safe_get(meta, "routes", []) or []

    forms_text = normalize_text(" ".join(forms))
    routes_text = normalize_text(" ".join(routes))

    score = 0.0

    # Siro / cốm / gói
    if qf.has_syrup and has_any(forms_text, ["siro", "syrup", "gom", "goi"]):
        score += 1.0

    # Viên nén / viên nang
    if has_any(forms_text, ["vien nen", "vien nang"]):
        # nếu query không nói gì, vẫn + nhẹ
        score += 0.3

    # Đường uống
    if "oral" in routes_text:
        if qf.has_oral:
            score += 0.7
        else:
            score += 0.3  # mặc định cho uống là common route

    # Đường tiêm
    if has_any(routes_text, ["iv", "inject", "tiem"]):
        if qf.has_inject:
            score += 1.0

    # Topical
    if has_any(routes_text, ["topical"]) or has_any(forms_text, ["cream", "gel", "ointment"]):
        if qf.has_topical:
            score += 0.8

    # Giới hạn
    return min(score, 1.5)


# ---- 4.3 Age / pediatric ----

def score_age(age: Optional[int], qf: QueryFeatures, meta: Dict[str, Any]) -> float:
    if age is None:
        return 0.0

    is_pedi_form = bool(meta.get("is_pediatric_form"))
    forms = safe_get(meta, "forms", []) or []
    forms_text = normalize_text(" ".join(forms))

    score = 0.0

    # Nếu là form trẻ em rõ ràng
    if is_pedi_form:
        if age < PEDIATRIC_AGE:
            score += 2.0
        else:
            score += 0.5  # vẫn dùng đc cho người lớn, nhưng nhẹ hơn

    # Nếu trẻ nhưng là viên nén to / mg cao
    if age < PEDIATRIC_AGE:
        # viên nén / viên nang
        if has_any(forms_text, ["vien nen", "vien nang"]):
            score -= 0.5
        # mg cao
        if any(x in forms_text for x in ["500mg", "750mg", "1000mg"]):
            score -= 1.0

        # Query có hint muốn cho trẻ
        if qf.has_pediatric_hint:
            # ưu tiên dạng siro/gói
            if has_any(forms_text, ["siro", "gom", "goi"]):
                score += 1.0

    return max(min(score, 2.0), -2.0)


# ---- 4.4 Specialty ----

def score_specialty(specialty: Optional[str], meta: Dict[str, Any]) -> float:
    if not specialty:
        return 0.0

    specialty = specialty.lower()
    specs = [s.lower() for s in safe_get(meta, "specialties", []) or []]

    if specialty in specs:
        return 1.0

    # Một số mapping mềm (ví dụ pediatrics + is_pediatric_form)
    if specialty == "pediatrics" and meta.get("is_pediatric_form"):
        return 0.7

    return 0.0


# ---- 4.5 Safety: contraindications, allergies, risk_tags ----

def penalty_contra(active_icd: List[str], meta: Dict[str, Any]) -> float:
    """
    Nếu chẩn đoán hiện tại trùng với CCĐ → phạt mạnh.
    Ở đây mình match text đơn giản, sau này bạn có thể cải tiến:
    - map ICD → keywords tiếng Việt
    """
    ccd_list = safe_get(meta, "contraindications", []) or []
    if not ccd_list or not active_icd:
        return 0.0

    ccd_text = normalize_text(" ".join(ccd_list))
    penalty = 0.0

    for icd in active_icd:
        icd_norm = normalize_text(icd)
        if icd_norm and icd_norm in ccd_text:
            penalty -= 2.0  # CCĐ match trực tiếp → phạt rất mạnh

    return penalty


def penalty_allergy(allergies: List[str], meta: Dict[str, Any]) -> float:
    if not allergies:
        return 0.0

    inn = normalize_text(safe_get(meta, "inn", "") or "")
    brands = " ".join(safe_get(meta, "brand", []) or [])
    brands_norm = normalize_text(brands)
    text = inn + " " + brands_norm

    penalty = 0.0

    for alg in allergies:
        a = normalize_text(alg)
        if a and a in text:
            penalty -= 2.0

    return penalty


def penalty_risk_tags(active_icd: List[str], meta: Dict[str, Any]) -> float:
    """
    Đơn giản: nếu risk_tags chứa 'hepatic', 'renal', 'bleeding' mà bệnh nền liên quan → phạt.
    Sau này bạn có thể build map chi tiết hơn.
    """
    tags = [t.lower() for t in safe_get(meta, "risk_tags", []) or []]
    if not tags or not active_icd:
        return 0.0

    penalty = 0.0

    joined_icd = normalize_text(" ".join(active_icd))

    if "hepatic" in tags and any(k in joined_icd for k in ["suy gan", "viem gan", "xơ gan"]):
        penalty -= 1.0
    if "renal" in tags and any(k in joined_icd for k in ["suy than", "benh than"]):
        penalty -= 1.0
    if "bleeding" in tags and any(k in joined_icd for k in ["xuat huyet", "loet da day"]):
        penalty -= 1.0

    return penalty


def compute_safety_penalty(ctx: PatientContext, meta: Dict[str, Any]) -> float:
    return (
        penalty_contra(ctx.active_icd, meta)
        + penalty_allergy(ctx.allergies, meta)
        + penalty_risk_tags(ctx.active_icd, meta)
    )


# ============================
# 5. FINAL SCORE
# ============================

def compute_rule_context_score(
    query: str,
    meta: Dict[str, Any],
    ctx: PatientContext,
) -> Dict[str, float]:
    """
    Trả về từng component + final_score để dễ debug.
    """
    qf = extract_query_features(query)

    qm = score_query_match(qf, meta)        # 0 → 2
    ds = score_dosage(qf, meta)             # 0 → 2
    fr = score_form_route(qf, meta)         # 0 → 1.5
    ag = score_age(ctx.age, qf, meta)       # -2 → 2
    sp = score_specialty(ctx.specialty, meta)  # 0 → 1
    sf = compute_safety_penalty(ctx, meta)  # -∞ → 0 (thực tế ~ -6 → 0)

    # Trọng số có thể tinh chỉnh sau khi pilot
    final = (
        0.25 * qm +
        0.20 * ds +
        0.20 * fr +
        0.15 * ag +
        0.10 * sp +
        0.10 * sf
    )

    return {
        "query_match": qm,
        "dosage": ds,
        "form_route": fr,
        "age": ag,
        "specialty": sp,
        "safety": sf,
        "final_score": final,
    }


# ============================
# 6. RERANK API
# ============================

def rerank_atc_candidates(
    query: str,
    candidates: List[Any],       # list mã ATC hoặc list dict có 'code'
    metadata: Dict[str, Dict[str, Any]],
    ctx: PatientContext,
    top_k: Optional[int] = None,
) -> List[Tuple[Any, Dict[str, float]]]:
    """
    Trả về list (candidate, score_detail) đã sort theo final_score giảm dần.
    """
    scored: List[Tuple[Any, Dict[str, float]]] = []

    for cand in candidates:
        if isinstance(cand, str):
            code = cand
            obj = cand
        else:
            code = cand.get("code")
            obj = cand

        if not code:
            continue

        meta = metadata.get(code)
        if not meta:
            continue

        score_detail = compute_rule_context_score(query, meta, ctx)

        # Nếu safety penalty cực lớn (ví dụ final_score << -5), bạn có thể loại luôn
        scored.append((obj, score_detail))

    # sort theo final_score
    scored.sort(key=lambda x: x[1]["final_score"], reverse=True)

    if top_k is not None:
        scored = scored[:top_k]

    return scored

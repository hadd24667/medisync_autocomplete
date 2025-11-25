# atc_form_ranker.py
"""
Tầng form-level cho ATC:
- Input: query + tier2_results (đã được LightGBM rank theo code)
- Output: các gợi ý ở mức form: "Paracetamol – Viên nén 500 mg", ...
"""

from typing import List, Dict, Any
import re
from unidecode import unidecode


# Ưu tiên dạng bào chế (dùng key KHÔNG DẤU)
FORM_PRIORITY = {
    "vien nen": 0.6,
    "vien nang": 0.6,
    "vien nhai": 0.6,
    "vien bao phim": 0.6,
    "vien bao duong": 0.6,

    "hon dich": 0.3,
    "siro": 0.3,
    "sirô": 0.3,          # phòng trường hợp form trong DB đã bỏ dấu
    "dung dich uong": 0.3,
    "goi bot pha": 0.3,

    "thuoc dat": 0.2,

    "tiem": 0.1,
    "tiem truyen": 0.1,

    "gel boi": 0.05,
    "kem boi": 0.05,
    "thuoc boi": 0.05,
    "thuoc xit": 0.05,
    "nho mat": 0.05,
    "nho mui": 0.05,
}


def detect_form_priority(form: str) -> float:
    """
    Tính priority cho 1 form:
    - Chuẩn hoá về không dấu để match được cả 'Viên nén' / 'viên nén'
      với key 'vien nen'
    """
    if not form:
        return 0.0
    f = unidecode(form.lower())
    for key, val in FORM_PRIORITY.items():
        if key in f:
            return val
    return 0.0


def _extract_numbers(s: str) -> List[str]:
    """
    Lấy tất cả số trong chuỗi, ví dụ:
      - 'Viên nén 500 mg' -> ['500']
      - 'Hỗn dịch uống 120 mg/5 ml' -> ['120', '5']
    """
    if not s:
        return []
    return re.findall(r"\d+(?:\.\d+)?", s)


def _get_query_numbers(query: str) -> List[str]:
    """
    Lấy tất cả số xuất hiện trong query, ví dụ:
      'para 500' -> ['500']
      'para 500mg' -> ['500']
    """
    nums: List[str] = []
    for token in query.split():
        nums.extend(_extract_numbers(token))
    return nums


def normalize_form_key(form: str) -> str:
    """
    Key chuẩn hoá cho set() tránh trùng:
    - lower
    - bỏ khoảng trắng thừa
    - chuẩn 'mg' / 'ml'
    """
    if not form:
        return ""
    s = form.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("mg ", " mg ").replace("ml ", " ml ")
    return s


def build_form_level_suggestions(
    query: str,
    tier2_results: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:

    if not tier2_results:
        return []

    query_norm = query.lower().strip()
    query_tokens = query_norm.split()
    query_nums = set(_get_query_numbers(query_norm))

    # ----------------------------------------------------
    # 1) GROUP RESULTS BY CODE
    # ----------------------------------------------------
    grouped: Dict[str, Dict[str, Any]] = {}

    for r in tier2_results:
        code = r.get("code")
        if not code:
            continue

        forms = r.get("forms") or []
        ranker_score = float(r.get("ranker_score", 0.0))

        label = r.get("label") or ""
        # inn_clean dùng để match prefix query (không dùng hiển thị)
        inn_clean = (
            label.split("–")[0]
            .split("[")[0]
            .split("(")[0]
            .strip()
            .lower()
        )

        if code not in grouped:
            grouped[code] = {
                "code": code,
                "inn_clean": inn_clean,
                "inn_raw": r.get("inn_raw"),
                "brand_names": r.get("brand_names") or [],
                "forms": set(),          # set of (norm_key, real_form)
                "base_score": ranker_score,
            }
        else:
            if ranker_score > grouped[code]["base_score"]:
                grouped[code]["base_score"] = ranker_score

        # gom tất cả form của code này, tránh trùng
        for f in forms:
            key = normalize_form_key(f)
            grouped[code]["forms"].add((key, f))

    # ----------------------------------------------------
    # 2) BUILD VARIANTS (code + form)
    # ----------------------------------------------------
    variants: List[Dict[str, Any]] = []

    for code, g in grouped.items():
        base_score = g["base_score"]
        inn_clean = g["inn_clean"]

        for _norm_key, form_str in g["forms"]:
            score = base_score

            # 2.1 FORM PRIORITY
            # Nếu query KHÔNG có số → form_priority gần như quyết định thứ tự
            score += detect_form_priority(form_str)

            # 2.2 MATCH HOẠT CHẤT PREFIX (para, ibu, amox...)
            for tok in query_tokens:
                if len(tok) >= 3 and inn_clean.startswith(tok):
                    score += 3.0
                    break

            # 2.3 MATCH SỐ LIỀU (chỉ chạy khi query có số)
            form_nums = _extract_numbers(form_str)
            if query_nums and form_nums:
                # match chính xác 500 ↔ 500
                if any(fn in query_nums for fn in form_nums):
                    score += 2.0
                else:
                    # match lỏng 500 ↔ 500/5
                    if any(
                        any(q in fn or fn in q for q in query_nums)
                        for fn in form_nums
                    ):
                        score += 0.5

            # 2.4 FORMAT LABEL HIỂN THỊ (INN gốc + brand)
            inn_raw = g.get("inn_raw") or inn_clean.capitalize()

            brands = g.get("brand_names") or []
            brand_display = [b[0].upper() + b[1:] for b in brands if b]

            if brand_display:
                inn_display = f"{inn_raw} ({', '.join(brand_display)})"
            else:
                inn_display = inn_raw

            display = f"{inn_display} – {form_str}"

            variants.append({
                "code": code,
                "label": display,
                "form": form_str,
                "type": "ATC",
                "score": score,
            })

    # ----------------------------------------------------
    # 3) SORT & TRẢ VỀ
    # ----------------------------------------------------
    variants.sort(key=lambda v: v["score"], reverse=True)
    return variants[:top_k]

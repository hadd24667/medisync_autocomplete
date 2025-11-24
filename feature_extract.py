import re
import json
import redis
import numpy as np
from unidecode import unidecode
from transformers import AutoTokenizer, AutoModel
import torch

# =========================================
# 0. PhoBERT setup
# =========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base").to(device)
model.eval()

def embed(text: str) -> np.ndarray:
    text = unidecode(text.lower().strip())
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype("float32")
        return emb

# =========================================
# 1. Redis
# =========================================

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def load_meta(atc_code: str) -> dict:
    raw = r.get(f"atc:meta:{atc_code}")
    if not raw:
        raise KeyError(f"Missing meta for {atc_code}")
    return json.loads(raw)

def load_vec(atc_code: str) -> np.ndarray:
    raw = r.get(f"atc:vec:{atc_code}")
    if not raw:
        raise KeyError(f"Missing embedding for {atc_code}")
    return np.frombuffer(bytes.fromhex(raw), dtype=np.float32)

# =========================================
# 2. Text utilities
# =========================================

def normalize_text(s: str) -> str:
    return unidecode(s.lower().strip())

def tokenize(s: str):
    s = normalize_text(s)
    s = re.sub(r"[^\w/]", " ", s)  # giữ số, chữ, / (cho 120mg/5ml)
    toks = [t for t in s.split() if t]
    return toks

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)

def overlap_score(tokens_a, tokens_b):
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    return len(set_a & set_b) / (len(set_a) + 1e-6)

# =========================================
# 3. Canonical form / route maps
# =========================================

FORM_CANON = {
    "tablet": {"vien nen", "vien", "tab"},
    "capsule": {"vien nang", "caps", "cap"},
    "syrup": {"siro", "sirup", "hon dich uong"},
    "suppository": {"thuoc dat", "dat truc trang"},
    "injection": {"thuoc tiem", "tiem truyen", "truyen tinh mach"},
}

ROUTE_CANON = {
    "oral": {"uống", "uong", "po", "oral"},
    "injection": {"tiem", "tiêm", "iv", "im", "sc", "injection"},
    "rectal": {"dat", "truc trang", "suppository"},
}

def detect_form_types_from_text(text: str):
    text_n = normalize_text(text)
    out = set()
    for canon, kws in FORM_CANON.items():
        for kw in kws:
            if kw in text_n:
                out.add(canon)
                break
    return out

def detect_route_from_text(text: str):
    text_n = normalize_text(text)
    out = set()
    for canon, kws in ROUTE_CANON.items():
        for kw in kws:
            if kw in text_n:
                out.add(canon)
                break
    return out

def infer_routes_from_forms(form_types):
    out = set()
    if any(ft in {"tablet", "capsule", "syrup"} for ft in form_types):
        out.add("oral")
    if "suppository" in form_types:
        out.add("rectal")
    if "injection" in form_types:
        out.add("injection")
    return out

# =========================================
# 4. Strength extraction (mg/ml)
# =========================================

def extract_strengths_from_tokens(tokens):
    """Return set of (value, unit) where unit in {mg, ml, None}."""
    res = set()
    for tok in tokens:
        m = re.match(r"(\d+)(mg|ml)", tok)
        if m:
            val = int(m.group(1))
            unit = m.group(2)
            res.add((val, unit))
        elif tok.isdigit():
            res.add((int(tok), None))
    return res

def extract_strengths_from_forms(forms):
    strengths = set()
    for f in forms:
        toks = tokenize(f)
        strengths |= extract_strengths_from_tokens(toks)
    return strengths

def strength_match(query_strengths, drug_strengths) -> int:
    if not query_strengths or not drug_strengths:
        return 0
    # exact unit+value match
    for qv, qu in query_strengths:
        for dv, du in drug_strengths:
            if qu is not None and du is not None:
                if qv == dv and qu == du:
                    return 1
            else:
                # nếu query chỉ có số, meta có số+đơn vị
                if qv == dv:
                    return 1
    return 0

# =========================================
# 5. Build candidate info from meta
# =========================================

def build_candidate_info(meta: dict):
    # synonyms token set
    syn_tokens = set()
    for s in meta.get("synonyms", []):
        syn_tokens |= set(tokenize(s))

    # brand
    brands_norm = [normalize_text(b) for b in meta.get("brand", [])]

    # forms
    forms = meta.get("forms", [])
    form_types = set()
    for f in forms:
        form_types |= detect_form_types_from_text(f)

    # routes from meta + forms
    routes_meta = set()
    for r in meta.get("routes", []):
        routes_meta |= detect_route_from_text(r)
        routes_meta.add(normalize_text(r))  # oral, injection...

    routes_inferred = infer_routes_from_forms(form_types)
    routes_all = routes_meta | routes_inferred

    # strengths
    strengths = extract_strengths_from_forms(forms)

    # drug class tokens
    drug_class = normalize_text(meta.get("drug_class", ""))
    drug_class_tokens = tokenize(drug_class) if drug_class else []

    contras = [normalize_text(c) for c in meta.get("contraindications", [])]

    return {
        "syn_tokens": syn_tokens,
        "brands": brands_norm,
        "form_types": form_types,
        "routes": routes_all,
        "strengths": strengths,
        "drug_class_tokens": drug_class_tokens,
        "contras": contras,
        "is_pediatric": bool(meta.get("is_pediatric_form", False)),
        "specialties": [normalize_text(s) for s in meta.get("specialties", [])],
    }

# =========================================
# 6. Main feature extractor
# =========================================

def extract_features(
    query: str,
    atc_code: str,
    context: str | None = None,
    current_specialty: str | None = None,
):
    """
    Trả về list 10–12 features cho (query, atc_code).
    context / current_specialty có thể dùng sau, giờ chưa bắt buộc.
    """
    q_norm = normalize_text(query)
    q_tokens = tokenize(query)
    q_strengths = extract_strengths_from_tokens(q_tokens)

    meta = load_meta(atc_code)
    cand = build_candidate_info(meta)

    # PhoBERT
    q_emb = embed(q_norm)
    d_emb = load_vec(atc_code)
    f6_semantic = cosine(q_emb, d_emb)

    # 1) Synonym overlap
    f1_syn_overlap = overlap_score(q_tokens, list(cand["syn_tokens"]))

    # 2) Brand match
    f2_brand = int(any(b and b in q_norm for b in cand["brands"]))

    # 3) Form match (canonical)
    query_form_types = detect_form_types_from_text(q_norm)
    f3_form = int(bool(query_form_types & cand["form_types"]))

    # 4) Route match (explicit + inferred)
    query_routes = detect_route_from_text(q_norm) | infer_routes_from_forms(query_form_types)
    f4_route = int(bool(query_routes & cand["routes"]))

    # 5) Strength match
    drug_strengths = cand["strengths"]
    f5_strength = strength_match(q_strengths, drug_strengths)

    # 7) Drug class match
    f7_drugclass = overlap_score(q_tokens, cand["drug_class_tokens"])

    # 8) Specialty boost (simple version)
    f8_spec = 0.0
    if current_specialty:
        cur = normalize_text(current_specialty)
        if any(s in cur for s in cand["specialties"]):
            f8_spec = 1.0

    # 9) Pediatric
    f9_ped = 0.0
    if cand["is_pediatric"] and any(k in q_norm for k in ["tre em", "nhi", "be", "bé"]):
        f9_ped = 1.0

    # 10) Contra penalty (query + context)
    full_text = q_norm + " " + (normalize_text(context) if context else "")
    f10_contra = 0.0
    if any(c in full_text for c in cand["contras"]):
        f10_contra = -1.0

    # Bạn có thể thêm context_semantic / context_disease sau này (f11, f12)
    return [
        f1_syn_overlap,   # 1
        f2_brand,         # 2
        f3_form,          # 3
        f4_route,         # 4
        f5_strength,      # 5
        f6_semantic,      # 6
        f7_drugclass,     # 7
        f8_spec,          # 8
        f9_ped,           # 9
        f10_contra,       # 10
    ]


if __name__ == "__main__":
    # Example usage
    query = "Paracetamol 500mg viên nén"
    atc_code = "N02BE01"  # Example ATC code

    features = extract_features(query, atc_code)
    print(f"Extracted features for query '{query}' and ATC '{atc_code}':")
    print(features)

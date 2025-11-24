import re
import json
import pandas as pd
from unidecode import unidecode

# ============================================
# Helper Normalize
# ============================================

def normalize(text: str) -> str:
    if text is None:
        return ""
    t = unidecode(str(text).lower().strip())

    # remove trailing punctuation
    t = re.sub(r"[.,;:\-]+$", "", t)

    # collapse multiple spaces → 1
    t = re.sub(r"\s+", " ", t)

    # remove space before punctuation
    t = re.sub(r"\s+([),])", r"\1", t)

    # remove space after "("
    t = re.sub(r"\(\s+", "(", t)

    # final strip
    return t.strip()

def clean_whitespace_list(lst):
    cleaned = []
    for x in lst:
        if not x:
            continue
        v = normalize(x)
        if v:
            cleaned.append(v)
    return sorted(list(set(cleaned)))



def load_list(val):
    """Parse postgres array: {a,b,c} → ['a','b','c']"""
    if val is None:
        return []
    s = str(val).strip()
    if s in ["", "{}", "nan", "none", "None"]:
        return []
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1]
        parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
        # normalized
        return [normalize(p) for p in parts if p]
    return [normalize(s)]


# ============================================
# Abbrev + dose normalizer
# ============================================

def generate_short_forms(inn: str):
    inn = normalize(inn)
    if not inn:
        return []
    tokens = []
    if len(inn) >= 4: tokens.append(inn[:4])
    if len(inn) >= 5: tokens.append(inn[:5])
    consonants = re.sub(r"[aeiou]", "", inn)
    if len(consonants) >= 4:
        tokens.append(consonants[:5])
    return list(set(tokens))


def normalize_strength(s: str):
    s = s.lower()
    mg = re.findall(r"(\d+)\s*mg", s)
    if mg: return f"{mg[0]}mg"

    g = re.findall(r"(\d+\.?\d*)\s*g", s)
    if g:
        mg = int(float(g[0]) * 1000)
        return f"{mg}mg"

    ml = re.findall(r"(\d+)\s*ml", s)
    if ml: return f"{ml[0]}ml"

    return None


# ============================================
# Routes
# ============================================

route_map = {
    "oral": ["po", "uong", "uong", "uống"],
    "injection": ["tiem", "tiêm", "inj", "iv", "im"],
}

def expand_routes(routes):
    out = set()
    for r in routes:
        r = normalize(r)
        if not r: continue
        out.add(r)
        for k, v in route_map.items():
            if r == k or r in v:
                out.update(v)
                out.add(k)
    return sorted(list(out))

# -------- Dose extractor --------
def extract_doses(forms):
    """
    Extract dose from forms:
    - 1.25mg / 2.5mg
    - 1g → 1000mg
    - 12 mcg / 200mcg
    - 1% / 0.1%
    - IU / iu / ui
    """
    doses = set()

    for f in forms:
        f = normalize(f)

        # --- 1) mg including decimals ---
        mg = re.findall(r'(\d+(?:\.\d+)?)\s*mg', f)
        for x in mg:
            doses.add(f"{x}mg")
            doses.add(x)

            # add integer mg version if decimal
            try:
                mg_int = str(int(float(x)))
                doses.add(f"{mg_int}mg")
                doses.add(mg_int)
            except:
                pass

        # --- 2) gram → convert to mg ---
        g = re.findall(r'(\d+(?:\.\d+)?)\s*g', f)
        for x in g:
            mg_val = int(float(x) * 1000)
            doses.add(f"{mg_val}mg")
            doses.add(str(mg_val))
            doses.add(x)

        # --- 3) mcg → keep as mcg ---
        mcg = re.findall(r'(\d+(?:\.\d+)?)\s*mcg', f)
        for x in mcg:
            doses.add(f"{x}mcg")
            doses.add(x)

        # --- 4) % (nồng độ) ---
        percent = re.findall(r'(\d+(?:\.\d+)?)\s*%', f)
        for x in percent:
            doses.add(f"{x}%")
            doses.add(x)

        # --- 5) IU / iu / ui ---
        iu = re.findall(r'(\d+(?:\.\d+)?)\s*(?:iu|ui)', f)
        for x in iu:
            doses.add(f"{x}iu")
            doses.add(x)

        # --- 6) “x/lieu”, ví dụ 100 mcg/lieu ---
        per_dose = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mcg|mg)/lieu', f)
        for x in per_dose:
            doses.add(x)

        # --- 7) số đứng riêng (nhưng chỉ nếu gắn liền form) ---
        standalone_num = re.findall(r'\b(\d+(?:\.\d+)?)\b', f)
        for x in standalone_num:
            # tránh lấy mã số không liên quan
            if "." in x or len(x) <= 4: 
                doses.add(x)

    return sorted(doses)


# ============================================
# Build synonyms
# ============================================

def build_synonyms(row):
    syn = set()

    # multiple INN
    inns = [i.strip() for i in row["inn_clean"].split(",") if i.strip()]
    for inn in inns:
        inn_norm = normalize(inn)
        syn.add(inn_norm)
        for s in generate_short_forms(inn_norm):
            syn.add(s)

    # brand
    for b in row["brand_clean"]:
        b = normalize(b)
        syn.add(b)
        if len(b) > 4:
            syn.add(b[:4])
            syn.add(b[:5])

    # aliases
    for a in row["aliases"]:
        syn.add(normalize(a))

    return sorted(list(syn))


# ============================================
# Generate JSON with full metadata
# ============================================

def generate_synonym_file(csv_path, output_path="atc_synonyms_meta.json"):
    df = pd.read_csv(csv_path)

    grouped = {}

    for _, row in df.iterrows():
        code = row["atc_code"]

        if code not in grouped:
            grouped[code] = {
                "inn": set(),
                "brand": set(),
                "forms": set(),
                "routes": set(),
                "doses": set(),
                "drug_class": set(),
                "risk_tags": set(),
                "contraindications": set(),
                "specialties": set(),
                "aliases": set(),
                "is_pediatric_form": set(),
                "age_min": set(),
            }

        grouped[code]["inn"].add(normalize(row["inn_clean"]))

        for b in load_list(row["brand_clean"]):
            grouped[code]["brand"].add(b)

        for f in load_list(row["forms_clean"]):
            grouped[code]["forms"].add(f)

        for r in load_list(row["routes"]):
            grouped[code]["routes"].add(r)

        grouped[code]["drug_class"].add(normalize(row.get("drug_class_clean", "")))

        for tag in load_list(row.get("risk_tags", "")):
            grouped[code]["risk_tags"].add(tag)

        # Contraindications clean
        ci = str(row.get("contraindications", "")).strip()
        if ci.lower() not in ["nan", "none", ""]:
            for part in ci.split(";"):
                c = normalize(part)
                if c:
                    grouped[code]["contraindications"].add(c)

        for sp in load_list(row.get("specialties", "")):
            grouped[code]["specialties"].add(sp)

        for al in load_list(row.get("aliases", "")):
            grouped[code]["aliases"].add(al)

        ped = row.get("is_pediatric_form", "")
        if str(ped).lower() not in ["nan", "none", ""]:
            grouped[code]["is_pediatric_form"].add(bool(ped))

        age = row.get("age_min", "")
        if str(age).lower() not in ["nan", "none", ""]:
            try:
                grouped[code]["age_min"].add(int(float(age)))
            except:
                pass

    # Build final JSON
    out = {}

    for code, data in grouped.items():
        inn_clean = ", ".join(sorted([i for i in data["inn"] if i]))
        drug_class_list = [d for d in data["drug_class"] if d]
        drug_class = drug_class_list[0] if drug_class_list else ""

        is_ped = True if True in data["is_pediatric_form"] else False

        age_vals = sorted([a for a in data["age_min"] if isinstance(a, int)])
        age_min = age_vals[0] if age_vals else None

        merged_row = {
            "inn_clean": inn_clean,
            "brand_clean": sorted(data["brand"]),
            "forms_clean": sorted(data["forms"]),
            "routes": sorted(data["routes"]),
            "aliases": sorted(data["aliases"]),
        }

        synonyms = build_synonyms(merged_row)
        synonyms = clean_whitespace_list(synonyms)


        out[code] = {
            "inn": inn_clean,
            "brand": merged_row["brand_clean"],
            "forms": merged_row["forms_clean"],
            "routes": merged_row["routes"],
            "doses": extract_doses(merged_row["forms_clean"]),
            "drug_class": drug_class,
            "risk_tags": sorted(data["risk_tags"]),
            "contraindications": sorted(data["contraindications"]),
            "specialties": sorted(data["specialties"]),
            "is_pediatric_form": is_ped,
            "age_min": age_min,
            "aliases": merged_row["aliases"],
            "synonyms": synonyms,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("✓ DONE generating", output_path)


if __name__ == "__main__":
    generate_synonym_file("atc_clean.csv")

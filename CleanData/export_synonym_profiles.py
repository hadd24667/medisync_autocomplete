import json
import re
import os
from unidecode import unidecode

INPUT_FILE = "../atc_synonyms_meta_cleaned.json"              # đổi tên nếu cần
OUTPUT_FILE = "atc_metadata_with_profiles.json"


def build_synonym_profiles(code: str, meta: dict):
    """
    Sinh synonym_profiles cho 1 ATC code, dựa vào:
      - inn (chuỗi CSV)
      - brand (list)
      - synonyms (list)
    Trả về dict:
      { normalized_syn: {"source": <inn|brand|code>, "value": <canonical>} }
    """
    profiles = {}

    raw_inn = meta.get("inn") or ""
    inn_list = [s.strip() for s in raw_inn.split(",") if s.strip()]
    brand_list = meta.get("brand") or []
    synonyms = meta.get("synonyms") or []

    def norm(s: str):
        return unidecode(s.lower().strip())

    # map brand_norm -> canonical brand
    brand_map = {norm(b): b for b in brand_list}

    # map inn_norm -> inn canonical (bỏ ngoặc)
    inn_map = {}
    for inn in inn_list:
        inn_norm = norm(inn)
        base = re.sub(r"\(.*?\)", "", inn).strip()
        inn_canon = base if base else inn
        inn_map[inn_norm] = inn_canon

    # fallback canonical
    if inn_list:
        cleaned = []
        for inn in inn_list:
            base = re.sub(r"\(.*?\)", "", inn).strip() or inn
            cleaned.append(base)
        default_canonical = min(cleaned, key=len)
    elif brand_list:
        default_canonical = brand_list[0]
    else:
        default_canonical = code

    for syn in synonyms:
        s_norm = norm(syn)
        if not s_norm:
            continue

        # (1) brand match
        matched = False
        for b_norm, b_canon in brand_map.items():
            if b_norm.startswith(s_norm) or s_norm.startswith(b_norm):
                profiles[s_norm] = {"source": "brand", "value": b_canon}
                matched = True
                break
        if matched:
            continue

        # (2) inn match
        for i_norm, i_canon in inn_map.items():
            if i_norm.startswith(s_norm) or s_norm.startswith(i_norm):
                profiles[s_norm] = {"source": "inn", "value": i_canon}
                matched = True
                break
        if matched:
            continue

        # (3) fallback
        profiles[s_norm] = {"source": "code", "value": default_canonical}

    return profiles


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Metadata file not found: {INPUT_FILE}")
        return

    print(f"📥 Loading: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = {}

    for code, meta in raw.items():
        code_norm = code.strip().lower()
        syn_profiles = build_synonym_profiles(code, meta)

        new_meta = dict(meta)
        new_meta["synonym_profiles"] = syn_profiles

        out[code_norm] = new_meta

    # ghi file mới
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Exported {len(out):,} codes → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

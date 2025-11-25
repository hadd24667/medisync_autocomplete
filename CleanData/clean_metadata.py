import json

# ==== CONFIG ====
INPUT_FILE  = "../atc_synonyms_meta.json"
OUTPUT_FILE = "atc_synonyms_meta_cleaned.json"

# ===== LOAD =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

# ===== STEP 1: Lấy toàn bộ prefix của INN =====
def normalize(s):
    return s.lower().strip()

def get_prefixes(text):
    text = normalize(text)
    toks = text.replace("(", " ").replace(")", " ").split()
    prefixes = set()
    for t in toks:
        if len(t) >= 4:
            prefixes.add(t[:4])     # prefix 4 ký tự
            prefixes.add(t[:5])     # prefix 5 ký tự
    return prefixes

all_inn_prefixes = set()

for code, m in meta.items():
    inn = m.get("inn", "")
    prefixes = get_prefixes(inn)
    all_inn_prefixes.update(prefixes)

# ===== STEP 2: Xóa synonym rác =====
removed_count = 0

for code, m in meta.items():
    inn_text = m.get("inn", "")
    inn_prefixes = get_prefixes(inn_text)

    clean_syns = []
    for syn in m.get("synonyms", []):
        s = normalize(syn)

        # Không xóa synonyms là chính INN của thuốc này
        if s in inn_prefixes:
            clean_syns.append(s)
            continue

        # Nếu synonym xuất phát từ prefix bị đụng INN thuốc khác → loại
        if s in all_inn_prefixes:
            removed_count += 1
            # print(f"REMOVE {s} from {code}")
            continue

        clean_syns.append(s)

    m["synonyms"] = clean_syns

# ===== SAVE =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"Done! Removed {removed_count} bad synonyms.")
print(f"Saved cleaned file → {OUTPUT_FILE}")

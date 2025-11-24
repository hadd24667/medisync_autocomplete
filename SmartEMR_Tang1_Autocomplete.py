import pandas as pd, time, random, string, pickle, os
from unidecode import unidecode
from underthesea import word_tokenize
from collections import defaultdict
from rapidfuzz import process, fuzz
import re

cache = {}
PERSIST_FILE = "smartemr_index.pkl"  # file lưu index đã build

# ============================================================
# 1️⃣ Đọc dữ liệu ICD-10
file_path = "icd10cm-codes-2026.txt"
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

data = []
for line in lines:
    line = line.strip()
    if not line:
        continue

    if "\t" in line:
        parts = line.split("\t")
    else:
        parts = re.split(r"\s{2,}", line)

    if len(parts) >= 2:
        code = parts[0].strip()
        desc_raw = parts[1].strip().lower()

        # ❗ xoá mã ICD nếu bị lặp ở đầu mô tả
        if desc_raw.startswith(code.lower()):
            desc = desc_raw[len(code):].strip()
        else:
            desc = desc_raw

        data.append({"code": code, "description": desc})


df = pd.DataFrame(data)
df.dropna(inplace=True)
df.to_csv("icd10cm_2026_clean.csv", index=False, encoding="utf-8-sig")

icd_df = pd.read_csv("icd10cm_2026_clean.csv")
icd_df.dropna(subset=["code", "description"], inplace=True)

# GHÉP MÃ + TÊN
all_terms = [
    f"{row['code']} {row['description']}"
    for _, row in icd_df.iterrows()
]



# ============================================================
# 2️⃣ Chuẩn hóa & viết tắt
abbr_map = {
    "đtd": "đái tháo đường",
    "tha": "tăng huyết áp",
    "copd": "bệnh phổi tắc nghẽn mạn tính",
    "suyth": "suy thận",
    "suytim": "suy tim",
    "tbmn": "tai biến mạch máu não",
}

def normalize_text(text):
    text = text.lower().strip()

    # mở rộng viết tắt
    for abbr, full in abbr_map.items():
        text = re.sub(r"\b" + re.escape(abbr) + r"\b", full, text)

    text = unidecode(text)

    # tách mã ICD riêng
    icd_codes = re.findall(r"[a-zA-Z]\d{2}(?:\.\d+)?", text)
    
    for code in icd_codes:
        text = text.replace(code, f" {code} ")

    tokens = word_tokenize(text)

    # hợp nhất token mã ICD lại
    tokens = [tok for tok in tokens if tok.strip()]
    
    return tokens


def make_fingerprint(text):
    bits = 0
    for ch in set(unidecode(text.lower())):
        if "a" <= ch <= "z":
            bits |= 1 << (ord(ch) - ord("a"))
    return bits

# ============================================================
# 3️⃣ Load lại index nếu có
if os.path.exists(PERSIST_FILE):
    with open(PERSIST_FILE, "rb") as f:
        ngram_index, inverted_index, term_fingerprints, prefix_cache = pickle.load(f)
    print(f"⚡ Đã load index từ {PERSIST_FILE}")
else:
    print("🚀 Chưa có index cache, đang build mới...")
    ngram_index = defaultdict(set)
    for idx, term in enumerate(all_terms):
        term_norm = unidecode(term.lower())
        for i in range(len(term_norm) - 2):
            ngram_index[term_norm[i:i+3]].add(idx)

    inverted_index = defaultdict(set)
    for idx, term in enumerate(all_terms):
        for tok in set(unidecode(term.lower()).split()):
            if len(tok) >= 2:
                inverted_index[tok[:2]].add(idx)
                if len(tok) >= 3:
                    inverted_index[tok[:3]].add(idx)

    term_fingerprints = [make_fingerprint(t) for t in all_terms]

    valid_prefixes = list(string.ascii_lowercase) + list(string.digits)

    prefix_cache = {
        ch: [t for t in all_terms if unidecode(t.lower()).startswith(ch)]
        for ch in valid_prefixes
    }


    # 🔹 Lưu lại để khởi động nhanh lần sau
    with open(PERSIST_FILE, "wb") as f:
        pickle.dump((ngram_index, inverted_index, term_fingerprints, prefix_cache), f)
    print(f"💾 Đã lưu index cache vào {PERSIST_FILE}")

print(f"Hoàn tất build 3-gram & prefix index ({len(inverted_index):,} keys).")

# ============================================================
# 4️⃣ Multi-Prefix Matching (MPM) & Fuzzy fallback
def mpm_score(query_tokens, term_tokens):
    m, n = len(query_tokens), len(term_tokens)
    if n == 0: return 0
    score_sum = 0
    for q in query_tokens:
        for t in term_tokens:
            if t.startswith(q):
                score_sum += len(q)/len(t)
                break
    return score_sum / n

def mpm_optimized(query_tokens):
    candidate_sets = []
    for tok in query_tokens:
        key = tok[:3] if len(tok) >= 3 else tok[:2]
        if key in inverted_index:
            candidate_sets.append(inverted_index[key])
    if not candidate_sets: return []
    candidate_ids = set.intersection(*candidate_sets)
    mask_q = make_fingerprint(" ".join(query_tokens))
    candidate_ids = [i for i in candidate_ids if (mask_q & term_fingerprints[i]) == mask_q]
    scored = []
    for i in candidate_ids:
        term_tokens = unidecode(all_terms[i].lower()).split()
        score = mpm_score(query_tokens, term_tokens)
        scored.append((all_terms[i], score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:30]]

def fuzzy_fallback(query):
    q = unidecode(query.lower())
    qgrams = {q[i:i+3] for i in range(len(q)-2)} if len(q) >= 3 else {q}
    candidate_ids = set()
    for g in qgrams:
        candidate_ids |= ngram_index.get(g, set())
    if len(candidate_ids) < 20:
        for key, ids in inverted_index.items():
            if key in q or q.startswith(key) or key.startswith(q[:2]):
                candidate_ids |= ids
    if not candidate_ids:
        candidate_ids = set(range(min(2000, len(all_terms))))
    subset = [all_terms[i] for i in list(candidate_ids)[:5000]]
    results = [x[0] for x in process.extract(q, subset, scorer=fuzz.token_set_ratio, limit=30, score_cutoff=75)]
    return results

# ============================================================
# 5️⃣ API chính
def format_results(candidates):
    results = []
    for term in candidates:
        parts = term.split(" ", 1)
        code = parts[0]
        desc = parts[1] if len(parts) > 1 else ""
        results.append({
            "code": code,
            "description": desc,
            "display": f"{code} – {desc}"
        })
    return results


def autocomplete(query):
    t0 = time.time()
    query_norm = unidecode(query.lower().strip())

    # 1️⃣ Prefix cache (a-z, 0-9)
    if len(query_norm) == 1 and query_norm in prefix_cache:
        results = format_results(prefix_cache[query_norm][:30])
        return results, "tier-0 cache", round((time.time() - t0) * 1000, 2)

    # 2️⃣ Cache full format
    if query_norm in cache:
        return cache[query_norm], "cache", round((time.time() - t0) * 1000, 2)

    # 3️⃣ Normalize → Tokens
    tokens = normalize_text(query)
    if not tokens:
        return [], "empty", 0

    # 4️⃣ MPM + fuzzy
    candidates = mpm_optimized(tokens)
    if not candidates:
        candidates = fuzzy_fallback(query_norm)

    # 5️⃣ Format output JSON
    results = format_results(candidates)

    # 6️⃣ Lưu cache (dạng JSON, không phải raw string)
    cache[query_norm] = results

    latency = round((time.time() - t0) * 1000, 2)
    return results, "retrieved", latency



def clear_cache_data():
    global cache
    n = len(cache)
    cache = {}
    return n

import os
import time
import re
import string
import pickle
from collections import defaultdict
import json
import pandas as pd
from sqlalchemy import create_engine
from unidecode import unidecode
import ast
from rapidfuzz import process, fuzz
from synonym_profile import (
    normalize_text,
    build_synonym_profiles,
    pick_matched_syn_from_query,
)

# ============================================
# 0. CONFIG
# ============================================

POSTGRES_URL = "postgresql://postgres:12345678@localhost:5432/medisync"

# File cache index (để khởi động nhanh hơn)
PERSIST_ICD_FILE = "smartemr_tier1_icd11_index.pkl"
PERSIST_ATC_FILE = "smartemr_tier1_atc_index.pkl"
ATC_META_FILE = "atc_synonyms_meta_cleaned.json" 

def load_atc_metadata(path: str):
    """
    Load metadata tầng 2 cho ATC từ JSON:
    { "N02BE01": { ... }, ... }
    → map về dict key là atc_code lower-case
      + tự sinh synonym_profiles.
    """
    if not os.path.exists(path):
        print(f"⚠️ ATC metadata file not found: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    meta_by_code = {}
    for code, meta in raw.items():
        if not code:
            continue

        meta = meta or {}
        # sinh synonym_profiles từ inn + brand + synonyms
        meta["synonym_profiles"] = build_synonym_profiles(code, meta)

        meta_by_code[code.lower()] = meta

    print(f"💊 Loaded ATC metadata JSON: {len(meta_by_code):,} codes")
    return meta_by_code


# ============================================
# 1. UTILS & NORMALIZATION
# ============================================

abbr_map = {
    # mở rộng một số viết tắt thường gặp trong hồ sơ bệnh án
    "đtd": "dai thao duong",
    "dtđ": "dai thao duong",
    "tha": "tang huyet ap",
    "copd": "benh phoi tac nghen man tinh",
    "suyth": "suy than",
    "suytim": "suy tim",
    "tbmn": "tai bien mach mau nao",
}


def normalize_query(text: str):
    """
    Chuẩn hoá query từ bác sĩ:
    - lowercase + strip
    - unidecode
    - mở rộng viết tắt (đtd, tha, ...)
    - tách đơn giản theo space
    """
    if not text:
        return []

    text = text.lower().strip()
    text = unidecode(text)

    # mở rộng viết tắt
    for abbr, full in abbr_map.items():
        text = re.sub(r"\b" + re.escape(abbr) + r"\b", full, text)

    # chỉ giữ chữ, số và space
    text = re.sub(r"[^a-z0-9\s\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [t for t in text.split(" ") if t]
    return tokens


def make_fingerprint(text: str) -> int:
    """
    Fingerprint 26-bit để lọc nhanh:
    - bit i = 1 nếu trong text có ký tự 'a'+i
    """
    bits = 0
    for ch in set(unidecode(text.lower())):
        if "a" <= ch <= "z":
            bits |= 1 << (ord(ch) - ord("a"))
    return bits

def _choose_canonical_inn(inn_raw: str) -> str:
    """
    Chọn 1 INN 'đẹp' nhất từ chuỗi inn_raw (phân tách bởi dấu phẩy).
    Ưu tiên dạng ngắn, không ngoặc.
    """
    if not inn_raw:
        return ""

    inns = [s.strip() for s in inn_raw.split(",") if s.strip()]
    if not inns:
        return ""

    cleaned = []
    for inn in inns:
        base = re.sub(r"\(.*?\)", "", inn).strip()
        cleaned.append(base if base else inn)

    # chọn cái ngắn nhất
    return min(cleaned, key=len)

def choose_display_name(code: str, matched_syn: str | None, meta: dict) -> str:
    inn_raw = meta.get("inn_raw") or ""
    brands_orig = meta.get("brand_names") or []
    profiles = meta.get("synonym_profiles") or {}
    print("[DEBUG] choose_display_name:")
    print(repr(inn_raw), repr(brands_orig))

    def _cap_first(s: str) -> str:
        if not s:
            return s
        return s[0].upper() + s[1:]

    # Chuẩn hoá lại case cho INN + brand để hiển thị
    inn_disp = _cap_first(inn_raw)
    brands_disp = [_cap_first(b) for b in brands_orig]

    # chuẩn hoá matched_syn
    key = None
    if matched_syn:
        key = unidecode(matched_syn.lower().strip())

    # Nếu synonym match → chỉ dùng để chọn canonical,
    # KHÔNG dùng value trong profile (vì đó là bản clean)
    if key and key in profiles:
        source = profiles[key].get("source")

        # Nếu match brand → hiển thị Brand (INN gốc)
        if source == "brand":
            if brands_disp:
                # ví dụ matched_syn = 'pana' → Panadol (Paracetamol)
                candidates = [b for b in brands_disp if key in unidecode(b.lower())]
                brand_show = candidates[0] if candidates else brands_disp[0]
                return f"{brand_show} ({inn_disp})"
            return inn_disp or code

        # Nếu match inn → dùng INN đã chuẩn hóa hoa/thường
        if source == "inn":
            return inn_disp or code

        # Nếu profile 'code' → fallback canonical
        return inn_disp or code

    # Không có synonym match → bản gốc (INN + brand) đã chuẩn hoá case
    if brands_disp:
        return f"{inn_disp} ({', '.join(brands_disp)})"

    return inn_disp or code


# ============================================
# 2. CORE ENGINE
# ============================================

class Tier1AutocompleteEngine:
    """
    Engine Tầng 1:
    - Multi-Prefix Matching (MPM) + inverted index
    - 3-gram index cho fuzzy fallback
    - 26-bit fingerprint filter
    - prefix cache cho query 1 ký tự
    """

    def __init__(self, all_terms, meta_by_index, persist_file=None):
        """
        all_terms: list[str]  - chuỗi index hoá, ví dụ: "1a00 benh ta"
        meta_by_index: list[dict] - metadata để trả về cho UI
        persist_file: đường dẫn file .pkl để lưu index
        """
        self.all_terms = all_terms
        self.meta_by_index = meta_by_index
        self.persist_file = persist_file
        self.cache = {}

        # index structures
        self.ngram_index = defaultdict(set)
        self.inverted_index = defaultdict(set)
        self.term_fingerprints = []
        self.prefix_cache = {}

        # build / load index
        self._init_index()

    # -----------------------------
    # 2.1. Build / Load index
    # -----------------------------
    def _init_index(self):
        if self.persist_file and os.path.exists(self.persist_file):
            with open(self.persist_file, "rb") as f:
                data = pickle.load(f)

            # Hỗ trợ cả version cũ (6 phần tử) và mới (7 phần tử)
            if len(data) == 6:
                (
                    self.all_terms,
                    self.meta_by_index,
                    self.ngram_index,
                    self.inverted_index,
                    self.term_fingerprints,
                    self.prefix_cache,
                ) = data
                # build token_vocab từ đầu
                self._build_token_vocab()
            else:
                (
                    self.all_terms,
                    self.meta_by_index,
                    self.ngram_index,
                    self.inverted_index,
                    self.term_fingerprints,
                    self.prefix_cache,
                    self.token_vocab,
                ) = data

            print(f"⚡ Loaded index from {self.persist_file}")
            return

        print("Building index from scratch ...")

        # 3-gram index cho fuzzy fallback
        for idx, term in enumerate(self.all_terms):
            norm = unidecode(term.lower())
            for i in range(len(norm) - 2):
                trigram = norm[i : i + 3]
                self.ngram_index[trigram].add(idx)

        # inverted index cho multi-prefix (2-3 ký tự đầu của token)
        for idx, term in enumerate(self.all_terms):
            tokens = unidecode(term.lower()).split()
            for tok in set(tokens):
                if len(tok) >= 2:
                    self.inverted_index[tok[:2]].add(idx)
                if len(tok) >= 3:
                    self.inverted_index[tok[:3]].add(idx)

        # fingerprint 26-bit/filter
        self.term_fingerprints = [make_fingerprint(t) for t in self.all_terms]

        # prefix cache cho query dài 1 ký tự (a-z, 0-9)
        valid_prefixes = list(string.ascii_lowercase) + list(string.digits)
        self.prefix_cache = {
            ch: [
                i
                for i, t in enumerate(self.all_terms)
                if unidecode(t.lower()).startswith(ch)
            ]
            for ch in valid_prefixes
        }

        # 🔹 build token vocab cho spell-correction
        self._build_token_vocab()

        # lưu lại
        if self.persist_file:
            with open(self.persist_file, "wb") as f:
                pickle.dump(
                    (
                        self.all_terms,
                        self.meta_by_index,
                        self.ngram_index,
                        self.inverted_index,
                        self.term_fingerprints,
                        self.prefix_cache,
                        self.token_vocab,
                    ),
                    f,
                )
            print(f" Saved index cache to {self.persist_file}")

        print(
            f" Done building index: {len(self.all_terms):,} terms, "
            f"{len(self.inverted_index):,} prefix keys."
        )
    

    # -----------------------------
    # 2.2. Multi-Prefix Matching
    # -----------------------------
    @staticmethod
    def _mpm_score(query_tokens, term_tokens):
        """
        Multi-Prefix-Matching score.
        Thuật toán ưu tiên:
        - Match prefix ngắn, gần giống hơn
        - Phạt từ dài
        - Phạt số lượng token nhiều
        - Thưởng match liên tục
        """
        if not term_tokens:
            return 0.0

        score = 0.0

        for q in query_tokens:
            best = 0.0

            for t in term_tokens:
                # prefix match mạnh
                if t.startswith(q):
                    # phần thưởng: match càng gần -> càng cao
                    hit = len(q) / len(t)

                    # phạt từ dài: t càng dài – penalize giảm
                    hit *= 1 / (1 + (len(t) - len(q)) * 0.4)

                else:
                    # không prefix: fuzzy (bonus rất nhẹ)
                    from rapidfuzz.fuzz import partial_ratio
                    pr = partial_ratio(q, t) / 100
                    hit = pr * 0.2      # fuzzy cap thấp

                best = max(best, hit)

            score += best

        # phạt theo số lượng token của term
        score = score / (len(term_tokens) ** 0.8)

        return score


    def _mpm_optimized(self, query_tokens):
        # B1: lấy candidate set cho từng token từ inverted index
        candidate_sets = []
        for tok in query_tokens:
            key = tok[:3] if len(tok) >= 3 else tok[:2]
            if key in self.inverted_index:
                candidate_sets.append(self.inverted_index[key])

        if not candidate_sets:
            return []

        # B2: giao các tập ứng viên
        candidate_ids = set.intersection(*candidate_sets)

        # B3: lọc nhanh bằng fingerprint
        mask_q = make_fingerprint(" ".join(query_tokens))
        candidate_ids = [
            i for i in candidate_ids if (mask_q & self.term_fingerprints[i]) == mask_q
        ]

        # B4: scoring
        scored = []
        for i in candidate_ids:
            term_tokens = unidecode(self.all_terms[i].lower()).split()
            score = self._mpm_score(query_tokens, term_tokens)
            if score > 0:
                scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [i for i, s in scored[:30]]
        return top_ids

    # -----------------------------
    # 2.3. Fuzzy fallback
    # -----------------------------
    def _fuzzy_fallback(self, query_norm: str):
        q = query_norm
        if len(q) >= 3:
            qgrams = {q[i : i + 3] for i in range(len(q) - 2)}
        else:
            qgrams = {q}

        candidate_ids = set()
        for g in qgrams:
            candidate_ids |= self.ngram_index.get(g, set())

        # nếu quá ít → mở rộng thêm inverted index
        if len(candidate_ids) < 20:
            for key, ids in self.inverted_index.items():
                if key in q or q.startswith(key) or key.startswith(q[:2]):
                    candidate_ids |= ids

        if not candidate_ids:
            # fallback rất mạnh: lấy 2k term đầu (vẫn nhanh vì đã load vào RAM)
            candidate_ids = set(range(min(2000, len(self.all_terms))))

        # RapidFuzz với choices = dict {idx: term} để giữ nguyên index
        choices = {i: self.all_terms[i] for i in candidate_ids}
        matches = process.extract(
            q,
            choices,
            scorer=fuzz.token_set_ratio,
            limit=30,
            score_cutoff=75,
        )

        # matches: list of (term_string, score, key)
        top_ids = [key for (_, score, key) in matches]
        return top_ids
    
    def _build_token_vocab(self):
        """
        Tạo vocabulary token từ toàn bộ all_terms để phục vụ spell-correction.
        """
        vocab = set()
        for term in self.all_terms:
            for tok in unidecode(term.lower()).split():
                if tok:
                    vocab.add(tok)
        self.token_vocab = vocab
        print(f"📌 Built token vocab: {len(self.token_vocab):,} unique tokens")

    def _correct_tokens(self, tokens):
        """
        Sửa lỗi chính tả đơn giản cho từng token:
        - Dùng RapidFuzz WRatio trên token_vocab
        - Chỉ sửa token dài >= 4
        - Ngưỡng similarity: 85
        """
        if not getattr(self, "token_vocab", None):
            return tokens

        corrected = []
        for t in tokens:
            if len(t) < 4:
                corrected.append(t)
                continue

            # tìm token gần nhất trong vocab
            match = process.extractOne(
                t,
                self.token_vocab,
                scorer=fuzz.WRatio,
                score_cutoff=85,
            )
            if match:
                best_tok, score, _ = match
                corrected.append(best_tok)
            else:
                corrected.append(t)

        return corrected


    # -----------------------------
    # 2.4. Format kết quả cho UI
    # -----------------------------
    def _format_results(self, idx_list, matched_syn=None):
        """
        Format kết quả hiển thị.
        matched_syn: chuỗi synonym mà user đã gõ (nếu có)
        """
        results = []
        for i in idx_list:
            meta = self.meta_by_index[i]
            term = self.all_terms[i]

            t_type = meta.get("type")
            code = meta.get("code", "")
            desc = meta.get("description", "")

            # -----------------------------
            #  CHỌN LABEL HIỂN THỊ (DYNAMIC)
            # -----------------------------
            if t_type == "ATC":
                # dynamic: canonical tùy vào synonym match
                display_name = choose_display_name(
                    code=code,
                    matched_syn=matched_syn,
                    meta=meta
                )
            else:
                # ICD11 giữ nguyên
                display_name = meta.get("label", "")

            # -----------------------------
            #  FORMAT FORMS (nếu là ATC)
            # -----------------------------
            forms = meta.get("forms") or []
            forms_str = ", ".join(forms) if forms else ""

            display = f"{code} – {display_name}".strip(" –")
            if t_type == "ATC" and forms_str:
                display = f"{display} [{forms_str}]"

            results.append(
                {
                    "type": t_type,
                    "code": code,
                    "label": display_name,      # tên hiển thị mới
                    "display_name": display_name,
                    "description": desc,
                    "forms": forms,
                    "display": display,
                    "matched_syn": matched_syn,   # để FE debug
                    "raw_term": term,
                }
            )
        return results


    # -----------------------------
    # 2.5. Public API
    # -----------------------------
    def autocomplete(self, query: str):
        """
        Autocomplete với pipeline:
        1. Query 1 ký tự → prefix cache
        2. Cache full query
        3. Normalize + tokenize
        4. MPM với tokens gốc
        5. MPM với tokens đã sửa lỗi chính tả
        6. Relax-query (bỏ 1 token) + MPM
        7. Fuzzy fallback
        """
        t0 = time.time()
        query_norm = unidecode(query.lower().strip())

        # Case 1: query 1 ký tự → dùng prefix cache
        if len(query_norm) == 1 and query_norm in self.prefix_cache:
            idxs = self.prefix_cache[query_norm][:30]
            results = self._format_results(idxs)
            latency = round((time.time() - t0) * 1000, 2)
            return results, "tier-0-prefix", latency

        # Case 2: cache full query
        if query_norm in self.cache:
            results = self.cache[query_norm]
            latency = round((time.time() - t0) * 1000, 2)
            return results, "cache", latency

        # Case 3: chuẩn hoá + token hoá query
        tokens = normalize_query(query)
        if not tokens:
            return [], "empty", 0.0

        # Case 4: Multi-Prefix Matching trên tokens gốc
        idxs = self._mpm_optimized(tokens)
        source = "mpm"

        # Case 5: Thử sửa lỗi chính tả cho từng token rồi MPM lại
        if not idxs:
            corrected_tokens = self._correct_tokens(tokens)
            if corrected_tokens != tokens:
                idxs = self._mpm_optimized(corrected_tokens)
                source = "mpm-corrected"

        # Case 6: Relax-query (bỏ 1 token) nếu vẫn chưa có gì
        if not idxs and len(tokens) >= 2:
            best_ids = []
            for i in range(len(tokens)):
                relaxed = tokens[:i] + tokens[i+1:]
                cand_ids = self._mpm_optimized(relaxed)
                if cand_ids and (not best_ids or len(cand_ids) > len(best_ids)):
                    best_ids = cand_ids
            if best_ids:
                idxs = best_ids
                source = "mpm-relax"

        # Case 7: Fuzzy fallback nếu tất cả MPM đều fail
        if not idxs:
            idxs = self._fuzzy_fallback(query_norm)
            source = "fuzzy"

        # lấy "synonym" mà user gõ – tạm thời dùng token đầu tiên
        matched_syn = None
        tokens_for_syn = normalize_query(query)
        if tokens_for_syn:
            matched_syn = tokens_for_syn[0]

        results = self._format_results(idxs, matched_syn=matched_syn)
        self.cache[query_norm] = results


        latency = round((time.time() - t0) * 1000, 2)
        return results, source, latency


# ============================================
# 3. LOAD DATA TỪ POSTGRES
# ============================================

def load_icd11_terms(engine):

    sql = """
        SELECT
            icd11_code,
            title_vn,
            title_vn_clean,
            title_vn_display,
            icd10_code_source
        FROM icd11_clean
    """

    df = pd.read_sql_query(sql, engine)

    all_terms = []
    meta = []

    for _, row in df.iterrows():
        code = (row["icd11_code"] or "").strip()

        title_vn_display = (row["title_vn_display"] or "").strip()
        title_vn = (row["title_vn"] or "").strip()
        title_vn_clean = (row["title_vn_clean"] or "").strip()

        # ===== LABEL is always Vietnamese =====
        label = title_vn_display or title_vn
        if not label:
            # fallback: dùng sạch không dấu
            label = title_vn_clean.capitalize()

        if not label:
            continue  # skip entry with no label

        # ===== SEARCH BASE: only Vietnamese no-diacritic =====
        base = unidecode(title_vn_clean.lower())

        if not base:
            continue

        term_str = f"{base}".strip()

        all_terms.append(term_str)
        meta.append({
            "type": "ICD11",
            "code": code,
            "label": label,   # FE hiển thị tên bệnh
            "description": "",
            "forms": []
        })

    print(f"📚 Loaded ICD-11 terms: {len(all_terms):,}")
    return all_terms, meta



def parse_pg_array(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val  # đã là list thì return luôn

    s = str(val).strip()

    # Remove outer braces
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]

    if not s:
        return []

    # Split respecting quoted strings
    items = []
    current = ''
    in_quotes = False

    for char in s:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            items.append(current.strip().strip('"'))
            current = ''
        else:
            current += char

    if current:
        items.append(current.strip().strip('"'))

    return [x for x in items if x]


def load_atc_terms(engine, atc_meta_by_code):

    sql = """
        SELECT
            atc_code,
            inn,
            inn_clean,
            drug_class,
            drug_class_clean,
            forms,
            forms_clean,
            brand_names,
            brand_clean
        FROM atc_clean
    """

    df = pd.read_sql_query(sql, engine)

    all_terms = []
    meta = []

    for _, row in df.iterrows():
        code = (row["atc_code"] or "").strip()
        if not code:
            continue

        inn = (row["inn"] or "").strip()
        inn_clean = (row["inn_clean"] or "").strip()

        forms = parse_pg_array(row["forms"])
        forms_clean = parse_pg_array(row["forms_clean"])
        brands = parse_pg_array(row["brand_names"])
        brands_clean = parse_pg_array(row["brand_clean"])

        drug_class = (row["drug_class"] or "").strip()
        drug_class_clean = (row["drug_class_clean"] or "").strip()

        # 🔹 lấy metadata tầng 2 nếu có
        meta_json = atc_meta_by_code.get(code.lower(), {})
        synonyms = meta_json.get("synonyms", []) or []
        synonym_profiles = meta_json.get("synonym_profiles", {}) or {}

        # label hiển thị tạm thời (phase 1, canonical sau)
        label = inn
        if brands:
            label = f"{inn} ({', '.join(brands)})"

        # search_base cho tầng 1 (đã có syn)
        brands_clean_str = " ".join(brands_clean)
        syn_str = " ".join(unidecode(s.lower()) for s in synonyms)

        search_base = f"{inn_clean} {drug_class_clean} {brands_clean_str} {syn_str}".strip()
        search_base = unidecode(search_base.lower())

        term_str = f"{code} {search_base}".strip()
        all_terms.append(term_str)

        meta.append({
            "type": "ATC",
            "code": code,
            "label": label,
            "description": drug_class,
            "forms": forms,
            # để dành cho Phase 2:
            "synonyms": synonyms,
            "synonym_profiles": synonym_profiles,
            "inn_raw": inn,
            "brand_names": brands,
        })

    print(f"💊 Loaded ATC terms: {len(all_terms):,}")
    return all_terms, meta




# ============================================
# 4. INIT ENGINES (ICD11 + ATC)
# ============================================

# tạo engine DB
_db_engine = create_engine(POSTGRES_URL)

# load metadata ATC tầng 2 từ JSON
_atc_meta_json = load_atc_metadata(ATC_META_FILE)

# load vocab từ DB
_icd_terms, _icd_meta = load_icd11_terms(_db_engine)
_atc_terms, _atc_meta = load_atc_terms(_db_engine, _atc_meta_json)

# tạo 2 engine tầng 1
icd_engine = Tier1AutocompleteEngine(
    _icd_terms,
    _icd_meta,
    persist_file=PERSIST_ICD_FILE,
)

atc_engine = Tier1AutocompleteEngine(
    _atc_terms,
    _atc_meta,
    persist_file=PERSIST_ATC_FILE,
)


# ============================================
# 5. PUBLIC API
# ============================================

def autocomplete_icd(query: str):
    """
    Autocomplete ICD-11
    return: (results, source, latency_ms)
    """
    return icd_engine.autocomplete(query)


def autocomplete_atc(query: str):
    """
    Autocomplete ATC/thuốc
    return: (results, source, latency_ms)
    """
    return atc_engine.autocomplete(query)


# ============================================
# 6. DEMO CLI
# ============================================

def benchmark_autocomplete():
    test_icd_queries = [
        "tieu duong",
        "benh ta",
        "viem ruot",
        "hen phe quan",
        "roi loan lo lang",
        "ung thu phoi",
        "xuat huyet nao",
        "gan nhiem mo",
        "dot quy",
        "viem gan b"
    ]

    test_atc_queries = [
        "paracetamol",
        "ibuprofen",
        "zovirax",
        "aciclovir",
        "metformin",
        "aspirin",
        "amoxicillin",
        "ceftriaxone",
        "omeprazol",
        "atorvastatin"
    ]

    print("\n==============================")
    print("🔥 BENCHMARK ICD-11 (10 queries)")
    print("==============================")
    icd_times = []

    for q in test_icd_queries:
        print(f"\n🔎 Query ICD: '{q}'")
        results, source, ms = autocomplete_icd(q)
        icd_times.append(ms)

        print(f"⏱  Latency: {ms} ms | Source: {source} | Total results: {len(results)}")
        print("📌  Results:")
        for idx, r in enumerate(results):
            print(f"  [{idx+1}] {r['display']}")

    print(f"\n👉 ICD avg latency: {sum(icd_times)/len(icd_times):.2f} ms\n")

    print("==============================")
    print("💊 BENCHMARK ATC (10 queries)")
    print("==============================")
    atc_times = []

    for q in test_atc_queries:
        print(f"\n🔎 Query ATC: '{q}'")
        results, source, ms = autocomplete_atc(q)
        atc_times.append(ms)

        print(f"⏱  Latency: {ms} ms | Source: {source} | Total results: {len(results)}")
        print("📌  Results:")
        for idx, r in enumerate(results):
            forms = f" [{', '.join(r['forms'])}]" if r['forms'] else ""
            print(f"  [{idx+1}] {r['display']}")

    print(f"\n👉 ATC avg latency: {sum(atc_times)/len(atc_times):.2f} ms\n")

def benchmark_misspell():
    """
    Benchmark các query bị sai chính tả cho ICD & ATC
    để đo khả năng chịu lỗi của tầng 1.
    """
    icd_misspell = [
        "tieu duonh",       # tiểu đường
        "benh ta1",         # bệnh tả
        "viem ruojt",       # viêm ruột
        "hen phe quanm",    # hen phế quản
        "roi loan lo lagn", # rối loạn lo âu
        "ung thu phok",     # ung thư phổi
        "xuat huyet nao0",  # xuất huyết não
        "gan nhiem mooj",   # gan nhiễm mỡ
        "dot quyy",         # đột quỵ
        "viem gan bb",      # viêm gan B
    ]

    atc_misspell = [
        "paracetemol",      # paracetamol
        "ibuprofeen",       # ibuprofen
        "zovriax",          # zovirax
        "acilcovir",        # aciclovir
        "metfromin",        # metformin
        "aspiriin",         # aspirin
        "amoxcillin",       # amoxicillin
        "ceftraixone",      # ceftriaxone
        "omeprazoll",       # omeprazol
        "atorvasatin",      # atorvastatin
    ]

    print("\n==============================")
    print("🔥 BENCHMARK ICD-11 (MISSPELL)")
    print("==============================")
    icd_times = []

    for q in icd_misspell:
        print(f"\n🔎 Query ICD (misspell): '{q}'")
        results, source, ms = autocomplete_icd(q)
        icd_times.append(ms)
        print(f"⏱  Latency: {ms} ms | Source: {source} | Total results: {len(results)}")
        print("📌  Top 3:")
        for idx, r in enumerate(results[:3]):
            print(f"  [{idx+1}] {r['display']}")

    print(f"\n👉 ICD misspell avg latency: {sum(icd_times)/len(icd_times):.2f} ms\n")

    print("==============================")
    print("💊 BENCHMARK ATC (MISSPELL)")
    print("==============================")
    atc_times = []

    for q in atc_misspell:
        print(f"\n🔎 Query ATC (misspell): '{q}'")
        results, source, ms = autocomplete_atc(q)
        atc_times.append(ms)
        print(f"⏱  Latency: {ms} ms | Source: {source} | Total results: {len(results)}")
        print("📌  Top 3:")
        for idx, r in enumerate(results[:3]):
            forms = f" [{', '.join(r['forms'])}]" if r['forms'] else ""
            print(f"  [{idx+1}] {r['display']}{forms}")

    print(f"\n👉 ATC misspell avg latency: {sum(atc_times)/len(atc_times):.2f} ms\n")


# Gọi benchmark sau khi khởi động engine
if __name__ == "__main__":
    print("SmartEMR Tier-1 Autocomplete Benchmark")
    benchmark_autocomplete()
    benchmark_misspell()


import pandas as pd
import re
from unidecode import unidecode
from sqlalchemy import create_engine  

# ==============================
# 1. UTILS
# ==============================

def clean_text(x):
    """
    Chuẩn hóa:
    - lowercase
    - unidecode (bỏ dấu)
    - strip whitespace
    - remove extra spaces
    """
    if pd.isna(x):
        return ""
    x = unidecode(str(x)).lower().strip()
    x = " ".join(x.split())
    return x


def tokenize(s):
    if not s:
        return []
    s = clean_text(s)
    s = re.sub(r"[^\w\s/%]", "", s)  # XÓA dấu ., ; …
    return [t for t in s.split(" ") if t]


def remove_hierarchy_prefix(title):
    """
    ICD-11 có các dòng bắt đầu bằng '- ', '- - ', '- - - ' cho block
    => bỏ hết prefix '-'
    """
    if pd.isna(title):
        return ""
    title = str(title).strip()
    while title.startswith("-"):
        title = title[1:].strip()
    return title


def split_list_clean(s):
    """
    Tách theo dấu ";" và chuẩn hóa từng mục.
    """
    if pd.isna(s):
        return []
    parts = [clean_text(p) for p in str(s).split(";") if p.strip() != ""]
    return parts


def normalize_forms_vn(s):
    """
    Chuẩn hóa dạng thuốc tiếng Việt:
    - Tách theo dấu ;
    - Ví dụ: "Viên nén 125 mg, 250 mg"
      -> ["Viên nén 125 mg", "Viên nén 250 mg"]
    - Giữ nguyên dạng đơn
    - Không xóa dấu thập phân (0.3%)
    - Bỏ item rỗng hoặc không có hàm lượng
    """

    if pd.isna(s):
        return []

    # Chỉ xoá dấu chấm ở cuối câu, không xoá 0.3%
    text = re.sub(r"\.$", "", str(s).strip())

    # Tách theo dấu ;
    parts = [p.strip() for p in text.split(";") if p.strip()]

    result = []

    for p in parts:
        # CASE 1: có dấu phẩy → nhiều hàm lượng
        if "," in p:
            items = [i.strip() for i in p.split(",") if i.strip()]

            first = items[0]

            # tìm base = phần trước số đầu tiên
            m = re.match(r"^(.*?)(\d+.*)$", first)
            if m:
                base = m.group(1).strip()
            else:
                tokens = first.split()
                idx = next((i for i, t in enumerate(tokens) if any(c.isdigit() for c in t)), None)
                base = " ".join(tokens[:idx]) if idx else first

            # Tạo từng item
            for it in items:
                if not re.search(r"\d", it):
                    continue  # Skip item không có số → tránh "Viên nén"

                qty = re.findall(r"\d+.*", it)
                if qty:
                    result.append(f"{base} {qty[0]}".strip())

        else:
            # CASE 2: dạng đơn
            if p.strip():
                result.append(p.strip())

    return result

def to_pg_array(lst):
    if not isinstance(lst, list):
        return "{}"
    return "{" + ",".join(f'"{str(x)}"' for x in lst) + "}"


# ========= ATC metadata detection =========

PEDIATRIC_KEYWORDS = ["siro", "hon dich", "com", "goi", "drops", "gói", "cốm", "giot"]


def detect_routes(forms):
    """
    Trả về list routes, không phải một route duy nhất.
    """
    routes = []
    forms_text = " ".join(forms).lower()

    if "tiem" in forms_text:
        routes.append("injection")

    if "xit" in forms_text or "khi dung" in forms_text or "hit" in forms_text:
        routes.append("inhalation")

    if "nho mat" in forms_text or "nho mui" in forms_text or "nho tai" in forms_text:
        routes.append("ophthalmic")

    # oral = mặc định nếu có dạng uống
    if any(x in forms_text for x in ["vien", "si ro", "hon dich", "goi", "com", "siro"]):
        routes.append("oral")

    # nếu rỗng, fallback oral
    if not routes:
        routes.append("oral")

    return list(set(routes))



def detect_pediatric(forms):
    """
    forms_clean: list string đã clean_text
    """
    forms_text = " ".join(forms).lower()
    return any(k in forms_text for k in PEDIATRIC_KEYWORDS)


def detect_age_min(contra):
    """
    Tìm tuổi tối thiểu từ cột Chống chỉ định. Các pattern ví dụ:
    - "Tre em duoi 18 tuoi"
    - "Khong dung cho tre < 2 tuoi"
    """
    text = clean_text(contra)

    patterns = [
        r"tre em duoi (\d+)",
        r"tre em < ?(\d+)",
        r"duoi (\d+) tuoi"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def detect_risk_tags(contra):
    """
    Từ chuỗi chống chỉ định → gán tag đơn giản:
    - 'gan'  → hepatic_toxic
    - 'than' → renal_adjust
    - 'thai' → pregnancy_risk
    """
    text = clean_text(contra)
    tags = []
    if "gan" in text:
        tags.append("hepatic_toxic")
    if "than" in text:
        tags.append("renal_adjust")
    if any(k in text for k in ["thai ky", "phu nu co thai", "thai nhi"]):
        tags.append("pregnancy_risk")
    return tags


# ========= ICD metadata detection =========

def detect_specialty(chapter_str: str):
    """
    Map chapter → specialty chính (sơ bộ, có thể refine sau).
    """
    chapter_str = str(chapter_str).zfill(2)
    if chapter_str.startswith("09"):
        return ["cardiology"]
    if chapter_str.startswith("01"):
        return ["infectious"]
    if chapter_str.startswith("05"):
        return ["neurology"]
    if chapter_str.startswith("12"):
        return ["gastro"]
    if chapter_str.startswith("26"):
        return ["obstetrics"]
    if chapter_str.startswith("28"):
        return ["pediatrics"]
    return []


def detect_icd_pediatric(title_vn_clean: str):
    """
    Đánh dấu bệnh nhi (thô, sẽ cải thiện sau).
    """
    t = title_vn_clean
    return any(k in t for k in ["tre em", "nhi", "so sinh"])


def detect_icd_chronic(title_vn_clean: str):
    """
    Đánh dấu bệnh mạn tính đơn giản.
    """
    t = title_vn_clean
    chronic_keywords = [
        "man tinh",
        "tang huyet ap",
        "dai thao duong",
        "copd",
        "suy than man",
        "benh ly man"
    ]
    return any(k in t for k in chronic_keywords)


# ==============================
# 2. CLEAN ICD-11
# ==============================

def clean_icd11(icd_file: str) -> pd.DataFrame:
    """
    Đọc file ICD11-Dataset.csv (theo cấu trúc WHO)
    → Chuẩn hóa cho search + thêm metadata tầng 2.
    """
    df = pd.read_csv(icd_file, encoding="utf-8", encoding_errors="replace")

    # Lọc bỏ dòng không có code (chapter/header)
    df = df[df["ICD11_Code"].notna() & (df["ICD11_Code"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)

    df_clean = pd.DataFrame()

    # Mã ICD11
    df_clean["icd11_code"] = df["ICD11_Code"].fillna("").astype(str).str.strip()

    # Tiêu đề tiếng Việt từ ICD10 (Title_VN)
    df_clean["title_vn"] = df["Title_VN"].fillna("").astype(str).str.strip()
    df_clean["title_vn_clean"] = df_clean["title_vn"].apply(clean_text)

    # ===== DISPLAY VERSION (TIẾNG VIỆT CÓ DẤU) =====
    df_clean["title_vn_display"] = (
        df["ICD11_Title_EN_VN"]
        .fillna("")
        .apply(remove_hierarchy_prefix)
        .astype(str)
        .str.strip()
    )

    # ICD11 Vietnamese official WHO (EN_VN)
    df_clean["icd11_title_en_vn"] = (
        df["ICD11_Title_EN_VN"]
        .fillna("")
        .apply(remove_hierarchy_prefix)
        .apply(clean_text)
    )

    # ICD11 English (loại prefix '-')
    df_clean["icd11_title_en"] = (
        df["ICD11_Title_EN"]
        .fillna("")
        .apply(remove_hierarchy_prefix)
        .apply(clean_text)
    )

    # ICD10 code & title EN
    df_clean["icd10_code_source"] = df["ICD_10_Code_Source"].fillna("").apply(clean_text)

    df_clean["title_en_icd10"] = (
        df["Title_EN_ICD10"]
        .fillna("")
        .apply(remove_hierarchy_prefix)
        .apply(clean_text)
    )

    # ======================
    # TOKENS (Tầng 1 search)
    # ======================
    df_clean["tokens"] = df_clean.apply(
        lambda r: list(set(
            tokenize(r["title_vn_clean"]) +
            tokenize(r["icd11_code"]) +
            tokenize(r["icd10_code_source"]) +
            tokenize(r["icd11_title_en"]) +
            tokenize(r["title_en_icd10"]) +
            tokenize(r["icd11_title_en_vn"])
        )),
        axis=1
    )

    # Aliases (hiện để trống, human-in-the-loop bổ sung sau)
    df_clean["aliases"] = [[] for _ in range(len(df_clean))]

    # ======================
    # Metadata cho tầng 2
    # ======================

    # Chapter, Block, ClassKind, isLeaf, Grouping...
    df_clean["chapter"] = df["ChapterNo"].astype(str)
    df_clean["block"] = df["BlockId"].fillna("").astype(str)

    # Nếu dataset có các cột này thì lấy, không thì để default
    df_clean["class_kind"] = df.get("ClassKind", pd.Series([""] * len(df))).astype(str)
    df_clean["is_leaf"] = df.get("isLeaf", pd.Series([False] * len(df))).astype(bool)

    for g_col in ["Grouping1", "Grouping2", "Grouping3", "Grouping4", "Grouping5"]:
        if g_col in df.columns:
            df_clean[g_col.lower()] = df[g_col].fillna("").astype(str)
        else:
            df_clean[g_col.lower()] = ""

    # Specialty, pediatric, chronic
    df_clean["primary_specialties"] = df_clean["chapter"].apply(detect_specialty)

    df_clean["primary_specialties"] = df_clean["primary_specialties"].apply(
        lambda x: x if x else []
    )
    df_clean["primary_specialties"] = df_clean["primary_specialties"].apply(to_pg_array)
    df_clean["is_pediatric"] = df_clean["title_vn_clean"].apply(detect_icd_pediatric)
    df_clean["chronic"] = df_clean["title_vn_clean"].apply(detect_icd_chronic)

    return df_clean


# ==============================
# 3. CLEAN ATC
# ==============================

# ==============================
# SPECIALTY MAPPING (ATC → chuyên khoa)
# ==============================

ATC_SPECIALTY_MAP = {
    # A — TIÊU HÓA, DINH DƯỠNG
    "A01": "dentistry",
    "A02": "gastro",
    "A03": "gastro",
    "A04": "oncology",
    "A05": "hepatology",
    "A06": "gastro",
    "A07": "gastro",
    "A08": "nutrition",
    "A09": "nutrition",

    # B — MÁU – ĐÔNG MÁU
    "B01": "hematology",
    "B02": "hematology",

    # C — TIM MẠCH
    "C01": "cardiology",
    "C02": "cardiology",
    "C03": "cardiology",
    "C04": "cardiology",
    "C05": "vascular",
    "C07": "cardiology",
    "C08": "cardiology",
    "C09": "cardiology",

    # D — DA LIỄU
    "D": "dermatology",

    # G — PHỤ KHOA
    "G01": "gynecology",
    "G02": "gynecology",
    "G03": "endocrinology",

    # H — NỘI TIẾT
    "H": "endocrinology",

    # J — KHÁNG SINH
    "J": "infectious",

    # L — UNG THƯ
    "L": "oncology",

    # M — CƠ – KHỚP
    "M01": "rheumatology",
    "M03": "neurology",
    "M04": "rheumatology",
    "M05": "rheumatology",

    # N — THẦN KINH – TÂM THẦN
    "N01": "anesthesiology",
    "N02": "pain",
    "N03": "neurology",
    "N04": "neurology",
    "N05": "psychiatry",
    "N06": "psychiatry",
    "N07": "neurology",

    # R — TAI MŨI HỌNG – HÔ HẤP
    "R01": "ent",
    "R02": "ent",
    "R03": "respiratory",
    "R05": "respiratory",
    "R06": "allergy",

    # S — MẮT, TAI
    "S01": "ophthalmology",
    "S02": "otology",
    "S03": "ent",
}

def detect_specialties_from_atc(atc_code: str):
    """
    Map từ mã ATC → chuyên khoa (ưu tiên)
    """
    specs = []
    if not atc_code:
        return specs

    code = atc_code.upper()

    # Mapping theo prefix từ dài → ngắn
    for prefix, spec in ATC_SPECIALTY_MAP.items():
        if code.startswith(prefix):
            specs.append(spec)

    return list(set(specs))


CARDIO_KEYWORDS = ["tang huyet ap", "huyet ap", "nhoi mau", "suy tim", "mach vanh",
                   "dau that nguc", "tim mach"]

RESP_KEYWORDS = ["ho hap", "hen", "phe quan", "suyen", "xo xoang", "viem pq"]

INFECTIOUS_KEYWORDS = ["khang sinh", "nhiem khuan", "viem", "virus", "lao"]

ENDO_KEYWORDS = ["tieu duong", "dai thao duong", "noitiet"]

GASTRO_KEYWORDS = ["da day", "dday", "dd", "tieu hoa", "ruot", "gast"]

def detect_specialties_from_keyword(drug_class_clean: str):
    """
    fallback nếu ATC prefix không match
    """
    t = drug_class_clean
    specs = []

    if any(k in t for k in CARDIO_KEYWORDS):
        specs.append("cardiology")
    if any(k in t for k in RESP_KEYWORDS):
        specs.append("respiratory")
    if any(k in t for k in INFECTIOUS_KEYWORDS):
        specs.append("infectious")
    if any(k in t for k in ENDO_KEYWORDS):
        specs.append("endocrinology")
    if any(k in t for k in GASTRO_KEYWORDS):
        specs.append("gastro")

    return list(set(specs))

def clean_atc(atc_file: str) -> pd.DataFrame:
    """
    Đọc ATC-Dataset-v2.csv theo cấu trúc bạn gửi
    → Chuẩn hóa cho search + thêm metadata (route, pediatric, age_min, risk_tags,...)
    """
    df = pd.read_csv(atc_file, encoding="utf-8", encoding_errors="replace")

    df_clean = pd.DataFrame()

    # Giữ dấu cho hiển thị
    df_clean["inn"] = df["Tên chung quốc tế"].fillna("").astype(str).str.strip()
    df_clean["drug_class"] = df["Loại thuốc"].fillna("").astype(str).str.strip()
    df_clean["brand_names"] = df["Tên (Chế phẩm/Biệt dược)"].apply(
        lambda x: [p.strip() for p in str(x).replace(".", "").split(",") if p.strip() != ""]
        if not pd.isna(x) else []
    )

    df_clean["forms"] = df["Dạng thuốc và hàm lượng"].apply(normalize_forms_vn)

    # Bản không dấu để search
    df_clean["inn_clean"] = df_clean["inn"].apply(clean_text)
    df_clean["atc_code"] = df["Mã ATC"].fillna("").apply(clean_text)
    df_clean["drug_class_clean"] = df_clean["drug_class"].apply(clean_text)
    df_clean["forms_clean"] = df_clean["forms"].apply(
        lambda lst: [clean_text(f).replace(" mg", "mg").replace(" ml", "ml").replace(" g", "g") for f in lst]
    )
    df_clean["brand_clean"] = df_clean["brand_names"].apply(lambda lst: [clean_text(x) for x in lst])

    # Tokens (tầng 1)
    df_clean["tokens"] = df_clean.apply(
        lambda r: list(set(
            tokenize(r["inn_clean"]) +
            tokenize(r["atc_code"]) +
            tokenize(r["drug_class_clean"]) +
            sum([tokenize(f) for f in r["forms_clean"]], []) +
            sum([tokenize(b) for b in r["brand_clean"]], [])
        )),
        axis=1
    )

    # Aliases (ví dụ: aciclovir → acyclovir)
    df_clean["aliases"] = df_clean["inn_clean"].apply(
        lambda x: ["acyclovir"] if x == "aciclovir" else []
    )

    # ======================
    # Metadata tầng 2
    # ======================

    # Route, pediatric form, age_min, risk_tags
    df_clean["routes"] = df_clean["forms_clean"].apply(detect_routes)
    df_clean["routes"] = df_clean["routes"].apply(to_pg_array)

    df_clean["is_pediatric_form"] = df_clean["forms_clean"].apply(detect_pediatric)
    df_clean["age_min"] = df["Chống chỉ định"].apply(detect_age_min)
    df_clean["risk_tags"] = df["Chống chỉ định"].apply(detect_risk_tags)

    # Để trống indications / contraindications để sau này refine nếu cần
    df_clean["contraindications"] = df["Chống chỉ định"].fillna("").astype(str).tolist()

    # SPECIALTIES (ƯU TIÊN ATC → KEYWORD → FALLBACK)
    df_clean["specialties"] = df_clean.apply(
        lambda r: detect_specialties_from_atc(r["atc_code"]) or
                detect_specialties_from_keyword(r["drug_class_clean"]),
        axis=1
    )

    # đảm bảo không NULL → empty array "{}"
    df_clean["specialties"] = df_clean["specialties"].apply(lambda x: x if x else [])
    df_clean["specialties"] = df_clean["specialties"].apply(to_pg_array)

    return df_clean


# ==============================
# 4. RUN
# ==============================

if __name__ == "__main__":
 
    ICD_FILE = "ICD11-Dataset.csv"
    ATC_FILE = "ATC-Dataset-v2.csv"

    print("=== Cleaning ICD-11 ===")
    icd = clean_icd11(ICD_FILE)
    print(icd.head(3).to_json(orient="records", force_ascii=False, indent=2))

    print("=== Cleaning ATC ===")
    atc = clean_atc(ATC_FILE)
    print(atc.head(3).to_json(orient="records", force_ascii=False, indent=2))

    # Xuất CSV sạch để debug / backup
    icd.to_csv("ICD11-clean-final.csv", index=False, encoding="utf-8-sig")
    atc.to_csv("ATC-clean-final.csv", index=False, encoding="utf-8-sig")

    print("✔ Saved ICD11-clean-final.csv & ATC-clean-final.csv")

    # ================= LOAD DB (OPTIONAL) =================
    # Nếu muốn load thẳng vào Postgres bằng pandas.to_sql
    # thì bật đoạn này + tạo bảng trước (hoặc dùng if_exists="replace")

    engine = create_engine("postgresql://postgres:12345678@localhost:5432/medisync")
    
    icd.to_sql("icd11_clean", con=engine, if_exists="append", index=False)
    atc.to_sql("atc_clean", con=engine, if_exists="append", index=False)
    
    print("✔ Loaded into Postgres tables icd11_clean & atc_clean")

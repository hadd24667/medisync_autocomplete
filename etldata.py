import pandas as pd
import re
from unidecode import unidecode

# ==============================
# 1. UTILS
# ==============================

def clean_text(x):
    """
    Chuẩn hóa:
    - lowercase
    - unidecode
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
    s = re.sub(r"[^\w\s/%]", "", s) # XÓA dấu ., ; …
    return [t for t in s.split(" ") if t]



def split_list_clean(s):
    """
    Tách theo dấu ";" và chuẩn hóa từng mục.
    """
    if pd.isna(s):
        return []
    parts = [clean_text(p) for p in str(s).split(";") if p.strip() != ""]
    return parts


def normalize_forms(forms):
    """
    Chuẩn hóa dạng thuốc:
    - 50 mg -> 50mg
    - 10 ml -> 10ml
    - Giữ định dạng unify
    """
    out = []
    for f in split_list_clean(forms):
        f = f.replace(" mg", "mg")
        f = f.replace(" ml", "ml")
        f = f.replace(" / ", "/")
        f = f.replace(" /", "/")
        f = f.replace("/ ", "/")
        out.append(f)
    return out


def remove_hierarchy_prefix(title):
    """
    Loại các prefix như '- - - ', '- - ', '- ' từ ICD-11 Excel
    """
    if pd.isna(title):
        return ""
    title = str(title).strip()
    while title.startswith("-"):
        title = title[1:].strip()
    return title

def split_forms_keep_vietnamese(s):
    if pd.isna(s):
        return []

    result = []
    parts = str(s).split(";")

    for p in parts:
        p = p.strip()
        if not p:
            continue

        # nếu dạng có nhiều hàm lượng: "Viên nén 200 mg, 400 mg"
        subparts = [sp.strip() for sp in p.replace(".", "").split(",")]
        
        if len(subparts) > 1:
            # tái tạo từng hàm lượng
            base = subparts[0].split(" ")[0:-2]  # "Viên nén"
            base = " ".join(base)
            for sp in subparts:
                if any(c.isdigit() for c in sp):
                    result.append(f"{base} {sp}")
        else:
            result.append(p)

    return result

def split_forms_vn(s):
    if pd.isna(s):
        return []

    # loại dấu chấm cuối
    s = s.replace(".", "")

    # tách theo ; hoặc xuống dòng
    parts = re.split(r"[;\n]", s)

    clean = []
    for p in parts:
        p = p.strip()
        if not p:
            continue

        # tách hàm lượng theo dấu phẩy
        sub = [x.strip() for x in p.split(",")]

        # tái cấu trúc: "Viên nén", "200 mg"
        if len(sub) > 1:
            base = " ".join(sub[0].split(" ")[:2])  # "Viên nén"
            for item in sub:
                if any(c.isdigit() for c in item):
                    clean.append(f"{base} {item}")
        else:
            clean.append(p)

    return clean

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

def detect_route(forms):
    forms_text = " ".join(forms).lower()
    if "tiem" in forms_text:
        return "injection"
    if any(x in forms_text for x in ["siro", "hon dich", "cốm", "gói", "drops"]):
        return "oral"
    return "oral"

def detect_pediatric(forms):
    forms_text = " ".join(forms).lower()
    return any(x in forms_text for x in ["siro", "hon dich", "cốm", "gói", "drops"])

def detect_age_min(contra):
    contra = clean_text(contra)
    m = re.search(r"tre em duoi (\d+)", contra)
    return int(m.group(1)) if m else None

def detect_risk_tags(contra):
    contra = clean_text(contra)
    tags = []
    if "gan" in contra:
        tags.append("hepatic_toxic")
    if "than" in contra:
        tags.append("renal_adjust")
    if "thai" in contra:
        tags.append("pregnancy_risk")
    return tags



# ==============================
# CLEAN ICD-11
# ==============================

def clean_icd11(icd_file):
    df = pd.read_csv(icd_file, encoding="utf-8", encoding_errors="replace")

    df_clean = pd.DataFrame()
    df_clean["icd11_code"] = df["ICD11_Code"].fillna("").astype(str).apply(clean_text)
    
    # VN title
    df_clean["title_vn"] = df["Title_VN"].fillna("").str.strip()
    df_clean["title_vn_clean"] = df_clean["title_vn"].apply(clean_text)

    # EN title (loại dấu '-' prefix rồi clean)
    df_clean["icd11_title_en"] = (
        df["ICD11_Title_EN"]
        .fillna("")
        .apply(remove_hierarchy_prefix)
        .apply(clean_text)
    )

    df_clean["icd11_title_en_vn"] = (
        df["ICD11_Title_EN_VN"]
        .fillna("")
        .apply(remove_hierarchy_prefix)
        .apply(clean_text)
    )

    df_clean["icd10_code_source"] = df["ICD_10_Code_Source"].fillna("").apply(clean_text)

    df_clean["title_en_icd10"] = (
        df["Title_EN_ICD10"]
        .fillna("")
        .apply(remove_hierarchy_prefix)
        .apply(clean_text)
    )

    # --------------------
    # TOKENS
    # --------------------
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

    # Aliases (hiện để trống)
    df_clean["aliases"] = [[] for _ in range(len(df_clean))]

    return df_clean


# ==============================
# CLEAN ATC
# ==============================

def clean_atc(atc_file):
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

    df_clean["aliases"] = df_clean["inn_clean"].apply(
        lambda x: ["acyclovir"] if x == "aciclovir" else []
    )
    df_clean["main_route"] = df_clean["forms_clean"].apply(detect_route)
    df_clean["is_pediatric_form"] = df_clean["forms_clean"].apply(detect_pediatric)
    df_clean["age_min"] = df["Chống chỉ định"].apply(detect_age_min)
    df_clean["risk_tags"] = df["Chống chỉ định"].apply(detect_risk_tags)


    return df_clean



# ==============================
# 4. RUN
# ==============================

if __name__ == "__main__":
    # icd = clean_icd11("ICD11-Dataset.csv")
    atc = clean_atc("ATC-Dataset-v2.csv")

    # print("=== ICD-11 CLEAN SAMPLE ===")
    # print(icd.head(3).to_json(orient="records", force_ascii=False, indent=2))

    print("=== ATC CLEAN SAMPLE ===")
    print(atc.head(3).to_json(orient="records", force_ascii=False, indent=2))

    # icd.to_csv("ICD11-clean.csv", index=False, encoding="utf-8-sig")
    atc.to_csv("ATC-clean.csv", index=False, encoding="utf-8-sig")

    print("Cleaned ATC data saved.")


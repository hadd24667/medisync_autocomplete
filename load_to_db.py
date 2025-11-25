import ast
import pandas as pd
from sqlalchemy import create_engine, text

POSTGRES_URL = "postgresql://postgres:12345678@localhost:5432/medisync"

engine = create_engine(POSTGRES_URL)

# ==============================
# 1. CREATE TABLES ĐÚNG SCHEMA MỚI
# ==============================

schema_sql = """
DROP TABLE IF EXISTS icd11_clean CASCADE;
DROP TABLE IF EXISTS atc_clean CASCADE;

CREATE TABLE icd11_clean (
    id SERIAL PRIMARY KEY,
    icd11_code TEXT,
    title_vn TEXT,
    title_vn_clean TEXT,
    title_vn_display TEXT,
    icd11_title_en_vn TEXT,
    icd11_title_en TEXT,
    icd10_code_source TEXT,
    title_en_icd10 TEXT,
    tokens TEXT[],
    aliases TEXT[],
    chapter TEXT,
    block TEXT,
    class_kind TEXT,
    is_leaf BOOLEAN,
    grouping1 TEXT,
    grouping2 TEXT,
    grouping3 TEXT,
    grouping4 TEXT,
    grouping5 TEXT,
    primary_specialties TEXT[],
    is_pediatric BOOLEAN,
    chronic BOOLEAN
);

CREATE TABLE atc_clean (
    id SERIAL PRIMARY KEY,
    inn TEXT,
    drug_class TEXT,
    brand_names TEXT[],
    forms TEXT[],
    inn_clean TEXT,
    atc_code TEXT,
    drug_class_clean TEXT,
    forms_clean TEXT[],
    brand_clean TEXT[],
    tokens TEXT[],
    aliases TEXT[],
    routes TEXT[],
    is_pediatric_form BOOLEAN,
    age_min INT,
    risk_tags TEXT[],
    contraindications TEXT,
    specialties TEXT[]
);
"""


with engine.begin() as conn:
    conn.execute(text(schema_sql))

print("✅ Tables (icd11_clean, atc_clean) created with NEW schema")


# ==============================
# 2. HÀM PARSE GIÁ TRỊ LIST / BOOL / NULL
# ==============================

def parse_list_like(val):
    """
    Chuẩn hóa các cột dạng list:
    - "['a', 'b']"  -> ['a','b']
    - "{"a","b"}"   -> ['a','b']
    - "[]" / "{}"   -> []
    """
    if not isinstance(val, str):
        return val

    s = val.strip()
    if s in ("", "nan", "NaN", "None", "null"):
        return None

    # Kiểu list Python: "['a', 'b']"
    if s.startswith("[") and s.endswith("]"):
        try:
            out = ast.literal_eval(s)
            # đảm bảo list string
            if isinstance(out, list):
                return [str(x) for x in out]
            return out
        except Exception:
            return s

    # Kiểu Postgres array: "{"a","b"}"
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        parts = inner.split(",")
        res = []
        for p in parts:
            item = p.strip().strip('"').strip("'")
            if item:
                res.append(item)
        return res

    return s


def parse_scalar(val):
    """
    Bool / null / số đơn giản.
    """
    if not isinstance(val, str):
        return val

    s = val.strip()
    if s.lower() in ("nan", "none", "null", ""):
        return None

    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False

    # Thử parse số (cho age_min kiểu "18.0")
    try:
        if "." in s:
            f = float(s)
            # nếu là số nguyên dạng 18.0 thì trả về int
            if f.is_integer():
                return int(f)
            return f
        else:
            i = int(s)
            return i
    except Exception:
        return s


def clean_dataframe(df, list_cols=None, scalar_cols=None):
    list_cols = list_cols or []
    scalar_cols = scalar_cols or []

    for col in df.columns:
        if df[col].dtype != object:
            continue

        if col in list_cols:
            df[col] = df[col].apply(parse_list_like)
        elif col in scalar_cols:
            df[col] = df[col].apply(parse_scalar)
        else:
            # vẫn xử lý null cơ bản
            df[col] = df[col].apply(
                lambda x: None
                if isinstance(x, str) and x.strip().lower() in ("nan", "none", "null", "")
                else x
            )

    return df


def load_csv_to_pg(csv_file, table_name, list_cols, scalar_cols):
    df = pd.read_csv(csv_file)
    
    df = clean_dataframe(df, list_cols=list_cols, scalar_cols=scalar_cols)

    df.to_sql(table_name, con=engine, if_exists="append", index=False)
    print(f"✅ Loaded into {table_name}: {len(df)} rows")


# ==============================
# 3. KHAI BÁO CỘT LIST & CỘT SCALAR ĐẶC BIỆT
# ==============================

ICD_LIST_COLS = ["tokens", "aliases", "primary_specialties"]
ICD_SCALAR_COLS = ["is_leaf", "is_pediatric", "chronic"]

ATC_LIST_COLS = [
    "brand_names",
    "forms",
    "forms_clean",
    "brand_clean",
    "tokens",
    "aliases",
    "routes",
    "risk_tags",
    "specialties",
]
ATC_SCALAR_COLS = ["is_pediatric_form", "age_min"]

# ==============================
# 4. LOAD VÀO DB
# ==============================

load_csv_to_pg("ICD11-clean-final.csv", "icd11_clean", ICD_LIST_COLS, ICD_SCALAR_COLS)
load_csv_to_pg("ATC-clean-final.csv", "atc_clean", ATC_LIST_COLS, ATC_SCALAR_COLS)

print("🎉 DONE: All clean data imported to PostgreSQL.")

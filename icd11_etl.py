import pandas as pd
import re
from unidecode import unidecode


# =========================
# Utils
# =========================
def clean_text(x):
    if pd.isna(x):
        return ""
    x = unidecode(str(x)).lower().strip()
    x = " ".join(x.split())
    return x

def tokenize(s):
    if not s:
        return []
    s = clean_text(s)
    s = re.sub(r"[^\w\s/%]", "", s)
    return [t for t in s.split() if t]

def remove_hierarchy_prefix(title):
    if pd.isna(title):
        return ""
    title = str(title).strip()
    while title.startswith("-"):
        title = title[1:].strip()
    return title
def to_pg_array(lst):
    if not isinstance(lst, list):
        return "{}"
    return "{" + ",".join(f'"{str(x)}"' for x in lst) + "}"

def detect_specialty(chapter):
    if chapter.startswith("09"):
        return ["cardiology"]
    if chapter.startswith("01"):
        return ["infectious"]
    if chapter.startswith("05"):
        return ["neurology"]
    if chapter.startswith("12"):
        return ["gastro"]
    return []

# =========================
# Clean ICD-11
# =========================
def clean_icd11(icd_file):
    df = pd.read_csv(icd_file, encoding="utf-8", encoding_errors="replace")

    df_clean = pd.DataFrame()

    # 1. ICD11 code
    df_clean["icd11_code"] = df["ICD11_Code"].fillna("").astype(str).str.strip()

    # bỏ chapter/block
    df_clean = df_clean[df_clean["icd11_code"] != ""].reset_index(drop=True)

    # 2. title_vn (ICD10 Vietnamese)
    df_clean["title_vn"] = df["Title_VN"].fillna("").astype(str).str.strip()

    # 3. ICD11 Vietnamese official WHO VN
    icd11_vn_raw = (
        df["ICD11_Title_EN_VN"]
        .fillna("")
        .apply(remove_hierarchy_prefix)
        .astype(str)
        .str.strip()
    )

    # 4. title_vn_final (ƯU TIÊN ICD10 → fallback ICD11 VN)
    df_clean["title_vn_final"] = df_clean.apply(
        lambda r: r["title_vn"] if r["title_vn"] else icd11_vn_raw[r.name],
        axis=1
    )

    df_clean["title_vn_final_clean"] = df_clean["title_vn_final"].apply(clean_text)

    # 5. English ICD11
    df_clean["icd11_title_en"] = (
        df["ICD11_Title_EN"]
        .fillna("")
        .apply(remove_hierarchy_prefix)
        .apply(clean_text)
    )

    # 6. English→Vietnamese (ICD11)
    df_clean["icd11_title_en_vn"] = icd11_vn_raw.apply(clean_text)

    # 7. ICD10 fields
    df_clean["icd10_code_source"] = df["ICD_10_Code_Source"].fillna("").apply(clean_text)

    df_clean["title_en_icd10"] = (
        df["Title_EN_ICD10"]
        .fillna("")
        .apply(remove_hierarchy_prefix)
        .apply(clean_text)
    )

    # 8. Tokens (Tier-1 Search)
    df_clean["tokens"] = df_clean.apply(
        lambda r: list(set(
            tokenize(r["title_vn_final_clean"]) +
            tokenize(r["icd11_code"]) +
            tokenize(r["icd11_title_en"]) +
            tokenize(r["icd10_code_source"]) +
            tokenize(r["title_en_icd10"])
        )),
        axis=1
    )

    # 9. Aliases (for human-in-loop later)
    df_clean["aliases"] = [[] for _ in range(len(df_clean))]

    # 10. Convert for PostgreSQL
    df_clean["tokens"] = df_clean["tokens"].apply(to_pg_array)
    df_clean["aliases"] = df_clean["aliases"].apply(to_pg_array)

    df_clean["chapter"] = df["ChapterNo"].astype(str)
    df_clean["block"] = df["BlockId"].astype(str)

    df_clean["primary_specialties"] = df_clean["chapter"].apply(detect_specialty)
    df_clean["is_pediatric"] = df_clean["title_vn_clean"].apply(lambda x: "tre em" in x or "nhi" in x)
    df_clean["chronic"] = df_clean["title_vn_clean"].apply(lambda x: any(k in x for k in ["man tinh","tang huyet ap","dai thao duong","copd"]))

    return df_clean


# =========================
# RUN
# =========================
if __name__ == "__main__":
    df = clean_icd11("ICD11-Dataset.csv")

    print(df.head(5).to_json(orient="records", force_ascii=False, indent=2))

    df.to_csv("ICD11-clean.csv", index=False, encoding="utf-8-sig")

    print("✔ Exported ICD11-clean.csv (NO title_vn_clean)")

# import pandas as pd
# from sqlalchemy import create_engine

# engine = create_engine("postgresql://postgres:12345678@localhost:5432/medisync")

# df = pd.read_csv("ICD11-clean.csv", encoding="utf-8")
# df.to_sql("icd11_clean", con=engine, if_exists="append", index=False)

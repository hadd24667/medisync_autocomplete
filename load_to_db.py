import pandas as pd
from sqlalchemy import create_engine, text

POSTGRES_URL = "postgresql://postgres:12345678@localhost:5432/medisync"

engine = create_engine(POSTGRES_URL)

# ==============================
# CREATE TABLES
# ==============================

schema_sql = """
CREATE TABLE IF NOT EXISTS icd11_clean (
    id SERIAL PRIMARY KEY,
    icd11_code TEXT,
    title_vn TEXT,
    title_vn_clean TEXT,
    icd11_title_en TEXT,
    icd11_title_en_vn TEXT,
    icd10_code_source TEXT,
    title_en_icd10 TEXT,
    tokens TEXT[],
    aliases TEXT[]
);

CREATE TABLE IF NOT EXISTS atc_clean (
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
    aliases TEXT[]
);
"""

with engine.begin() as conn:
    conn.execute(text(schema_sql))

print("Tables created successfully!")


# ==============================
# LOAD CLEANED CSV
# ==============================

def load_csv_to_pg(csv_file, table_name):
    df = pd.read_csv(csv_file)

    # convert list string → real Python list
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else x)

    df.to_sql(table_name, con=engine, if_exists="append", index=False)
    print(f"Loaded into {table_name}: {len(df)} rows")


# ==============================
# RUN
# ==============================

load_csv_to_pg("ICD11-clean.csv", "icd11_clean")
load_csv_to_pg("ATC-clean.csv", "atc_clean")

print("DONE: All clean data imported to PostgreSQL.")

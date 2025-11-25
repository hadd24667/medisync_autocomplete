import pandas as pd

# --- 1. Load file ---
df = pd.read_csv("train_features_v8.csv")

print(f"Loaded: {len(df)} rows")

# --- 2. Random sample 100 rows ---
sample = df.sample(n=100, random_state=42)

# --- 3. Print ra console ---
print("\n===== SAMPLE 100 ROWS =====\n")
print(sample.to_string(index=False))

# --- 4. Save file ---
OUT_FILE = "train_features_sample100.csv"
sample.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

print(f"\nSaved sample → {OUT_FILE}")

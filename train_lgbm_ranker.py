# ===========================================================
# SMARTEMR — Tier-2 ATC Ranker Training (LightGBM Ranker)
# Version: v8 (rich feature pipeline, 19 features)
# ===========================================================

import pandas as pd
import lightgbm as lgb
import numpy as np

# -----------------------------------------------------------
# 1. LOAD FEATURE TABLE
# -----------------------------------------------------------
FEATURE_COLS = [
    "f_semantic",
    "f_semantic_sq",
    "f_inn_sub",
    "f_brand_sub",
    "f_syn_sub",
    "f_len_ratio",
    "f_char_overlap",
    "f_has_number",
    "f_match_brand",
    "f_match_inn",
    "f_match_alias",
    "f_match_exact",
    "f_syn_ambiguity",
    "f_dose_match",
    "f_dose_diff_min",
    "f_route_match",
    "f_is_combo_drug",
    "f_is_pediatric_form",
    "f_risk_hepatic",
]

LABEL_COL = "label"
GROUP_COL = "group_id"

print("Loading train_features_v8.csv ...")
df = pd.read_csv("train_features_v8.csv")

print("Total samples:", len(df))
print("Total groups :", df[GROUP_COL].nunique())
print("Feature columns:", FEATURE_COLS)

# -----------------------------------------------------------
# 2. BUILD FEATURE MATRIX + GROUP VECTOR
# -----------------------------------------------------------
X = df[FEATURE_COLS].values.astype(np.float32)
y = df[LABEL_COL].values.astype(np.float32)

# group sizes ordered by group_id
groups = df.groupby(GROUP_COL).size().tolist()

print("✔ Feature matrix shape:", X.shape)
print("✔ Label vector shape  :", y.shape)
print("✔ Group vector length :", len(groups))

train_dataset = lgb.Dataset(
    X,
    y,
    group=groups,
    feature_name=FEATURE_COLS,
)

# -----------------------------------------------------------
# 3. TRAIN LIGHTGBM RANKER (TUNED FOR ~100K ROWS)
# -----------------------------------------------------------
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3, 5, 10],
    "learning_rate": 0.05,
    "num_leaves": 63,          # hơi lớn hơn tí cho feature nhiều
    "max_depth": -1,
    "min_data_in_leaf": 50,    # tránh overfit với 100k rows
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 0.2,
    "min_gain_to_split": 0.01,
    "verbosity": -1,
}

print("\nTraining LightGBM Ranker v8 ...\n")

model = lgb.train(
    params=params,
    train_set=train_dataset,
    num_boost_round=350,   # có thể tăng/giảm sau khi xem ndcg
)

# -----------------------------------------------------------
# 4. SAVE MODEL
# -----------------------------------------------------------
MODEL_FILE = "lgbm_ranker_v8feat.txt"
model.save_model(MODEL_FILE)

print("\n===================================")
print("\nFeature importance (gain):")
for name, score in zip(FEATURE_COLS, model.feature_importance(importance_type="gain")):
    print(f"{name:20s}  {score:.2f}")
print("✓ DONE — Model saved as:", MODEL_FILE)
print("===================================")

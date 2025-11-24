# ===========================================================
# SMARTEMR — Tier-2 ATC Ranker Training (LightGBM Ranker)
# Version: v7 (7 feature pipeline)
# ===========================================================

import pandas as pd
import lightgbm as lgb

# -----------------------------------------------------------
# Load feature table
# -----------------------------------------------------------
print("Loading train_features.csv ...")
df = pd.read_csv("train_features.csv")

# 7 feature columns
FEATURE_COLS = [
    "f_semantic",
    "f_inn_sub",
    "f_brand_sub",
    "f_syn_sub",
    "f_len_ratio",
    "f_char_overlap",
    "f_edit_prefix",
]

LABEL_COL = "label"
GROUP_COL = "group_id"

print("Total samples:", len(df))
print("Total groups :", df[GROUP_COL].nunique())
print("Feature columns:", FEATURE_COLS)

# -----------------------------------------------------------
# Build feature matrix + group vector
# -----------------------------------------------------------
X = df[FEATURE_COLS].values
y = df[LABEL_COL].values

# group sizes ordered by group_id
groups = df.groupby(GROUP_COL).size().tolist()

print("✔ Feature matrix shape:", X.shape)
print("✔ Label vector shape  :", y.shape)
print("✔ Group vector length :", len(groups))

# -----------------------------------------------------------
# LightGBM Ranker — hyperparameters tuned for small feature set
# -----------------------------------------------------------
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3, 5],
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 30,
    "max_depth": -1,
    "lambda_l2": 0.2,
    "min_gain_to_split": 0.01,
    "verbosity": -1,
}

train_dataset = lgb.Dataset(
    X, 
    y, 
    group=groups,
    feature_name=FEATURE_COLS
)

print("\nTraining LightGBM Ranker ...\n")

model = lgb.train(
    params=params,
    train_set=train_dataset,
    num_boost_round=300,
)

# -----------------------------------------------------------
# Save model
# -----------------------------------------------------------
MODEL_FILE = "lgbm_ranker_v7feat.txt"
model.save_model(MODEL_FILE)

print("\n===================================")
print("\nFeature importance (gain):")
for name, score in zip(FEATURE_COLS, model.feature_importance(importance_type="gain")):
    print(f"{name:15s}  {score:.2f}")
print("✓ DONE — Model saved as:", MODEL_FILE)
print("===================================")

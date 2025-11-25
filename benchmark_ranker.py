# benchmark_ranker.py
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

# --- CONFIG ---
FEAT_FILE = "benchmark_features_v8.csv"
MODEL_FILE = "lgbm_ranker_v8feat.txt"

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


def dcg_at_k(rels, k):
    """rels: list/array relevance (0/1), đã được sort theo thứ tự rank."""
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    # log2(i+2) vì index bắt đầu từ 0
    discounts = np.log2(np.arange(2, rels.size + 2))
    return np.sum((2 ** rels - 1) / discounts)


def ndcg_at_k(rels, k):
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def main():
    print("=== Benchmark LightGBM Ranker v8 ===")
    print(f"Loading features from: {FEAT_FILE}")
    df = pd.read_csv(FEAT_FILE)

    print(f"Total rows         : {len(df)}")
    n_groups = df["group_id"].nunique()
    print(f"Total query groups : {n_groups}")

    print(f"Loading model      : {MODEL_FILE}")
    model = lgb.Booster(model_file=MODEL_FILE)

    # --- Metrics aggregate ---
    hits_at1 = []
    recalls_at5 = []
    recalls_at10 = []
    mrrs = []
    ndcg3_list = []
    ndcg5_list = []
    ndcg10_list = []

    # Nếu dataset lớn quá và bạn muốn test nhanh trước:
    # sample_groups = df["group_id"].drop_duplicates().sample(n=200, random_state=42)
    # df = df[df["group_id"].isin(sample_groups)]

    grouped = df.groupby("group_id", sort=False)

    for group_id, g in tqdm(grouped, desc="Evaluating groups"):
        # mỗi group = 1 query, nhiều candidate code
        labels = g["label"].values.astype(float)
        X = g[FEATURE_COLS].values.astype(np.float32)

        # skip nếu không có relevant
        if labels.max() <= 0:
            continue

        scores = model.predict(X)
        order = np.argsort(scores)[::-1]  # sort desc
        labels_sorted = labels[order]

        # --- Hit@1 ---
        hit1 = 1.0 if labels_sorted[0] > 0 else 0.0
        hits_at1.append(hit1)

        # --- Recall@5, Recall@10 ---
        top5 = labels_sorted[:5]
        top10 = labels_sorted[:10]
        recalls_at5.append(1.0 if top5.max() > 0 else 0.0)
        recalls_at10.append(1.0 if top10.max() > 0 else 0.0)

        # --- MRR ---
        pos_idx = np.where(labels_sorted > 0)[0]
        if len(pos_idx) > 0:
            mrrs.append(1.0 / (pos_idx[0] + 1))
        else:
            mrrs.append(0.0)

        # --- NDCG ---
        ndcg3_list.append(ndcg_at_k(labels_sorted, 3))
        ndcg5_list.append(ndcg_at_k(labels_sorted, 5))
        ndcg10_list.append(ndcg_at_k(labels_sorted, 10))

    def avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    print("\n===== OFFLINE METRICS (train_features_v8) =====")
    print(f"#groups evaluated : {len(hits_at1)}")
    print(f"Hit@1             : {avg(hits_at1):.4f}")
    print(f"Recall@5          : {avg(recalls_at5):.4f}")
    print(f"Recall@10         : {avg(recalls_at10):.4f}")
    print(f"MRR               : {avg(mrrs):.4f}")
    print(f"NDCG@3            : {avg(ndcg3_list):.4f}")
    print(f"NDCG@5            : {avg(ndcg5_list):.4f}")
    print(f"NDCG@10           : {avg(ndcg10_list):.4f}")
    print("===============================================")


if __name__ == "__main__":
    main()

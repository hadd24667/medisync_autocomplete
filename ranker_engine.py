# ranker_engine.py
# Tier-2 LightGBM ranker runtime

import numpy as np
import lightgbm as lgb
from unidecode import unidecode

from ranker_utils import embed_query
from extract_features_infer import extract_feats_from_vec

# Thứ tự feature phải giống 100% lúc train
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

# Model v8 dùng 19 feature
model = lgb.Booster(model_file="lgbm_ranker_v8feat.txt")


def rank_candidates(
    query: str,
    codes,
    matched_syn: str | None = None,
    top_k: int = 20,
):
    """
    Rank list of ATC codes cho query.
    Trả về list[(code, score)] sort desc theo score.
    - query: text bác sĩ gõ
    - codes: list candidate ATC code từ Tier-1
    - matched_syn: (optional) token synonym mà Tier-1 detect được, ví dụ 'para', 'tylenol'...
    """

    if not codes:
        return []

    # Embed query ONCE
    q_vec = embed_query(query)

    rows = []
    kept_codes = []

    # codes_in_group = toàn bộ candidate cùng group cho việc build brand_index_for_group
    codes_in_group = list(codes)

    for code in codes_in_group:
        try:
            feats = extract_feats_from_vec(
                q_vec=q_vec,
                query=query,
                code=code,
                codes_in_group=codes_in_group,
                matched_syn=matched_syn,  # có thể None, extract_features_infer sẽ tự pick từ query
            )
            if feats is None:
                print(f"[WARN] feats is None for code {code}")
                continue

            # Safety check: phải đúng 19 features
            if len(feats) != len(FEATURE_COLS):
                # Có gì đó sai: skip để tránh vỡ model.predict
                print(f"[WARN] wrong feature length for {code}: {len(feats)}")
                continue

            rows.append(feats)
            kept_codes.append(code)
        except Exception as e:
            # Không để 1 candidate lỗi làm vỡ cả ranking
            print(f"[ERROR] rank_candidates error for {code}: {e}")
            continue

    if not rows:
        return []

    X = np.array(rows, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    preds = model.predict(X)

    sorted_list = sorted(
        zip(kept_codes, preds),
        key=lambda x: x[1],
        reverse=True,
    )

    return sorted_list[:top_k]

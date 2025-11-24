import json
from tier2_ranker import (
    PatientContext,
    rerank_atc_candidates,
    compute_rule_context_score
)

# =====================
# LOAD METADATA
# =====================
with open("atc_synonyms_meta.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# =====================
# TEST CASES
# =====================
tests = [
    {
        "title": "1) CREON – người lớn – khoa Nutrition",
        "query": "creon",
        "ctx": PatientContext(age=40, specialty="nutrition"),
        "candidates": ["A09AA02", "N07BA01", "A11HA04"]
    },
    {
        "title": "2) CREON – BN viêm tụy cấp (contraindication)",
        "query": "creon",
        "ctx": PatientContext(age=40, active_icd=["viem tuy cap"]),
        "candidates": ["A09AA02", "N07BA01", "A11HA04"]
    },
    {
        "title": "3) VIT B5 5MG – match hàm lượng + brand",
        "query": "vit b5 5mg",
        "ctx": PatientContext(age=25),
        "candidates": ["A11HA04", "N07BA01"]
    },
    {
        "title": "4) VIT B5 – allergy riboflavin",
        "query": "vitamin b5",
        "ctx": PatientContext(age=25, allergies=["riboflavin"]),
        "candidates": ["A11HA04"]
    },
    {
        "title": "5) NIC – khoa thần kinh",
        "query": "nic",
        "ctx": PatientContext(age=35, specialty="neurology"),
        "candidates": ["N07BA01", "A11HA04"]
    },
    {
        "title": "6) SIRO PARA – BN 3 tuổi",
        "query": "siro para",
        "ctx": PatientContext(age=3, specialty="pediatrics"),
        "candidates": ["A09AA02", "A11HA04"]
    },
    {
        "title": "7) PARA 500 – người lớn",
        "query": "para 500",
        "ctx": PatientContext(age=45),
        "candidates": ["A09AA02", "A11HA04"]
    }
]

# =====================
# RUN ALL TESTS
# =====================
for t in tests:
    print("\n==============================")
    print(t["title"])
    print("==============================")

    ranked = rerank_atc_candidates(
        query=t["query"],
        candidates=t["candidates"],
        metadata=metadata,
        ctx=t["ctx"],
        top_k=None
    )

    for cand, score in ranked:
        print(f"- {cand}: final_score={score['final_score']:.4f}")
        print(f"  details={score}")

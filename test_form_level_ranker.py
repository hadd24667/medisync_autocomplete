# test_form_level_ranker.py
"""
Test tầng form-level ranker với 10 thuốc mẫu.
Không cần DB, không cần Tier-1/Tier-2.
Dữ liệu mock giống format output tier2_results.
"""

from atc_form_ranker import build_form_level_suggestions


# ==========================================
# MOCK 10 THUỐC MẪU CÓ DẠNG + HÀM LƯỢNG
# ==========================================
mock_tier2 = [

    # 1. Paracetamol
    {
        "code": "N02BE01",
        "label": "paracetamol [Viên nén 500 mg, Viên nén 650 mg, Hỗn dịch uống 120 mg/5 ml]",
        "forms": [
            "Viên nén 500 mg",
            "Viên nén 650 mg",
            "Hỗn dịch uống 120 mg/5 ml",
        ],
        "type": "ATC",
        "ranker_score": 1.0,
    },

    # 2. Ibuprofen
    {
        "code": "M01AE01",
        "label": "ibuprofen [Viên nén 200 mg, Viên nén 400 mg, Hỗn dịch uống 100 mg/5 ml]",
        "forms": [
            "Viên nén 200 mg",
            "Viên nén 400 mg",
            "Hỗn dịch uống 100 mg/5 ml",
        ],
        "type": "ATC",
        "ranker_score": 0.8,
    },

    # 3. Amoxicillin
    {
        "code": "J01CA04",
        "label": "amoxicillin [Viên nang 250 mg, Viên nang 500 mg, Bột pha hỗn dịch uống 250 mg/5 ml]",
        "forms": [
            "Viên nang 250 mg",
            "Viên nang 500 mg",
            "Bột pha hỗn dịch uống 250 mg/5 ml",
        ],
        "type": "ATC",
        "ranker_score": 0.6,
    },

    # 4. Cefixime
    {
        "code": "J01DD08",
        "label": "cefixime [Viên nén 200 mg, Viên nén 400 mg, Bột pha hỗn dịch uống 100 mg/5 ml]",
        "forms": [
            "Viên nén 200 mg",
            "Viên nén 400 mg",
            "Bột pha hỗn dịch uống 100 mg/5 ml",
        ],
        "type": "ATC",
        "ranker_score": 0.5,
    },

    # 5. Metronidazole
    {
        "code": "J01XD01",
        "label": "metronidazole [Viên nén 250 mg, Viên nén 500 mg]",
        "forms": [
            "Viên nén 250 mg",
            "Viên nén 500 mg",
        ],
        "type": "ATC",
        "ranker_score": 0.4,
    },

    # 6. Azithromycin
    {
        "code": "J01FA10",
        "label": "azithromycin [Viên nén 250 mg, Viên nén 500 mg, Bột pha hỗn dịch uống 200 mg/5 ml]",
        "forms": [
            "Viên nén 250 mg",
            "Viên nén 500 mg",
            "Bột pha hỗn dịch uống 200 mg/5 ml",
        ],
        "type": "ATC",
        "ranker_score": 0.7,
    },

    # 7. Loratadine
    {
        "code": "R06AX13",
        "label": "loratadin [Viên nén 10 mg, Siro 5 mg/5 ml]",
        "forms": [
            "Viên nén 10 mg",
            "Siro 5 mg/5 ml",
        ],
        "type": "ATC",
        "ranker_score": 0.3,
    },

    # 8. Cetirizine
    {
        "code": "R06AE07",
        "label": "cetirizin [Viên nén 10 mg, Siro 5 mg/5 ml]",
        "forms": [
            "Viên nén 10 mg",
            "Siro 5 mg/5 ml",
        ],
        "type": "ATC",
        "ranker_score": 0.35,
    },

    # 9. Omeprazole
    {
        "code": "A02BC01",
        "label": "omeprazol [Viên nang 20 mg, Viên nang 40 mg]",
        "forms": [
            "Viên nang 20 mg",
            "Viên nang 40 mg",
        ],
        "type": "ATC",
        "ranker_score": 0.45,
    },

    # 10. Diazepam
    {
        "code": "N05BA01",
        "label": "diazepam [Viên nén 2 mg, Viên nén 5 mg, Viên nén 10 mg]",
        "forms": [
            "Viên nén 2 mg",
            "Viên nén 5 mg",
            "Viên nén 10 mg",
        ],
        "type": "ATC",
        "ranker_score": 0.25,
    },
]


# ==========================================
# TEST: NHIỀU QUERY KHÁC NHAU
# ==========================================
queries = [
    "para 500",
    "ibu 400",
    "amox 250",
    "cefi 400",
    "metro 500",
    "azit 500",
    "cet 10",
    "lora 10",
    "ome 40",
    "diaz 5",
]

for q in queries:
    print("====================================")
    print(f"🔎 QUERY: {q}")
    print("====================================")

    results = build_form_level_suggestions(
        query=q,
        tier2_results=mock_tier2,
        top_k=5,
    )

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['label']} | score={r['score']}")
    print("\n")

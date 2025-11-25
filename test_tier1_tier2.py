import csv
from Medisync_Autocomplete import autocomplete_atc   # Tier 1
from ranker_engine import rank_candidates            # Tier 2


# ==============================
# 1. LIST QUERY CẦN TEST
# ==============================
QUERIES = [
    "para 500",
    "prctm",
    "efferalgan 150",
    "paracetamol 80mg dat",
    "metformin 500",
    "glucophage 850",
    "amlo 5mg",
    "coveram 5/5",
    "creon 25000",
    "omeprazol 20",
    "pantoloc 40",
]


def autocomplete_atc_with_ranker(query: str,
                                 top_k_t1: int = 30,
                                 top_k_t2: int = 10):
    """
    Chạy Tier-1 lấy ứng viên → Tier-2 re-rank → trả về kết quả cuối.
    """

    # === TIER 1: RETRIEVAL ===
    t1_results, t1_source, t1_ms = autocomplete_atc(query)

    # Chỉ giữ ATC (phòng sau này bạn có ICD chung trong 1 hàm)
    atc_results = [r for r in t1_results if r.get("type") == "ATC"]

    if not atc_results:
        return {
            "query": query,
            "tier1_source": t1_source,
            "tier1_latency_ms": t1_ms,
            "tier1_candidates": [],
            "tier2_results": [],
        }

    # Cắt bớt K ứng viên để đưa qua ranker
    atc_results = atc_results[:top_k_t1]
    codes = [r["code"] for r in atc_results]

    # Tier-1 thường gán cùng matched_syn cho cả group
    matched_syn = atc_results[0].get("matched_syn")

    # === TIER 2: RANKING ===
    ranked = rank_candidates(
        query=query,
        codes=codes,
        matched_syn=matched_syn,
        top_k=top_k_t2,
    )

    if not ranked:
        # Nếu Tier-2 fail (thiếu vec/meta) → trả về raw Tier-1
        return {
            "query": query,
            "tier1_source": t1_source,
            "tier1_latency_ms": t1_ms,
            "tier1_candidates": atc_results,
            "tier2_results": [],
        }

    # Map score -> code
    score_map = {code: float(score) for code, score in ranked}

    # Join lại meta từ Tier-1 + score Ranker
    reranked = [r for r in atc_results if r["code"] in score_map]
    for r in reranked:
        r["ranker_score"] = score_map[r["code"]]

    # Sort theo score giảm dần
    reranked.sort(key=lambda x: x["ranker_score"], reverse=True)

    return {
        "query": query,
        "tier1_source": t1_source,
        "tier1_latency_ms": t1_ms,
        "tier1_candidates": atc_results,
        "tier2_results": reranked[:top_k_t2],
    }


def run_batch(queries, output_csv="batch_tier1_tier2_results.csv",
              top_k_t1=30, top_k_t2=10):
    """
    Chạy toàn bộ list query, in kết quả + lưu CSV.
    """
    rows = []

    for q in queries:
        print("\n======================================")
        print(f"🔎 QUERY: '{q}'")
        print("======================================")

        res = autocomplete_atc_with_ranker(q, top_k_t1=top_k_t1, top_k_t2=top_k_t2)

        print(f"Tier-1 source : {res['tier1_source']}")
        print(f"Tier-1 latency: {res['tier1_latency_ms']} ms")

        # In top-10 Tier-1 raw
        print("\n--- Tier-1 candidates (raw) ---")
        for i, r in enumerate(res["tier1_candidates"][:10], start=1):
            forms = f" [{', '.join(r['forms'])}]" if r.get("forms") else ""
            print(f"{i:2d}. {r['code']} – {r['display']}{forms}")

        # In Tier-2
        if not res["tier2_results"]:
            print("\n(No Tier-2 re-ranking applied, dùng Tier-1 luôn)")
            # vẫn ghi ra CSV từ Tier-1
            rank_list = res["tier1_candidates"][:top_k_t2]
            for rank, r in enumerate(rank_list, start=1):
                rows.append({
                    "query": q,
                    "tier": "tier1_only",
                    "rank": rank,
                    "code": r["code"],
                    "display": r["display"],
                    "forms": "|".join(r.get("forms") or []),
                    "score": "",
                    "tier1_source": res["tier1_source"],
                    "tier1_latency_ms": res["tier1_latency_ms"],
                })
            continue

        print("\n--- Tier-2 ranked (LightGBM) ---")
        for i, r in enumerate(res["tier2_results"], start=1):
            forms = f" [{', '.join(r['forms'])}]" if r.get("forms") else ""
            score = r.get("ranker_score", 0.0)
            print(f"{i:2d}. {r['code']} – {r['display']}{forms} | score={score:.4f}")

            # Lưu từng dòng vào list để ghi CSV
            rows.append({
                "query": q,
                "tier": "tier2",
                "rank": i,
                "code": r["code"],
                "display": r["display"],
                "forms": "|".join(r.get("forms") or []),
                "score": score,
                "tier1_source": res["tier1_source"],
                "tier1_latency_ms": res["tier1_latency_ms"],
            })

    # Ghi CSV tổng hợp
    fieldnames = [
        "query", "tier", "rank", "code", "display", "forms",
        "score", "tier1_source", "tier1_latency_ms"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\n======================================")
    print(f"✅ DONE. Saved batch results to: {output_csv}")
    print("======================================")


if __name__ == "__main__":
    run_batch(QUERIES)

from flask import Flask, request, jsonify
from flask_cors import CORS

# Tier-1
from Medisync_Autocomplete import (
    icd_engine,
    atc_engine,
    autocomplete_icd,
    autocomplete_atc,
)

# Tier-2 (LightGBM Ranker)
from ranker_engine import rank_candidates

app = Flask(__name__)
CORS(app)


# ============================
# Load Tier-2 LightGBM ranker
# ============================
print("🔰 Loading LightGBM Ranker...")
RANKER = rank_candidates
print("✅ Tier-2 Ready!")


# =========================================================
# Helper: ATC + Ranker (giống test_tier1_tier2.py)
# =========================================================
def autocomplete_atc_with_ranker(query: str,
                                 top_k_t1: int = 30,
                                 top_k_t2: int = 10):
    """Chạy Tier-1 lấy ứng viên → Tier-2 re-rank → trả về kết quả cuối."""

    # === TIER 1: RETRIEVAL ===
    t1_results, t1_source, t1_ms = autocomplete_atc(query)

    # Chỉ giữ ATC (phòng sau này autocomplete_atc trả mixed)
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
    ranked = RANKER(
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
    # Giữ nguyên từng 'variant' của cùng 1 code (paracetamol / acetaminophen...)
    reranked = [r for r in atc_results if r["code"] in score_map]
    for r in reranked:
        r["ranker_score"] = score_map[r["code"]]

    # Sort theo score giảm dần (sort ổn định → giữ nguyên order giữa các variant)
    reranked.sort(key=lambda x: x["ranker_score"], reverse=True)

    return {
        "query": query,
        "tier1_source": t1_source,
        "tier1_latency_ms": t1_ms,
        "tier1_candidates": atc_results,
        "tier2_results": reranked[:top_k_t2],
    }


# =========================================
# API: /autocomplete (Tier-1 + Tier-2)
# =========================================
@app.get("/autocomplete")
def api_autocomplete():
    q = request.args.get("q", "").strip()
    mode = request.args.get("type", "icd")  # icd / atc
    top_k = int(request.args.get("top_k", 10))

    if not q:
        return jsonify({
            "query": q,
            "mode": mode,
            "results": [],
            "tier1_source": "empty",
            "tier1_latency_ms": 0,
            "tier2_used": False,
        })

    # -------------------------
    # ICD: hiện mới dùng Tier-1
    # -------------------------
    if mode == "icd":
        tier1_results, t_source, t_latency = autocomplete_icd(q)
        return jsonify({
            "query": q,
            "mode": mode,
            "results": tier1_results[:top_k],
            "tier1_source": t_source,
            "tier1_latency_ms": t_latency,
            "tier2_used": False,
        })

    # -------------------------
    # ATC: dùng Tier-2 LightGBM
    # -------------------------
    res = autocomplete_atc_with_ranker(
        query=q,
        top_k_t1=max(top_k, 30),
        top_k_t2=top_k,
    )

    t_source = res["tier1_source"]
    t_latency = res["tier1_latency_ms"]

    # Nếu Tier-2 không chạy được → fallback Tier-1
    if not res["tier2_results"]:
        final_results = []
        for r in res["tier1_candidates"][:top_k]:
            final_results.append({
                "code": r.get("code"),
                "label": r.get("label") or r.get("display") or "",
                "forms": r.get("forms") or [],
                "routes": r.get("routes") or [],
                "type": r.get("type", "ATC"),
            })

        return jsonify({
            "query": q,
            "mode": mode,
            "results": final_results,
            "tier1_source": t_source,
            "tier1_latency_ms": t_latency,
            "tier2_used": False,
        })

    # Ngược lại: dùng kết quả Tier-2
    final_results = []
    for r in res["tier2_results"]:
        final_results.append({
            "code": r.get("code"),
            "label": r.get("label") or r.get("display") or "",
            "forms": r.get("forms") or [],
            "routes": r.get("routes") or [],
            "type": r.get("type", "ATC"),
            "score": float(r.get("ranker_score", 0.0)),
        })

    return jsonify({
        "query": q,
        "mode": mode,
        "results": final_results,
        "tier1_source": t_source,
        "tier1_latency_ms": t_latency,
        "tier2_used": True,
    })


# =========================================
# API: Clear Cache Tier-1
# =========================================
@app.post("/clear_cache")
def clear_cache():
    icd_cleared = icd_engine.clear_cache()
    atc_cleared = atc_engine.clear_cache()
    return jsonify({
        "cleared_icd": icd_cleared,
        "cleared_atc": atc_cleared,
    })


# =========================================
# Run Server
# =========================================
if __name__ == "__main__":
    print("🚀 SmartEMR Autocomplete API running on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)

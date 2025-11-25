# test_ranker_v7.py
import time
from Medisync_Autocomplete import autocomplete_atc

tests = [
    "prctm",
    "para 500",
    "paracetamol 500mg",
    "creon",
    "pancrease",
    "nicorette",
    "siro ho",
    "vit b5 5mg",
    "riboflavin",
]

if __name__ == "__main__":
    for q in tests:
        print("\n===============================")
        print("🔍 QUERY:", q)

        t0 = time.perf_counter()
        results, source, latency_ms = autocomplete_atc(q)
        t1 = time.perf_counter()

        print(f"⏱  Engine latency: {latency_ms:.2f} ms")
        print(f"⏱  Wall-clock    : {(t1 - t0) * 1000:.2f} ms")
        print(f"📦 Source        : {source}")

        # In top 5 cho gọn
        for item in results[:5]:
            code = (
                item.get("code")
                or item.get("atc_code")
                or item.get("id")
                or "?"
            )
            label = (
                item.get("label")
                or item.get("name")
                or item.get("display")
                or item.get("inn")
                or ""
            )
            score = item.get("rank_score")

            if score is not None:
                print(f"{code:10s} | {score:7.4f} | {label}")
            else:
                print(f"{code:10s} | {label}")

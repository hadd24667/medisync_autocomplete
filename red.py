import redis, numpy as np

r = redis.Redis()

raw = r.get("atc:vec:N02BE01")
v = np.frombuffer(bytes.fromhex(raw.decode()), dtype=np.float32)
print(v.shape)

from extract_features_train import extract_feats
print(extract_feats("para 500", "N02BE01"))

# from rapidfuzz.distance import JaroWinkler, Levenshtein
# from unidecode import unidecode

# def norm(s):
#     return unidecode(s.lower().strip())

# def similarity(a, b):
#     a = norm(a)
#     b = norm(b)

#     jw = JaroWinkler.normalized_similarity(a, b)
#     lev = 1 - Levenshtein.normalized_distance(a, b)

#     overlap = len(set(a) & set(b)) / max(len(a), len(b))

#     return jw, lev, overlap


# if __name__ == "__main__":
#     query = "prctm"

#     inns = {
#         "PARACETAMOL": "paracetamol",
#         "ACETAMINOPHEN": "acetaminophen"
#     }

#     print("\n=== SIMILARITY USING ONLY INN ===\n")
#     for label, inn in inns.items():
#         jw, lev, ov = similarity(query, inn)

#         print(f"{label}")
#         print(f"  Jaro-Winkler : {jw:.4f}")
#         print(f"  Levenshtein  : {lev:.4f}")
#         print(f"  Overlap      : {ov:.4f}")
#         print("-" * 40)


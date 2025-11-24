import json
import redis

# Kết nối Redis
r = redis.Redis(host="localhost", port=6379, db=0)

# Load file JSON
with open("atc_synonyms_meta.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Ghi từng ATC vào Redis
for atc_code, meta in data.items():
    key = f"atc:meta:{atc_code}"
    r.set(key, json.dumps(meta, ensure_ascii=False))

print("✓ Đã import toàn bộ ATC metadata vào Redis")

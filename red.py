import redis, numpy as np

r = redis.Redis(host='localhost', port=6379)
raw = r.get("atc:vec:N02BE01")
vec = np.frombuffer(bytes.fromhex(raw.decode()), dtype=np.float32)
print(vec.shape)


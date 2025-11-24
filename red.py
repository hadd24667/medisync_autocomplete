import redis, numpy as np

r = redis.Redis(host='localhost', port=6379)

v = r.get("atc:vec:A09AA02")
arr = np.frombuffer(bytes.fromhex(v.decode("utf-8")), dtype=np.float32)
print(arr.shape)

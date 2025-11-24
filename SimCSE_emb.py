import json
import redis
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from unidecode import unidecode


# ============================================
# 0. Load SimCSE-PhoBERT
# ============================================

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"

print("Loading SimCSE PhoBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


# ============================================
# 1. Embedding helper (mean pooling)
# ============================================

def simcse_embed(text: str):
    text = unidecode(text.lower().strip())
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    for k in inputs:
        inputs[k] = inputs[k].to(device)

    with torch.no_grad():
        out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype("float32")
        return emb  # (768,)


# ============================================
# 2. Load ATC metadata
# ============================================

with open("atc_synonyms_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)


# ============================================
# 3. Connect Redis
# ============================================

r = redis.Redis(host="localhost", port=6379)


# ============================================
# 4. Build text for embedding
# ============================================

def build_text(m):
    """Combine metadata fields into 1 text string for semantic embedding."""
    inn = m["inn"]
    brand = " ".join(m["brand"])
    forms = " ".join(m["forms"])
    routes = " ".join(m["routes"])
    drug_class = m["drug_class"]
    return f"{inn} {brand} {forms} {routes} {drug_class}"


# ============================================
# 5. Generate & save embeddings to Redis
# ============================================

print("\nGenerating embeddings using SimCSE-PhoBERT...\n")

for code, m in meta.items():
    txt = build_text(m)
    emb = simcse_embed(txt)  # numpy float32 vector (768,)

    # Convert to hex for Redis storage
    emb_hex = emb.tobytes().hex()

    r.set(f"atc:vec:{code}", emb_hex)

    print(f"✓ Saved vector for ATC: {code}")

print("\n DONE — All SimCSE embeddings saved to Redis!\n")

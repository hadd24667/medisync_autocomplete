import torch
from transformers import AutoTokenizer, AutoModel
from unidecode import unidecode
import numpy as np


# =============================
# LOAD MODELS
# =============================

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SIMCSE = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
MODEL_PHOBERT = "vinai/phobert-base"

print("Loading SimCSE PhoBERT...")
tokenizer_s = AutoTokenizer.from_pretrained(MODEL_SIMCSE)
model_s = AutoModel.from_pretrained(MODEL_SIMCSE).to(device)
model_s.eval()

print("Loading PhoBERT base...")
tokenizer_p = AutoTokenizer.from_pretrained(MODEL_PHOBERT)
model_p = AutoModel.from_pretrained(MODEL_PHOBERT).to(device)
model_p.eval()


# =============================
# EMBEDDING FUNCTION
# =============================
def embed(model, tokenizer, text: str):
    text = unidecode(text.lower().strip())
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    for k in inputs:
        inputs[k] = inputs[k].to(device)

    with torch.no_grad():
        out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype("float32")
        return emb


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =============================
# TEST PAIRS
# =============================
TEST_PAIRS = [
    ("paracetamol", "acetaminophen"),
    ("paracetamol", "panadol"),
    ("paracetamol 500mg", "efferalgan"),
    ("paracetamol 500mg", "thuoc giam dau ha sot"),
    ("ibuprofen", "advil"),
    ("ibuprofen", "thuoc giam dau khang viem"),
    ("amoxicillin", "thuoc khang sinh"),
    ("azithromycin", "zithromax"),
    ("guaifenesin", "mucinex"),
    ("vitamin c", "ascorbic acid"),
    ("vitamin b1 b6 b12", "vitamin b complex"),
    ("salbutamol", "ventolin"),
    ("omeprazol", "thuoc giam acid da day"),
    ("loratadin", "thuoc di ung"),
]


# =============================
# MAIN
# =============================
print("\n🔍 Comparing PhoBERT vs SimCSE-PhoBERT...\n")
print(f"{'Query':25s} | {'Target':25s} | PhoBERT | SimCSE")
print("-" * 80)

for a, b in TEST_PAIRS:
    emb_s_a = embed(model_s, tokenizer_s, a)
    emb_s_b = embed(model_s, tokenizer_s, b)
    score_s = cosine(emb_s_a, emb_s_b)

    emb_p_a = embed(model_p, tokenizer_p, a)
    emb_p_b = embed(model_p, tokenizer_p, b)
    score_p = cosine(emb_p_a, emb_p_b)

    print(f"{a:25s} | {b:25s} | {score_p:7.4f} | {score_s:7.4f}")

print("\n✓ DONE\n")

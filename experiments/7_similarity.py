import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pickle
from src.gru_model import GRUModel

print("\n[SIM] Loading data...")

X_domain, X_concept, X_time, y = pickle.load(
    open("data/processed/gru_tensors.pkl","rb")
)

model = GRUModel(num_concepts=50000)
model.load_state_dict(torch.load("data/processed/gru_model.pt"))
model.eval()

# =========================================================
# STEP 1 — COMPUTE ALL EMBEDDINGS
# =========================================================
print("[SIM] Computing embeddings...")

embeddings = []

with torch.no_grad():

    for i in range(len(X_concept)):

        x = torch.tensor(X_concept[i]).unsqueeze(0)

        emb = model.get_patient_embedding(x).squeeze(0)

        embeddings.append(emb.numpy())

embeddings = np.array(embeddings)

print("[SIM] Embeddings shape:", embeddings.shape)

# =========================================================
# STEP 2 — SIMILARITY FUNCTION
# =========================================================
def top_k_similar(idx, k=5):

    query = embeddings[idx]

    scores = []

    for j in range(len(embeddings)):

        dist = np.linalg.norm(query - embeddings[j])

        scores.append((j, dist))

    return sorted(scores, key=lambda x: x[1])[:k]

# =========================================================
# STEP 3 — TEST
# =========================================================
idx = 99

print("\nTop similar patients to:", idx)
print(top_k_similar(idx, k=5))

print("Embedding check:")

e0 = embeddings[0]
e1 = embeddings[1]
e2 = embeddings[2]

print(np.linalg.norm(e0 - e1))
print(np.linalg.norm(e0 - e2))

print("Patient 0:", np.unique(X_concept[0])[:10])
print("Patient 1:", np.unique(X_concept[1])[:10])
print("Patient 2:", np.unique(X_concept[2])[:10])
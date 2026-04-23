import pickle
import numpy as np

print("\n[PRECOMPUTE] Starting optimized pipeline...")

# =========================================================
# LOAD DATA
# =========================================================
print("[LOAD] Loading tensors + labels...")

X_domain, X_concept, X_time, y = pickle.load(
    open("data/processed/gru_tensors.pkl", "rb")
)

labels_dict = pickle.load(open("data/processed/labels.pkl", "rb"))

# Convert labels → array (FASTER)
patient_ids = np.array(list(labels_dict.keys()))
labels = np.array(list(labels_dict.values()), dtype=np.int8)

N = len(patient_ids)

print(f"[INFO] Number of patients: {N}")

# =========================================================
# 1. SIMILARITY MATRIX (TOP-K ONLY, NUMPY)
# =========================================================
print("\n[SIMILARITY] Building fast similarity index...")

TOP_K = 20

# Simple embedding proxy (mean concept id)
embeddings = np.mean(X_concept, axis=1).astype(np.float32)

similarity_matrix = np.zeros((N, TOP_K, 2), dtype=np.float32)
# columns: [patient_index, distance]

for i in range(N):

    if i % 50 == 0:
        print(f"[SIM] Processing patient {i}/{N}")

    dist = np.abs(embeddings[i] - embeddings)

    idx = np.argsort(dist)[1:TOP_K+1]  # skip self

    similarity_matrix[i, :, 0] = idx
    similarity_matrix[i, :, 1] = dist[idx]

# =========================================================
# 2. DIGITAL TWIN (VECTOR FORM)
# =========================================================
print("\n[DIGITAL TWIN] Precomputing scenarios...")

digital_twin = np.zeros((N, 3), dtype=np.float32)
# columns: [baseline, drug_boost, early_remove]

for i in range(N):

    if i % 100 == 0:
        print(f"[DT] Patient {i}/{N}")

    base = embeddings[i]

    digital_twin[i, 0] = base
    digital_twin[i, 1] = base * 1.1
    digital_twin[i, 2] = base * 0.9

# =========================================================
# 3. GRU EXPLANATIONS (VECTOR FORMAT)
# =========================================================
print("\n[EXPLAIN] Precomputing event importance...")

MAX_EVENTS = 50

event_index = np.zeros((N, MAX_EVENTS), dtype=np.int16)
importance = np.zeros((N, MAX_EVENTS), dtype=np.float32)

for i in range(N):

    if i % 100 == 0:
        print(f"[EXPLAIN] Patient {i}/{N}")

    seq_len = min(MAX_EVENTS, len(X_concept[i]))

    event_index[i, :seq_len] = np.arange(seq_len)

    # simple stable importance proxy
    imp = np.abs(np.diff(X_concept[i][:seq_len], prepend=0))

    importance[i, :seq_len] = imp.astype(np.float32)

# =========================================================
# 4. SAVE EVERYTHING (COMPACT FORMAT)
# =========================================================
print("\n[SAVE] Writing optimized files...")

np.save("data/processed/patient_ids.npy", patient_ids)

np.save("data/processed/similarity.npy", similarity_matrix)
np.save("data/processed/digital_twin.npy", digital_twin)

np.save("data/processed/event_index.npy", event_index)
np.save("data/processed/importance.npy", importance)

pickle.dump(labels_dict, open("data/processed/labels.pkl", "wb"))

print("\n[PRECOMPUTE] DONE ✔")
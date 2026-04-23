import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
from src.embedding import build_embeddings

print("[DTW PIPELINE] Loading TRAIN tensors only...")

train_tensors = pickle.load(open("data/processed/train_tensors.pkl", "rb"))

print("[DTW PIPELINE] Building embeddings...")
embeddings = build_embeddings(train_tensors)

pids = list(embeddings.keys())
seqs = list(embeddings.values())

from src.dtw import compute_dtw_matrix

print("[DTW PIPELINE] Computing DTW matrix...")
dtw_matrix = compute_dtw_matrix(seqs)

np.save("data/processed/dtw_matrix_train.npy", dtw_matrix)

pickle.dump(pids, open("data/processed/train_patient_ids.pkl", "wb"))

print("[DTW PIPELINE] Done ✔")
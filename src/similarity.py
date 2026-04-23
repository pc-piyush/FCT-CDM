import numpy as np

def cosine_similarity(seq1, seq2):
    v1 = np.array(seq1)
    v2 = np.array(seq2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
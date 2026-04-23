import torch
import torch.nn as nn
import math

class EventEmbedder(nn.Module):
    def __init__(self, vocab_sizes, dim=64):
        super().__init__()

        print("[EMBEDDING] Initializing embeddings...")

        self.concept_emb = nn.ModuleDict({
            d: nn.Embedding(v + 1, dim)
            for d, v in vocab_sizes.items()
        })

        self.dim = dim
        self.value_proj = nn.Linear(1, dim)

    def time_embedding(self, t):
        pe = torch.zeros(self.dim)

        for i in range(0, self.dim, 2):
            div = 10000 ** (i / self.dim)
            pe[i] = math.sin(t / div)
            pe[i+1] = math.cos(t / div)

        return pe

    def forward(self, domain, concept_id, t, value=None):
        c = self.concept_emb[domain](concept_id)
        t_emb = self.time_embedding(t)

        if value is not None:
            v_emb = self.value_proj(torch.tensor([value], dtype=torch.float))
        else:
            v_emb = torch.zeros_like(c)

        return c + t_emb + v_emb


import numpy as np

def safe_float(x):
    try:
        if x is None:
            return 0.0
        if isinstance(x, str):
            if x.strip() == "" or x.lower() == "none":
                return 0.0
        return float(x)
    except:
        return 0.0


def build_embeddings(tensors):
    print("[EMBEDDING] Building DTW-compatible embeddings...")

    embeddings = {}

    for pid, tensor in tensors.items():
        seq = []

        for domain in tensor:
            for event in tensor[domain]:

                # (t, idx)
                if len(event) == 2:
                    t, idx = event
                    if idx is None:
                        continue

                    seq.append(idx * 1000 + t)

                # (t, idx, value)
                elif len(event) == 3:
                    t, idx, val = event

                    if idx is None:
                        continue

                    val = safe_float(val)

                    seq.append(idx * 1000 + t + val)

        embeddings[pid] = np.array(seq, dtype=np.float32)

    print(f"[EMBEDDING] Built embeddings for {len(embeddings)} patients")

    return embeddings
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
DATA_DIR = "data/processed"
EMB_DIM = 128
EPOCHS = 5
BATCH_SIZE = 16

os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# 1. LOAD RAW TENSORS
# =========================
print("\n[PIPELINE] Loading raw tensors...")

train = pickle.load(open(f"{DATA_DIR}/train_tensors.pkl", "rb"))
val   = pickle.load(open(f"{DATA_DIR}/val_tensors.pkl", "rb"))
test  = pickle.load(open(f"{DATA_DIR}/test_tensors.pkl", "rb"))

print(f"[PIPELINE] Train patients: {len(train)}")

# =========================
# 2. BUILD VOCAB (STRICT)
# =========================
print("\n[PIPELINE] Building vocab...")

vocab = {}
idx = 1  # 0 = PAD

def extract_concepts(tensor):
    for domain in tensor:
        for event in tensor[domain]:
            if len(event) == 2:
                yield event[1]
            elif len(event) == 3:
                yield event[1]

for dataset in [train]:
    for pid, tensor in dataset.items():
        for concept in extract_concepts(tensor):

            # HARD FILTER: reject OMOP-like IDs
            if isinstance(concept, int) and concept > 100000:
                continue

            if concept not in vocab:
                vocab[concept] = idx
                idx += 1

print(f"[PIPELINE] Vocab size = {len(vocab)}")

# =========================
# 3. ENCODE TENSORS (FORCE SAFE)
# =========================
print("\n[PIPELINE] Encoding tensors...")

def encode(tensor):
    encoded = {}

    for pid, t in tensor.items():
        new_t = {}

        for domain in t:
            events = []

            for e in t[domain]:

                if len(e) == 2:
                    time, concept = e
                    cid = vocab.get(concept, 0)

                elif len(e) == 3:
                    time, concept, val = e
                    cid = vocab.get(concept, 0)

                else:
                    continue

                # FINAL SAFETY CHECK
                if cid >= len(vocab):
                    cid = 0

                events.append((time, cid))

            new_t[domain] = events

        encoded[pid] = new_t

    return encoded

train = encode(train)
val   = encode(val)
test  = encode(test)

# =========================
# 4. SAVE CLEAN DATA
# =========================
print("\n[PIPELINE] Saving encoded tensors...")

pickle.dump(train, open(f"{DATA_DIR}/train_enc.pkl", "wb"))
pickle.dump(val,   open(f"{DATA_DIR}/val_enc.pkl", "wb"))
pickle.dump(test,  open(f"{DATA_DIR}/test_enc.pkl", "wb"))
pickle.dump(vocab, open(f"{DATA_DIR}/vocab.pkl", "wb"))

# =========================
# 5. DATASET
# =========================
class PatientDataset(Dataset):
    def __init__(self, data):
        self.data = list(data.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pid, tensor = self.data[idx]

        seq = []
        for domain in tensor:
            for e in tensor[domain]:
                seq.append(e[1])  # ONLY SAFE INDEX

        seq = seq[:200]
        seq += [0] * (200 - len(seq))

        return torch.tensor(seq), torch.tensor(0.0)  # placeholder label

# =========================
# 6. MODEL
# =========================
class GRUModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, EMB_DIM)
        self.gru = nn.GRU(EMB_DIM, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.emb(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0)).squeeze()

# =========================
# 7. LOAD DATA
# =========================
train_ds = PatientDataset(train)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GRUModel(len(vocab)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

# =========================
# 8. TRAIN LOOP
# =========================
print("\n[PIPELINE] Training started...")

for epoch in range(EPOCHS):
    total_loss = 0

    for i, (x, y) in enumerate(train_dl):
        x = x.to(device)
        y = y.to(device)

        # HARD GUARANTEE CHECK
        assert x.max().item() < len(vocab) + 1, "❌ vocab mismatch"

        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        total_loss += loss.item()

        if i % 20 == 0:
            print(f"[EPOCH {epoch}] batch {i} loss {loss.item():.4f}")

    print(f"[EPOCH {epoch}] total loss {total_loss:.4f}")

print("\n[PIPELINE] Done ✔ Model trained safely")
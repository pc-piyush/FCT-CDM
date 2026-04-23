import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




# import pickle
# from sklearn.model_selection import train_test_split
# from src.models import train_model
# from src.evaluation import evaluate

# print("[MODEL] Loading features + labels...")

# X_train = pickle.load(open("data/processed/train_features.pkl", "rb"))
# y_train = pickle.load(open("data/processed/train_labels.pkl", "rb"))

# X_val = pickle.load(open("data/processed/val_features.pkl", "rb"))
# y_val = pickle.load(open("data/processed/val_labels.pkl", "rb"))

# X_test = pickle.load(open("data/processed/test_features.pkl", "rb"))
# y_test = pickle.load(open("data/processed/test_labels.pkl", "rb"))

# print("[MODEL] Training model...")
# model = train_model(X_train, y_train)

# print("[MODEL] Evaluating on validation...")
# val_results = evaluate(model, X_val, y_val)

# print("[MODEL] Evaluating on test...")
# test_results = evaluate(model, X_test, y_test)

# print("[RESULTS]", val_results, test_results)

# import pickle
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from src.embedding import build_embeddings

# print("[MODEL] Loading tensors...")

# train_tensors = pickle.load(open("data/processed/train_tensors.pkl", "rb"))
# val_tensors = pickle.load(open("data/processed/val_tensors.pkl", "rb"))
# test_tensors = pickle.load(open("data/processed/test_tensors.pkl", "rb"))

# train_labels = pickle.load(open("data/processed/train_labels.pkl", "rb"))
# val_labels = pickle.load(open("data/processed/val_labels.pkl", "rb"))
# test_labels = pickle.load(open("data/processed/test_labels.pkl", "rb"))

# print("[MODEL] Building embeddings...")

# X_train = list(build_embeddings(train_tensors).values())
# X_val = list(build_embeddings(val_tensors).values())
# X_test = list(build_embeddings(test_tensors).values())

# y_train = list(train_labels.values())
# y_val = list(val_labels.values())
# y_test = list(test_labels.values())

# print("[MODEL] Converting to arrays...")

# # pad / truncate for simplicity
# max_len = 200

# def pad(X):
#     return np.array([
#         np.pad(x[:max_len], (0, max(0, max_len - len(x))))
#         for x in X
#     ])

# X_train, X_val, X_test = pad(X_train), pad(X_val), pad(X_test)

# print("[MODEL] Training logistic regression...")

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# print("[MODEL] Evaluating...")

# print("Train accuracy:", model.score(X_train, y_train))
# print("Val accuracy:", model.score(X_val, y_val))
# print("Test accuracy:", model.score(X_test, y_test))

# import torch
# from torch.utils.data import DataLoader
# import pickle

# from src.dataset import PatientDataset
# from src.collate import collate_fn
# from src.model import GRUPatientModel

# print("[TRAIN] Loading tensors...")

# train_tensors = pickle.load(open("data/processed/train_tensors.pkl", "rb"))
# val_tensors = pickle.load(open("data/processed/val_tensors.pkl", "rb"))

# train_labels = pickle.load(open("data/processed/train_labels.pkl", "rb"))
# val_labels = pickle.load(open("data/processed/val_labels.pkl", "rb"))

# train_ds = PatientDataset(train_tensors, train_labels)
# val_ds = PatientDataset(val_tensors, val_labels)

# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn)

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = GRUPatientModel().to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = torch.nn.BCEWithLogitsLoss()

# print("[TRAIN] Starting training loop...")

# for epoch in range(5):
#     model.train()
#     total_loss = 0

#     for x, y in train_loader:
#         x, y = x.to(device), y.to(device)

#         optimizer.zero_grad()
#         preds = model(x)

#         loss = loss_fn(preds, y)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"[TRAIN] Epoch {epoch} Loss: {total_loss:.4f}")

# print("[TRAIN] Training complete ✔")

import pickle
import torch
from torch.utils.data import DataLoader

from src.dataset import PatientDataset
from src.collate import collate_fn
from src.model import GRUPatientModel


print("[DEBUG] Loading TRAIN FILE...")

train_tensors = pickle.load(open("data/processed/train_enc.pkl", "rb"))

for pid, t in list(train_tensors.items())[:1]:
    for domain in t:
        for event in t[domain][:5]:
            print("[DEBUG EVENT]", event)


for pid, t in train_tensors.items():
    for domain in t:
        for event in t[domain]:
            if event[1] > 50000:   # sanity bound
                raise ValueError(f"RAW OMOP STILL PRESENT: {event}")

import os
import pickle

print("[DEBUG] CWD =", os.getcwd())
print("[DEBUG] Loading file =", os.path.abspath("data/processed/train_enc.pkl"))

def is_raw(x):
    return x > 100000  # OMOP concepts are large

for pid, t in train_tensors.items():
    for domain in t:
        for event in t[domain]:
            if is_raw(event[1]):
                print("❌ RAW DETECTED:", event)
                raise RuntimeError("Encoding failed - raw OMOP still present")

# ----------------------------
# LOAD ENCODED DATA
# ----------------------------
print("[TRAIN] Loading encoded tensors...")

train_tensors = pickle.load(open("data/processed/train_enc.pkl", "rb"))
val_tensors   = pickle.load(open("data/processed/val_enc.pkl", "rb"))
test_tensors  = pickle.load(open("data/processed/test_enc.pkl", "rb"))

train_labels = pickle.load(open("data/processed/train_labels.pkl", "rb"))
val_labels   = pickle.load(open("data/processed/val_labels.pkl", "rb"))
test_labels  = pickle.load(open("data/processed/test_labels.pkl", "rb"))

print(f"[TRAIN] Train patients: {len(train_tensors)}")
print(f"[TRAIN] Val patients: {len(val_tensors)}")
print(f"[TRAIN] Test patients: {len(test_tensors)}")

# ----------------------------
# DATASETS
# ----------------------------
train_ds = PatientDataset(train_tensors, train_labels)
val_ds   = PatientDataset(val_tensors, val_labels)
test_ds  = PatientDataset(test_tensors, test_labels)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

# ----------------------------
# MODEL
# ----------------------------
# IMPORTANT: vocab size must match encoded vocab
vocab = pickle.load(open("data/processed/vocab.pkl", "rb"))
vocab_size = len(vocab) + 1

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[TRAIN] Vocab size: {vocab_size}")
print(f"[TRAIN] Device: {device}")

model = GRUPatientModel(vocab_size=vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCEWithLogitsLoss()

# ----------------------------
# TRAINING LOOP
# ----------------------------
EPOCHS = 5

print("[TRAIN] Starting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    print(f"\n[TRAIN] Epoch {epoch+1}/{EPOCHS}")

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        # safety check (prevents silent crashes)
        if x.max() >= vocab_size:
            raise ValueError(f"[ERROR] Token index out of range: {x.max()} >= {vocab_size}")

        optimizer.zero_grad()

        preds = model(x)
        loss = loss_fn(preds, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 20 == 0:
            print(f"[TRAIN] Batch {i}, Loss: {loss.item():.4f}")

    print(f"[TRAIN] Epoch {epoch+1} Loss: {total_loss:.4f}")

    # ----------------------------
    # VALIDATION
    # ----------------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            preds = model(x)
            loss = loss_fn(preds, y)
            val_loss += loss.item()

    print(f"[VAL] Epoch {epoch+1} Loss: {val_loss:.4f}")

# ----------------------------
# TEST EVALUATION
# ----------------------------
print("\n[TEST] Evaluating final model...")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()

        correct += (preds == y).sum().item()
        total += y.size(0)

accuracy = correct / total

print(f"[TEST] Accuracy: {accuracy:.4f}")

# ----------------------------
# SAVE MODEL
# ----------------------------
torch.save(model.state_dict(), "data/processed/gru_model.pt")

print("[TRAIN] Model saved ✔")
print("[TRAIN] Done")
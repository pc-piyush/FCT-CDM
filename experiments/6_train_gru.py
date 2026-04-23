import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from src.gru_model import GRUModel

print("\n[TRAIN-GRU] Loading data...")

X_domain, X_concept, X_time, y = pickle.load(
    open("data/processed/gru_tensors.pkl","rb")
)

device = "cpu"

# =========================================================
# MODEL
# =========================================================
model = GRUModel(num_concepts=50000)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCEWithLogitsLoss()

X_concept = torch.tensor(X_concept).long()
y = torch.tensor(y).float()

for epoch in range(10):

    model.train()

    opt.zero_grad()

    out = model(None, X_concept, None)

    loss = loss_fn(out, y)

    loss.backward()
    opt.step()

    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# =========================================================
# EVALUATION (FIXED PART)
# =========================================================
print("\n[GRU] Evaluating model...")

# model.eval()
# with torch.no_grad():

#     logits = model(X_domain, X_concept, X_time).squeeze()
#     probs = torch.sigmoid(logits).numpy()

#     y_true = y.numpy()
#     y_pred_proba = probs

model.eval()

with torch.no_grad():

    logits = model(None, X_concept, None)

    probs = torch.sigmoid(logits).numpy()

    y_true = y.numpy()
    y_pred_proba = probs

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_true, probs)

print("GRU AUC:", auc)

gru_auc_score = roc_auc_score(y_true, y_pred_proba)
gru_acc_score = accuracy_score(y_true, (y_pred_proba > 0.5).astype(int))

print(f"[GRU] AUC: {gru_auc_score:.4f}")
print(f"[GRU] ACC: {gru_acc_score:.4f}")

# =========================================================
# SAVE MODEL
# =========================================================
torch.save(model.state_dict(), "data/processed/gru_model.pt")

print("[TRAIN-GRU] Model saved ✔")

# =========================================================
# SAVE INTO RESULTS FILE
# =========================================================
print("[GRU] Saving results into model_results.pkl")

results = pickle.load(open("data/processed/model_results.pkl", "rb"))

results["GRU_AUC"] = float(gru_auc_score)
results["GRU_ACC"] = float(gru_acc_score)

pickle.dump(results, open("data/processed/model_results.pkl", "wb"))

print("[GRU] Added to model_results.pkl ✔")

print("Positive rate:", y.mean())
print("logits std:", logits.std().item())
print("logits mean:", logits.mean().item())
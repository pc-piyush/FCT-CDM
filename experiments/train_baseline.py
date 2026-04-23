import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import json

print("[BASELINE] Loading features...")

X_train, y_train = pickle.load(open("data/processed/train_features.pkl", "rb"))
X_val, y_val = pickle.load(open("data/processed/val_features.pkl", "rb"))
X_test, y_test = pickle.load(open("data/processed/test_features.pkl", "rb"))

model = LogisticRegression(max_iter=1000)

print("[BASELINE] Training...")
model.fit(X_train, y_train)

def evaluate(X, y, name):
    probs = model.predict_proba(X)[:,1]
    preds = (probs > 0.5).astype(int)

    return {
        "auc": roc_auc_score(y, probs),
        "accuracy": accuracy_score(y, preds)
    }

print("[BASELINE] Evaluating...")

metrics = {
    "train": evaluate(X_train, y_train, "train"),
    "val": evaluate(X_val, y_val, "val"),
    "test": evaluate(X_test, y_test, "test"),
}

print(metrics)

with open("data/processed/metrics_baseline.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("[BASELINE] Done ✔")
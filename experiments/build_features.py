import pickle
import numpy as np

print("[FEATURES] Loading encoded tensors...")

train = pickle.load(open("data/processed/train_enc.pkl", "rb"))
val   = pickle.load(open("data/processed/val_enc.pkl", "rb"))
test  = pickle.load(open("data/processed/test_enc.pkl", "rb"))

def build_features(data):
    X = []
    y = []

    for pid, t in data.items():

        # simple aggregated features
        num_events = 0
        num_conditions = 0
        num_drugs = 0
        num_measurements = 0

        for domain in t:
            num_events += len(t[domain])

            if domain == "condition":
                num_conditions += len(t[domain])

            if domain == "drug":
                num_drugs += len(t[domain])

            if domain == "measurement":
                num_measurements += len(t[domain])

        X.append([
            num_events,
            num_conditions,
            num_drugs,
            num_measurements
        ])

        y.append(0)  # placeholder (replace with real labels)

    return np.array(X), np.array(y)

print("[FEATURES] Building train features...")
X_train, y_train = build_features(train)
X_val, y_val = build_features(val)
X_test, y_test = build_features(test)

import pickle

pickle.dump((X_train, y_train), open("data/processed/train_features.pkl", "wb"))
pickle.dump((X_val, y_val), open("data/processed/val_features.pkl", "wb"))
pickle.dump((X_test, y_test), open("data/processed/test_features.pkl", "wb"))

print("[FEATURES] Saved ✔")
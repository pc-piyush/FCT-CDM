import pickle
import numpy as np

print("\n[FEATURES] Loading tensor data...")

data = pickle.load(open("data/processed/tensors.pkl","rb"))
labels = pickle.load(open("data/processed/labels.pkl","rb"))

X, y = [], []

for i, (pid, t) in enumerate(data.items()):

    print(f"[FEATURES] Patient {i+1}/{len(data)}")

    f = []

    for d in ["condition","drug","measurement","procedure","visit"]:

        events = t[d]

        f.append(len(events))
        f.append(len(set([e[1] for e in events])))

    X.append(f)
    y.append(labels[pid])

X = np.array(X)
y = np.array(y)

print("[FEATURES] Shape:", X.shape)

pickle.dump((X,y), open("data/processed/features.pkl","wb"))

print("[FEATURES] Done ✔")
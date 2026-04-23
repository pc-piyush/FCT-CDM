import pickle
import numpy as np

print("\n[TENSOR] Building GRU-ready tensors...")

data = pickle.load(open("data/processed/tensors.pkl","rb"))
labels = pickle.load(open("data/processed/labels.pkl","rb"))

domain_map = {"condition":0,"drug":1,"procedure":2,"measurement":3,"visit":4}

X_domain, X_concept, X_time, y = [], [], [], []

for i,(pid, events) in enumerate(data.items()):

    print(f"[TENSOR] Patient {i+1}/{len(data)}")

    d_seq, c_seq, t_seq = [], [], []

    for e in events:

        # CASE 1: already tuple (correct format)
        if isinstance(e, tuple) or isinstance(e, list):
            domain = e[0]
            concept = e[1]

        # CASE 2: string "domain|concept|time"
        elif isinstance(e, str):
            parts = e.split("|")
            if len(parts) < 2:
                continue
            domain = parts[0]
            concept = parts[1]

        else:
            continue

        # HARD SAFETY CHECK
        if domain not in domain_map:
            print("[SKIP INVALID DOMAIN]", domain)
            continue

        d_seq.append(domain_map[domain])
        c_seq.append(int(concept))
        t_seq.append(1.0)

    # padding
    max_len = 200

    def pad(x):
        return (x + [0]*max(0, max_len-len(x)))[:max_len]

    X_domain.append(pad(d_seq))
    X_concept.append(pad(c_seq))
    X_time.append(pad(t_seq))

    y.append(labels[pid])

X_domain = np.array(X_domain)
X_concept = np.array(X_concept)
X_time = np.array(X_time)
y = np.array(y)

pickle.dump((X_domain, X_concept, X_time, y),
            open("data/processed/gru_tensors.pkl","wb"))

print("[TENSOR] Done ✔")
import pickle
import random

print("[COHORT] Loading tensors...")

tensors = pickle.load(open("data/processed/train_tensors.pkl", "rb"))

T2D_CONCEPTS = set([
    201826, 201820, 443238
])

cases = {}
controls = {}

print("[COHORT] Building labels...")

for pid, t in tensors.items():

    has_t2d = False

    for domain in t:
        for event in t[domain]:

            concept = event[1] if len(event) >= 2 else None

            if concept in T2D_CONCEPTS:
                has_t2d = True
                break

    if has_t2d:
        cases[pid] = t
    else:
        controls[pid] = t

print(f"[COHORT] Cases: {len(cases)}")
print(f"[COHORT] Controls: {len(controls)}")

# ----------------------------
# BALANCED SAMPLING
# ----------------------------
n = min(len(cases), len(controls))

cases_sample = dict(random.sample(list(cases.items()), n))
controls_sample = dict(random.sample(list(controls.items()), n))

balanced = {**cases_sample, **controls_sample}

labels = {}
for pid in cases_sample:
    labels[pid] = 1
for pid in controls_sample:
    labels[pid] = 0

print(f"[COHORT] Final balanced cohort: {len(balanced)}")

pickle.dump(balanced, open("data/processed/cohort.pkl", "wb"))
pickle.dump(labels, open("data/processed/labels.pkl", "wb"))

print("[COHORT] Done ✔")
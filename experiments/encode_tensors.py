import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pickle
from src.vocab import VocabBuilder

print("[ENCODE] Loading train tensors...")

train = pickle.load(open("data/processed/train_tensors.pkl", "rb"))
val = pickle.load(open("data/processed/val_tensors.pkl", "rb"))
test = pickle.load(open("data/processed/test_tensors.pkl", "rb"))

vocab = VocabBuilder().fit(train)

def encode_tensor(tensor):
    encoded = {}

    for pid, t in tensor.items():
        new_t = {}

        for domain in t:
            new_events = []

            for event in t[domain]:
                if len(event) == 2:
                    time, concept = event
                    new_events.append((time, vocab.encode(concept)))

                elif len(event) == 3:
                    time, concept, val = event
                    new_events.append((time, vocab.encode(concept), val))

            new_t[domain] = new_events

        encoded[pid] = new_t

    return encoded

print("[ENCODE] Encoding datasets...")

train_enc = encode_tensor(train)
val_enc = encode_tensor(val)
test_enc = encode_tensor(test)

pickle.dump(train_enc, open("data/processed/train_enc.pkl", "wb"))
pickle.dump(val_enc, open("data/processed/val_enc.pkl", "wb"))
pickle.dump(test_enc, open("data/processed/test_enc.pkl", "wb"))
pickle.dump(vocab.token_to_id, open("data/processed/vocab.pkl", "wb"))

print("[ENCODE] Done ✔")
import pickle

def extract_concepts(tensor):
    for domain in tensor:
        for event in tensor[domain]:
            if len(event) == 2:
                yield event[1]
            elif len(event) == 3:
                yield event[1]


def build_vocab(train):
    print("[VOCAB] Building clean vocab...")

    vocab = {}
    idx = 1  # 0 = padding

    for pid, tensor in train.items():
        for concept in extract_concepts(tensor):

            # ❗ HARD GUARD: prevent raw OMOP leakage
            if isinstance(concept, int) and concept > 100000:
                continue

            if concept not in vocab:
                vocab[concept] = idx
                idx += 1

    print(f"[VOCAB] Final size = {len(vocab)}")
    return vocab


def encode_tensor(tensor, vocab):
    encoded = {}

    for pid, t in tensor.items():
        new_t = {}

        for domain in t:
            new_events = []

            for event in t[domain]:

                if len(event) == 2:
                    time, concept = event

                elif len(event) == 3:
                    time, concept, val = event

                else:
                    continue

                # SAFE ENCODING ONLY
                idx = vocab.get(concept, 0)

                new_events.append((time, idx))

            new_t[domain] = new_events

        encoded[pid] = new_t

    return encoded


def run_encoding():
    print("[ENCODE] Loading tensors...")

    train = pickle.load(open("data/processed/train_tensors.pkl", "rb"))
    val   = pickle.load(open("data/processed/val_tensors.pkl", "rb"))
    test  = pickle.load(open("data/processed/test_tensors.pkl", "rb"))

    vocab = build_vocab(train)

    print("[ENCODE] Encoding safely...")

    train_enc = encode_tensor(train, vocab)
    val_enc   = encode_tensor(val, vocab)
    test_enc  = encode_tensor(test, vocab)

    pickle.dump(train_enc, open("data/processed/train_enc.pkl", "wb"))
    pickle.dump(val_enc, open("data/processed/val_enc.pkl", "wb"))
    pickle.dump(test_enc, open("data/processed/test_enc.pkl", "wb"))
    pickle.dump(vocab, open("data/processed/vocab.pkl", "wb"))

    print("[ENCODE] Done ✔")

run_encoding()
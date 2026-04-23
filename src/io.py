import pickle

def load_split_tensors():
    print("[IO] Loading train/val/test tensors...")

    train = pickle.load(open("data/processed/train_tensors.pkl", "rb"))
    val = pickle.load(open("data/processed/val_tensors.pkl", "rb"))
    test = pickle.load(open("data/processed/test_tensors.pkl", "rb"))

    print(f"[IO] Train: {len(train)} patients")
    print(f"[IO] Val: {len(val)} patients")
    print(f"[IO] Test: {len(test)} patients")

    return train, val, test


def load_split_labels():
    print("[IO] Loading train/val/test labels...")

    train = pickle.load(open("data/processed/train_labels.pkl", "rb"))
    val = pickle.load(open("data/processed/val_labels.pkl", "rb"))
    test = pickle.load(open("data/processed/test_labels.pkl", "rb"))

    return train, val, test
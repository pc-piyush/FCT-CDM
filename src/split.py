import numpy as np
import pandas as pd

def split_cohort(cohort, config):
    print("[SPLIT] Splitting cohort into train/val/test...")

    seed = config["split"]["seed"]
    np.random.seed(seed)

    cohort = cohort.sample(frac=1).reset_index(drop=True)  # shuffle

    n = len(cohort)

    train_end = int(n * config["split"]["train_ratio"])
    val_end = train_end + int(n * config["split"]["val_ratio"])

    train = cohort.iloc[:train_end]
    val = cohort.iloc[train_end:val_end]
    test = cohort.iloc[val_end:]

    print(f"[SPLIT] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    return train, val, test
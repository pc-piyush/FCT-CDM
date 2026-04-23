# import pandas as pd

# T2D_CODES = [201826, 201820, 443238]  # expand later

# def label_t2d(data, cutoff="2020-01-01"):
#     cond = data["condition_occurrence"]
#     cond["condition_start_date"] = pd.to_datetime(cond["condition_start_date"])

#     cutoff = pd.to_datetime(cutoff)

#     labels = {}

#     for pid, group in cond.groupby("person_id"):
#         pre = group[group["condition_start_date"] < cutoff]
#         post = group[group["condition_start_date"] >= cutoff]

#         # ONLY label based on post-period → prevents leakage
#         has_t2d_post = post["condition_concept_id"].isin(T2D_CODES).any()

#         labels[pid] = int(has_t2d_post)

#     return labels

import pandas as pd

T2D_CODES = [201826, 201820, 443238]

def label_t2d(data, cutoff="2020-01-01"):
    print("[LABELS] Generating T2D labels...")

    cond = data["condition_occurrence"]
    cond["condition_start_date"] = pd.to_datetime(cond["condition_start_date"])
    cutoff = pd.to_datetime(cutoff)

    labels = {}

    for pid, g in cond.groupby("person_id"):
        post = g[g["condition_start_date"] >= cutoff]
        labels[pid] = int(post["condition_concept_id"].isin(T2D_CODES).any())

    print(f"[LABELS] Generated labels for {len(labels)} patients")
    return labels
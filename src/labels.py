import pandas as pd

def label_aki(data):
    """
    KDIGO AKI:
    - creatinine increase >= 0.3 in 48h OR 1.5x baseline
    """
    meas = data["measurement"]

    creat = meas[meas["measurement_concept_id"] == 3020460]  # serum creatinine

    labels = {}

    for pid, group in creat.groupby("person_id"):
        group = group.sort_values("measurement_date")

        baseline = group["value_as_number"].rolling(window=5).mean()

        aki_flag = False
        for i in range(1, len(group)):
            if group.iloc[i]["value_as_number"] >= 1.5 * baseline.iloc[i-1]:
                aki_flag = True
                break

        labels[pid] = int(aki_flag)

    return labels


def label_t2d(data):
    cond = data["condition_occurrence"]
    t2d_codes = [201826]  # example concept_id

    labels = {}
    for pid, group in cond.groupby("person_id"):
        labels[pid] = int(group["condition_concept_id"].isin(t2d_codes).any())

    return labels
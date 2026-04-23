import pandas as pd
import numpy as np

CREATININE_CONCEPT_ID = 3020460  # OMOP standard


def compute_baseline(group):
    """
    Rolling baseline:
    lowest creatinine in past 7–365 days
    """
    values = group["value_as_number"].values
    dates = pd.to_datetime(group["measurement_date"]).values

    baseline = []

    for i in range(len(group)):
        current_date = dates[i]

        mask = (dates < current_date) & \
               (dates >= current_date - np.timedelta64(365, 'D')) & \
               (dates <= current_date - np.timedelta64(7, 'D'))

        if mask.sum() > 0:
            baseline.append(values[mask].min())
        else:
            baseline.append(np.nan)

    return np.array(baseline)


def detect_aki(group):
    """
    Apply KDIGO rules
    """
    group = group.sort_values("measurement_date").copy()

    group["baseline"] = compute_baseline(group)

    aki_flag = 0
    aki_date = None

    for i in range(len(group)):
        current = group.iloc[i]
        curr_val = current["value_as_number"]
        curr_date = current["measurement_date"]

        # --- Rule 1: 0.3 increase in 48 hours ---
        window = group[
            (group["measurement_date"] < curr_date) &
            (group["measurement_date"] >= curr_date - pd.Timedelta(days=2))
        ]

        if not window.empty:
            if curr_val - window["value_as_number"].min() >= 0.3:
                aki_flag = 1
                aki_date = curr_date
                break

        # --- Rule 2: 1.5x baseline ---
        baseline = current["baseline"]
        if not np.isnan(baseline) and curr_val >= 1.5 * baseline:
            aki_flag = 1
            aki_date = curr_date
            break

    return aki_flag, aki_date


def label_aki_kdigo(data, cohort, cutoff_date=None):
    meas = data["measurement"]

    # filter creatinine
    creat = meas[meas["measurement_concept_id"] == CREATININE_CONCEPT_ID].copy()
    creat["measurement_date"] = pd.to_datetime(creat["measurement_date"])

    if cutoff_date:
        cutoff_date = pd.to_datetime(cutoff_date)
        creat = creat[creat["measurement_date"] < cutoff_date]

    labels = {}
    event_dates = {}

    for pid in cohort["person_id"]:
        group = creat[creat["person_id"] == pid]

        if len(group) < 2:
            labels[pid] = 0
            event_dates[pid] = None
            continue

        aki_flag, aki_date = detect_aki(group)

        labels[pid] = aki_flag
        event_dates[pid] = aki_date

    return labels, event_dates
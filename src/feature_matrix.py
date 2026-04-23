import pandas as pd

def build_feature_matrix(data):
    cond = data["condition_occurrence"]

    features = cond.groupby(["person_id", "condition_concept_id"]).size().unstack(fill_value=0)

    return features
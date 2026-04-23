import pandas as pd
import numpy as np

class MeasurementNormalizer:
    def __init__(self):
        self.stats = {}

    def fit(self, measurement_df):
        grouped = measurement_df.groupby("measurement_concept_id")["value_as_number"]

        for concept, values in grouped:
            values = values.dropna()
            self.stats[concept] = {
                "mean": values.mean(),
                "std": values.std() + 1e-6
            }

    def transform(self, concept_id, value):
        if concept_id not in self.stats:
            return value  # fallback

        mean = self.stats[concept_id]["mean"]
        std = self.stats[concept_id]["std"]

        return (value - mean) / std
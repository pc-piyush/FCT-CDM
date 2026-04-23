# import numpy as np
# import pandas as pd
# from collections import defaultdict

# class HealthTensorBuilder:
#     def __init__(self, config):
#         self.config = config
#         self.vocab = defaultdict(dict)

#     def _get_time_index(self, birth_date, event_date):
#         raw_days = (event_date - birth_date).days

#         bin_size = self.config["tensor"]["time_bin_size"]
#         time_bin = raw_days // bin_size

#         return max(time_bin, 0)
    
#     import numpy as np

#     def time_bin(self, days, bin_size="week"):
#         if bin_size == "day":
#             return days

#         if bin_size == "week":
#             return days // 7

#         if bin_size == "month":
#             return days // 30

#         if bin_size == "quarter":
#             return days // 90

#         return days

#     def _encode(self, domain, concept_id):
#         if pd.isna(concept_id):
#             return None
#         if concept_id not in self.vocab[domain]:
#             self.vocab[domain][concept_id] = len(self.vocab[domain])
#         return self.vocab[domain][concept_id]

#     def build_tensor(self, cohort, data):
#         patient_tensors = {}

#         # Pre-group all tables (IMPORTANT for speed)
#         grouped = {
#             "condition": data["condition_occurrence"].groupby("person_id"),
#             "drug": data["drug_exposure"].groupby("person_id"),
#             "measurement": data["measurement"].groupby("person_id"),
#             "procedure": data["procedure_occurrence"].groupby("person_id"),
#             "visit": data["visit_occurrence"].groupby("person_id"),
#         }

#         for _, person in cohort.iterrows():
#             pid = person["person_id"]
#             birth = pd.to_datetime(person["birth_datetime"])

#             tensor = defaultdict(list)

#             # --------------------
#             # CONDITION
#             # --------------------
#             if pid in grouped["condition"].groups:
#                 df = grouped["condition"].get_group(pid)
#                 for _, row in df.iterrows():
#                     t = self._get_time_index(birth, pd.to_datetime(row["condition_start_date"]))
#                     idx = self._encode("condition", row["condition_concept_id"])
#                     if idx is not None and t >= 0:
#                         tensor["condition"].append((t, idx))

#             # --------------------
#             # DRUG
#             # --------------------
#             if pid in grouped["drug"].groups:
#                 df = grouped["drug"].get_group(pid)
#                 for _, row in df.iterrows():
#                     t = self._get_time_index(birth, pd.to_datetime(row["drug_exposure_start_date"]))
#                     idx = self._encode("drug", row["drug_concept_id"])
#                     if idx is not None and t >= 0:
#                         tensor["drug"].append((t, idx))

#             # --------------------
#             # MEASUREMENT
#             # --------------------
#             if pid in grouped["measurement"].groups:
#                 df = grouped["measurement"].get_group(pid)
#                 for _, row in df.iterrows():
#                     t = self._get_time_index(birth, pd.to_datetime(row["measurement_date"]))
#                     concept_idx = self._encode("measurement", row["measurement_concept_id"])

#                     # include value (important!)
#                     value = row.get("value_as_number", None)

#                     if concept_idx is not None and t >= 0:
#                         tensor["measurement"].append((t, concept_idx, value))

#             # --------------------
#             # PROCEDURE
#             # --------------------
#             if pid in grouped["procedure"].groups:
#                 df = grouped["procedure"].get_group(pid)
#                 for _, row in df.iterrows():
#                     t = self._get_time_index(birth, pd.to_datetime(row["procedure_date"]))
#                     idx = self._encode("procedure", row["procedure_concept_id"])
#                     if idx is not None and t >= 0:
#                         tensor["procedure"].append((t, idx))

#             # --------------------
#             # VISIT
#             # --------------------
#             if pid in grouped["visit"].groups:
#                 df = grouped["visit"].get_group(pid)
#                 for _, row in df.iterrows():
#                     t = self._get_time_index(birth, pd.to_datetime(row["visit_start_date"]))
#                     idx = self._encode("visit", row["visit_concept_id"])
#                     if idx is not None and t >= 0:
#                         tensor["visit"].append((t, idx))

#             patient_tensors[pid] = tensor

#         return patient_tensors

import pandas as pd
from collections import defaultdict

class HealthTensorBuilder:
    def __init__(self, config):
        self.config = config
        self.vocab = defaultdict(dict)

        print("[TENSOR] Initialized builder")

    def _time_bin(self, birth_date, event_date):

        birth_date = pd.to_datetime(birth_date)
        event_date = pd.to_datetime(event_date)

        days = (event_date - birth_date).days

        bin_size = self.config["tensor"]["time_bin_size"]
        return max(days // bin_size, 0)

    def _encode(self, domain, concept_id):
        if pd.isna(concept_id):
            return None
        if concept_id not in self.vocab[domain]:
            self.vocab[domain][concept_id] = len(self.vocab[domain])
        return self.vocab[domain][concept_id]

    def build_tensor(self, cohort, data):
        print("[TENSOR] Building patient tensors...")

        grouped = {
            "condition": data["condition_occurrence"].groupby("person_id"),
            "drug": data["drug_exposure"].groupby("person_id"),
            "measurement": data["measurement"].groupby("person_id"),
            "procedure": data["procedure_occurrence"].groupby("person_id"),
            "visit": data["visit_occurrence"].groupby("person_id"),
        }

        tensors = {}

        for i, row in cohort.iterrows():
            pid = row["person_id"]
            birth = row["birth_datetime"]

            if i % 100 == 0:
                print(f"[TENSOR] Processing patient {i}/{len(cohort)}")

            tensor = defaultdict(list)

            # ---------------- CONDITION
            if pid in grouped["condition"].groups:
                df = grouped["condition"].get_group(pid)
                for _, r in df.iterrows():
                    t = self._time_bin(birth, pd.to_datetime(r["condition_start_date"]))
                    idx = self._encode("condition", r["condition_concept_id"])
                    tensor["condition"].append((t, idx))

            # ---------------- DRUG
            if pid in grouped["drug"].groups:
                df = grouped["drug"].get_group(pid)
                for _, r in df.iterrows():
                    t = self._time_bin(birth, pd.to_datetime(r["drug_exposure_start_date"]))
                    idx = self._encode("drug", r["drug_concept_id"])
                    tensor["drug"].append((t, idx))

            # ---------------- MEASUREMENT
            if pid in grouped["measurement"].groups:
                df = grouped["measurement"].get_group(pid)
                for _, r in df.iterrows():
                    t = self._time_bin(birth, pd.to_datetime(r["measurement_date"]))
                    idx = self._encode("measurement", r["measurement_concept_id"])
                    val = r.get("value_as_number", None)
                    tensor["measurement"].append((t, idx, val))

            # ---------------- PROCEDURE
            if pid in grouped["procedure"].groups:
                df = grouped["procedure"].get_group(pid)
                for _, r in df.iterrows():
                    t = self._time_bin(birth, pd.to_datetime(r["procedure_date"]))
                    idx = self._encode("procedure", r["procedure_concept_id"])
                    tensor["procedure"].append((t, idx))

            # ---------------- VISIT
            if pid in grouped["visit"].groups:
                df = grouped["visit"].get_group(pid)
                for _, r in df.iterrows():
                    t = self._time_bin(birth, pd.to_datetime(r["visit_start_date"]))
                    idx = self._encode("visit", r["visit_concept_id"])
                    tensor["visit"].append((t, idx))

            tensors[pid] = tensor

        print(f"[TENSOR] Completed {len(tensors)} patient tensors")
        return tensors
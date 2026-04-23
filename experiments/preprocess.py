import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# import yaml
# import pickle
# from src.data_loader import OMOPLoader
# from src.tensor_builder import HealthTensorBuilder
# from src.labels_t2d import label_t2d

# config = yaml.safe_load(open("config/config.yaml"))

# print("[PIPELINE] Starting preprocessing...")

# loader = OMOPLoader(config)

# person = loader.get_cohort(config["cohort"]["n_patients"])

# data = {
#     "condition_occurrence": loader.load_table_for_cohort("condition_occurrence", person["person_id"]),
#     "drug_exposure": loader.load_table_for_cohort("drug_exposure", person["person_id"]),
#     "measurement": loader.load_table_for_cohort("measurement", person["person_id"]),
#     "procedure_occurrence": loader.load_table_for_cohort("procedure_occurrence", person["person_id"]),
#     "visit_occurrence": loader.load_table_for_cohort("visit_occurrence", person["person_id"]),
# }

# builder = HealthTensorBuilder(config)
# tensors = builder.build_tensor(person, data)

# labels = label_t2d(data)

# print("[PIPELINE] Saving outputs...")

# pickle.dump(tensors, open("data/processed/tensors.pkl", "wb"))
# pickle.dump(labels, open("data/processed/labels.pkl", "wb"))

# print("[PIPELINE] Done ✔")

import yaml
import pickle
from src.data_loader import OMOPLoader
from src.tensor_builder import HealthTensorBuilder
from src.labels_t2d import label_t2d
from src.split import split_cohort

print("[PIPELINE] Starting preprocessing...")

config = yaml.safe_load(open("config/config.yaml"))

loader = OMOPLoader(config)

# -------------------------
# 1. LOAD COHORT
# -------------------------
person = loader.get_cohort(config["cohort"]["n_patients"])

# -------------------------
# 2. SPLIT COHORT (NEW)
# -------------------------
train_cohort, val_cohort, test_cohort = split_cohort(person, config)

# -------------------------
# 3. LOAD DATA PER COHORT (NO LEAKAGE)
# -------------------------
def load_data(cohort):
    ids = cohort["person_id"].tolist()

    return {
        "condition_occurrence": loader.load_table_for_cohort("condition_occurrence", ids),
        "drug_exposure": loader.load_table_for_cohort("drug_exposure", ids),
        "measurement": loader.load_table_for_cohort("measurement", ids),
        "procedure_occurrence": loader.load_table_for_cohort("procedure_occurrence", ids),
        "visit_occurrence": loader.load_table_for_cohort("visit_occurrence", ids),
    }

print("[PIPELINE] Loading train data...")
train_data = load_data(train_cohort)

print("[PIPELINE] Loading val data...")
val_data = load_data(val_cohort)

print("[PIPELINE] Loading test data...")
test_data = load_data(test_cohort)

# -------------------------
# 4. BUILD TENSORS
# -------------------------
builder = HealthTensorBuilder(config)

print("[PIPELINE] Building train tensors...")
train_tensors = builder.build_tensor(train_cohort, train_data)

print("[PIPELINE] Building val tensors...")
val_tensors = builder.build_tensor(val_cohort, val_data)

print("[PIPELINE] Building test tensors...")
test_tensors = builder.build_tensor(test_cohort, test_data)

# -------------------------
# 5. LABELS
# -------------------------
print("[PIPELINE] Creating labels...")
train_labels = label_t2d(train_data)
val_labels = label_t2d(val_data)
test_labels = label_t2d(test_data)

# -------------------------
# 6. SAVE OUTPUTS
# -------------------------
print("[PIPELINE] Saving outputs...")

pickle.dump(train_tensors, open("data/processed/train_tensors.pkl", "wb"))
pickle.dump(val_tensors, open("data/processed/val_tensors.pkl", "wb"))
pickle.dump(test_tensors, open("data/processed/test_tensors.pkl", "wb"))

pickle.dump(train_labels, open("data/processed/train_labels.pkl", "wb"))
pickle.dump(val_labels, open("data/processed/val_labels.pkl", "wb"))
pickle.dump(test_labels, open("data/processed/test_labels.pkl", "wb"))

print("[PIPELINE] Done ✔")
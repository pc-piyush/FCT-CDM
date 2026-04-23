# import duckdb
# import pickle

# print("[COHORT] Loading raw OMOP...")

# con = duckdb.connect("data/duckdb/omop.db")

# T2D_CONCEPT = ('201826', '40475049', '44833365')  # replace with full concept set

# # -------------------------
# # CASES (T2D patients)
# # -------------------------
# cases_query = f"""
# SELECT DISTINCT person_id
# FROM condition_occurrence
# --WHERE condition_concept_id in {T2D_CONCEPT}
# WHERE (condition_source_value like 'E11.%' OR condition_source_value like '250.%')
# and YEAR(CAST(condition_start_date AS DATE)) < 2020
# """

# cases = set(con.execute(cases_query).df()["person_id"])

# # -------------------------
# # CONTROLS (no T2D ever)
# # -------------------------
# controls_query = f"""
# SELECT DISTINCT person_id
# FROM condition_occurrence
# WHERE person_id NOT IN (
#     SELECT DISTINCT person_id
#     FROM condition_occurrence
#     --WHERE condition_concept_id in {T2D_CONCEPT}
#     WHERE (condition_source_value like 'E11.%' OR condition_source_value = '250.%')
# )
# """

# controls = set(con.execute(controls_query).df()["person_id"])

# print(f"[COHORT] Cases: {len(cases)}")
# print(f"[COHORT] Controls: {len(controls)}")

# # -------------------------
# # BALANCE 50:50
# # -------------------------
# n = min(len(cases), len(controls))

# cases_sample = set(list(cases)[:n])
# controls_sample = set(list(controls)[:n])

# cohort = list(cases_sample | controls_sample)

# labels = {pid: 1 for pid in cases_sample}
# labels.update({pid: 0 for pid in controls_sample})

# pickle.dump(cohort, open("data/processed/cohort_ids.pkl", "wb"))
# pickle.dump(labels, open("data/processed/labels.pkl", "wb"))

# print("[COHORT] Done ✔")

import duckdb
import pandas as pd
import pickle
import random
import numpy as np
from datetime import date

print("[COHORT] Loading raw OMOP...")

con = duckdb.connect("data/duckdb/omop.db")

T2D_CONCEPT = 201826

# -----------------------------
# LOAD RAW DATA (NO DATE FILTER IN SQL)
# -----------------------------
df = con.execute("""
SELECT person_id, condition_concept_id, condition_start_date
FROM condition_occurrence where (condition_source_value like 'E11.%' OR condition_source_value like '250.%')
""").df()

print(f"[COHORT] Loaded rows: {len(df)}")
print(df.dtypes)
# -----------------------------
# FIX DATE IN PYTHON
# -----------------------------
df["date"] = pd.to_datetime(df["condition_start_date"], format='%Y-%m-%d', errors="coerce")
start = pd.to_datetime("2011-01-01",format='%Y-%m-%d')
end = pd.to_datetime("2019-12-31",format='%Y-%m-%d')
diff = end - start
print(f"[COHORT] Date range in raw data: {df['date'].min()} to {df['date'].max()}")
print(f"[COHORT] Date range: {start} to {end}")
print(f"[COHORT] Date range difference: {diff}")

# remove bad dates
df = df.dropna(subset=["date"])
print(df.head())

# -----------------------------
# APPLY TIME FILTER (2011–2019)
# -----------------------------
# df = df[(df["date"] >= np.datetime64('2011-01-01', 'us')) & (df["date"] <= np.datetime64('2019-12-31', 'us'))]
# df = df[(df["date"] >= date(2011, 1, 1)) & (df["date"] <= date(2019, 12, 31))]
df = df[(df["date"] >= start) & (df["date"] <= end)]

print(f"[COHORT] After date filter: {len(df)}")

# -----------------------------
# BUILD CASES / CONTROLS
# -----------------------------
cases = set(df[df["condition_concept_id"] == T2D_CONCEPT]["person_id"])
all_patients = set(df["person_id"])
controls = all_patients - cases

print(f"[COHORT] Cases: {len(cases)}")
print(f"[COHORT] Controls: {len(controls)}")

# -----------------------------
# BALANCE 50:50
# -----------------------------
n = min(len(cases), len(controls))

cases_sample = set(random.sample(list(cases), n))
controls_sample = set(random.sample(list(controls), n))

cohort_ids = list(cases_sample | controls_sample)

labels = {pid: 1 for pid in cases_sample}
labels.update({pid: 0 for pid in controls_sample})

print(f"[COHORT] Final cohort size: {len(cohort_ids)}")

# -----------------------------
# SAVE
# -----------------------------
pickle.dump(cohort_ids, open("data/processed/cohort_ids.pkl", "wb"))
pickle.dump(labels, open("data/processed/labels.pkl", "wb"))

print("[COHORT] Done ✔")
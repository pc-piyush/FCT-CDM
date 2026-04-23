# import duckdb
# import pickle
# import pandas as pd

# print("[COVID COHORT] Loading OMOP...")

# con = duckdb.connect("data/duckdb/omop.db")

# COVID_CONCEPT = (

#     710706,
# 710708,
# 756023,
# 756031,
# 756039,
# 756044,
# 756061,
# 756081,
# 766502,
# 766503,
# 37310269,
# 37311061,
# 931072,
# 931073,
# 957638,
# 45756093
# )


# #37311061  # example (replace with your actual COVID concept)

# # -------------------------
# # INDEX DATE = FIRST COVID DIAGNOSIS
# # -------------------------
# covid_df = con.execute(f"""
# SELECT person_id, MIN(condition_start_date) AS index_date
# FROM condition_occurrence
# WHERE condition_concept_id = {COVID_CONCEPT}
# GROUP BY person_id
# """).df()

# print("[COHORT] Patients:", len(covid_df))

# # -------------------------
# # SEVERITY LABEL
# # Example: ICU visit after index
# # -------------------------
# icu_df = con.execute("""
# SELECT person_id, visit_start_date
# FROM visit_occurrence
# WHERE visit_concept_id IN (9201)  -- ICU concept example
# """).df()

# icu_df = icu_df.merge(covid_df, on="person_id")

# # label = ICU within 30 days
# icu_df["label"] = (
#     (icu_df["visit_start_date"] > icu_df["index_date"]) &
#     (icu_df["visit_start_date"] <= icu_df["index_date"] + pd.Timedelta(days=30))
# )

# labels = {pid: 0 for pid in covid_df["person_id"]}

# for _, row in icu_df.iterrows():
#     if row["label"]:
#         labels[row["person_id"]] = 1

# index_dates = dict(zip(covid_df["person_id"], covid_df["index_date"]))

# pickle.dump(labels, open("data/processed/labels.pkl", "wb"))
# pickle.dump(index_dates, open("data/processed/index_dates.pkl", "wb"))

# print("[COHORT] Done ✔")

# import duckdb
# import pickle
# import random

# print("[COHORT] Building COVID cohort...")

# con = duckdb.connect("data/duckdb/omop.db")

# COVID_CONCEPT = (

#     710706,
# 710708,
# 756023,
# 756031,
# 756039,
# 756044,
# 756061,
# 756081,
# 766502,
# 766503,
# 37310269,
# 37311061,
# 931072,
# 931073,
# 957638,
# 45756093
# )

# # ----------------------------
# # COVID patients (already given dataset, but still define)
# # ----------------------------
# covid_query = f"""
# SELECT DISTINCT person_id
# FROM condition_occurrence
# WHERE condition_concept_id IN {COVID_CONCEPT}  -- example COVID concept
# """

# covid_ids = list(con.execute(covid_query).df()["person_id"])

# # ----------------------------
# # SAMPLE 1000 PATIENTS
# # ----------------------------
# N = 1000
# covid_ids = covid_ids[:N]

# print(f"[COHORT] Using {len(covid_ids)} patients")

# # ----------------------------
# # LABEL: SEVERE OUTCOME
# # ----------------------------
# # Example concepts (adjust to your OMOP vocab)
# SEVERE_VISITS = [9201]   # ICU visit
# VENTILATION = [3007461]  # ventilation procedure

# labels = {}

# for pid in covid_ids:

#     severe = con.execute(f"""
#         SELECT COUNT(*) as cnt
#         FROM visit_occurrence
#         WHERE person_id = {pid}
#         AND visit_concept_id IN (9201)
#     """).fetchone()[0]

#     vent = con.execute(f"""
#         SELECT COUNT(*) as cnt
#         FROM procedure_occurrence
#         WHERE person_id = {pid}
#         AND procedure_concept_id IN (3007461)
#     """).fetchone()[0]

#     labels[pid] = 1 if (severe > 0 or vent > 0) else 0

# pickle.dump(covid_ids, open("data/processed/cohort_ids.pkl", "wb"))
# pickle.dump(labels, open("data/processed/labels.pkl", "wb"))

# print("[COHORT] Done ✔")

import duckdb, pickle, random
from datetime import timedelta

con = duckdb.connect("data/duckdb/omop.db")

COVID_CONCEPT = (

   710706,
710708,
756023,
756031,
756039,
756044,
756061,
756081,
766502,
766503,
37310269,
37311061,
931072,
931073,
957638,
45756093
)
N_PATIENTS = 100
print("[COHORT] Loading COVID patients...")

# # -------------------------
# # Get COVID patients
# # -------------------------
# df = con.execute(f"""
# SELECT person_id, condition_start_date AS covid_date
# FROM condition_occurrence
# WHERE condition_concept_id = {COVID_CONCEPT}
# """).df()

# df = df.drop_duplicates("person_id").head(N_PATIENTS)

# patients = df["person_id"].tolist()

# # -------------------------
# # Adverse outcome concepts (example placeholders)
# # -------------------------
ADVERSE = [#'U07.1','U07.2',
           'J12.82','J80','J96.00',
    'I51.4',
    'I26.99',
    'I21.9',
    'I80.1',
    'I74.3',
    'N17.9',
    #'G93.3',
    'R65.20',
    #'U09.9',
    'M35.81']  # replace properly

# labels = {}

# for _, row in df.iterrows():
#     pid = row["person_id"]
#     index_date = row["covid_date"]

#     outcome = con.execute(f"""
#     SELECT COUNT(*) as cnt
#     FROM condition_occurrence
#     WHERE person_id = {pid}
#     --AND condition_concept_id IN {tuple(ADVERSE)}
#     AND condition_source_value IN {tuple(ADVERSE)}
#     AND condition_start_date BETWEEN '{index_date}' 
#     AND DATE '{index_date}' + INTERVAL '180 days'
#     """).fetchone()[0]

#     labels[pid] = 1 if outcome > 0 else 0

# pickle.dump(patients, open("data/processed/cohort_ids.pkl", "wb"))
# pickle.dump(labels, open("data/processed/labels.pkl", "wb"))

# print("Done cohort ✔")

# -------------------------
# Get COVID patients
# -------------------------
df = con.execute(f"""
SELECT person_id, MIN(CAST(condition_start_date AS DATE)) as covid_date
FROM condition_occurrence
WHERE CAST(condition_concept_id AS BIGINT) in {COVID_CONCEPT}
GROUP BY person_id
LIMIT {N_PATIENTS}
""").df()

print(f"[COHORT] Selected patients: {len(df)}")

patients = df["person_id"].tolist()

labels = {}

print("\n[COHORT] Generating labels (6-month outcome window)...")

for i, row in df.iterrows():
    pid = row["person_id"]
    index_date = row["covid_date"]

    print(f"[COHORT] Processing patient {i+1}/{len(df)} | PID={pid}")

    # -------------------------
    # ICU (visit)
    # -------------------------
    icu = con.execute(f"""
    SELECT COUNT(*) FROM visit_occurrence
    WHERE person_id = {pid}
    AND CAST(visit_concept_id AS BIGINT) in (9201,9203)
    AND CAST(visit_start_date AS DATE) BETWEEN '{index_date}'
    AND DATE '{index_date}' + INTERVAL '30 days'
    """).fetchone()[0]

    # -------------------------
    # Ventilation (procedure)
    # -------------------------
    vent = con.execute(f"""
    SELECT COUNT(*) FROM procedure_occurrence
    WHERE person_id = {pid}
    AND procedure_concept_id IN (3007461)
    AND CAST(procedure_date AS DATE) BETWEEN '{index_date}'
    AND DATE '{index_date}' + INTERVAL '30 days'
    """).fetchone()[0]

    # -------------------------
    # AKI (condition)
    # -------------------------
    aki = con.execute(f"""
    SELECT COUNT(*) FROM condition_occurrence
    WHERE person_id = {pid}
    AND condition_source_value IN {tuple(ADVERSE)}
    AND CAST(condition_start_date AS DATE) BETWEEN '{index_date}'
    AND DATE '{index_date}' + INTERVAL '30 days'
    """).fetchone()[0]

    # -------------------------
    # Death
    # -------------------------
    death = con.execute(f"""
    SELECT COUNT(*) FROM death
    WHERE person_id = {pid}
    AND CAST(death_date AS DATE) BETWEEN '{index_date}'
    AND DATE '{index_date}' + INTERVAL '30 days'
    """).fetchone()[0]

    # -------------------------
    # Final label
    # -------------------------
    labels[pid] = 1 if (aki + death) > 0 else 0
    print(f"   -> ICU={icu}, VENT={vent}, AE={aki}, death={death}, LABEL={labels[pid]}")

print("\n[COHORT] Class distribution:")
print("Positive:", sum(labels.values()))
print("Negative:", len(labels) - sum(labels.values()))



pickle.dump(df, open("data/processed/cohort_index.pkl","wb"))
pickle.dump(labels, open("data/processed/labels.pkl","wb"))

print("Cohort built ✔")

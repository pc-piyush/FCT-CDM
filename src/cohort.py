# import pandas as pd

# def build_cohort(data, config):
#     person = data["person"]

#     # sample patients
#     cohort = person.sample(n=config["cohort"]["n_patients"], random_state=42)

#     cutoff = pd.to_datetime(config["cohort"]["cutoff_date"])

#     def filter_table(df, date_col):
#         df[date_col] = pd.to_datetime(df[date_col])
#         return df[df[date_col] < cutoff]

#     data["condition_occurrence"] = filter_table(data["condition_occurrence"], "condition_start_date")
#     data["drug_exposure"] = filter_table(data["drug_exposure"], "drug_exposure_start_date")
#     data["measurement"] = filter_table(data["measurement"], "measurement_date")
#     data["procedure_occurrence"] = filter_table(data["procedure_occurrence"], "procedure_date")
#     data["visit_occurrence"] = filter_table(data["visit_occurrence"], "visit_start_date")

#     return cohort, data

import pandas as pd

def build_cohort(person_df):
    print("[COHORT] Building cohort...")

    person_df["birth_datetime"] = pd.to_datetime(person_df["birth_datetime"])

    print(f"[COHORT] Cohort size: {len(person_df)}")
    return person_df
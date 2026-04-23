# import duckdb
# import pandas as pd

# class OMOPLoader:
#     def __init__(self, config):
#         self.config = config
#         self.conn = duckdb.connect(config["data"]["duckdb_path"])

#     def get_cohort_persons(self, n):
#         query = f"""
#         SELECT *
#         FROM person
#         USING SAMPLE {n}
#         """
#         return self.conn.execute(query).df()

#     def load_table_for_cohort(self, table, person_ids, date_col=None, cutoff=None):
#         ids_str = ",".join(map(str, person_ids))

#         query = f"""
#         SELECT *
#         FROM {table}
#         WHERE person_id IN ({ids_str})
#         """

#         if date_col and cutoff:
#             query += f" AND {date_col} < '{cutoff}'"

#         return self.conn.execute(query).df()

#     def load_all_for_cohort(self, cohort, cutoff):
#         person_ids = cohort["person_id"].tolist()

#         data = {}

#         data["condition_occurrence"] = self.load_table_for_cohort(
#             "condition_occurrence", person_ids, "condition_start_date", cutoff
#         )

#         data["drug_exposure"] = self.load_table_for_cohort(
#             "drug_exposure", person_ids, "drug_exposure_start_date", cutoff
#         )

#         data["measurement"] = self.load_table_for_cohort(
#             "measurement", person_ids, "measurement_date", cutoff
#         )

#         data["procedure_occurrence"] = self.load_table_for_cohort(
#             "procedure_occurrence", person_ids, "procedure_date", cutoff
#         )

#         data["visit_occurrence"] = self.load_table_for_cohort(
#             "visit_occurrence", person_ids, "visit_start_date", cutoff
#         )

#         return data

import duckdb
import pandas as pd

class OMOPLoader:
    def __init__(self, config):
        self.config = config
        self.conn = duckdb.connect(config["data"]["duckdb_path"])
        print("[DATA LOADER] Connected to DuckDB")

    def get_cohort(self, n):
        print(f"[DATA LOADER] Sampling {n} patients...")
        df = self.conn.execute(f"""
            SELECT * FROM person USING SAMPLE {n}
        """).df()

        print(f"[DATA LOADER] Cohort loaded: {len(df)} patients")
        return df

    def load_table_for_cohort(self, table, ids, date_col=None, cutoff=None):
        print(f"[DATA LOADER] Loading {table} for cohort...")

        id_str = ",".join(map(str, ids))

        query = f"""
        SELECT *
        FROM {table}
        WHERE person_id IN ({id_str})
        """

        df = self.conn.execute(query).df()

        if cutoff and date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df[df[date_col] < cutoff]

        print(f"[DATA LOADER] {table}: {len(df)} rows after filtering")
        return df
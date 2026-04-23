import duckdb
import pickle
from datetime import timedelta

print("\n[TENSOR] Starting tensor construction...")

con = duckdb.connect("data/duckdb/omop.db")

df = pickle.load(open("data/processed/cohort_index.pkl","rb"))
patients = df["person_id"].tolist()

tensors = {}

domains = {
    "condition": "condition_occurrence",
    "drug": "drug_exposure",
    "measurement": "measurement",
    "procedure": "procedure_occurrence",
    "visit": "visit_occurrence"
}

print(f"[TENSOR] Processing {len(patients)} patients...")

for i, row in df.iterrows():


    pid = row["person_id"]
    index_date = row["covid_date"]

    print(f"\n[TENSOR] Patient {i+1}/{len(df)} | PID={pid}")

    tensors[pid] = {d: [] for d in domains}

    for domain, table in domains.items():

        print(f"  [DOMAIN] {domain}")

        if domain == "drug":
            date_col = f"{table}_start_date"
        elif domain == "measurement":
            date_col = f"{table}_date"
        elif domain == "visit":
            date_col = f"{domain}_start_date"
        elif domain == "procedure":
            date_col = f"{domain}_date"
        else:
            date_col = f"{domain}_start_date"

        q = f"""
        SELECT person_id, {domain}_concept_id as cid, {date_col} as date
        FROM {table}
        WHERE CAST(person_id AS BIGINT) = '{pid}'
        AND CAST({date_col} AS DATE) < CAST('{index_date}' AS DATE) 
        -- AND CAST('{index_date}' AS DATE) + INTERVAL 7 DAY
         --                         --AND DATE '{index_date}' + INTERVAL '180 days'
        """

        try:
            df2 = con.execute(q).df()
        except:
            continue

        print(f"     rows={len(df2)}")

        for _, r in df2.iterrows():
            tensors[pid][domain].append((r["date"], r["cid"]))

print("\n[TENSOR] Saving tensors...")
pickle.dump(tensors, open("data/processed/tensors.pkl","wb"))
# print(df["person_id"].head())
print("[TENSOR] Done ✔")

# pid = patients[0]
# index_date = df[df["person_id"] == pid]["covid_date"].iloc[0]

# print(pid, index_date)

# q = f"""
# SELECT COUNT(*)
# FROM condition_occurrence
# WHERE person_id = {pid}
# """

# print(con.execute(q).fetchone())
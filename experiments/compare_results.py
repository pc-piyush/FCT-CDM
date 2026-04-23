import json

gru = json.load(open("data/processed/metrics_gru.json"))
baseline = json.load(open("data/processed/metrics_baseline.json"))

print("\n====================")
print("MODEL COMPARISON")
print("====================\n")

print("GRU MODEL:")
print(gru)

print("\nBASELINE MODEL:")
print(baseline)

print("\nIMPROVEMENT (TEST AUC):",
      gru["test"]["auc"] - baseline["test"]["auc"])
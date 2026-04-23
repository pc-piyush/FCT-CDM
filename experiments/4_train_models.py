import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

print("\n[TRAIN] Loading data...")

X,y = pickle.load(open("data/processed/features.pkl","rb"))

models = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=200),
    "mlp": MLPClassifier(hidden_layer_sizes=(64,32)),
    "xgb": xgb.XGBClassifier(eval_metric="logloss"),
    "lgb": lgb.LGBMClassifier()
}

results = {}

print("\n[TRAIN] Training models...")

for name, model in models.items():

    print(f"\n[MODEL] {name}")

    model.fit(X,y)
    preds = model.predict_proba(X)[:,1]

    auc = roc_auc_score(y, preds)

    results[name] = auc

    print(f"   AUC = {auc:.4f}")

print("\n[TRAIN] Final Results:", results)

pickle.dump(results, open("data/processed/model_results.pkl","wb"))

print("[TRAIN] Done ✔")
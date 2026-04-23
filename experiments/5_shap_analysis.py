import pickle
import shap
from sklearn.ensemble import RandomForestClassifier

print("\n[SHAP] Loading data...")

X,y = pickle.load(open("data/processed/features.pkl","rb"))

model = RandomForestClassifier(n_estimators=100)
model.fit(X,y)

print("[SHAP] Computing explanations...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

pickle.dump(shap_values, open("data/processed/shap_values.pkl","wb"))

print("[SHAP] Done ✔")
from sklearn.metrics import roc_auc_score

def evaluate(model, X_test, y_test):
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    return {"auc": auc}
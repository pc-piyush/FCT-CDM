from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train, model_type="rf"):
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    return model
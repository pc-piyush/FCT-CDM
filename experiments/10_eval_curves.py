import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_curves(y_true, feature_probs, gru_probs):

    print("[EVAL] Generating ROC + PR curves...")

    # ROC
    fpr1, tpr1, _ = roc_curve(y_true, feature_probs)
    fpr2, tpr2, _ = roc_curve(y_true, gru_probs)

    plt.figure()
    plt.plot(fpr1, tpr1, label="Feature Model")
    plt.plot(fpr2, tpr2, label="GRU Model")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # PR
    p1, r1, _ = precision_recall_curve(y_true, feature_probs)
    p2, r2, _ = precision_recall_curve(y_true, gru_probs)

    plt.figure()
    plt.plot(r1, p1, label="Feature Model")
    plt.plot(r2, p2, label="GRU Model")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
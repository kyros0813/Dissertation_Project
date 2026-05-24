"""
04_classification.py — Model Training & Evaluation
Εκπαιδεύει LR, RF και XGBoost. Αξιολογεί στο test set. Αποθηκεύει αποτελέσματα.
"""

import os
import pickle
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

OUT_DIR = "outputs"
ART_PATH = os.path.join(OUT_DIR, "artifacts.pkl")
RES_PATH = os.path.join(OUT_DIR, "results.pkl")


with open(ART_PATH, "rb") as f:
    art = pickle.load(f)

X_train_sc = art["X_train_sc"]

X_test_sc = art["X_test_sc"]
y_train = art["y_train"]
y_test = art["y_test"]

imb_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance ratio: {imb_ratio:.1f}:1")

models = {
    "LR": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=42,
    ),
    "RF": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    ),
    "XGB": XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        scale_pos_weight=imb_ratio,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    ),
}


print("\nModel comparison")
print(f"{'Model':<6} {'Test AUC':>10} {'PR-AUC':>8} {'F1':>7}")
print("-" * 50)

trained_models = {}
y_probs = {}
classification_results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    probs = model.predict_proba(X_test_sc)[:, 1]
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_test, preds)
    test_auc = round(float(roc_auc_score(y_test, probs)), 4)
    pr_auc   = round(float(average_precision_score(y_test, probs)), 4)
    f1       = round(float(f1_score(y_test, preds, pos_label=1)), 4)
    precision = round(cm[1, 1] / (cm[0, 1] + cm[1, 1]), 4) if (cm[0, 1] + cm[1, 1]) else 0.0
    recall    = round(cm[1, 1] / (cm[1, 0] + cm[1, 1]), 4) if (cm[1, 0] + cm[1, 1]) else 0.0

    trained_models[name] = model
    y_probs[name] = probs
    classification_results[name] = {
        "test_auc": test_auc,
        "prauc":    pr_auc,
        "f1":       f1,
        "precision_at_05": precision,
        "recall_at_05":    recall,
        "confusion_matrix": cm.tolist(),
    }

    print(f"{name:<6} {test_auc:>10.4f} {pr_auc:>8.4f} {f1:>7.4f}")


# XGBoost χρησιμοποιείται για threshold optimization και SHAP (λειτουργεί καλύτερα με SHAP TreeExplainer)
decision_model = "XGB"

art["models"]         = {decision_model: trained_models[decision_model]}
art["y_probs"]        = y_probs
art["decision_model"] = decision_model

with open(ART_PATH, "wb") as f:
    pickle.dump(art, f)

try:
    with open(RES_PATH, "rb") as f:
        all_results = pickle.load(f)
except FileNotFoundError:
    all_results = {}

all_results["classification"] = classification_results
all_results["decision_model"] = decision_model

with open(RES_PATH, "wb") as f:
    pickle.dump(all_results, f)

print(f"\n[OK] Classification results saved. Decision model: {decision_model}")

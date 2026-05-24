"""Compare decision thresholds using a simple profit function."""

import os
import pickle

import numpy as np
from sklearn.metrics import f1_score

OUT_DIR = "outputs"
ART_PATH = os.path.join(OUT_DIR, "artifacts.pkl")
RES_PATH = os.path.join(OUT_DIR, "results.pkl")


with open(ART_PATH, "rb") as f:
    art = pickle.load(f)

y_test = art["y_test"]
test_clusters = art["test_clusters"]
decision_model = art.get("decision_model", "XGB")
model_prob = art["y_probs"][decision_model]
C_call = art["C_call"]
V_deposit = art["V_deposit"]

thresholds = np.arange(0.01, 1.00, 0.01)
total_positives = int(y_test.sum())


# -- Strategy 1: default threshold 0.50 --
t = 0.50
pred = (model_prob >= t).astype(int)
tp = int(((pred == 1) & (y_test == 1)).sum())
fp = int(((pred == 1) & (y_test == 0)).sum())
calls = tp + fp
profit = tp * V_deposit - calls * C_call
s_default = {
    "threshold": round(float(t), 2),
    "profit": profit,
    "calls": calls,
    "recall": round(tp / total_positives, 3) if total_positives else 0.0,
    "precision": round(tp / calls, 3) if calls else 0.0,
    "roi": round(profit / (calls * C_call) * 100, 0) if calls else 0.0,
}


# -- Strategy 2: one global profit-maximizing threshold --
# Δοκιμάζουμε όλα τα thresholds και κρατάμε αυτό με το μεγαλύτερο κέρδος
best_global_t = thresholds[0]
best_global_profit = None
for t in thresholds:
    pred = (model_prob >= t).astype(int)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    profit = tp * V_deposit - (tp + fp) * C_call
    if best_global_profit is None or profit > best_global_profit:
        best_global_profit = profit
        best_global_t = t

pred = (model_prob >= best_global_t).astype(int)
tp = int(((pred == 1) & (y_test == 1)).sum())
fp = int(((pred == 1) & (y_test == 0)).sum())
calls = tp + fp
profit = tp * V_deposit - calls * C_call
s_global = {
    "threshold": round(float(best_global_t), 2),
    "profit": profit,
    "calls": calls,
    "recall": round(tp / total_positives, 3) if total_positives else 0.0,
    "precision": round(tp / calls, 3) if calls else 0.0,
    "roi": round(profit / (calls * C_call) * 100, 0) if calls else 0.0,
}


# -- Strategy 3: threshold that maximizes F1-score --
best_f1_t = thresholds[0]
best_f1_value = None
for t in thresholds:
    pred = (model_prob >= t).astype(int)
    f1_value = f1_score(y_test, pred, zero_division=0)
    if best_f1_value is None or f1_value > best_f1_value:
        best_f1_value = f1_value
        best_f1_t = t

pred = (model_prob >= best_f1_t).astype(int)
tp = int(((pred == 1) & (y_test == 1)).sum())
fp = int(((pred == 1) & (y_test == 0)).sum())
calls = tp + fp
profit = tp * V_deposit - calls * C_call
s_maxf1 = {
    "threshold": round(float(best_f1_t), 2),
    "profit": profit,
    "calls": calls,
    "recall": round(tp / total_positives, 3) if total_positives else 0.0,
    "precision": round(tp / calls, 3) if calls else 0.0,
    "roi": round(profit / (calls * C_call) * 100, 0) if calls else 0.0,
}


# -- Strategy 4: one profit-maximizing threshold per cluster --
cluster_thresholds = {}
cluster_threshold_results = {}
total_profit = 0
total_calls = 0
total_tp = 0

for c in sorted(np.unique(test_clusters)):
    mask = test_clusters == c
    y_c = y_test[mask]
    prob_c = model_prob[mask]

    # Βρίσκουμε το threshold με το μεγαλύτερο κέρδος για αυτό το cluster
    best_t = thresholds[0]
    best_t_profit = None
    for t in thresholds:
        pred_c = (prob_c >= t).astype(int)
        tp_t = int(((pred_c == 1) & (y_c == 1)).sum())
        fp_t = int(((pred_c == 1) & (y_c == 0)).sum())
        profit_t = tp_t * V_deposit - (tp_t + fp_t) * C_call
        if best_t_profit is None or profit_t > best_t_profit:
            best_t_profit = profit_t
            best_t = t

    pred_c = (prob_c >= best_t).astype(int)
    tp = int(((pred_c == 1) & (y_c == 1)).sum())
    fp = int(((pred_c == 1) & (y_c == 0)).sum())
    calls = tp + fp
    profit = tp * V_deposit - calls * C_call

    cluster_thresholds[int(c)] = round(float(best_t), 2)
    cluster_threshold_results[int(c)] = {
        "threshold": round(float(best_t), 2),
        "n_test": int(mask.sum()),
        "response_rate": round(float(y_c.mean() * 100), 1),
        "tp": tp,
        "fp": fp,
        "calls": calls,
        "profit": profit,
        "recall": round(tp / int(y_c.sum()), 3) if y_c.sum() else 0.0,
        "precision": round(tp / calls, 3) if calls else 0.0,
    }

    total_profit += profit
    total_calls += calls
    total_tp += tp

s_cluster = {
    "profit": total_profit,
    "calls": total_calls,
    "recall": round(total_tp / total_positives, 3),
    "precision": round(total_tp / total_calls, 3) if total_calls else 0.0,
    "roi": round(total_profit / (total_calls * C_call) * 100, 0) if total_calls else 0.0,
    "per_cluster": cluster_threshold_results,
}


print("=" * 78)
print(f"Threshold optimization using {decision_model}")
print(f"{'Strategy':<24} {'Profit EUR':>12} {'ROI%':>8} {'Calls':>8} {'Recall':>8} {'Prec':>8}")
print("-" * 78)
for label, summary in [
    ("Default t=0.50", s_default),
    (f"Global t*={s_global['threshold']:.2f}", s_global),
    (f"Max-F1 t={s_maxf1['threshold']:.2f}", s_maxf1),
    ("Cluster t*", s_cluster),
]:
    print(
        f"{label:<24} {summary['profit']:>12,.0f} {summary['roi']:>8.0f} "
        f"{summary['calls']:>8,} {summary['recall']:>8.3f} {summary['precision']:>8.3f}"
    )

print("\nCluster-specific thresholds:")
print(f"{'C':<3} {'N_test':>7} {'Rate%':>7} {'t*':>6} {'TP':>5} {'FP':>6} {'Calls':>7} {'Profit':>10}")
print("-" * 65)
for c, row in cluster_threshold_results.items():
    print(
        f"C{c:<2} {row['n_test']:>7,} {row['response_rate']:>7.1f} "
        f"{row['threshold']:>6.2f} {row['tp']:>5} {row['fp']:>6} "
        f"{row['calls']:>7,} {row['profit']:>10,.0f}"
    )

art["cluster_thresholds"] = cluster_thresholds

with open(ART_PATH, "wb") as f:
    pickle.dump(art, f)

with open(RES_PATH, "rb") as f:
    all_results = pickle.load(f)

all_results["threshold"] = {
    "model": decision_model,
    "default": s_default,
    "global": s_global,
    "max_f1": s_maxf1,
    "cluster": s_cluster,
    "cluster_thresholds": cluster_thresholds,
}

with open(RES_PATH, "wb") as f:
    pickle.dump(all_results, f)

improvement = (total_profit - s_default["profit"]) / s_default["profit"] * 100
print(f"\n[OK] Threshold results saved. Improvement over default: +{improvement:.1f}%")

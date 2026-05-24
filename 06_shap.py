"""Calculate SHAP feature importance for the final XGBoost model."""

import os
import pickle

import numpy as np
import xgboost as xgb

OUT_DIR = "outputs"
ART_PATH = os.path.join(OUT_DIR, "artifacts.pkl")
RES_PATH = os.path.join(OUT_DIR, "results.pkl")


with open(ART_PATH, "rb") as f:
    art = pickle.load(f)

X_test_sc = art["X_test_sc"]
test_clusters = art["test_clusters"]
y_test = art["y_test"]
feature_names = art["feature_names"]
decision_model = art.get("decision_model", "XGB")
model = art["models"][decision_model]

print(f"Computing SHAP values for {decision_model}...")

np.random.seed(42)
sample_size = min(2000, len(X_test_sc))
idx_sample = np.random.choice(len(X_test_sc), size=sample_size, replace=False)
X_sample = X_test_sc[idx_sample]

if decision_model != "XGB":
    raise ValueError("This simplified SHAP script expects XGB as the decision model.")

# The last SHAP column is the bias term, so it is excluded.
dmatrix = xgb.DMatrix(X_sample, feature_names=feature_names)
shap_values = model.get_booster().predict(dmatrix, pred_contribs=True)[:, :-1]

mean_abs = np.abs(shap_values).mean(axis=0)
order = np.argsort(mean_abs)[::-1]

print("\nTop 15 features by mean |SHAP|:")
print(f"{'#':<4} {'Feature':<40} {'Mean |SHAP|':>12}")
print("-" * 58)

global_shap = []
for rank, i in enumerate(order[:15], start=1):
    value = round(float(mean_abs[i]), 4)
    global_shap.append(
        {"rank": rank, "feature": feature_names[i], "mean_abs_shap": value}
    )
    print(f"{rank:<4} {feature_names[i]:<40} {value:>12.4f}")


tc_sample = test_clusters[idx_sample]
y_sample = y_test[idx_sample]

print("\nTop 5 features per cluster:")
per_cluster_shap = {}
for c in sorted(np.unique(tc_sample)):
    mask = tc_sample == c
    if mask.sum() < 10:
        continue

    cluster_mean_abs = np.abs(shap_values[mask]).mean(axis=0)
    top_features = np.argsort(cluster_mean_abs)[::-1][:5]
    response_rate = round(float(y_sample[mask].mean() * 100), 1)

    cluster_list = []
    for i in top_features:
        cluster_list.append({
            "feature": feature_names[i],
            "mean_abs_shap": round(float(cluster_mean_abs[i]), 4),
        })
    per_cluster_shap[int(c)] = cluster_list

    print(f"  C{c} (n={mask.sum()}, response~{response_rate:.1f}%):")
    for entry in per_cluster_shap[int(c)]:
        print(f"    {entry['feature']:<40}: {entry['mean_abs_shap']:.4f}")


with open(RES_PATH, "rb") as f:
    all_results = pickle.load(f)

all_results["shap"] = {
    "model": decision_model,
    "global_top15": global_shap,
    "per_cluster_top5": per_cluster_shap,
}

with open(RES_PATH, "wb") as f:
    pickle.dump(all_results, f)

print("\n[OK] SHAP results saved.")

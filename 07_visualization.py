"""
07_visualization.py — Data Visualization
Φορτώνει results.pkl + artifacts.pkl, αποθηκεύει plots στο outputs/figures/
"""

import numpy as np
import os, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

#Load
with open(os.path.join(OUT_DIR, "results.pkl"), "rb") as f:
    res = pickle.load(f)

with open(os.path.join(OUT_DIR, "artifacts.pkl"), "rb") as f:
    art = pickle.load(f)

eda    = res["eda"]
clust  = res["clustering"]
clf    = res["classification"]
thr    = res["threshold"]
shap_r = res["shap"]
shap_model = shap_r.get("model", res.get("decision_model", "XGB"))

y_test  = art["y_test"]
y_probs = art["y_probs"]

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

C4 = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]   #one colour per cluster

#Εδώ χρησιμοποιώ τα ήδη αποθηκευμένα αποτελέσματα από results/artifacts, χωρίς νέα εκπαίδευση μοντέλου.
#EDA overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Exploratory Data Analysis", fontsize=16, fontweight="bold")

#class distribution
ax = axes[0, 0]
labels = ["NO", "YES"]
sizes  = [eda["n_no"], eda["n_yes"]]
bars   = ax.bar(labels, sizes, color=["#90CAF9", "#1565C0"], width=0.4)
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
ax.set_ylim(0, max(sizes) * 1.18)
for bar, s in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
            f"{s:,}\n({s / eda['n_total'] * 100:.1f}%)",
            ha="center", fontsize=10)

#correlation with target
ax = axes[0, 1]
corr = eda["correlations"]
#Ταξινόμηση των συσχετίσεων κατά τιμή (αύξουσα), χωρίς lambda
pairs = []
for name in corr:
    pairs.append([corr[name], name])
pairs.sort()
names_c = []
vals_c = []
for value, name in pairs:
    vals_c.append(value)
    names_c.append(name)
colors_c = ["#EF9A9A" if v < 0 else "#90CAF9" for v in vals_c]
ax.barh(names_c, vals_c, color=colors_c)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Correlation with Target")
ax.set_xlabel("Pearson r")
for i, v in enumerate(vals_c):
    ax.text(v + (0.003 if v >= 0 else -0.003), i,
            f"{v:+.3f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8)

#response rate by month
ax = axes[1, 0]
month_order = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
ms = eda["month_stats"]
months = [m for m in month_order if m in ms]
rates_m = [ms[m]["rate"] for m in months]
x = np.arange(len(months))
bars = ax.bar(x, rates_m, color="#42A5F5")
ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in months], rotation=45, ha="right")
ax.set_title("Response Rate by Month (%)")
ax.set_ylabel("Response Rate (%)")
for bar, r in zip(bars, rates_m):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{r:.1f}%", ha="center", fontsize=8)

#response rate by poutcome
ax = axes[1, 1]
po = eda["poutcome_stats"]
po_names = list(po.keys())
po_rates = [po[p]["rate"] for p in po_names]
po_ns    = [po[p]["n"]    for p in po_names]
colors_po = ["#1565C0", "#42A5F5", "#90CAF9"][: len(po_names)]
bars = ax.bar(po_names, po_rates, color=colors_po, width=0.4)
ax.set_title("Response Rate by Previous Outcome (%)")
ax.set_ylabel("Response Rate (%)")
ax.set_ylim(0, max(po_rates) * 1.2)
for bar, r, n in zip(bars, po_rates, po_ns):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{r:.1f}%\n(n={n:,})", ha="center", fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_eda.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 01_eda.png")

#Clustering metrics (K=2..8)
metrics_k = clust["metrics"]
ks  = sorted(metrics_k.keys())
sil = [metrics_k[k]["silhouette"]        for k in ks]
db  = [metrics_k[k]["davies_bouldin"]    for k in ks]
ch  = [metrics_k[k]["calinski_harabasz"] for k in ks]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("K-Means Clustering — Metric Evaluation (K=2..8)",
             fontsize=14, fontweight="bold")

for ax, vals, title, ylabel, best_fn in [
    (axes[0], sil, "Silhouette Score",        "Score (higher = better)", max),
    (axes[1], db,  "Davies-Bouldin Index",    "Score (lower = better)",  min),
    (axes[2], ch,  "Calinski-Harabasz Index", "Score (higher = better)", max),
]:
    best_k = ks[vals.index(best_fn(vals))]
    ax.plot(ks, vals, "o-", color="#2196F3", linewidth=2, markersize=7)
    ax.axvline(4, color="#FF9800", linestyle="--", linewidth=1.5, label="K=4 (chosen)")
    ax.scatter([best_k], [best_fn(vals)], color="red", zorder=5, s=90,
               label=f"Best K={best_k}")
    ax.set_title(title)
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ks)
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_clustering_metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 02_clustering_metrics.png")

#Cluster profiles
profiles = clust["profiles"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Customer Segments (K=4) — Profiles", fontsize=14, fontweight="bold")

#response rate per cluster
ax = axes[0]
c_rates = [profiles[c]["response_rate"] for c in range(4)]
c_ns    = [profiles[c]["n"]             for c in range(4)]
bars = ax.bar([f"C{c}" for c in range(4)], c_rates, color=C4, width=0.5)
ax.set_title("Response Rate per Cluster (%)")
ax.set_ylabel("Response Rate (%)")
ax.set_ylim(0, max(c_rates) * 1.25)
for bar, r, n in zip(bars, c_rates, c_ns):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{r:.1f}%\n(n={n:,})", ha="center", fontsize=9)

#characteristics heatmap
ax = axes[1]
char_keys   = ["avg_age", "single_pct", "married_pct",
               "poutcome_success_pct", "housing_no_pct"]
char_labels = ["Avg Age", "Single %", "Married %",
               "Prev Success %", "No Housing %"]
#Τα χαρακτηριστικά κάθε cluster 
matrix = np.zeros((4, len(char_keys)))
for c in range(4):
    for j in range(len(char_keys)):
        matrix[c, j] = profiles[c][char_keys[j]]
col_range = matrix.max(axis=0) - matrix.min(axis=0)
col_range[col_range == 0] = 1.0
matrix_norm = (matrix - matrix.min(axis=0)) / col_range

im = ax.imshow(matrix_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(char_labels)))
ax.set_xticklabels(char_labels, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(4))
ax.set_yticklabels([f"C{c}" for c in range(4)])
ax.set_title("Cluster Characteristics (column-normalized)")
for i in range(4):
    for j in range(len(char_keys)):
        ax.text(j, i, f"{matrix[i, j]:.1f}",
                ha="center", va="center", fontsize=9,
                color="white" if matrix_norm[i, j] > 0.6 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_cluster_profiles.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 03_cluster_profiles.png")

#Model comparison (grouped bar)
model_names   = list(clf.keys())
metric_keys   = ["test_auc", "prauc", "f1"]
metric_labels = ["Test AUC", "PR-AUC", "F1"]
bar_colors    = ["#1565C0", "#42A5F5", "#66BB6A", "#FF7043"]

fig, ax = plt.subplots(figsize=(10, 6))
x     = np.arange(len(model_names))
width = 0.22

for i, (mk, ml, bc) in enumerate(zip(metric_keys, metric_labels, bar_colors)):
    vals = [clf[mn][mk] for mn in model_names]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, label=ml, color=bc)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=7, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=12)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
ax.set_title("Model Comparison — LR / RF / XGBoost", fontsize=14, fontweight="bold")
ax.legend(loc="upper right")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 04_model_comparison.png")


#ROC curves
fig, ax = plt.subplots(figsize=(8, 7))
roc_colors = ["#2196F3", "#FF9800", "#4CAF50"]

for (name, prob), color in zip(y_probs.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name}  (AUC = {roc_val:.4f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim((0.0, 1.0))
ax.set_ylim((0.0, 1.02))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "05_roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 05_roc_curves.png")


#Confusion matrices
fig, axes = plt.subplots(1, len(model_names), figsize=(14, 5))
fig.suptitle("Confusion Matrices (threshold = 0.50)", fontsize=14, fontweight="bold")

for ax, name in zip(axes, model_names):
    cm = np.array(clf[name]["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NO", "YES"])
    ax.set_yticklabels(["NO", "YES"])
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, f"{val:,}\n({val / total * 100:.1f}%)",
                    ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "06_confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 06_confusion_matrices.png")


#Threshold strategy comparison
s_labels = [
    "Default\nt=0.50",
    f"Global\nt*={thr['global']['threshold']}",
    f"Max-F1\nt={thr['max_f1']['threshold']}",
    "Cluster\nt*",
]
s_data = [thr["default"], thr["global"], thr["max_f1"], thr["cluster"]]
profits = [s["profit"] for s in s_data]
calls_l = [s["calls"]  for s in s_data]
rois    = [s["roi"]    for s in s_data]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("Threshold Optimization — Strategy Comparison",
             fontsize=14, fontweight="bold")

bar_s = ["#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1"]
for ax, vals, title, ylabel in [
    (axes[0], profits, "Net Profit (€)",     "Profit (€)"),
    (axes[1], calls_l, "Number of Calls",    "Calls"),
    (axes[2], rois,    "ROI on Call Cost (%)", "ROI (%)"),
]:
    bars = ax.bar(s_labels, vals, color=bar_s, width=0.5)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(max(vals)) * 0.01,
                f"{v:,.0f}", ha="center", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "07_threshold_strategies.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 07_threshold_strategies.png")


#SHAP — global top-15

global_shap = shap_r["global_top15"]
sh_names = [e["feature"]       for e in global_shap]
sh_vals  = [e["mean_abs_shap"] for e in global_shap]

fig, ax = plt.subplots(figsize=(10, 7))
n = len(sh_names)
colors_sh = plt.get_cmap("Blues")(np.linspace(0.35, 0.85, n))[::-1]
ax.barh(range(n), sh_vals, color=colors_sh)
ax.set_yticks(range(n))
ax.set_yticklabels(sh_names, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP| Value")
ax.set_title(f"SHAP Global Feature Importance (Top 15) - {shap_model} Model",
             fontsize=13, fontweight="bold")
for i, v in enumerate(sh_vals):
    ax.text(v + max(sh_vals) * 0.01, i, f"{v:.4f}", va="center", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "08_shap_global.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 08_shap_global.png")


#SHAP — per-cluster top-5

per_cluster = shap_r["per_cluster_top5"]
cluster_ids = sorted(per_cluster.keys())

fig, axes = plt.subplots(1, len(cluster_ids), figsize=(16, 5))
fig.suptitle(f"SHAP Top-5 Features per Cluster - {shap_model} Model",
             fontsize=13, fontweight="bold")

for c, ax in zip(cluster_ids, axes):
    entries  = per_cluster[c]
    fnames   = [e["feature"]       for e in entries]
    fvals    = [e["mean_abs_shap"] for e in entries]
    rate     = profiles[c]["response_rate"]

    ax.barh(range(len(fnames)), fvals, color=C4[c])
    ax.set_yticks(range(len(fnames)))
    ax.set_yticklabels(fnames, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"C{c}  (response {rate:.1f}%)", fontsize=11, fontweight="bold")
    for i, v in enumerate(fvals):
        ax.text(v + max(fvals) * 0.02, i, f"{v:.4f}", va="center", fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "09_shap_per_cluster.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 09_shap_per_cluster.png")


print(f"\n[OK] All 9 figures saved to {FIG_DIR}/")

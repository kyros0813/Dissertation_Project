"""
03_clustering.py — K-Means Customer Segmentation
Φορτώνει artifacts.pkl, τρέχει K-Means για K=2..8, αποθηκεύει cluster assignments + profiles
"""

import pandas as pd
import numpy as np
import os, pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

DATA_PATH = "data/bank-additional-full.csv"
OUT_DIR   = "outputs"

#Load
art_path = os.path.join(OUT_DIR, "artifacts.pkl")
with open(art_path, "rb") as f:
    art = pickle.load(f)

X_train_sc          = art["X_train_sc"]
X_test_sc           = art["X_test_sc"]
y_train             = art["y_train"]
y_test              = art["y_test"]
cluster_feature_idx = art["cluster_feature_idx"]

X_train_cl = X_train_sc[:, cluster_feature_idx]
X_test_cl  = X_test_sc[:,  cluster_feature_idx]

#Evaluate K=2..8
print("K-Means evaluation:")
print(f"{'K':<4} {'Silhouette':>12} {'Davies-Bouldin':>16} {'Calinski-Harabasz':>20}")
print("-" * 55)

clustering_metrics = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_train_cl)
    sil = round(float(silhouette_score(X_train_cl, labels, sample_size=5000, random_state=42)), 4)
    db  = round(float(davies_bouldin_score(X_train_cl, labels)), 4)
    ch  = round(float(calinski_harabasz_score(X_train_cl, labels)), 1)
    clustering_metrics[k] = {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}
    print(f"K={k}  {sil:>12}  {db:>16}  {ch:>20}")

#Fit final K=4
print("\nFitting K=4 (final model)...")
km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
train_clusters = km4.fit_predict(X_train_cl)
test_clusters  = km4.predict(X_test_cl)

#Cluster profiles
#Φορτώνω τα αρχικά δεδομένα μόνο για να βγάλω κατανοητά προφίλ των clusters.
df_orig = pd.read_csv(DATA_PATH, sep=";")
idx_train = art["idx_train"]
idx_test  = art["idx_test"]

df_train_orig = df_orig.iloc[idx_train].copy().reset_index(drop=True)
df_train_orig["cluster"] = train_clusters
df_train_orig["y_bin"]   = y_train

df_test_orig = df_orig.iloc[idx_test].copy().reset_index(drop=True)
df_test_orig["cluster"] = test_clusters
df_test_orig["y_bin"]   = y_test

print("\nCluster profiles (TRAIN):")
print(f"{'C':<3} {'N':>7} {'%':>6} {'Response':>10} {'Age':>5} {'Top Job':<15} {'Marital':<10} {'pout_succ%':>11} {'single%':>8} {'married%':>9}")
print("-" * 95)

cluster_profiles = {}
for c in range(4):
    sub  = df_train_orig[df_train_orig["cluster"] == c]
    n_c  = len(sub)
    pct  = round(n_c / len(df_train_orig) * 100, 1)
    rate = round(float(sub["y_bin"].mean() * 100), 1)
    age  = round(float(sub["age"].mean()), 1)
    top_job     = sub["job"].value_counts().index[0]
    top_marital = sub["marital"].value_counts().index[0]
    pout_s = round(float((sub["poutcome"] == "success").mean() * 100), 1)
    single = round(float((sub["marital"] == "single").mean() * 100), 1)
    married= round(float((sub["marital"] == "married").mean() * 100), 1)
    housing_no = round(float((sub["housing"] == "no").mean() * 100), 1)

    cluster_profiles[c] = {
        "n": n_c, "pct": pct, "response_rate": rate, "avg_age": age,
        "top_job": top_job, "top_marital": top_marital,
        "poutcome_success_pct": pout_s,
        "single_pct": single, "married_pct": married,
        "housing_no_pct": housing_no,
    }
    print(f"C{c}  {n_c:>7,} {pct:>6.1f}% {rate:>9.1f}% {age:>5.1f} {top_job:<15} {top_marital:<10} {pout_s:>11.1f}% {single:>7.1f}% {married:>8.1f}%")

print("\nCluster distribution (TEST):")
test_cluster_stats = {}
for c in range(4):
    sub  = df_test_orig[df_test_orig["cluster"] == c]
    n_c  = len(sub)
    pct  = round(n_c / len(df_test_orig) * 100, 1)
    rate = round(float(sub["y_bin"].mean() * 100), 1)
    test_cluster_stats[c] = {"n": n_c, "pct": pct, "response_rate": rate}
    print(f"  C{c}: N={n_c:,} ({pct:.1f}%)  Response={rate:.1f}%")

#Save
#Αποθηκεύω μόνο όσα χρειάζονται τα επόμενα βήματα και η εφαρμογή.
art["kmeans"]        = km4
art["test_clusters"] = test_clusters

with open(art_path, "wb") as f:
    pickle.dump(art, f)

res_path = os.path.join(OUT_DIR, "results.pkl")
with open(res_path, "rb") as f:
    all_results = pickle.load(f)
all_results["clustering"] = {
    "metrics": clustering_metrics,
    "profiles": cluster_profiles,
    "test_stats": test_cluster_stats,
}
with open(res_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"\n[OK] Clustering saved to {art_path}")

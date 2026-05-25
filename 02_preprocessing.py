"""
02_preprocessing.py — Data Preprocessing
Φορτώνει raw data, κάνει encoding/scaling, αποθηκεύει train/test sets στο outputs/artifacts.pkl
"""

import pandas as pd
import numpy as np
import os, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = "data/bank-additional-full.csv"
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

#Load
df = pd.read_csv(DATA_PATH, sep=";")
y  = (df["y"] == "yes").astype(int)
y_arr = y.to_numpy(dtype=int)
df_proc = df.drop(columns=["duration", "y"])
print(f"Loaded {len(df):,} rows. Removed 'duration'.")

#Ordinal encoding for education
edu_order = [
    "illiterate", "basic.4y", "basic.6y", "basic.9y",
    "high.school", "professional.course", "university.degree", "unknown"
]
#Κρατάω σταθερή τη σειρά κωδικοποίησης του education για να είναι συνεπή εκπαίδευση και εφαρμογή.
edu_map = {}
for i in range(len(edu_order)):
    edu_map[edu_order[i]] = i
df_proc["education"] = df_proc["education"].map(edu_map).fillna(len(edu_order) - 1).astype(int)
print("Education ordinal encoded.")

#One-hot encoding
nominal_cols = ["job", "marital", "default", "housing", "loan",
                "contact", "month", "day_of_week", "poutcome"]
df_proc = pd.get_dummies(df_proc, columns=nominal_cols, drop_first=False)
feature_names = df_proc.columns.tolist()
print(f"After encoding: {len(feature_names)} features")

#Train / Test split (stratified)
all_idx = np.arange(len(df_proc))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    df_proc.values, y_arr, all_idx,
    test_size=0.20, random_state=42, stratify=y_arr
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Train YES%: {y_train.mean()*100:.2f}% | Test YES%: {y_test.mean()*100:.2f}%")

#Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("StandardScaler applied.")

#Clustering feature indices (exclude macro)
macro_names = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
#Στο clustering εξαιρώ τους μακροοικονομικούς δείκτες για να μην αλλοιώνουν τεχνητά τα τμήματα πελατών.
cluster_feature_idx = []
for i in range(len(feature_names)):
    name = feature_names[i]
    is_macro = False
    for m in macro_names:
        if m in name:
            is_macro = True
    if not is_macro:
        cluster_feature_idx.append(i)
print(f"Clustering features: {len(cluster_feature_idx)} (excluded {len(feature_names) - len(cluster_feature_idx)} macro)")

#Metadata for app.py    
num_cols = ["age", "campaign", "pdays", "previous", "education",
            "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
cat_cols = ["job", "marital", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome"]

#Save artifacts
artifacts = {
    "X_train_sc":          X_train_sc,
    "X_test_sc":           X_test_sc,
    "y_train":             y_train,
    "y_test":              y_test,
    "idx_train":           idx_train,
    "idx_test":            idx_test,
    "feature_names":       feature_names,
    "scaler":              scaler,
    "cluster_feature_idx": cluster_feature_idx,
    "C_call":              1.25,
    "V_deposit":           50.0,
    "cat_cols":            cat_cols,
    "num_cols":            num_cols,
    "edu_order":           edu_order,
}

art_path = os.path.join(OUT_DIR, "artifacts.pkl")
with open(art_path, "wb") as f:
    pickle.dump(artifacts, f)

print(f"\n[OK] Artifacts saved to {art_path}")

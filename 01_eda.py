"""
01_eda.py — Exploratory Data Analysis
Τρέξε πρώτο. Εκτυπώνει στατιστικά και αποθηκεύει eda_results στο outputs/results.pkl
"""

import pandas as pd
import numpy as np
import os, pickle

#Paths
DATA_PATH = "data/bank-additional-full.csv"
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

#Load
df = pd.read_csv(DATA_PATH, sep=";")
y  = (df["y"] == "yes").astype(int)
print(f"Dataset: {len(df):,} rows x {len(df.columns)} columns")

#Class Distribution
n_yes = int(y.sum())
n_no  = int((y == 0).sum())
ratio = round(n_no / n_yes, 1)
print(f"\nClass distribution:")
print(f"  YES: {n_yes:,} ({n_yes/len(df)*100:.2f}%)")
print(f"  NO:  {n_no:,}  ({n_no/len(df)*100:.2f}%)")
print(f"  Imbalance ratio 1:{ratio}")

#Correlations with target
#Ελέγχω πρώτα τις βασικές αριθμητικές μεταβλητές για μια γρήγορη αρχική εικόνα.
num_cols = ["age", "campaign", "pdays", "previous",
            "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
print("\nCorrelation with target:")
correlations = {}
for col in num_cols:
    r = round(float(df[col].corr(y)), 3)
    correlations[col] = r
    print(f"  {col:<22}: r = {r:+.3f}")

#Multicollinearity
macro_cols = ["euribor3m", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "nr.employed"]
macro_corr = df[macro_cols].corr()
macro_corr_values = macro_corr.to_numpy(dtype=float)
print("\nMulticollinearity among macro features (|r| > 0.70):")
high_corr = {}
for i in range(len(macro_cols)):
    for j in range(i + 1, len(macro_cols)):
        r = round(float(macro_corr_values[i, j]), 3)
        if abs(r) > 0.70:
            pair = f"{macro_cols[i]} <-> {macro_cols[j]}"
            high_corr[pair] = r
            print(f"  {pair}: r = {r:.3f}")

#Month response rates
print("\nResponse rate by month:")
month_stats = {}
for month, grp in df.groupby("month"):
    n     = len(grp)
    rate  = round(float((grp["y"] == "yes").mean() * 100), 1)
    month_stats[month] = {"n": n, "rate": rate}
    print(f"  {month:<5}: n={n:>6,}  rate={rate:.1f}%")

#Contact channel
print("\nResponse rate by contact type:")
contact_stats = {}
for ctype, grp in df.groupby("contact"):
    n    = len(grp)
    rate = round(float((grp["y"] == "yes").mean() * 100), 1)
    contact_stats[ctype] = {"n": n, "rate": rate}
    print(f"  {ctype:<12}: n={n:>6,}  rate={rate:.1f}%")

#Previous campaign outcome
print("\nResponse rate by poutcome:")
poutcome_stats = {}
for po, grp in df.groupby("poutcome"):
    n    = len(grp)
    rate = round(float((grp["y"] == "yes").mean() * 100), 1)
    poutcome_stats[po] = {"n": n, "rate": rate}
    print(f"  {po:<15}: n={n:>6,}  rate={rate:.1f}%")

#Unknown value counts
print("\nUnknown values per categorical feature:")
unknown_counts = {}
for col in ["job", "marital", "education", "default", "housing", "loan"]:
    n   = int((df[col] == "unknown").sum())
    pct = round(n / len(df) * 100, 1)
    unknown_counts[col] = {"n": n, "pct": pct}
    if n > 0:
        print(f"  {col:<12}: {n:>5,} ({pct:.1f}%)")

#Macro feature ranges
print("\nMacro feature ranges:")
macro_ranges = {}
for col in macro_cols:
    mn = round(float(df[col].min()), 3)
    mx = round(float(df[col].max()), 3)
    macro_ranges[col] = {"min": mn, "max": mx}
    print(f"  {col:<22}: [{mn}, {mx}]")

#Save results
eda_results = {
    "n_total": len(df),
    "n_yes": n_yes,
    "n_no": n_no,
    "imbalance_ratio": ratio,
    "correlations": correlations,
    "high_corr_pairs": high_corr,
    "month_stats": month_stats,
    "contact_stats": contact_stats,
    "poutcome_stats": poutcome_stats,
    "unknown_counts": unknown_counts,
    "macro_ranges": macro_ranges,
    "macro_cols": macro_cols,
}

results_path = os.path.join(OUT_DIR, "results.pkl")
#Αν υπάρχει ήδη αρχείο αποτελεσμάτων, το ενημερώνω χωρίς να σβήνω τα υπόλοιπα στάδια.
if os.path.exists(results_path):
    with open(results_path, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = {}

all_results["eda"] = eda_results

with open(results_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"\n[OK] EDA results saved to {results_path}")

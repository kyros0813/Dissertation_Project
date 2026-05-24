# Bank Telemarketing DSS

Πτυχιακή εργασία για την ανάπτυξη ενός απλού Συστήματος Υποστήριξης
Αποφάσεων (DSS) για τηλεφωνικές τραπεζικές καμπάνιες.

## Δομή project

```text
project/
├── data/
│   └── bank-additional-full.csv
├── outputs/
│   ├── artifacts.pkl
│   ├── results.pkl
│   └── figures/
├── 01_eda.py
├── 02_preprocessing.py
├── 03_clustering.py
├── 04_classification.py
├── 05_threshold.py
├── 06_shap.py
├── 07_visualization.py
├── run_all.py
└── app.py
```

## Βήματα ανάλυσης

1. `01_eda.py`: βασική διερευνητική ανάλυση.
2. `02_preprocessing.py`: αφαίρεση `duration`, encoding, scaling και train/test split.
3. `03_clustering.py`: K-Means clustering και επιλογή K=4 για customer segments.
4. `04_classification.py`: σύγκριση Logistic Regression, Random Forest και XGBoost.
5. `05_threshold.py`: βελτιστοποίηση decision thresholds με βάση το αναμενόμενο κέρδος.
6. `06_shap.py`: ερμηνεία του τελικού μοντέλου με SHAP values.
7. `07_visualization.py`: δημιουργία των βασικών γραφημάτων.

## Εγκατάσταση

Με conda:

```bash
conda env create -f environment.yml
conda activate dss_project
```

Εναλλακτικά με pip:

```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib
```

## Εκτέλεση

Για ολόκληρη τη ροή:

```bash
python run_all.py
```

Για εκτέλεση βήμα-βήμα:

```bash
python 01_eda.py
python 02_preprocessing.py
python 03_clustering.py
python 04_classification.py
python 05_threshold.py
python 06_shap.py
python 07_visualization.py
```

Για την εφαρμογή DSS:

```bash
streamlit run app.py
```

## Σημειώσεις

- Το `duration` αφαιρείται γιατί δεν είναι διαθέσιμο πριν γίνει η κλήση και μπορεί να οδηγήσει σε data leakage.
- Το XGBoost χρησιμοποιείται ως τελικό decision model, ώστε να συνδέεται καθαρά με το SHAP analysis.
- Το `outputs/artifacts.pkl` λειτουργεί ως απλό model bundle για την εφαρμογή. Περιέχει το μοντέλο, τον scaler, το K-Means και τα thresholds.

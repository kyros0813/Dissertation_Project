# Copilot instructions

## Commands

### Environment setup

```bash
conda env create -f environment.yml
conda activate dss_project
```

Fallback if Conda is not being used:

```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib
```

### Main entrypoints

Run the full analysis pipeline in the intended order:

```bash
python run_all.py
```

Run a single pipeline stage:

```bash
python 01_eda.py
python 02_preprocessing.py
python 03_clustering.py
python 04_classification.py
python 05_threshold.py
python 06_shap.py
python 07_visualization.py
```

Run the DSS UI:

```bash
streamlit run app.py
```

There is no automated test suite or lint command defined in this repository.

## High-level architecture

This repository is a linear analytics pipeline over `data/bank-additional-full.csv`, plus a small Streamlit inference UI.

- `run_all.py` is the canonical orchestration entrypoint. It executes the numbered scripts in order and stops on the first failure.
- `01_eda.py` reads the raw CSV and writes the `eda` section into `outputs/results.pkl`.
- `02_preprocessing.py` creates the shared modeling contract in `outputs/artifacts.pkl`: scaled train/test arrays, `feature_names`, the fitted `StandardScaler`, feature-group metadata for the app, clustering feature indices, and business constants.
- `03_clustering.py` reads `artifacts.pkl`, clusters only the non-macro feature subset, writes the fitted `kmeans` model and `test_clusters` back into `artifacts.pkl`, and stores clustering summaries in `results.pkl`.
- `04_classification.py` trains Logistic Regression, Random Forest, and XGBoost on the scaled features. It keeps all test-set probabilities in `artifacts.pkl["y_probs"]`, but only persists XGBoost in `artifacts.pkl["models"]` as the final `decision_model`.
- `05_threshold.py` uses the chosen model probabilities, the test clusters, and the business-cost constants to compare threshold strategies and save cluster-specific profit-maximizing thresholds into `artifacts.pkl["cluster_thresholds"]`.
- `06_shap.py` assumes the final decision model is XGBoost and stores summary SHAP outputs in `results.pkl`.
- `07_visualization.py` expects every earlier stage to have populated `results.pkl` and `artifacts.pkl`; it generates the final figures in `outputs/figures`.
- `app.py` is intentionally thin: it loads `outputs/artifacts.pkl`, manually rebuilds a single feature row using the stored metadata, scales it, predicts the customer cluster, and applies that cluster's threshold to the selected decision model.

## Key conventions

- The numbered scripts are stateful pipeline stages, not independent modules. Later stages assume earlier pickle keys already exist.
- `outputs/artifacts.pkl` is the runtime bundle for downstream scripts and `app.py`. Extend it compatibly instead of replacing its structure.
- `outputs/results.pkl` is an analysis bundle keyed by stage name (`eda`, `clustering`, `classification`, `threshold`, `shap`). `07_visualization.py` reads those keys directly.
- `duration` is intentionally dropped before modeling to avoid leakage. Do not add it back into training features or the Streamlit input flow.
- `education` is the only categorical field treated as ordinal; it must stay aligned with `edu_order`. Other categorical inputs are represented through one-hot columns.
- Clustering intentionally excludes the macroeconomic variables (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`) through `cluster_feature_idx`.
- Threshold selection is business-profit driven, not accuracy-driven. The threshold stage uses `C_call=1.25` and `V_deposit=50.0` from `artifacts.pkl`.
- If you change model or preprocessing artifacts, preserve the keys consumed by `app.py`: `feature_names`, `num_cols`, `cat_cols`, `edu_order`, `scaler`, `cluster_feature_idx`, `kmeans`, `models`, and `cluster_thresholds`.

# Project: TCR-H Reproduction (Epitope-Hard Split)

Reproduce TCR-H paper (Front Immunol 2024) epitope-hard split results.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/train_final.py
```

## Output

All results go to `results/final/`:
- `results_table.csv` — metrics for all models
- `performance_bar_chart.png` — grouped bar chart
- `roc_curves.png` — ROC curve overlay
- `tcr_he_features_removed.json` — TCR-HE correlated feature removal log
- `models/*.pkl` — saved models + scalers

## Models

| Name | Features | Description |
|------|----------|-------------|
| RF | All 194 | Random Forest (default params) |
| GBT | All 194 | Gradient Boosting (default params) |
| XGB | All 194 | XGBoost (conservative params) |
| SVM-RBF (TCR-Hb) | All 194 | SVM RBF kernel, C=1.0, class_weight=balanced |
| TCR-HE | Removed correlated >0.8 | SVM RBF on reduced feature set |

## ROC-AUC Methodology

ROC-AUC is computed using `predict()` (class labels 0/1) to match the paper's
methodology. `AUC (proba)` using `predict_proba()` / `decision_function()` is
also reported for reference.

## Epitope-Hard Split

Train: 199,988 rows, 816 unique epitopes
Test:  54,449 rows, 65 unique epitopes (no overlap with train)

## Key Results vs Paper

| Model | Paper | Ours |
|-------|-------|------|
| RF | 0.50 | 0.5000 |
| GBT | 0.54 | ~0.49 |
| XGB | 0.51 | ~0.51 |
| SVM-RBF (TCR-Hb) | 0.80 | ~0.86 |
| TCR-HE | — | ~0.87 |

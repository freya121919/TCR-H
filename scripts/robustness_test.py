"""
Robustness test: 3× extra splits for epitope hard, TCR hard, strict.
Generates supplementary table 2 equivalent.
"""
import os, sys, time, warnings, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
RESULTS_DIR = "results/final"
CORR_THRESHOLD = 0.8
SVM_SUBSAMPLE = 30000
EXTRA_SEEDS = [43, 44, 45]

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Specificity": round(spec, 4),
        "F1-score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC (predict)": round(roc_auc_score(y_true, y_pred), 4),
    }

def train_svm(X_train, y_train, X_test, y_test, seed):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=SVM_SUBSAMPLE, random_state=seed)
    idx = next(sss.split(X_tr_sc, y_train))[0]
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
              random_state=seed, cache_size=2000)
    t0 = time.time()
    svm.fit(X_tr_sc[idx], y_train[idx])
    y_pred = svm.predict(X_te_sc)
    metrics = compute_metrics(y_test, y_pred)
    metrics["train_time"] = round(time.time() - t0, 1)
    return metrics

def correlated_feature_removal(X_train, X_test, seed):
    rng = np.random.default_rng(seed)
    corr = pd.DataFrame(X_train).corr()
    n = len(corr.columns)
    drop = set()
    for i in range(n):
        for j in range(i):
            if abs(corr.iloc[i, j]) > CORR_THRESHOLD:
                pair = (corr.columns[i], corr.columns[j])
                keep = rng.choice(pair, 1)[0]
                drop.add(pair[0] if keep == pair[1] else pair[1])
    drop_cols = sorted(drop)
    keep_cols = [c for c in corr.columns if c not in drop]
    return pd.DataFrame(X_train).drop(columns=drop_cols).values, \
           pd.DataFrame(X_test).drop(columns=drop_cols).values, drop_cols, keep_cols

# Load all data
train_ep = pd.read_csv("data/splits_tchard/epitope_hard_train.csv", low_memory=False)
test_ep  = pd.read_csv("data/splits_tchard/epitope_hard_test.csv", low_memory=False)
feat_cols = [c for c in train_ep.columns if c.startswith(("cdr3_", "epitope_"))]
full = pd.concat([train_ep, test_ep], ignore_index=True)
X_all = full[feat_cols].fillna(0.0).astype(float)
y_all = full["label"].astype(int).values
cdr3_all = full["CDR3.beta"]
epi_all = full["antigen_epitope"]

from src.data.splits import epitope_hard_split, tcr_hard_split, strict_split

results = []

# ── 1. Epitope hard split × 3 (TCR-HE) ──
print("="*60)
print("TCR-HE (epitope hard split) × 3")
print("="*60)
for seed in EXTRA_SEEDS:
    train_idx, _, _ = epitope_hard_split(X_all, pd.Series(y_all), epi_all, test_size=0.2, random_seed=seed)
    test_idx = np.setdiff1d(np.arange(len(X_all)), train_idx)
    X_tr = X_all.iloc[train_idx]; y_tr = y_all[train_idx]
    X_te = X_all.iloc[test_idx]; y_te = y_all[test_idx]
    X_tr_r, X_te_r, _, _ = correlated_feature_removal(X_tr, X_te, seed)
    m = train_svm(X_tr_r, y_tr, X_te_r, y_te, seed)
    print(f"  seed={seed}  AUC={m['AUC (predict)']:.4f}  Acc={m['Accuracy']:.4f}  Prec={m['Precision']:.4f}  Recall={m['Recall']:.4f}  Spec={m['Specificity']:.4f}  F1={m['F1-score']:.4f}")
    m["Model"] = f"TCR-HE (seed {seed})"; m["Split"] = "Epitope hard"
    results.append(m)

# ── 2. TCR hard split × 3 (TCR-Hβ) ──
print("="*60)
print("TCR-Hβ (TCR hard split) × 3")
print("="*60)
for seed in EXTRA_SEEDS:
    train_idx, _, _ = tcr_hard_split(X_all, pd.Series(y_all), cdr3_all, test_size=0.2, random_seed=seed)
    test_idx = np.setdiff1d(np.arange(len(X_all)), train_idx)
    X_tr = X_all.iloc[train_idx]; y_tr = y_all[train_idx]
    X_te = X_all.iloc[test_idx]; y_te = y_all[test_idx]
    X_tr_r, X_te_r, _, _ = correlated_feature_removal(X_tr, X_te, seed)
    m = train_svm(X_tr_r, y_tr, X_te_r, y_te, seed)
    print(f"  seed={seed}  AUC={m['AUC (predict)']:.4f}  Acc={m['Accuracy']:.4f}  Prec={m['Precision']:.4f}  Recall={m['Recall']:.4f}  Spec={m['Specificity']:.4f}  F1={m['F1-score']:.4f}")
    m["Model"] = f"TCR-Hβ (seed {seed})"; m["Split"] = "TCR hard"
    results.append(m)

# ── 3. Strict split × 3 (TCR-HβE) ──
print("="*60)
print("TCR-HβE (strict split) × 3")
print("="*60)
for seed in EXTRA_SEEDS:
    train_idx, _, _ = strict_split(X_all, pd.Series(y_all), epi_all, cdr3_all, test_size=0.2, random_seed=seed)
    test_idx = np.setdiff1d(np.arange(len(X_all)), train_idx)
    X_tr = X_all.iloc[train_idx]; y_tr = y_all[train_idx]
    X_te = X_all.iloc[test_idx]; y_te = y_all[test_idx]
    X_tr_r, X_te_r, _, _ = correlated_feature_removal(X_tr, X_te, seed)
    m = train_svm(X_tr_r, y_tr, X_te_r, y_te, seed)
    print(f"  seed={seed}  AUC={m['AUC (predict)']:.4f}  Acc={m['Accuracy']:.4f}  Prec={m['Precision']:.4f}  Recall={m['Recall']:.4f}  Spec={m['Specificity']:.4f}  F1={m['F1-score']:.4f}")
    m["Model"] = f"TCR-HβE (seed {seed})"; m["Split"] = "Strict"
    results.append(m)

# ── Summary table ──
print("\n" + "="*100)
print("SUPPLEMENTARY TABLE 2 — Robustness Test Summary")
print("="*100)
print(f"{'Split':<16} {'Seed':<6} {'AUC':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'Spec':<8} {'F1':<8} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6}")
print("-"*100)
for r in results:
    print(f"{r['Split']:<16} {r['Model'][-2:]:<6} {r['AUC (predict)']:<8} {r['Accuracy']:<8} {r['Precision']:<8} {r['Recall']:<8} {r['Specificity']:<8} {r['F1-score']:<8} {r['TP']:<6} {r['TN']:<6} {r['FP']:<6} {r['FN']:<6}")

# Summary stats
print("\n" + "="*60)
print("RANGE ACROSS SEEDS")
print("="*60)
for split_name in ["Epitope hard", "TCR hard", "Strict"]:
    aucs = [r["AUC (predict)"] for r in results if r["Split"] == split_name]
    print(f"  {split_name:<16} AUC range: {min(aucs):.4f} - {max(aucs):.4f}  (mean: {np.mean(aucs):.4f})")

# Save
df_out = pd.DataFrame(results)
df_out.to_csv(os.path.join(RESULTS_DIR, "robustness_test.csv"), index=False)
with open(os.path.join(RESULTS_DIR, "robustness_test.json"), "w") as f:
    json.dump([{k: v for k, v in r.items() if not isinstance(v, (np.integer, np.floating)) or isinstance(v, (int, float))} for r in results], f, indent=2, default=str)
print(f"\nSaved: {RESULTS_DIR}/robustness_test.csv")

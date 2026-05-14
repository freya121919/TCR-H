"""
Test equal-probability random epitope split vs sqrt-weighted greedy split.

Result (2026-05-14): equal-probability produces consistently worse splits
(176 test epitopes, AUC ~0.50-0.76, extreme class imbalance in 12/14 seeds).
The sqrt-weighted greedy approach is retained as the default because it
produces test sets with stable sample sizes (~20% of total) and better
class balance.

See report.md §"分割方法比较" for discussion.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
RESULTS_DIR = "results/final"
CORR_THRESHOLD = 0.8
SVM_SUBSAMPLE = 30000

def train_svm_auc(X_train, y_train, X_test, y_test, seed):
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
    auc = roc_auc_score(y_test, y_pred)
    return auc, round(time.time() - t0, 1)

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
           pd.DataFrame(X_test).drop(columns=drop_cols).values, keep_cols

def simple_epitope_split(epitopes, test_size, random_seed):
    """Equal-probability random split of unique epitopes (no sqrt-weighting)."""
    unique = epitopes.unique()
    rng = np.random.default_rng(random_seed)
    n_test = max(1, round(len(unique) * test_size))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    test_ep = set(shuffled[:n_test])
    train_ep = set(shuffled[n_test:])
    train_mask = epitopes.isin(train_ep)
    test_mask = epitopes.isin(test_ep)
    return np.where(train_mask)[0], np.where(test_mask)[0], train_ep, test_ep

# Load data
train_ep = pd.read_csv("data/splits_tchard/epitope_hard_train.csv", low_memory=False)
test_ep = pd.read_csv("data/splits_tchard/epitope_hard_test.csv", low_memory=False)
feat_cols = [c for c in train_ep.columns if c.startswith(("cdr3_", "epitope_"))]
full = pd.concat([train_ep, test_ep], ignore_index=True)
X_all = full[feat_cols].fillna(0.0).astype(float)
y_all = full["label"].astype(int).values
epi_all = full["antigen_epitope"]

# CSV baseline
print(f"{'seed':<8} {'test_pos':>9} {'test_neg':>9} {'ratio':>7} {'epi':>4} {'AUC':>8} {'time':>6}")
print("-"*55)
print(f"{'CSV':<8} {sum(test_ep['label']):>9} {len(test_ep)-sum(test_ep['label']):>9} "
      f"{(len(test_ep)-sum(test_ep['label']))/max(sum(test_ep['label']),1):>7.2f} "
      f"{test_ep['antigen_epitope'].nunique():>4}  -    -   ")

for s in range(42, 56):
    train_idx, test_idx, train_ep_set, test_ep_set = simple_epitope_split(epi_all, 0.2, s)
    y_tr = y_all[train_idx]
    y_te = y_all[test_idx]
    X_tr = X_all.iloc[train_idx]
    X_te = X_all.iloc[test_idx]
    pos = int(sum(y_te))
    neg = int(len(y_te) - pos)
    ratio = neg / max(pos, 1)
    nep = len(test_ep_set)

    # Feature removal + SVM
    X_tr_r, X_te_r, keep = correlated_feature_removal(X_tr, X_te, s)
    auc, t = train_svm_auc(X_tr_r, y_tr, X_te_r, y_te, s)

    flag = " <--" if ratio > 3 or ratio < 0.3 else ""
    print(f"{f'seed={s}':<8} {pos:>9} {neg:>9} {ratio:>7.2f} {nep:>4} {auc:>8.4f} {t:>5.1f}{flag}")

# Also show original greedy results for comparison
print()
print("=== 原始 greedy (from earlier robustness test) ===")
greedy_data = {43: 0.8590, 44: 0.9124, 45: 0.5784}
for s, auc in greedy_data.items():
    print(f"  seed={s}  AUC={auc:.4f}")

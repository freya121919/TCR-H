"""
Imbalance stress test for TCR-HE — AUC(predict) to match paper methodology.
For high ratios, downsamples positives (not enough negatives available).
"""
import os, sys, time, warnings, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
RESULTS_DIR = "results/final"
CORR_THRESHOLD = 0.8
SVM_SUBSAMPLE = 30000

def correlated_feature_removal(X_train, X_test):
    rng = np.random.default_rng(RANDOM_SEED)
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
    return pd.DataFrame(X_train).drop(columns=drop_cols).values, \
           pd.DataFrame(X_test).drop(columns=drop_cols).values

train_ep = pd.read_csv("data/splits_tchard/epitope_hard_train.csv", low_memory=False)
test_ep = pd.read_csv("data/splits_tchard/epitope_hard_test.csv", low_memory=False)
feat_cols = [c for c in train_ep.columns if c.startswith(("cdr3_", "epitope_"))]
X_tr = train_ep[feat_cols].fillna(0.0).astype(float)
y_tr = train_ep["label"].astype(int).values
X_te = test_ep[feat_cols].fillna(0.0).astype(float)
y_te = test_ep["label"].astype(int).values
X_tr_r, X_te_r = correlated_feature_removal(X_tr, X_te)

pos_total = int((y_tr == 1).sum())
neg_total = int((y_tr == 0).sum())

# For low ratios: subsample negatives
# For high ratios: subsample positives (not enough neg samples)
scenarios = [
    ("0.5:1 (more pos)", int(pos_total * 0.5), pos_total),
    ("1:1 (balanced)",   pos_total,             pos_total),
    ("1.64:1 (original)", neg_total,            pos_total),
    ("3:1",              neg_total,             int(neg_total / 3)),
    ("5:1",              neg_total,             int(neg_total / 5)),
    ("10:1",             neg_total,             int(neg_total / 10)),
]

results = []
for desc, n_neg, n_pos in scenarios:
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    rng = np.random.default_rng(RANDOM_SEED)
    pos_sub = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    neg_sub = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    train_idx = np.concatenate([pos_sub, neg_sub])
    rng.shuffle(train_idx)

    X_tr_sub = X_tr_r[train_idx]
    y_tr_sub = y_tr[train_idx]
    actual_ratio = round(float((y_tr_sub == 0).sum()) / float((y_tr_sub == 1).sum()), 2)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_sub)
    X_te_sc = scaler.transform(X_te_r)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(SVM_SUBSAMPLE, len(X_tr_sub)),
                                  random_state=RANDOM_SEED)
    idx = next(sss.split(X_tr_sc, y_tr_sub))[0]
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
              random_state=RANDOM_SEED, cache_size=2000)
    t0 = time.time()
    svm.fit(X_tr_sc[idx], y_tr_sub[idx])
    y_pred = svm.predict(X_te_sc)
    auc_pred = round(float(roc_auc_score(y_te, y_pred)), 4)
    results.append({"scenario": desc, "actual_ratio": actual_ratio,
                    "neg": int(n_neg), "pos": int(n_pos), "AUC_predict": auc_pred,
                    "time": round(time.time() - t0, 1)})
    print(f"  {desc:<22} neg={int(n_neg):<6} pos={int(n_pos):<6} ratio={actual_ratio:<6} AUC(pred)={auc_pred:.4f}")

print("\n=== Imbalance Stress Test ===")
print(f"{'Scenario':<22} {'Ratio':>6} {'AUC(pred)':>10}")
for r in results:
    print(f"{r['scenario']:<22} {r['actual_ratio']:>6} {r['AUC_predict']:>10}")

fig, ax = plt.subplots(figsize=(9, 5))
lbls = [r["scenario"] for r in results]
vals = [r["AUC_predict"] for r in results]
ax.plot(range(len(lbls)), vals, "o-", color="#E74C3C", linewidth=2, markersize=8)
ax.axhline(y=0.8735, color="gray", linestyle="--", alpha=0.5, label="TCR-HE original")
ax.set_xticks(range(len(lbls)))
ax.set_xticklabels(lbls, fontsize=8, rotation=15)
ax.set_xlabel("Neg:Pos Ratio")
ax.set_ylabel("AUC (predict)")
ax.set_title("TCR-HE Imbalance Stress Test")
ax.set_ylim([0.5, 1.0])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "imbalance_stress_test.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved. Paper range: 0.72-0.853")

with open(os.path.join(RESULTS_DIR, "imbalance_stress_test.json"), "w") as f:
    json.dump({"results": results}, f, indent=2)

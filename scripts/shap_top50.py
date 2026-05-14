"""
Regenerate SHAP plots with top-50 features from saved models.
"""
import os, sys, logging, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
logging.getLogger("shap").setLevel(logging.WARNING)

RANDOM_SEED = 42
RESULTS_DIR = "results/final"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# Data
train_ep = pd.read_csv("data/splits_tchard/epitope_hard_train.csv", low_memory=False)
test_ep = pd.read_csv("data/splits_tchard/epitope_hard_test.csv", low_memory=False)
feat_cols = [c for c in train_ep.columns if c.startswith(("cdr3_", "epitope_"))]

full = pd.concat([train_ep, test_ep], ignore_index=True)
X_all = full[feat_cols].fillna(0.0).astype(float)
y_all = full["label"].astype(int).values
cdr3_all = full["CDR3.beta"]
epi_all = full["antigen_epitope"]

from src.data.splits import epitope_hard_split, tcr_hard_split, strict_split, random_split

RNG = np.random.default_rng(RANDOM_SEED)
CORR_THRESHOLD = 0.8

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
    return [c for c in corr.columns if c not in drop]

def shap_analysis_top50(model_name, svm, scaler, X_train, X_test, y_test, feature_names):
    print(f"SHAP top-50 for {model_name} ...")
    X_tr_sc = scaler.transform(X_train)
    X_te_sc = scaler.transform(X_test)
    n_bg = min(30, len(X_tr_sc))
    bg_idx = RNG.choice(len(X_tr_sc), size=n_bg, replace=False)
    n_explain = min(100, len(X_te_sc))
    te_idx = RNG.choice(len(X_te_sc), size=n_explain, replace=False)
    X_explain = X_te_sc[te_idx]
    explainer = shap.KernelExplainer(svm.decision_function, X_tr_sc[bg_idx])
    shap_values = explainer.shap_values(X_explain, nsamples=50, l1_reg=False)
    X_display = pd.DataFrame(X_explain, columns=feature_names)
    shap.summary_plot(shap_values, X_display, max_display=50, show=False)
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()
    shap.summary_plot(shap_values, X_display, plot_type="bar", max_display=50, show=False)
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {model_name} SHAP top-50 done")

# TCR-HE
keep_cols = correlated_feature_removal(
    train_ep[feat_cols].fillna(0.0).astype(float),
    test_ep[feat_cols].fillna(0.0).astype(float))
X_tr = train_ep[feat_cols].fillna(0.0).astype(float)[keep_cols]
y_tr = train_ep["label"].astype(int).values
X_te = test_ep[feat_cols].fillna(0.0).astype(float)[keep_cols]
y_te = test_ep["label"].astype(int).values
with open(os.path.join(MODELS_DIR, "tcr_he.pkl"), "rb") as f:
    d = pickle.load(f)
shap_analysis_top50("TCR-HE", d["model"], d["scaler"], X_tr.values, X_te.values, y_te, keep_cols)

# TCR-Hβ
train_idx, _, _ = tcr_hard_split(X_all, pd.Series(y_all), cdr3_all, test_size=0.2, random_seed=RANDOM_SEED)
test_idx = np.setdiff1d(np.arange(len(X_all)), train_idx)
X_tr_tcr = X_all.iloc[train_idx]; y_tr_tcr = y_all[train_idx]
X_te_tcr = X_all.iloc[test_idx]; y_te_tcr = y_all[test_idx]
keep_hb = correlated_feature_removal(X_tr_tcr, X_te_tcr)
with open(os.path.join(MODELS_DIR, "tcr_hb.pkl"), "rb") as f:
    d = pickle.load(f)
shap_analysis_top50("TCR-Hβ", d["model"], d["scaler"], X_tr_tcr[keep_hb].values, X_te_tcr[keep_hb].values, y_te_tcr, keep_hb)

# TCR-HβE
test_cdr3_set = set(test_ep["CDR3.beta"].unique())
train_mask = ~train_ep["CDR3.beta"].isin(test_cdr3_set)
X_tr_s = train_ep.loc[train_mask, feat_cols].fillna(0.0).astype(float)
y_tr_s = train_ep.loc[train_mask, "label"].astype(int).values
X_te_s = test_ep[feat_cols].fillna(0.0).astype(float)
y_te_s = test_ep["label"].astype(int).values
keep_hbe = correlated_feature_removal(X_tr_s, X_te_s)
with open(os.path.join(MODELS_DIR, "tcr_hbe.pkl"), "rb") as f:
    d = pickle.load(f)
shap_analysis_top50("TCR-HβE", d["model"], d["scaler"], X_tr_s[keep_hbe].values, X_te_s[keep_hbe].values, y_te_s, keep_hbe)

# TCR-RS
train_idx_r, _, _ = random_split(X_all, pd.Series(y_all), test_size=0.2, random_seed=RANDOM_SEED)
test_idx_r = np.setdiff1d(np.arange(len(X_all)), train_idx_r)
X_tr_rs = X_all.iloc[train_idx_r]; y_tr_rs = y_all[train_idx_r]
X_te_rs = X_all.iloc[test_idx_r]; y_te_rs = y_all[test_idx_r]
keep_rs = correlated_feature_removal(X_tr_rs, X_te_rs)
with open(os.path.join(MODELS_DIR, "tcr_rs.pkl"), "rb") as f:
    d = pickle.load(f)
shap_analysis_top50("TCR-RS", d["model"], d["scaler"], X_tr_rs[keep_rs].values, X_te_rs[keep_rs].values, y_te_rs, keep_rs)

print("All SHAP top-50 plots regenerated.")

"""
Final training: all models and splits.

Table 2 (epitope-hard split, all 194 features):
  RF, GBT, XGB, SVM-RBF

Table 3 (uncorrelated features, remove corr > 0.8):
  TCR-Hβ (TCR hard), TCR-HβE (strict), TCR-RS (random), TCR-HE (epitope hard)

Output -> results/final/
  - results_table.csv    — all models' metrics
  - performance_bar_chart.png
  - roc_curves.png
  - models/*.pkl
  - *_shap_*.png         — SHAP beeswarm + bar plots
"""

import json, logging, os, pickle, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
logging.getLogger("shap").setLevel(logging.WARNING)

RESULTS_DIR = "results/final"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED = 42
SVM_SUBSAMPLE = 30000
CORR_THRESHOLD = 0.8

EPI_HARD_TRAIN = "data/splits_tchard/epitope_hard_train.csv"
EPI_HARD_TEST  = "data/splits_tchard/epitope_hard_test.csv"


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


def train_svm(name, X_train, y_train, X_test, y_test):
    logger.info("Training %s ...", name)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=SVM_SUBSAMPLE, random_state=RANDOM_SEED)
    idx = next(sss.split(X_tr_sc, y_train))[0]

    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
              random_state=RANDOM_SEED, cache_size=2000)
    t0 = time.time()
    svm.fit(X_tr_sc[idx], y_train[idx])
    logger.info("%s trained in %.1f s (subsample %d)", name, time.time() - t0, SVM_SUBSAMPLE)

    y_pred = svm.predict(X_te_sc)
    y_score = svm.decision_function(X_te_sc)
    metrics = compute_metrics(y_test, y_pred)
    metrics["AUC (proba)"] = round(roc_auc_score(y_test, y_score), 4)
    return metrics, y_score, scaler, svm


def train_sklearn_model(name, clf, X_train, y_train, X_test, y_test):
    """Train sklearn-compatible classifier (RF, GBT, XGB) with all features."""
    logger.info("Training %s ...", name)
    t0 = time.time()
    clf.fit(X_train, y_train)
    logger.info("%s trained in %.1f s", name, time.time() - t0)
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    else:
        y_score = clf.decision_function(X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics["AUC (proba)"] = round(roc_auc_score(y_test, y_score), 4)
    return metrics, y_score, clf


def correlated_feature_removal(X_train, X_test):
    """Remove features with >0.8 correlation, randomly choose which to drop."""
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
    keep_cols = [c for c in corr.columns if c not in drop]
    logger.info("Corr > %.1f: %d removed, %d kept", CORR_THRESHOLD, len(drop_cols), len(keep_cols))
    X_tr_r = pd.DataFrame(X_train).drop(columns=drop_cols).values
    X_te_r = pd.DataFrame(X_test).drop(columns=drop_cols).values
    return X_tr_r, X_te_r, drop_cols, keep_cols


def plot_roc(all_results, y_test_dict, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.tab10
    for i, (name, m) in enumerate(all_results.items()):
        score = m.get("_score")
        if score is None or name not in y_test_dict:
            continue
        y_test = y_test_dict[name]
        fpr, tpr, _ = roc_curve(y_test, score)
        ax.plot(fpr, tpr, color=cmap(i), lw=2,
                label=f"{name} (AUC = {roc_auc_score(y_test, score):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Split Types", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("ROC saved: %s", out_path)


def plot_bar(df, out_path, title="Model Performance Comparison"):
    metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1-score",
               "AUC (predict)"]
    metric_labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1-score",
                     "AUC"]
    models = df["Model"].values
    n_metrics = len(metrics)
    n_models = len(models)
    x = np.arange(n_metrics)
    w = 0.85 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = df[df["Model"] == model][metrics].values[0]
        bars = ax.bar(x + i * w, vals, w, label=model, color=color, edgecolor="grey", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=45)
    ax.set_ylabel("Score"); ax.set_title(title, fontsize=13)
    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim([0, 1.08]); ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Bar chart saved: %s", out_path)


def main():
    print("=" * 70)
    print("  TCR-H Final — All Models")
    print("  Baseline: RF, GBT, XGB, SVM-RBF (all 194 features)")
    print("  TCR-H:    TCR-Hβ, TCR-HβE, TCR-RS, TCR-HE (uncorrelated features)")
    print("=" * 70)

    all_results = {}
    all_y_test = {}
    all_y_score = []

    # ── Load epi-hard data (used by TCR-HE and as base for strict) ──
    train_ep = pd.read_csv(EPI_HARD_TRAIN, low_memory=False)
    test_ep  = pd.read_csv(EPI_HARD_TEST, low_memory=False)
    feat_cols = [c for c in train_ep.columns if c.startswith(("cdr3_", "epitope_"))]

    # ── Baseline models on epi-hard split (all 194 features) ──
    X_tr_base = train_ep[feat_cols].fillna(0.0).astype(float)
    y_tr_base = train_ep["label"].astype(int).values
    X_te_base = test_ep[feat_cols].fillna(0.0).astype(float)
    y_te_base = test_ep["label"].astype(int).values

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    m_rf, score_rf, _ = train_sklearn_model("RF", rf_clf, X_tr_base, y_tr_base, X_te_base, y_te_base)
    m_rf["_score"] = score_rf; all_results["RF"] = m_rf; all_y_test["RF"] = y_te_base

    gbt_clf = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
    m_gbt, score_gbt, _ = train_sklearn_model("GBT", gbt_clf, X_tr_base, y_tr_base, X_te_base, y_te_base)
    m_gbt["_score"] = score_gbt; all_results["GBT"] = m_gbt; all_y_test["GBT"] = y_te_base

    xgb_clf = XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0)
    m_xgb, score_xgb, _ = train_sklearn_model("XGB", xgb_clf, X_tr_base, y_tr_base, X_te_base, y_te_base)
    m_xgb["_score"] = score_xgb; all_results["XGB"] = m_xgb; all_y_test["XGB"] = y_te_base

    m_svm_base, score_svm_base, scaler_svm_base, svm_svm_base = train_svm("SVM-RBF", X_tr_base, y_tr_base, X_te_base, y_te_base)
    m_svm_base["_score"] = score_svm_base
    all_results["SVM-RBF"] = m_svm_base
    all_y_test["SVM-RBF"] = y_te_base
    _save(svm_svm_base, scaler_svm_base, "svm_rbf_baseline")

    # ── 1. TCR-HE (epitope hard + feature removal) ──
    logger.info("--- TCR-HE ---")
    X_tr = X_tr_base.copy()
    y_tr = y_tr_base.copy()
    X_te = X_te_base.copy()
    y_te = y_te_base.copy()

    X_tr_r, X_te_r, drop_cols, keep_cols = correlated_feature_removal(X_tr, X_te)
    with open(os.path.join(RESULTS_DIR, "tcr_he_features_removed.json"), "w") as f:
        json.dump({"n_removed": len(drop_cols), "n_kept": len(keep_cols),
                    "removed": drop_cols, "kept": keep_cols}, f, indent=2)

    m_he, score_he, scaler_he, svm_he = train_svm("TCR-HE", X_tr_r, y_tr, X_te_r, y_te)
    m_he["_score"] = score_he
    all_results["TCR-HE"] = m_he
    all_y_test["TCR-HE"] = y_te
    _save(svm_he, scaler_he, "tcr_he")
    shap_analysis("TCR-HE", svm_he, scaler_he, X_tr_r, X_te_r, y_te, keep_cols, RESULTS_DIR)

    # ── Full dataset (for TCR hard & random splits) ──
    full = pd.concat([train_ep, test_ep], ignore_index=True)
    X_all = full[feat_cols].fillna(0.0).astype(float)
    y_all = full["label"].astype(int).values
    cdr3_all = full["CDR3.beta"]
    epi_all = full["antigen_epitope"]

    # ── 2. TCR-Hβ (TCR hard split) ──
    logger.info("--- TCR-Hβ (TCR hard split) ---")
    from src.data.splits import tcr_hard_split
    train_idx, _, info = tcr_hard_split(X_all, pd.Series(y_all), cdr3_all,
                                         test_size=0.2, random_seed=RANDOM_SEED)
    test_idx = np.setdiff1d(np.arange(len(X_all)), train_idx)
    logger.info("TCR-Hβ split: train=%d test=%d", len(train_idx), len(test_idx))
    X_tr_tcr = X_all.iloc[train_idx]
    y_tr_tcr = y_all[train_idx]
    X_te_tcr = X_all.iloc[test_idx]
    y_te_tcr = y_all[test_idx]
    X_tr_tcr_r, X_te_tcr_r, _, keep_cols_hb = correlated_feature_removal(X_tr_tcr, X_te_tcr)
    m_hb, score_hb, scaler_hb, svm_hb = train_svm("TCR-Hβ", X_tr_tcr_r, y_tr_tcr, X_te_tcr_r, y_te_tcr)
    m_hb["_score"] = score_hb
    all_results["TCR-Hβ"] = m_hb
    all_y_test["TCR-Hβ"] = y_te_tcr
    _save(svm_hb, scaler_hb, "tcr_hb")
    shap_analysis("TCR-Hβ", svm_hb, scaler_hb, X_tr_tcr_r, X_te_tcr_r, y_te_tcr, keep_cols_hb, RESULTS_DIR)

    # ── 3. TCR-HβE (strict split) ──
    logger.info("--- TCR-HβE (strict split) ---")
    # Same epitope assignment as epitope hard, remove CDR3 overlap from TRAIN only
    test_cdr3_set = set(test_ep["CDR3.beta"].unique())
    train_mask = ~train_ep["CDR3.beta"].isin(test_cdr3_set)
    X_tr_s = train_ep.loc[train_mask, feat_cols].fillna(0.0).astype(float)
    y_tr_s = train_ep.loc[train_mask, "label"].astype(int).values
    # Test unchanged (same as epi hard test)
    X_te_s = X_te.copy()
    y_te_s = y_te.copy()
    logger.info("Strict train: %d (removed %d from %d), test: %d (unchanged)",
                len(X_tr_s), (~train_mask).sum(), len(train_ep), len(X_te_s))
    X_tr_s_r, X_te_s_r, _, keep_cols_hbe = correlated_feature_removal(X_tr_s, X_te_s)
    m_hbe, score_hbe, scaler_hbe, svm_hbe = train_svm("TCR-HβE", X_tr_s_r, y_tr_s, X_te_s_r, y_te_s)
    m_hbe["_score"] = score_hbe
    all_results["TCR-HβE"] = m_hbe
    all_y_test["TCR-HβE"] = y_te_s
    _save(svm_hbe, scaler_hbe, "tcr_hbe")
    shap_analysis("TCR-HβE", svm_hbe, scaler_hbe, X_tr_s_r, X_te_s_r, y_te_s, keep_cols_hbe, RESULTS_DIR)

    # ── 4. TCR-RS (random split) ──
    logger.info("--- TCR-RS (random split) ---")
    from src.data.splits import random_split
    train_idx_r, _, info = random_split(X_all, pd.Series(y_all), test_size=0.2, random_seed=RANDOM_SEED)
    test_idx_r = np.setdiff1d(np.arange(len(X_all)), train_idx_r)
    logger.info("Random split: train=%d test=%d", len(train_idx_r), len(test_idx_r))
    X_tr_rs = X_all.iloc[train_idx_r]
    y_tr_rs = y_all[train_idx_r]
    X_te_rs = X_all.iloc[test_idx_r]
    y_te_rs = y_all[test_idx_r]
    X_tr_rs_r, X_te_rs_r, _, keep_cols_rs = correlated_feature_removal(X_tr_rs, X_te_rs)
    m_rs, score_rs, scaler_rs, svm_rs = train_svm("TCR-RS", X_tr_rs_r, y_tr_rs, X_te_rs_r, y_te_rs)
    m_rs["_score"] = score_rs
    all_results["TCR-RS"] = m_rs
    all_y_test["TCR-RS"] = y_te_rs
    _save(svm_rs, scaler_rs, "tcr_rs")
    shap_analysis("TCR-RS", svm_rs, scaler_rs, X_tr_rs_r, X_te_rs_r, y_te_rs, keep_cols_rs, RESULTS_DIR)

    # ── Build table ──
    order = ["RF", "GBT", "XGB", "SVM-RBF", "TCR-Hβ", "TCR-HβE", "TCR-RS", "TCR-HE"]
    cols_show = ["Model", "AUC (predict)", "TP", "TN", "FP", "FN",
                  "Accuracy", "Precision", "Recall", "Specificity", "F1-score"]
    rows = []
    for name in order:
        m = all_results[name]
        row = {"Model": name}
        for c in cols_show[1:]:
            row[c] = m.get(c, "")
        rows.append(row)
    df = pd.DataFrame(rows)

    print("\n" + "=" * 130)
    print("  RESULTS — All Models")
    print("=" * 130)
    print(df.to_string(index=False))
    print("=" * 130)

    df.to_csv(os.path.join(RESULTS_DIR, "results_table.csv"), index=False)

    # ── Paper comparison ──
    paper = {
        "RF":      {"AUC (predict)": 0.50},
        "GBT":     {"AUC (predict)": 0.54},
        "XGB":     {"AUC (predict)": 0.51},
        "SVM-RBF": {"AUC (predict)": 0.80},
        "TCR-Hβ":  {"AUC (predict)": 0.92, "TP": 18760, "TN": 28780, "FP": 766, "FN": 2994,
                     "Accuracy": 0.93, "Precision": 0.96, "Recall": 0.86, "Specificity": 0.97, "F1-score": 0.91},
        "TCR-HβE": {"AUC (predict)": 0.89, "TP": 29570, "TN": 19143, "FP": 3328, "FN": 2393,
                     "Accuracy": 0.89, "Precision": 0.898, "Recall": 0.92, "Specificity": 0.85, "F1-score": 0.91},
        "TCR-RS":  {"AUC (predict)": 0.92, "TP": 18382, "TN": 28771, "FP": 691, "FN": 3045,
                     "Accuracy": 0.93, "Precision": 0.96, "Recall": 0.86, "Specificity": 0.98, "F1-score": 0.91},
        "TCR-HE":  {"AUC (predict)": 0.87, "TP": 29567, "TN": 18297, "FP": 4174, "FN": 2396,
                     "Accuracy": 0.879, "Precision": 0.876, "Recall": 0.925, "Specificity": 0.814, "F1-score": 0.900},
    }
    print("\n\n" + "=" * 130)
    print("  PAPER vs OURS")
    print("=" * 130)
    print(f"{'Model':<10} {'Source':<6} {'AUC':<8} {'TP':<7} {'TN':<7} {'FP':<7} {'FN':<7} "
          f"{'Acc':<7} {'Prec':<7} {'Recall':<7} {'Spec':<7} {'F1':<7}")
    print("-" * 130)
    for name in order:
        p = paper.get(name, {})
        o = all_results[name]
        for src, m in [("Paper", p), ("Ours", o)]:
            print(f"{name:<10} {src:<6} {m.get('AUC (predict)', ''):<8} {m.get('TP', ''):<7} "
                  f"{m.get('TN', ''):<7} {m.get('FP', ''):<7} {m.get('FN', ''):<7} "
                  f"{m.get('Accuracy', ''):<7} {m.get('Precision', ''):<7} "
                  f"{m.get('Recall', ''):<7} {m.get('Specificity', ''):<7} {m.get('F1-score', ''):<7}")
    print("=" * 130)

    # ── Plots ──
    tcr_models = ["TCR-Hβ", "TCR-HβE", "TCR-RS", "TCR-HE"]
    base_models = ["RF", "GBT", "XGB", "SVM-RBF"]
    plot_bar(df[df["Model"].isin(tcr_models)], os.path.join(RESULTS_DIR, "performance_bar_chart_tcr.png"),
             title="TCR-H Models Performance")
    plot_bar(df[df["Model"].isin(base_models)], os.path.join(RESULTS_DIR, "performance_bar_chart_baseline.png"),
             title="Baseline Models Performance")
    # ROC needs scores — rebuild with clean score dict
    plot_data = {}
    for name in order:
        m = all_results[name]
        if "_score" in m:
            plot_data[name] = m
    roc_scores = {name: all_results[name]["_score"] for name in order if "_score" in all_results[name]}
    # Simple ROC
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_roc = plt.cm.tab10
    for i, name in enumerate(order):
        if name not in roc_scores:
            continue
        yt = all_y_test[name]
        fpr, tpr, _ = roc_curve(yt, roc_scores[name])
        ax.plot(fpr, tpr, color=cmap_roc(i), lw=2,
                label=f"{name} (AUC = {roc_auc_score(yt, roc_scores[name]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves"); ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Split info ──
    print("\nSplit sizes:")
    splits_info = {
        "TCR-Hβ": (len(X_tr_tcr), len(X_te_tcr), int(y_tr_tcr.sum()), int((1-y_tr_tcr).sum()), int(y_te_tcr.sum()), int((1-y_te_tcr).sum())),
        "TCR-HβE": (len(X_tr_s), len(X_te_s), int(y_tr_s.sum()), int(len(X_tr_s)-y_tr_s.sum()), int(y_te_s.sum()), int(len(X_te_s)-y_te_s.sum())),
        "TCR-RS": (len(X_tr_rs), len(X_te_rs), int(y_tr_rs.sum()), int(len(X_tr_rs)-y_tr_rs.sum()), int(y_te_rs.sum()), int(len(X_te_rs)-y_te_rs.sum())),
        "TCR-HE": (len(X_tr), len(X_te), int(y_tr.sum()), int(len(X_tr)-y_tr.sum()), int(y_te.sum()), int(len(X_te)-y_te.sum())),
    }
    for name, (tr, te, tr_p, tr_n, te_p, te_n) in splits_info.items():
        print(f"  {name:>8}: train={tr} (pos={tr_p} neg={tr_n}), test={te} (pos={te_p} neg={te_n})")

    print("\nDone. Results saved to", RESULTS_DIR)


def _save(model, scaler, name):
    with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)


def shap_analysis(model_name, svm, scaler, X_train, X_test, y_test, feature_names, out_dir):
    """SHAP KernelExplainer for SVM-RBF model."""
    logger.info("SHAP analysis for %s ...", model_name)
    rng = np.random.default_rng(RANDOM_SEED)

    X_tr_sc = scaler.transform(X_train)
    X_te_sc = scaler.transform(X_test)

    n_bg = min(30, len(X_tr_sc))
    bg_idx = rng.choice(len(X_tr_sc), size=n_bg, replace=False)
    background = X_tr_sc[bg_idx]

    n_explain = min(100, len(X_te_sc))
    te_idx = rng.choice(len(X_te_sc), size=n_explain, replace=False)
    X_explain = X_te_sc[te_idx]
    y_explain = y_test[te_idx]

    explainer = shap.KernelExplainer(svm.decision_function, background)
    shap_values = explainer.shap_values(X_explain, nsamples=50, l1_reg=False)

    np.save(os.path.join(out_dir, f"{model_name}_shap_values.npy"), shap_values)
    np.save(os.path.join(out_dir, f"{model_name}_shap_data.npy"), X_explain)

    X_display = pd.DataFrame(X_explain, columns=feature_names)

    shap.summary_plot(shap_values, X_display, show=False)
    plt.savefig(os.path.join(out_dir, f"{model_name}_shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP beeswarm saved for %s", model_name)

    shap.summary_plot(shap_values, X_display, plot_type="bar", show=False)
    plt.savefig(os.path.join(out_dir, f"{model_name}_shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP bar saved for %s", model_name)


if __name__ == "__main__":
    main()

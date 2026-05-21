"""
Final training: all models and splits — matching paper methodology.

Key changes vs earlier version (to match paper):
  1. SVM: FULL training data (no 30K subsample)
  2. SVM: GridSearchCV(cv=None) = 5-fold CV → refit on full data
  3. SVM: SVC(probability=True)
  4. Correlation removal: deterministic (keep lower-index feature)
  5. SHAP: KernelExplainer(model.predict_proba, link='logit')

Output -> results/final_hpc/
  - results_table.csv, performance_bar_chart_*.png, roc_curves.png
  - models/*.pkl
  - *_shap_*.png
"""

import json, logging, os, pickle, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
    make_scorer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
logging.getLogger("shap").setLevel(logging.WARNING)

RESULTS_DIR = "results/final_hpc"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED = 42
CORR_THRESHOLD = 0.8

EPI_HARD_TRAIN = "data/splits_tchard/epitope_hard_train.csv"
EPI_HARD_TEST  = "data/splits_tchard/epitope_hard_test.csv"

# Paper's scorer: AUC from class predictions (matches paper's evaluate-with-predict)
roc_scorer = make_scorer(roc_auc_score)


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


def train_svm_paper(name, X_train, y_train, X_test, y_test):
    """
    Train SVM exactly as the paper does:
      - FULL training data (no subsampling)
      - GridSearchCV with cv=None (default 5-fold CV) + refit on full data
      - SVC(probability=True)
      - ROC scorer for CV selection
      - Final evaluation with predict() (class labels)
    """
    logger.info("Training %s on FULL data (%d samples) with GridSearchCV ...",
                name, len(X_train))
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    parameters = {
        'C': [1.0],
        'gamma': ['scale'],
        'kernel': ['rbf'],
        'class_weight': ['balanced'],
    }
    classifier = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_SEED),
        parameters,
        cv=None,            # paper uses cv=None (default 5-fold in sklearn 1.x)
        scoring=roc_scorer,  # paper uses AUC scorer for CV
        n_jobs=-1,         # 全核并行（~500GB内存，随便跑）
    )

    t0 = time.time()
    classifier.fit(X_tr_sc, y_train)
    elapsed = time.time() - t0
    logger.info("%s trained in %.1f s | CV score = %.4f | best_params = %s",
                name, elapsed, classifier.best_score_, classifier.best_params_)

    # Paper evaluates with predict() → class labels → AUC
    y_pred = classifier.predict(X_te_sc)
    y_score = classifier.decision_function(X_te_sc)
    metrics = compute_metrics(y_test, y_pred)
    metrics["AUC (proba)"] = round(roc_auc_score(y_test, y_score), 4)
    # Store the fitted GridSearchCV for SHAP
    metrics["_classifier"] = classifier
    metrics["_scaler"] = scaler
    return metrics, y_score, scaler, classifier


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


def correlated_feature_removal_deterministic(X_train, X_test):
    """
    Remove features with >0.8 correlation — DETERMINISTIC (paper's method).

    Paper's correlation() function:
      for i in range(len(columns)):
        for j in range(i):
          if corr[i,j] > threshold:
            drop columns[i]   # higher index → keeps lower-index feature
    """
    rng = np.random.default_rng(RANDOM_SEED)
    corr = pd.DataFrame(X_train).corr()
    n = len(corr.columns)
    drop = set()
    for i in range(n):
        for j in range(i):
            if abs(corr.iloc[i, j]) > CORR_THRESHOLD:
                # Paper: keep lower-index (j), drop higher-index (i)
                drop.add(corr.columns[i])
    drop_cols = sorted(drop)
    keep_cols = [c for c in corr.columns if c not in drop]
    logger.info("Corr > %.1f (deterministic): %d removed, %d kept",
                CORR_THRESHOLD, len(drop_cols), len(keep_cols))
    X_tr_r = pd.DataFrame(X_train).drop(columns=drop_cols).values
    X_te_r = pd.DataFrame(X_test).drop(columns=drop_cols).values
    return X_tr_r, X_te_r, drop_cols, keep_cols


def shap_analysis_paper(model_name, classifier, scaler,
                        X_train, X_test, y_test, feature_names, out_dir):
    """
    SHAP analysis matching paper:
      - KernelExplainer(model.predict_proba, data=background, link='logit')
      - Paper uses ALL training data as background (infeasible for large data)
      - We use a random subset (n_background samples) as approximation
      - Explain a random subset of test samples (n_explain)
    """
    logger.info("SHAP analysis for %s (paper method: KernelExplainer + link='logit') ...",
                model_name)

    X_tr_sc = scaler.transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # Use subset for practicality (paper uses full training data)
    rng = np.random.default_rng(RANDOM_SEED)
    n_bg = min(200, len(X_tr_sc))
    bg_idx = rng.choice(len(X_tr_sc), size=n_bg, replace=False)
    background = X_tr_sc[bg_idx]

    n_explain = min(100, len(X_te_sc))
    te_idx = rng.choice(len(X_te_sc), size=n_explain, replace=False)
    X_explain = X_te_sc[te_idx]
    y_explain = y_test[te_idx]

    # Paper's KernelExplainer with predict_proba + link='logit'
    model = classifier.predict_proba
    explainer = shap.KernelExplainer(model, background, link='logit')
    shap_values = explainer.shap_values(X_explain, nsamples=100, l1_reg=False)

    np.save(os.path.join(out_dir, f"{model_name}_shap_values.npy"), shap_values)
    np.save(os.path.join(out_dir, f"{model_name}_shap_data.npy"), X_explain)

    X_display = pd.DataFrame(X_explain, columns=feature_names)

    shap.summary_plot(shap_values, X_display, max_display=50, show=False)
    plt.savefig(os.path.join(out_dir, f"{model_name}_shap_beeswarm.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_display, plot_type="bar", max_display=50, show=False)
    plt.savefig(os.path.join(out_dir, f"{model_name}_shap_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP plots saved for %s", model_name)


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
        bars = ax.bar(x + i * w, vals, w, label=model, color=color,
                      edgecolor="grey", linewidth=0.5)
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
    print("  TCR-H Final — All Models (Paper Methodology)")
    print("  SVM: full data + GridSearchCV(cv=None) + probability=True")
    print("  Correlation removal: deterministic (keep lower-index)")
    print("  SHAP: KernelExplainer + link='logit'")
    print("=" * 70)

    all_results = {}
    all_y_test = {}
    all_y_score = []

    # ── Load epi-hard data ──
    train_ep = pd.read_csv(EPI_HARD_TRAIN, low_memory=False)
    test_ep  = pd.read_csv(EPI_HARD_TEST, low_memory=False)
    feat_cols = [c for c in train_ep.columns if c.startswith(("cdr3_", "epitope_"))]

    # ── Baseline models on epi-hard split (all 194 features) ──
    X_tr_base = train_ep[feat_cols].fillna(0.0).astype(float)
    y_tr_base = train_ep["label"].astype(int).values
    X_te_base = test_ep[feat_cols].fillna(0.0).astype(float)
    y_te_base = test_ep["label"].astype(int).values

    # RF (full data, no change needed)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    m_rf, score_rf, _ = train_sklearn_model("RF", rf_clf, X_tr_base, y_tr_base, X_te_base, y_te_base)
    m_rf["_score"] = score_rf; all_results["RF"] = m_rf; all_y_test["RF"] = y_te_base

    # GBT (full data, no change needed)
    gbt_clf = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
    m_gbt, score_gbt, _ = train_sklearn_model("GBT", gbt_clf, X_tr_base, y_tr_base, X_te_base, y_te_base)
    m_gbt["_score"] = score_gbt; all_results["GBT"] = m_gbt; all_y_test["GBT"] = y_te_base

    # XGB (full data, no change needed)
    xgb_clf = XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0)
    m_xgb, score_xgb, _ = train_sklearn_model("XGB", xgb_clf, X_tr_base, y_tr_base, X_te_base, y_te_base)
    m_xgb["_score"] = score_xgb; all_results["XGB"] = m_xgb; all_y_test["XGB"] = y_te_base

    # SVM-RBF baseline (CHANGED: full data + GridSearchCV + probability=True)
    m_svm_base, score_svm_base, scaler_svm_base, clf_svm_base = \
        train_svm_paper("SVM-RBF", X_tr_base, y_tr_base, X_te_base, y_te_base)
    m_svm_base["_score"] = score_svm_base
    m_svm_base["_clf"] = clf_svm_base
    m_svm_base["_scaler"] = scaler_svm_base
    all_results["SVM-RBF"] = m_svm_base
    all_y_test["SVM-RBF"] = y_te_base
    _save(clf_svm_base, scaler_svm_base, "svm_rbf_baseline")

    # ── 1. TCR-HE (epitope hard + deterministic feature removal) ──
    logger.info("--- TCR-HE ---")
    X_tr_he, X_te_he, drop_cols_he, keep_cols_he = \
        correlated_feature_removal_deterministic(X_tr_base, X_te_base)
    with open(os.path.join(RESULTS_DIR, "tcr_he_features_removed.json"), "w") as f:
        json.dump({"n_removed": len(drop_cols_he), "n_kept": len(keep_cols_he),
                    "removed": drop_cols_he, "kept": keep_cols_he}, f, indent=2)

    m_he, score_he, scaler_he, clf_he = \
        train_svm_paper("TCR-HE", X_tr_he, y_tr_base, X_te_he, y_te_base)
    m_he["_score"] = score_he
    m_he["_clf"] = clf_he
    m_he["_scaler"] = scaler_he
    all_results["TCR-HE"] = m_he
    all_y_test["TCR-HE"] = y_te_base
    _save(clf_he, scaler_he, "tcr_he")
    shap_analysis_paper("TCR-HE", clf_he, scaler_he,
                        X_tr_he, X_te_he, y_te_base, keep_cols_he, RESULTS_DIR)

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
    X_tr_tcr_r, X_te_tcr_r, _, keep_cols_hb = \
        correlated_feature_removal_deterministic(X_tr_tcr, X_te_tcr)
    m_hb, score_hb, scaler_hb, clf_hb = \
        train_svm_paper("TCR-Hβ", X_tr_tcr_r, y_tr_tcr, X_te_tcr_r, y_te_tcr)
    m_hb["_score"] = score_hb
    m_hb["_clf"] = clf_hb
    m_hb["_scaler"] = scaler_hb
    all_results["TCR-Hβ"] = m_hb
    all_y_test["TCR-Hβ"] = y_te_tcr
    _save(clf_hb, scaler_hb, "tcr_hb")
    shap_analysis_paper("TCR-Hβ", clf_hb, scaler_hb,
                        X_tr_tcr_r, X_te_tcr_r, y_te_tcr, keep_cols_hb, RESULTS_DIR)

    # ── 3. TCR-HβE (strict split) ──
    logger.info("--- TCR-HβE (strict split) ---")
    test_cdr3_set = set(test_ep["CDR3.beta"].unique())
    train_mask = ~train_ep["CDR3.beta"].isin(test_cdr3_set)
    X_tr_s = train_ep.loc[train_mask, feat_cols].fillna(0.0).astype(float)
    y_tr_s = train_ep.loc[train_mask, "label"].astype(int).values
    X_te_s = X_te_base.copy()
    y_te_s = y_te_base.copy()
    logger.info("Strict train: %d (removed %d from %d), test: %d (unchanged)",
                len(X_tr_s), (~train_mask).sum(), len(train_ep), len(X_te_s))
    X_tr_s_r, X_te_s_r, _, keep_cols_hbe = \
        correlated_feature_removal_deterministic(X_tr_s, X_te_s)
    m_hbe, score_hbe, scaler_hbe, clf_hbe = \
        train_svm_paper("TCR-HβE", X_tr_s_r, y_tr_s, X_te_s_r, y_te_s)
    m_hbe["_score"] = score_hbe
    m_hbe["_clf"] = clf_hbe
    m_hbe["_scaler"] = scaler_hbe
    all_results["TCR-HβE"] = m_hbe
    all_y_test["TCR-HβE"] = y_te_s
    _save(clf_hbe, scaler_hbe, "tcr_hbe")
    shap_analysis_paper("TCR-HβE", clf_hbe, scaler_hbe,
                        X_tr_s_r, X_te_s_r, y_te_s, keep_cols_hbe, RESULTS_DIR)

    # ── 4. TCR-RS (random split, matching paper's random_state=1) ──
    logger.info("--- TCR-RS (random split) ---")
    from sklearn.model_selection import train_test_split
    # Paper uses random_state=1 for random split
    X_tr_rs, X_te_rs, y_tr_rs, y_te_rs = train_test_split(
        X_all, y_all, test_size=0.2, random_state=1, stratify=y_all)
    logger.info("Random split: train=%d test=%d", len(X_tr_rs), len(X_te_rs))
    X_tr_rs_r, X_te_rs_r, _, keep_cols_rs = \
        correlated_feature_removal_deterministic(X_tr_rs, X_te_rs)
    m_rs, score_rs, scaler_rs, clf_rs = \
        train_svm_paper("TCR-RS", X_tr_rs_r, y_tr_rs, X_te_rs_r, y_te_rs)
    m_rs["_score"] = score_rs
    m_rs["_clf"] = clf_rs
    m_rs["_scaler"] = scaler_rs
    all_results["TCR-RS"] = m_rs
    all_y_test["TCR-RS"] = y_te_rs
    _save(clf_rs, scaler_rs, "tcr_rs")
    shap_analysis_paper("TCR-RS", clf_rs, scaler_rs,
                        X_tr_rs_r, X_te_rs_r, y_te_rs, keep_cols_rs, RESULTS_DIR)

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
    print("  RESULTS — All Models (Paper Methodology)")
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
    plot_bar(df[df["Model"].isin(tcr_models)],
             os.path.join(RESULTS_DIR, "performance_bar_chart_tcr.png"),
             title="TCR-H Models Performance")
    plot_bar(df[df["Model"].isin(base_models)],
             os.path.join(RESULTS_DIR, "performance_bar_chart_baseline.png"),
             title="Baseline Models Performance")

    # ROC curves
    roc_scores = {name: all_results[name]["_score"] for name in order
                  if "_score" in all_results[name]}
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
        "TCR-HE": (len(X_tr_base), len(X_te_base), int(y_tr_base.sum()), int(len(X_tr_base)-y_tr_base.sum()), int(y_te_base.sum()), int(len(X_te_base)-y_te_base.sum())),
    }
    for name, (tr, te, tr_p, tr_n, te_p, te_n) in splits_info.items():
        print(f"  {name:>8}: train={tr} (pos={tr_p} neg={tr_n}), test={te} (pos={te_p} neg={te_n})")

    print("\nDone. Results saved to", RESULTS_DIR)


def _save(model, scaler, name):
    with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)


if __name__ == "__main__":
    main()

"""
TCR-H: run a single SVM model (for SLURM array job).
Usage: python scripts/run_one_model.py <model_name>
  model_name: svm_rbf | tcr_he | tcr_hb | tcr_hbe | tcr_rs
"""
import json, logging, os, pickle, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, make_scorer,
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


def train_svm(name, X_train, y_train, X_test, y_test):
    logger.info("%s: training on %d samples (GridSearchCV + probability=True) ...",
                name, len(X_train))
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    params = {'C': [1.0], 'gamma': ['scale'], 'kernel': ['rbf'], 'class_weight': ['balanced']}
    clf = GridSearchCV(SVC(probability=True, random_state=RANDOM_SEED), params,
                       cv=None, scoring=roc_scorer, n_jobs=-1)
    t0 = time.time()
    clf.fit(X_tr_sc, y_train)
    logger.info("%s done in %.1f s | CV=%.4f", name, time.time()-t0, clf.best_score_)
    y_pred = clf.predict(X_te_sc)
    y_score = clf.decision_function(X_te_sc)
    metrics = compute_metrics(y_test, y_pred)
    metrics["AUC (proba)"] = round(roc_auc_score(y_test, y_score), 4)
    return metrics, clf, scaler


def corr_remove(X_train, X_test):
    corr = pd.DataFrame(X_train).corr()
    drop = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > CORR_THRESHOLD:
                drop.add(corr.columns[i])
    drop_cols = sorted(drop)
    keep = [c for c in corr.columns if c not in drop]
    logger.info("Corr>0.8: %d removed, %d kept", len(drop_cols), len(keep))
    X_tr_r = pd.DataFrame(X_train).drop(columns=drop_cols).values
    X_te_r = pd.DataFrame(X_test).drop(columns=drop_cols).values
    return X_tr_r, X_te_r, keep


def do_shap(name, clf, scaler, X_train, X_test, y_test, feat_names):
    logger.info("SHAP for %s ...", name)
    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)
    rng = np.random.default_rng(RANDOM_SEED)
    bg = X_tr_s[rng.choice(len(X_tr_s), 200, replace=False)]
    te = X_te_s[rng.choice(len(X_te_s), 100, replace=False)]
    explainer = shap.KernelExplainer(clf.predict_proba, bg, link='logit')
    sv = explainer.shap_values(te, nsamples=100, l1_reg=False)
    np.save(os.path.join(RESULTS_DIR, f"{name}_shap_values.npy"), sv)
    np.save(os.path.join(RESULTS_DIR, f"{name}_shap_data.npy"), te)
    Xdisp = pd.DataFrame(te, columns=feat_names)
    shap.summary_plot(sv, Xdisp, max_display=50, show=False)
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()
    shap.summary_plot(sv, Xdisp, plot_type="bar", max_display=50, show=False)
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "all"
    import matplotlib.pyplot as plt

    train_ep = pd.read_csv(EPI_HARD_TRAIN, low_memory=False)
    test_ep  = pd.read_csv(EPI_HARD_TEST, low_memory=False)
    feat_cols = [c for c in train_ep.columns if c.startswith(("cdr3_", "epitope_"))]
    X_tr_b = train_ep[feat_cols].fillna(0.0).astype(float)
    y_tr_b = train_ep["label"].astype(int).values
    X_te_b = test_ep[feat_cols].fillna(0.0).astype(float)
    y_te_b = test_ep["label"].astype(int).values

    results = {}

    # ── 1. SVM-RBF (all features) ──
    if model in ("all", "svm_rbf"):
        m, c, s = train_svm("SVM-RBF", X_tr_b, y_tr_b, X_te_b, y_te_b)
        results["SVM-RBF"] = m
        with open(os.path.join(MODELS_DIR, "svm_rbf.pkl"), "wb") as f:
            pickle.dump({"model": c, "scaler": s}, f)
        print(f"SVM-RBF: AUC={m['AUC (predict)']}")

    # ── 2. TCR-HE ──
    if model in ("all", "tcr_he"):
        X_tr_r, X_te_r, keep = corr_remove(X_tr_b, X_te_b)
        m, c, s = train_svm("TCR-HE", X_tr_r, y_tr_b, X_te_r, y_te_b)
        results["TCR-HE"] = m
        with open(os.path.join(MODELS_DIR, "tcr_he.pkl"), "wb") as f:
            pickle.dump({"model": c, "scaler": s}, f)
        with open(os.path.join(RESULTS_DIR, "tcr_he_features_removed.json"), "w") as f:
            json.dump({"keep": keep}, f)
        do_shap("TCR-HE", c, s, X_tr_r, X_te_r, y_te_b, keep)
        print(f"TCR-HE: AUC={m['AUC (predict)']}")

    # ── 3. TCR-Hβ ──
    if model in ("all", "tcr_hb"):
        from src.data.splits import tcr_hard_split
        full = pd.concat([train_ep, test_ep], ignore_index=True)
        X_a = full[feat_cols].fillna(0.0).astype(float); y_a = full["label"].astype(int).values
        ti, _, _ = tcr_hard_split(X_a, pd.Series(y_a), full["CDR3.beta"], 0.2, RANDOM_SEED)
        tsti = np.setdiff1d(np.arange(len(X_a)), ti)
        X_tr_t, y_tr_t = X_a.iloc[ti], y_a[ti]
        X_te_t, y_te_t = X_a.iloc[tsti], y_a[tsti]
        X_tr_r, X_te_r, keep = corr_remove(X_tr_t, X_te_t)
        m, c, s = train_svm("TCR-Hβ", X_tr_r, y_tr_t, X_te_r, y_te_t)
        results["TCR-Hβ"] = m
        with open(os.path.join(MODELS_DIR, "tcr_hb.pkl"), "wb") as f:
            pickle.dump({"model": c, "scaler": s}, f)
        do_shap("TCR-Hβ", c, s, X_tr_r, X_te_r, y_te_t, keep)
        print(f"TCR-Hβ: AUC={m['AUC (predict)']}")

    # ── 4. TCR-HβE ──
    if model in ("all", "tcr_hbe"):
        test_cdr3 = set(test_ep["CDR3.beta"].unique())
        mask = ~train_ep["CDR3.beta"].isin(test_cdr3)
        X_tr_s = train_ep.loc[mask, feat_cols].fillna(0.0).astype(float)
        y_tr_s = train_ep.loc[mask, "label"].astype(int).values
        X_tr_r, X_te_r, keep = corr_remove(X_tr_s, X_te_b)
        m, c, s = train_svm("TCR-HβE", X_tr_r, y_tr_s, X_te_r, y_te_b)
        results["TCR-HβE"] = m
        with open(os.path.join(MODELS_DIR, "tcr_hbe.pkl"), "wb") as f:
            pickle.dump({"model": c, "scaler": s}, f)
        do_shap("TCR-HβE", c, s, X_tr_r, X_te_r, y_te_b, keep)
        print(f"TCR-HβE: AUC={m['AUC (predict)']}")

    # ── 5. TCR-RS ──
    if model in ("all", "tcr_rs"):
        from sklearn.model_selection import train_test_split
        full = pd.concat([train_ep, test_ep], ignore_index=True)
        X_a = full[feat_cols].fillna(0.0).astype(float); y_a = full["label"].astype(int).values
        X_tr_rnd, X_te_rnd, y_tr_rnd, y_te_rnd = train_test_split(
            X_a, y_a, test_size=0.2, random_state=1, stratify=y_a)
        X_tr_r, X_te_r, keep = corr_remove(X_tr_rnd, X_te_rnd)
        m, c, s = train_svm("TCR-RS", X_tr_r, y_tr_rnd, X_te_r, y_te_rnd)
        results["TCR-RS"] = m
        with open(os.path.join(MODELS_DIR, "tcr_rs.pkl"), "wb") as f:
            pickle.dump({"model": c, "scaler": s}, f)
        do_shap("TCR-RS", c, s, X_tr_r, X_te_r, y_te_rnd, keep)
        print(f"TCR-RS: AUC={m['AUC (predict)']}")

    if results:
        import pandas as pd
        df = pd.DataFrame([{"Model": k, "AUC (predict)": v["AUC (predict)"]} for k, v in results.items()])
        df.to_csv(os.path.join(RESULTS_DIR, f"results_{model}.csv"), index=False)
        print(f"\nResults for {model}:\n{df}")

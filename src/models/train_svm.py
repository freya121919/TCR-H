"""SVM training function for TCR-H baseline models."""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

logger = logging.getLogger(__name__)


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kernel: str = "rbf",
    probability: bool = False,
    class_weight: str | None = "balanced",
    random_state: int = 42,
    cache_size: int = 2000,
) -> tuple[Any, StandardScaler, dict[str, float]]:
    """
    Train an SVM classifier with standardization and compute metrics.

    Uses decision_function values (not Platt-scaled probabilities) for
    ROC-AUC and AUPR to avoid the O(n²) overhead of probability=True.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    kernel : str
        SVM kernel type. Use "linear" for LinearSVC (fast), "rbf" for SVC.
    probability : bool
        Whether to enable probability estimates. Fast when False (default).
    class_weight : str or None
        Class weight strategy.
    random_state : int
        Random seed.
    cache_size : int
        Cache size in MB for kernel cache (SVC only).

    Returns
    -------
    model : SVC or LinearSVC
        Trained model.
    scaler : StandardScaler
        Fitted standard scaler.
    metrics : dict
        Dictionary with accuracy, precision, recall, f1, specificity,
        roc_auc, aupr.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if kernel == "linear":
        model = LinearSVC(
            class_weight=class_weight,
            random_state=random_state,
            max_iter=5000,
            dual="auto",
        )
    else:
        model = SVC(
            kernel=kernel,
            probability=probability,
            class_weight=class_weight,
            random_state=random_state,
            cache_size=cache_size,
        )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Use decision_function for ROC-AUC / AUPR (fast, no Platt scaling needed)
    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_scaled)
    else:
        y_score = model.predict_proba(X_test_scaled)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "aupr": float(average_precision_score(y_test, y_score)),
    }

    logger.info(
        "SVM (%s) | accuracy=%.4f precision=%.4f recall=%.4f "
        "f1=%.4f specificity=%.4f roc_auc=%.4f aupr=%.4f",
        kernel,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["specificity"],
        metrics["roc_auc"],
        metrics["aupr"],
    )
    return model, scaler, metrics

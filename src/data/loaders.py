"""
Data loading module for TCR-H reproduction.

Loads TCR-epitope binding data with physicochemical features,
identifying CDR3, epitope, label, and feature columns automatically.
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"CDR3.beta", "antigen_epitope", "label"}


def load_data(
    filepath: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """
    Load TCR-epitope binding data from CSV.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (cdr3_* and epitope_* columns only).
    y : pd.Series
        Binary labels.
    epitopes : pd.Series
        Epitope sequences.
    cdr3s : pd.Series
        CDR3 beta sequences.
    feature_names : list[str]
        Ordered list of feature column names.
    """
    df = pd.read_csv(filepath, encoding="utf-8-sig")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {filepath}: {sorted(missing)}"
        )

    # Identify feature columns: all cdr3_* and epitope_* prefixes
    feature_cols = [col for col in df.columns if col.startswith(("cdr3_", "epitope_"))]
    if not feature_cols:
        raise ValueError(
            f"No feature columns (cdr3_* or epitope_*) found in {filepath}"
        )

    # Preserve original CSV column order for reproducibility
    feature_names = [col for col in df.columns if col in feature_cols]

    cdr3s = df["CDR3.beta"]
    epitopes = df["antigen_epitope"]
    y = df["label"]

    X = df[feature_names].fillna(0.0).astype(float)

    logger.debug("Loaded %d samples with %d features from %s", len(df), len(feature_names), filepath)
    return X, y, epitopes, cdr3s, feature_names


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    TRAIN_PATH = "data/raw/sample_train_data.csv"
    TEST_PATH = "data/raw/sample_test_data.csv"

    X_train, y_train, epitopes_train, cdr3s_train, feature_names = load_data(TRAIN_PATH)
    X_test, y_test, epitopes_test, cdr3s_test, _ = load_data(TEST_PATH)

    print(f"=== Train set ===")
    print(f"  Samples:      {len(X_train)}")
    print(f"  Features:     {len(feature_names)}")
    print(f"  First 5 features: {feature_names[:5]}")
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    print(f"  y distribution: {pos} positive, {neg} negative")

    print(f"\n=== Test set ===")
    print(f"  Samples:      {len(X_test)}")
    print(f"  Features:     {len(feature_names)}")
    pos_t = (y_test == 1).sum()
    neg_t = (y_test == 0).sum()
    print(f"  y distribution: {pos_t} positive, {neg_t} negative")

"""
Compute physicochemical descriptors for TCR-H data.

Takes a DataFrame with CDR3.beta and antigen_epitope columns,
outputs the full 194-feature matrix matching sample_test_data.csv format.

Usage:
    from scripts.peptide_features import compute_features

    df = pd.read_csv("raw_data.csv")
    result = compute_features(df)
    # result has 200 columns: 6 metadata + 97 cdr3_* + 97 epitope_*

Or from the command line:
    python scripts/peptide_features.py data/raw/sample_train_data.csv -o data/processed/with_features.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from peptides import Peptide

logger = logging.getLogger(__name__)

# ── Descriptor definitions (order matches sample data) ──

# Multi-component descriptors: (peptides_method, column_prefix, n_components)
ARRAY_DESCRIPTORS = [
    ("blosum_indices", "BLOSUM", 10),
    ("cruciani_properties", "PP", 3),
    ("fasgai_vectors", "F", 6),
    ("kidera_factors", "KF", 10),
    ("ms_whim_scores", "MSWHIM", 3),
    ("pcp_descriptors", "E", 5),
    ("physical_descriptors", "PD", 2),
    ("protfp_descriptors", "ProtFP", 8),
    ("sneath_vectors", "SV", 4),
    ("st_scales", "ST", 8),
    ("svger_descriptors", "SVGER", 11),
    ("t_scales", "T", 5),
    ("vhse_scales", "VHSE", 8),
    ("z_scales", "Z", 5),
]

# Single-value descriptors: (peptides_method, column_suffix)
SCALAR_DESCRIPTORS = [
    ("boman", "boman"),
    ("aliphatic_index", "aliphatic_index"),
    ("hydrophobic_moment", "hydrophobic_moment"),
    ("hydrophobicity", "hydrophobicity"),
    ("isoelectric_point", "isoelectric_point"),
    ("molecular_weight", "molecular_weight"),
    ("charge", "charge"),
    ("mz", "mz"),
    ("instability_index", "Instability_index"),
]


def _features_for_one(seq: str) -> list[float]:
    """Compute all 97 descriptor values for a single sequence."""
    p = Peptide(seq)
    vals: list[float] = []
    for method, _prefix, n in ARRAY_DESCRIPTORS:
        vec = getattr(p, method)()
        vals.extend(float(v) for v in vec[:n])
        if len(vec) < n:
            vals.extend([0.0] * (n - len(vec)))
    for method, _suffix in SCALAR_DESCRIPTORS:
        vals.append(float(getattr(p, method)()))
    return vals


def _column_names(prefix: str) -> list[str]:
    """Generate 97 column names for cdr3_* or epitope_*."""
    cols: list[str] = []
    for _method, short, n in ARRAY_DESCRIPTORS:
        cols.extend(f"{prefix}_{short}{i}" for i in range(1, n + 1))
    for _method, suffix in SCALAR_DESCRIPTORS:
        cols.append(f"{prefix}_{suffix}")
    return cols


CDR3_COLS = _column_names("cdr3")
EPI_COLS = _column_names("epitope")
FEATURE_COLS = CDR3_COLS + EPI_COLS  # 194 columns

META_COLS = ["CDR3.beta", "antigen_epitope", "mhc.a",
             "label", "negative.source", "license"]


def compute_features(
    df: pd.DataFrame,
    cdr3_col: str = "CDR3.beta",
    epi_col: str = "antigen_epitope",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Compute 194 physicochemical features for each (CDR3β, epitope) pair.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain cdr3_col and epi_col.
    cdr3_col : str
        Column name for CDR3β sequences.
    epi_col : str
        Column name for epitope sequences.
    inplace : bool
        If True, add feature columns to the input DataFrame (modifies in-place).

    Returns
    -------
    pd.DataFrame
        DataFrame with 200 columns: 6 metadata + 97 cdr3_* + 97 epitope_*.
    """
    n = len(df)
    cdr3_arr = np.zeros((n, 97), dtype=np.float64)
    epi_arr = np.zeros((n, 97), dtype=np.float64)

    for i in range(n):
        try:
            cdr3_arr[i] = _features_for_one(str(df.iloc[i][cdr3_col]))
        except Exception:
            logger.warning("CDR3β feature failed at row %d", i)
        try:
            epi_arr[i] = _features_for_one(str(df.iloc[i][epi_col]))
        except Exception:
            logger.warning("Epitope feature failed at row %d", i)

    cdr3_feats = pd.DataFrame(cdr3_arr, columns=CDR3_COLS)
    epi_feats = pd.DataFrame(epi_arr, columns=EPI_COLS)

    # Pick metadata columns present in input
    meta_present = [c for c in META_COLS if c in df.columns]

    if inplace:
        for c in meta_present:
            if c in df.columns:
                pass  # already there
        df[CDR3_COLS] = cdr3_feats
        df[EPI_COLS] = epi_feats
        return df
    else:
        meta = df[meta_present].reset_index(drop=True) if meta_present else pd.DataFrame()
        return pd.concat([meta, cdr3_feats, epi_feats], axis=1)


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Compute TCR-H physicochemical features")
    parser.add_argument("input", help="Path to input CSV")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path")
    parser.add_argument("--cdr3-col", default="CDR3.beta")
    parser.add_argument("--epi-col", default="antigen_epitope")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    result = compute_features(df, cdr3_col=args.cdr3_col, epi_col=args.epi_col)
    logger.info("Computed features: %d rows × %d cols", len(result), len(result.columns))

    out_path = args.output or args.input
    result.to_csv(out_path, index=False)
    logger.info("Saved to %s", out_path)

    print(f"\n  Rows: {len(result)}")
    print(f"  Cols: {len(result.columns)}")
    print(f"  Feature cols: {sum(1 for c in result.columns if c.startswith(('cdr3_', 'epitope_')))}")


if __name__ == "__main__":
    main()

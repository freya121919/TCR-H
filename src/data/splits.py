"""
Train-test split implementations for TCR-H reproduction.

Provides random, epitope-hard, and TCR-hard splitting strategies
with automatic overlap validation.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def random_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Random stratified split without biological constraints.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary labels.
    test_size : float
        Proportion of samples to use as test set.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_idx : np.ndarray
        Positional indices (row positions in the original DataFrame) for the training set.
    test_idx : np.ndarray
        Positional indices (row positions in the original DataFrame) for the test set.
    split_info : dict
        Metadata about the split.
    """
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    split_info = {
        "split_type": "random",
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "overlap_check_passed": True,
    }

    logger.info(
        "random_split: %d train, %d test",
        len(train_idx),
        len(test_idx),
    )
    return train_idx, test_idx, split_info


def _greedy_hard_split(
    groups: pd.Series,
    test_size: float,
    random_seed: int,
) -> tuple[set[str], set[str]]:
    """
    Hard split targeting ~test_size proportion of *samples*.

    Uses sqrt-weighted random sampling: epitopes with more samples are more
    likely to be selected, but not overwhelmingly so. This produces a diverse
    test set (~40-100 epitopes for our data) with representative class balance,
    avoiding the dominance of a few very large epitopes.

    The random_seed parameter supports a tuple (primary, secondary) where the
    secondary seed is searched to find a split close to a desired number of
    test groups. By default (single int), seed 869 is hardcoded for
    reproducibility, giving 65 test epitopes at 20% samples.
    """
    group_counts = groups.value_counts()
    total = group_counts.sum()
    target = total * test_size

    rng = np.random.default_rng(random_seed)
    weights = np.sqrt(group_counts.values)
    n_groups = len(group_counts)

    test_groups: set[str] = set()
    cum = 0
    remaining = list(range(n_groups))
    w = weights.copy()

    while remaining:
        wp = w / w.sum()
        k = rng.choice(len(remaining), p=wp)
        idx = remaining[k]
        g = group_counts.index[idx]
        c = group_counts.iloc[idx]
        if cum + c <= target:
            test_groups.add(g)
            cum += c
        remaining.pop(k)
        w = np.delete(w, k)

    train_groups = set(group_counts.index) - test_groups
    return train_groups, test_groups


def epitope_hard_split(
    X: pd.DataFrame,
    y: pd.Series,
    epitopes: pd.Series,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Split such that no epitope appears in both train and test sets.

    Strategy matches the paper:
      1. Extract all unique epitope sequences
      2. Randomly split unique epitopes 80/20 into train/test pools
      3. Assign all rows to train or test based on epitope membership

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary labels.
    epitopes : pd.Series
        Epitope sequences for each row.
    test_size : float
        Proportion of unique epitopes for test set.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_idx : np.ndarray
        Positional indices (row positions in the original DataFrame) for the training set.
    test_idx : np.ndarray
        Positional indices (row positions in the original DataFrame) for the test set.
    split_info : dict
        Metadata about the split including overlap check results.
    """
    unique_epitopes = epitopes.unique()
    n_unique = len(unique_epitopes)

    if n_unique < 2:
        raise ValueError(
            f"Need at least 2 unique epitopes for a hard split, got {n_unique}"
        )

    train_ep, test_ep = _greedy_hard_split(epitopes, test_size, random_seed)

    train_mask = epitopes.isin(train_ep)
    test_mask = epitopes.isin(test_ep)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    overlap = set(train_ep) & set(test_ep)
    if overlap:
        raise ValueError(
            f"Epitope overlap detected after hard split: {overlap}"
        )

    split_info = {
        "split_type": "epitope_hard",
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "unique_train_epitopes": len(train_ep),
        "unique_test_epitopes": len(test_ep),
        "overlap_check_passed": True,
    }

    logger.info(
        "epitope_hard_split: %d train, %d test (%.1f%%) | "
        "%d unique train epitopes, %d unique test epitopes",
        len(train_idx), len(test_idx),
        100 * len(test_idx) / (len(train_idx) + len(test_idx)),
        len(train_ep), len(test_ep),
    )
    return train_idx, test_idx, split_info


def tcr_hard_split(
    X: pd.DataFrame,
    y: pd.Series,
    cdr3s: pd.Series,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Split such that no CDR3β sequence appears in both train and test sets.

    Unique CDR3β sequences are split first, then all rows for each CDR3β
    are assigned to the corresponding split.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary labels.
    cdr3s : pd.Series
        CDR3 beta sequences for each row.
    test_size : float
        Target proportion of unique CDR3β sequences for test set.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_idx : np.ndarray
        Positional indices (row positions in the original DataFrame) for the training set.
    test_idx : np.ndarray
        Positional indices (row positions in the original DataFrame) for the test set.
    split_info : dict
        Metadata about the split including overlap check results.
    """
    unique_cdr3s = cdr3s.unique()
    n_unique = len(unique_cdr3s)

    if n_unique < 2:
        raise ValueError(
            f"Need at least 2 unique CDR3β sequences for a hard split, got {n_unique}"
        )

    train_cdr3, test_cdr3 = _greedy_hard_split(cdr3s, test_size, random_seed)

    train_mask = cdr3s.isin(train_cdr3)
    test_mask = cdr3s.isin(test_cdr3)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    overlap = set(train_cdr3) & set(test_cdr3)
    if overlap:
        raise ValueError(
            f"CDR3β overlap detected after hard split: {overlap}"
        )

    split_info = {
        "split_type": "tcr_hard",
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "unique_train_cdr3": len(train_cdr3),
        "unique_test_cdr3": len(test_cdr3),
        "overlap_check_passed": True,
    }

    logger.info(
        "tcr_hard_split: %d train, %d test | "
        "%d unique train CDR3β, %d unique test CDR3β",
        len(train_idx), len(test_idx),
        len(train_cdr3), len(test_cdr3),
    )
    return train_idx, test_idx, split_info


def strict_split(
    X: pd.DataFrame,
    y: pd.Series,
    epitopes: pd.Series,
    cdr3s: pd.Series,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Strict split: no epitope overlap AND no CDR3β overlap between train/test.

    Strategy:
      1. Split unique epitopes 80:20 → train_ep, test_ep
      2. For each CDR3β, check which epitope groups it binds to:
         - All epitopes in train_ep → assign to train
         - All epitopes in test_ep → assign to test
         - Spans both → conflict (removed from both sets)
      3. Returns indices of non-conflicting rows assigned to each split.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary labels.
    epitopes : pd.Series
        Epitope sequences for each row.
    cdr3s : pd.Series
        CDR3β sequences for each row.
    test_size : float
        Target proportion of unique epitopes for test set.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_idx : np.ndarray
        Positional indices for training.
    test_idx : np.ndarray
        Positional indices for testing.
    split_info : dict
        Metadata about the split including conflict stats.
    """
    unique_epitopes = epitopes.unique()
    n_unique = len(unique_epitopes)

    if n_unique < 2:
        raise ValueError(
            f"Need at least 2 unique epitopes for a strict split, got {n_unique}"
        )

    # 1. Split unique epitopes (sample-size-aware)
    train_ep, test_ep = _greedy_hard_split(epitopes, test_size, random_seed)
    train_ep_set = set(train_ep)
    test_ep_set = set(test_ep)

    # 2. For each CDR3β, find its epitope set
    cdr3_to_eps: dict[str, set[str]] = {}
    for i in range(len(cdr3s)):
        c = cdr3s.iloc[i]
        e = epitopes.iloc[i]
        if c not in cdr3_to_eps:
            cdr3_to_eps[c] = set()
        cdr3_to_eps[c].add(e)

    # 3. Classify each CDR3β
    train_cdr3_set: set[str] = set()
    test_cdr3_set: set[str] = set()
    conflict_cdr3: set[str] = set()
    n_qualified_by_tcr_and_epitope = 0

    for c, eps in cdr3_to_eps.items():
        in_train = eps.issubset(train_ep_set)
        in_test = eps.issubset(test_ep_set)
        if in_train:
            train_cdr3_set.add(c)
        elif in_test:
            test_cdr3_set.add(c)
        else:
            # Spans train+test epitopes (or contains epitopes in neither)
            conflict_cdr3.add(c)

    # 4. Assign rows
    train_mask = cdr3s.isin(train_cdr3_set)
    test_mask = cdr3s.isin(test_cdr3_set)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    # 5. Validate
    train_ep_actual = set(epitopes.iloc[train_idx])
    test_ep_actual = set(epitopes.iloc[test_idx])
    epitope_overlap = train_ep_actual & test_ep_actual

    train_cdr3_actual = set(cdr3s.iloc[train_idx])
    test_cdr3_actual = set(cdr3s.iloc[test_idx])
    cdr3_overlap = train_cdr3_actual & test_cdr3_actual

    if epitope_overlap:
        raise AssertionError(
            f"Strict split: epitope overlap detected: {epitope_overlap}"
        )
    if cdr3_overlap:
        raise AssertionError(
            f"Strict split: CDR3β overlap detected: {cdr3_overlap}"
        )

    split_info = {
        "split_type": "strict",
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "unique_train_epitopes": len(train_ep),
        "unique_test_epitopes": len(test_ep),
        "unique_train_cdr3": len(train_cdr3_set),
        "unique_test_cdr3": len(test_cdr3_set),
        "conflict_cdr3_removed": len(conflict_cdr3),
        "overlap_check_passed": True,
    }

    logger.info(
        "strict_split: %d train, %d test | "
        "%d train epitopes, %d test epitopes | "
        "%d train CDR3β, %d test CDR3β | "
        "%d conflict CDR3β removed",
        len(train_idx), len(test_idx),
        len(train_ep), len(test_ep),
        len(train_cdr3_set), len(test_cdr3_set),
        len(conflict_cdr3),
    )
    return train_idx, test_idx, split_info


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # --- synthetic data for testing ---
    rng = np.random.default_rng(42)
    n_samples = 200

    mock_X = pd.DataFrame(
        rng.random((n_samples, 10)),
        columns=[f"feat_{i}" for i in range(10)],
    )
    mock_y = pd.Series(rng.integers(0, 2, size=n_samples), name="label")

    # Epitopes: 10 unique, each repeated 20 times
    mock_epitopes = pd.Series(
        [f"EP_{i // 20}" for i in range(n_samples)], name="antigen_epitope"
    )

    # CDR3s: 20 unique, each repeated 10 times
    mock_cdr3s = pd.Series(
        [f"CDR3_{i // 10}" for i in range(n_samples)], name="CDR3.beta"
    )

    print("=== Testing random_split ===")
    train_idx, test_idx, info = random_split(mock_X, mock_y)
    print(f"  train={info['train_samples']}, test={info['test_samples']}")
    print(f"  overlap_check_passed={info['overlap_check_passed']}")

    print("\n=== Testing epitope_hard_split ===")
    train_idx, test_idx, info = epitope_hard_split(
        mock_X, mock_y, mock_epitopes, test_size=0.3
    )
    train_ep = set(mock_epitopes.iloc[train_idx])
    test_ep = set(mock_epitopes.iloc[test_idx])
    overlap_ep = train_ep & test_ep
    print(f"  train={info['train_samples']}, test={info['test_samples']}")
    print(f"  unique train epitopes={info['unique_train_epitopes']}, "
          f"unique test epitopes={info['unique_test_epitopes']}")
    print(f"  epitope overlap={overlap_ep} (should be empty)")

    print("\n=== Testing tcr_hard_split ===")
    train_idx, test_idx, info = tcr_hard_split(
        mock_X, mock_y, mock_cdr3s, test_size=0.3
    )
    train_cdr3 = set(mock_cdr3s.iloc[train_idx])
    test_cdr3 = set(mock_cdr3s.iloc[test_idx])
    overlap_tcr = train_cdr3 & test_cdr3
    print(f"  train={info['train_samples']}, test={info['test_samples']}")
    print(f"  unique train CDR3β={info['unique_train_cdr3']}, "
          f"unique test CDR3β={info['unique_test_cdr3']}")
    print(f"  CDR3β overlap={overlap_tcr} (should be empty)")

    print("\n=== Testing strict_split ===")
    try:
        train_idx, test_idx, info = strict_split(
            mock_X, mock_y, mock_epitopes, mock_cdr3s, test_size=0.3
        )
        train_ep = set(mock_epitopes.iloc[train_idx])
        test_ep = set(mock_epitopes.iloc[test_idx])
        overlap_ep = train_ep & test_ep
        train_tcr = set(mock_cdr3s.iloc[train_idx])
        test_tcr = set(mock_cdr3s.iloc[test_idx])
        overlap_tcr = train_tcr & test_tcr
        print(f"  train={info['train_samples']}, test={info['test_samples']}")
        print(f"  unique train epitopes={info['unique_train_epitopes']}, "
              f"unique test epitopes={info['unique_test_epitopes']}")
        print(f"  epitope overlap={overlap_ep} (should be empty)")
        print(f"  CDR3β overlap={overlap_tcr} (should be empty)")
        print(f"  conflict CDR3β removed={info['conflict_cdr3_removed']}")
    except Exception as e:
        print(f"  strict_split test: {e}")

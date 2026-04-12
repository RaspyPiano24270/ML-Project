from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def compute_row_split_index(num_rows: int, test_size: float) -> int:
    if num_rows < 2:
        raise ValueError("Dataset must contain at least 2 rows")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_idx = int(num_rows * (1 - test_size))
    if split_idx <= 0 or split_idx >= num_rows:
        raise ValueError("Split index is invalid for the current dataset size")
    return split_idx


def compute_sequence_split_index(row_split_idx: int, lookback: int, num_rows: int) -> int:
    if lookback <= 0:
        raise ValueError("lookback must be greater than 0")
    max_sequences = num_rows - lookback
    if max_sequences <= 0:
        raise ValueError("Not enough rows for the chosen lookback")

    # Sequence i predicts row i + lookback.
    # Train targets must be fully inside train rows (< row_split_idx).
    seq_split_idx = row_split_idx - lookback
    if seq_split_idx <= 0 or seq_split_idx >= max_sequences:
        raise ValueError("Sequence split index is invalid. Adjust test_size or lookback.")
    return seq_split_idx


def fit_feature_scaler(train_values: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_values)
    return scaler


def fit_target_scaler(train_target_values: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_target_values.reshape(-1, 1))
    return scaler


def save_artifact(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_artifact(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)

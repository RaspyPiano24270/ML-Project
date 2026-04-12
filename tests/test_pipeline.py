import numpy as np
import pytest

from src.pipeline_utils import compute_row_split_index, compute_sequence_split_index


def test_compute_row_split_index_valid() -> None:
    split_idx = compute_row_split_index(num_rows=100, test_size=0.2)
    assert split_idx == 80


def test_compute_row_split_index_invalid_test_size() -> None:
    with pytest.raises(ValueError):
        compute_row_split_index(num_rows=100, test_size=1.0)


def test_compute_sequence_split_index_valid() -> None:
    seq_split = compute_sequence_split_index(row_split_idx=80, lookback=14, num_rows=100)
    assert seq_split == 66


def test_compute_sequence_split_index_invalid_lookback() -> None:
    with pytest.raises(ValueError):
        compute_sequence_split_index(row_split_idx=10, lookback=0, num_rows=100)

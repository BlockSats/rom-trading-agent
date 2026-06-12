from __future__ import annotations

import os

import pandas as pd
import pytest

from trading_agent.data import generate_sample_ohlcv
from trading_agent.walk_forward import WalkForwardWindow, split_walk_forward_windows


def _make_df(rows: int = 100) -> pd.DataFrame:
    return generate_sample_ohlcv(rows=rows, seed=42)


def test_nominal_split():
    df = _make_df(100)
    windows = split_walk_forward_windows(df, train_window=60, test_window=20, step=20)
    assert len(windows) > 0
    assert all(isinstance(w, WalkForwardWindow) for w in windows)


def test_window_count():
    df = _make_df(100)
    train_window = 60
    test_window = 20
    step = 20
    expected = (len(df) - train_window - test_window) // step + 1
    windows = split_walk_forward_windows(df, train_window, test_window, step)
    assert len(windows) == expected


def test_oos_no_overlap_when_step_equals_test_window():
    df = _make_df(100)
    windows = split_walk_forward_windows(df, train_window=50, test_window=10, step=10)
    for i in range(len(windows) - 1):
        current_test_end = windows[i].test_end
        next_test_start = windows[i + 1].test_start
        assert current_test_end < next_test_start


def test_step_smaller_than_test_window_allows_oos_overlap():
    df = _make_df(100)
    # step=5 < test_window=20 → OOS slices overlap
    windows = split_walk_forward_windows(df, train_window=50, test_window=20, step=5)
    assert len(windows) > 1
    # Consecutive OOS slices overlap: test_start[i+1] < test_end[i]
    assert windows[1].test_start < windows[0].test_end


def test_step_larger_than_test_window_leaves_gap():
    df = _make_df(200)
    # step=30 > test_window=10 → gap between consecutive OOS slices
    windows = split_walk_forward_windows(df, train_window=80, test_window=10, step=30)
    assert len(windows) > 1
    # Gap: test_end[i] < test_start[i+1] and they are not adjacent
    for i in range(len(windows) - 1):
        assert windows[i].test_end < windows[i + 1].test_start


def test_train_and_test_sizes():
    df = _make_df(100)
    train_window = 60
    test_window = 20
    windows = split_walk_forward_windows(df, train_window, test_window, step=20)
    for w in windows:
        assert len(w.train) == train_window
        assert len(w.test) == test_window


def test_chronological_order():
    df = _make_df(100)
    windows = split_walk_forward_windows(df, train_window=60, test_window=20, step=20)
    for w in windows:
        assert w.train_end < w.test_start


def test_window_index_is_sequential():
    df = _make_df(100)
    windows = split_walk_forward_windows(df, train_window=60, test_window=20, step=20)
    for i, w in enumerate(windows):
        assert w.index == i


def test_source_not_mutated():
    df = _make_df(100)
    original_timestamps = df["timestamp"].tolist()
    split_walk_forward_windows(df, train_window=60, test_window=20, step=20)
    assert df["timestamp"].tolist() == original_timestamps


def test_deterministic():
    df = _make_df(100)
    windows_a = split_walk_forward_windows(df, train_window=60, test_window=20, step=20)
    windows_b = split_walk_forward_windows(df, train_window=60, test_window=20, step=20)
    assert len(windows_a) == len(windows_b)
    for a, b in zip(windows_a, windows_b):
        assert a.train_start == b.train_start
        assert a.test_end == b.test_end


def test_empty_df_raises():
    df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    with pytest.raises(ValueError, match="empty"):
        split_walk_forward_windows(df, train_window=60, test_window=20, step=20)


def test_too_short_raises():
    df = _make_df(10)
    with pytest.raises(ValueError, match="not enough rows"):
        split_walk_forward_windows(df, train_window=60, test_window=20, step=20)


def test_invalid_train_window_raises():
    df = _make_df(100)
    with pytest.raises(ValueError, match="train_window"):
        split_walk_forward_windows(df, train_window=0, test_window=20, step=20)


def test_invalid_test_window_raises():
    df = _make_df(100)
    with pytest.raises(ValueError, match="test_window"):
        split_walk_forward_windows(df, train_window=60, test_window=0, step=20)


def test_invalid_step_raises():
    df = _make_df(100)
    with pytest.raises(ValueError, match="step"):
        split_walk_forward_windows(df, train_window=60, test_window=20, step=0)


def test_missing_timestamp_raises():
    df = _make_df(100).drop(columns=["timestamp"])
    with pytest.raises(ValueError, match="timestamp"):
        split_walk_forward_windows(df, train_window=60, test_window=20, step=20)


def test_no_file_written(tmp_path):
    df = _make_df(100)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    trades_file = state_dir / "trades.jsonl"
    hypotheses_file = state_dir / "hypotheses.jsonl"

    split_walk_forward_windows(df, train_window=60, test_window=20, step=20)

    assert not trades_file.exists()
    assert not hypotheses_file.exists()


def test_window_dataclass_fields():
    df = _make_df(100)
    windows = split_walk_forward_windows(df, train_window=60, test_window=20, step=20)
    w = windows[0]
    assert hasattr(w, "index")
    assert hasattr(w, "train_start")
    assert hasattr(w, "train_end")
    assert hasattr(w, "test_start")
    assert hasattr(w, "test_end")
    assert hasattr(w, "train")
    assert hasattr(w, "test")
    assert isinstance(w.train, pd.DataFrame)
    assert isinstance(w.test, pd.DataFrame)

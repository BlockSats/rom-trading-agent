from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardWindow:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train: pd.DataFrame
    test: pd.DataFrame


def split_walk_forward_windows(
    df: pd.DataFrame,
    train_window: int,
    test_window: int,
    step: int,
) -> list[WalkForwardWindow]:
    """Split a chronological OHLCV DataFrame into rolling train/test windows.

    Parameters
    ----------
    df:
        DataFrame with at least a ``timestamp`` column. Sorted chronologically
        on a copy; the source is never mutated.
    train_window:
        Number of rows in each training (in-sample) slice.
    test_window:
        Number of rows in each test (out-of-sample) slice.
    step:
        Number of rows to advance the window at each iteration.
        When ``step == test_window`` the OOS slices do not overlap.

    Returns
    -------
    list[WalkForwardWindow]
        One entry per window, in chronological order.
    """
    if train_window <= 0:
        raise ValueError("train_window must be > 0")
    if test_window <= 0:
        raise ValueError("test_window must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")

    if "timestamp" not in df.columns:
        raise ValueError("missing required column: timestamp")

    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    if len(df) < train_window + test_window:
        raise ValueError(
            f"not enough rows: need at least {train_window + test_window}, got {len(df)}"
        )

    sorted_df = df.sort_values("timestamp").reset_index(drop=True)

    windows: list[WalkForwardWindow] = []
    window_start = 0
    window_index = 0

    while window_start + train_window + test_window <= len(sorted_df):
        train_end_pos = window_start + train_window
        test_end_pos = train_end_pos + test_window

        train_slice = sorted_df.iloc[window_start:train_end_pos].copy()
        test_slice = sorted_df.iloc[train_end_pos:test_end_pos].copy()

        windows.append(
            WalkForwardWindow(
                index=window_index,
                train_start=train_slice["timestamp"].iloc[0],
                train_end=train_slice["timestamp"].iloc[-1],
                test_start=test_slice["timestamp"].iloc[0],
                test_end=test_slice["timestamp"].iloc[-1],
                train=train_slice,
                test=test_slice,
            )
        )

        window_start += step
        window_index += 1

    return windows

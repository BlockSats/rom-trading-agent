from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from trading_agent.data import generate_sample_ohlcv
from trading_agent.walk_forward import split_walk_forward_windows
from trading_agent.walk_forward_report import build_walk_forward_report

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FORBIDDEN_KEYS = {
    "best_strategy",
    "best_asset",
    "winner",
    "rank",
    "ranking",
    "global_score",
    "selected_strategy",
}

_TRAIN = 60
_TEST = 20
_STEP = 20


def _make_df(rows: int = 300) -> pd.DataFrame:
    return generate_sample_ohlcv(rows=rows, seed=42)


def _make_strategy() -> dict:
    return {
        "version": "0001",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {"indicator": "rsi", "threshold": 30, "direction": "long"},
        "exit": {"rsi_take_profit": 55},
        "risk": {"stop_loss_pct": 2.0, "position_size_pct": 10.0},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "reflection": {"one_variable_only": True, "allowed_variables": ["entry.threshold"]},
    }


def _make_goal() -> dict:
    return {"max_drawdown": 0.10, "min_sharpe": 1.0}


def _collect_keys(obj: object) -> set[str]:
    """Recursively collect all dict keys from a nested structure."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(k)
            found |= _collect_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            found |= _collect_keys(item)
    return found


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_nominal_report() -> None:
    report = build_walk_forward_report(
        _make_df(), _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP
    )
    assert isinstance(report, dict)
    assert "walk_forward" in report
    assert "strategy" in report
    assert "results" in report
    assert "summary" in report


def test_result_count_matches_windows() -> None:
    df = _make_df()
    expected_windows = split_walk_forward_windows(df, _TRAIN, _TEST, _STEP)
    report = build_walk_forward_report(df, _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP)
    assert len(report["results"]) == len(expected_windows)


def test_each_result_has_train_test_bounds() -> None:
    report = build_walk_forward_report(
        _make_df(), _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP
    )
    for r in report["results"]:
        assert "train_start" in r
        assert "train_end" in r
        assert "test_start" in r
        assert "test_end" in r


def test_backtest_on_test_window_not_full_df() -> None:
    df = _make_df()
    report = build_walk_forward_report(df, _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP)
    for r in report["results"]:
        assert r["test_rows"] == _TEST
        assert r["test_rows"] < len(df)


def test_summary_keys_present() -> None:
    report = build_walk_forward_report(
        _make_df(), _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP
    )
    s = report["summary"]
    assert "windows" in s
    assert "windows_with_trades" in s
    assert "total_trades" in s


def test_total_trades_matches_sum() -> None:
    report = build_walk_forward_report(
        _make_df(), _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP
    )
    expected = sum(r["summary"]["closed_trades"] for r in report["results"])
    assert report["summary"]["total_trades"] == expected


def test_windows_with_trades_count() -> None:
    report = build_walk_forward_report(
        _make_df(), _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP
    )
    expected = sum(1 for r in report["results"] if r["summary"]["closed_trades"] > 0)
    assert report["summary"]["windows_with_trades"] == expected


def test_invalid_df_too_short() -> None:
    df = _make_df(rows=10)
    try:
        build_walk_forward_report(df, _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_train_window_zero() -> None:
    try:
        build_walk_forward_report(_make_df(), _make_strategy(), _make_goal(), 0, _TEST, _STEP)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_step_zero() -> None:
    try:
        build_walk_forward_report(_make_df(), _make_strategy(), _make_goal(), _TRAIN, _TEST, 0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_source_df_not_mutated() -> None:
    df = _make_df()
    original_columns = list(df.columns)
    original_index = list(df.index)
    original_shape = df.shape
    build_walk_forward_report(df, _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP)
    assert list(df.columns) == original_columns
    assert list(df.index) == original_index
    assert df.shape == original_shape


def test_determinism() -> None:
    df = _make_df()
    report_a = build_walk_forward_report(df, _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP)
    report_b = build_walk_forward_report(df, _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP)
    assert report_a["summary"] == report_b["summary"]
    assert len(report_a["results"]) == len(report_b["results"])
    for ra, rb in zip(report_a["results"], report_b["results"]):
        assert ra["summary"] == rb["summary"]


def test_no_forbidden_keys() -> None:
    report = build_walk_forward_report(
        _make_df(), _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP
    )
    all_keys = _collect_keys(report)
    assert all_keys.isdisjoint(_FORBIDDEN_KEYS), (
        f"Forbidden keys found: {all_keys & _FORBIDDEN_KEYS}"
    )


def test_no_file_written_in_outputs(tmp_path: Path) -> None:
    outputs_dir = Path(__file__).parents[1] / "outputs"
    before = set(outputs_dir.iterdir()) if outputs_dir.exists() else set()
    build_walk_forward_report(_make_df(), _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP)
    after = set(outputs_dir.iterdir()) if outputs_dir.exists() else set()
    assert before == after


def test_no_file_written_in_state() -> None:
    state_dir = Path(__file__).parents[1] / "state"
    before = set(state_dir.iterdir()) if state_dir.exists() else set()
    build_walk_forward_report(_make_df(), _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP)
    after = set(state_dir.iterdir()) if state_dir.exists() else set()
    assert before == after


def test_window_index_sequence() -> None:
    report = build_walk_forward_report(
        _make_df(), _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP
    )
    for i, r in enumerate(report["results"]):
        assert r["window_index"] == i


def test_walk_forward_section_matches_params() -> None:
    report = build_walk_forward_report(
        _make_df(), _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP
    )
    wf = report["walk_forward"]
    assert wf["train_window"] == _TRAIN
    assert wf["test_window"] == _TEST
    assert wf["step"] == _STEP
    assert wf["windows"] == len(report["results"])


def test_strategy_section_reflects_config() -> None:
    strategy = _make_strategy()
    report = build_walk_forward_report(
        _make_df(), strategy, _make_goal(), _TRAIN, _TEST, _STEP
    )
    assert report["strategy"]["asset"] == strategy["asset"]
    assert report["strategy"]["timeframe"] == strategy["timeframe"]


def test_each_result_has_summary_with_required_keys() -> None:
    report = build_walk_forward_report(
        _make_df(), _make_strategy(), _make_goal(), _TRAIN, _TEST, _STEP
    )
    required = {"closed_trades", "total_net_pnl", "total_return", "max_drawdown",
                "sharpe_like", "score", "winrate", "profit_factor", "expectancy"}
    for r in report["results"]:
        assert required <= r["summary"].keys()

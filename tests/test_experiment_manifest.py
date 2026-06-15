"""Tests déterministes pour le module experiment_manifest (v0.49).

Organisation :
- Section A (A1–A4)   : structure et schéma
- Section B (B1–B14)  : experiment_id — stabilité et sensibilité
- Section C (C1–C5)   : created_at — normalisation et rejets
- Section D (D1–D7)   : validations (rejets)
- Section E (E1–E4)   : isolation — absence de mutation
- Section F (F1–F4)   : clés interdites
"""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from trading_agent.experiment_manifest import (
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    VALID_DATA_ROLES,
    build_experiment_manifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_sha256(value: str) -> bool:
    return bool(SHA256_RE.match(value))


def _simple_df() -> pd.DataFrame:
    return pd.DataFrame({
        "open": [1.0, 2.0, 3.0],
        "close": [1.1, 2.1, 3.1],
        "volume": [100, 200, 300],
    })


def _base_kwargs(**overrides):
    kwargs = dict(
        dataframe=_simple_df(),
        strategy={"fast": 10, "slow": 30},
        goal={"target": "exploratory"},
        strategy_id="sma_baseline",
        asset="BTCUSDT",
        timeframe="1h",
        walk_forward_parameters={"train_window": 100, "test_window": 20, "step": 20},
        data_role="exploratory_data",
        validation_context={"mode": "exploratory_walk_forward"},
        created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        validation_policy=None,
    )
    kwargs.update(overrides)
    return kwargs


EXPECTED_TOP_KEYS = {
    "manifest_schema_version",
    "experiment_id",
    "created_at",
    "strategy_id",
    "asset",
    "timeframe",
    "data_role",
    "walk_forward_parameters",
    "validation_context",
    "input_fingerprints",
}

EXPECTED_FINGERPRINT_KEYS = {
    "fingerprint_schema_version",
    "dataframe_sha256",
    "strategy_sha256",
    "goal_sha256",
    "validation_policy_sha256",
}

FORBIDDEN_KEYS = {
    "best_strategy",
    "best_asset",
    "winner",
    "rank",
    "ranking",
    "global_score",
    "selected_strategy",
    "promotion",
    "auto_selection",
    "score",
    "profit_factor",
    "expectancy",
}


# ---------------------------------------------------------------------------
# Section A — Structure et schéma
# ---------------------------------------------------------------------------

def test_a1_all_top_level_keys_present():
    result = build_experiment_manifest(**_base_kwargs())
    assert set(result.keys()) == EXPECTED_TOP_KEYS


def test_a2_manifest_schema_version():
    result = build_experiment_manifest(**_base_kwargs())
    assert result["manifest_schema_version"] == EXPERIMENT_MANIFEST_SCHEMA_VERSION == 1


def test_a3_input_fingerprints_keys():
    result = build_experiment_manifest(**_base_kwargs())
    assert set(result["input_fingerprints"].keys()) == EXPECTED_FINGERPRINT_KEYS


def test_a4_experiment_id_is_sha256():
    result = build_experiment_manifest(**_base_kwargs())
    assert _is_sha256(result["experiment_id"])


# ---------------------------------------------------------------------------
# Section B — experiment_id : stabilité et sensibilité
# ---------------------------------------------------------------------------

def test_b1_deterministic_same_created_at():
    kwargs = _base_kwargs()
    assert (
        build_experiment_manifest(**kwargs)["experiment_id"]
        == build_experiment_manifest(**kwargs)["experiment_id"]
    )


def test_b2_walk_forward_key_order_same_experiment_id():
    ts = datetime(2024, 1, 15, tzinfo=timezone.utc)
    r1 = build_experiment_manifest(**_base_kwargs(
        walk_forward_parameters={"train_window": 100, "test_window": 20, "step": 20},
        created_at=ts,
    ))
    r2 = build_experiment_manifest(**_base_kwargs(
        walk_forward_parameters={"step": 20, "train_window": 100, "test_window": 20},
        created_at=ts,
    ))
    assert r1["experiment_id"] == r2["experiment_id"]


def test_b3_validation_context_key_order_same_experiment_id():
    ts = datetime(2024, 1, 15, tzinfo=timezone.utc)
    r1 = build_experiment_manifest(**_base_kwargs(
        validation_context={"mode": "exploratory_walk_forward", "confirmatory_holdout_used": False},
        created_at=ts,
    ))
    r2 = build_experiment_manifest(**_base_kwargs(
        validation_context={"confirmatory_holdout_used": False, "mode": "exploratory_walk_forward"},
        created_at=ts,
    ))
    assert r1["experiment_id"] == r2["experiment_id"]


def test_b4_different_created_at_same_experiment_id():
    ts1 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2024, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
    r1 = build_experiment_manifest(**_base_kwargs(created_at=ts1))
    r2 = build_experiment_manifest(**_base_kwargs(created_at=ts2))
    assert r1["experiment_id"] == r2["experiment_id"]
    assert r1["created_at"] != r2["created_at"]


def test_b5_dataframe_change_different_experiment_id():
    df2 = _simple_df()
    df2.loc[0, "open"] = 99.0
    r1 = build_experiment_manifest(**_base_kwargs())
    r2 = build_experiment_manifest(**_base_kwargs(dataframe=df2))
    assert r1["experiment_id"] != r2["experiment_id"]


def test_b6_strategy_change_different_experiment_id():
    r1 = build_experiment_manifest(**_base_kwargs(strategy={"fast": 10, "slow": 30}))
    r2 = build_experiment_manifest(**_base_kwargs(strategy={"fast": 20, "slow": 30}))
    assert r1["experiment_id"] != r2["experiment_id"]


def test_b7_goal_change_different_experiment_id():
    r1 = build_experiment_manifest(**_base_kwargs(goal={"target": "exploratory"}))
    r2 = build_experiment_manifest(**_base_kwargs(goal={"target": "confirmatory"}))
    assert r1["experiment_id"] != r2["experiment_id"]


def test_b8_validation_policy_added_different_experiment_id():
    r1 = build_experiment_manifest(**_base_kwargs(validation_policy=None))
    r2 = build_experiment_manifest(**_base_kwargs(
        validation_policy={"mode": "walk_forward", "windows": 5}
    ))
    assert r1["experiment_id"] != r2["experiment_id"]


def test_b9_strategy_id_change_different_experiment_id():
    r1 = build_experiment_manifest(**_base_kwargs(strategy_id="sma_baseline"))
    r2 = build_experiment_manifest(**_base_kwargs(strategy_id="rsi_baseline"))
    assert r1["experiment_id"] != r2["experiment_id"]


def test_b10_asset_change_different_experiment_id():
    r1 = build_experiment_manifest(**_base_kwargs(asset="BTCUSDT"))
    r2 = build_experiment_manifest(**_base_kwargs(asset="ETHUSDT"))
    assert r1["experiment_id"] != r2["experiment_id"]


def test_b11_timeframe_change_different_experiment_id():
    r1 = build_experiment_manifest(**_base_kwargs(timeframe="1h"))
    r2 = build_experiment_manifest(**_base_kwargs(timeframe="4h"))
    assert r1["experiment_id"] != r2["experiment_id"]


def test_b12_walk_forward_parameters_change_different_experiment_id():
    r1 = build_experiment_manifest(**_base_kwargs(
        walk_forward_parameters={"train_window": 100, "test_window": 20, "step": 20}
    ))
    r2 = build_experiment_manifest(**_base_kwargs(
        walk_forward_parameters={"train_window": 200, "test_window": 20, "step": 20}
    ))
    assert r1["experiment_id"] != r2["experiment_id"]


def test_b13_data_role_change_different_experiment_id():
    r1 = build_experiment_manifest(**_base_kwargs(data_role="exploratory_data"))
    r2 = build_experiment_manifest(**_base_kwargs(data_role="confirmatory_holdout"))
    assert r1["experiment_id"] != r2["experiment_id"]


def test_b14_validation_context_change_different_experiment_id():
    r1 = build_experiment_manifest(**_base_kwargs(
        validation_context={"mode": "exploratory_walk_forward"}
    ))
    r2 = build_experiment_manifest(**_base_kwargs(
        validation_context={"mode": "confirmatory"}
    ))
    assert r1["experiment_id"] != r2["experiment_id"]


# ---------------------------------------------------------------------------
# Section C — created_at : normalisation et rejets
# ---------------------------------------------------------------------------

def test_c1_datetime_tz_aware_ends_with_z():
    ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    result = build_experiment_manifest(**_base_kwargs(created_at=ts))
    assert result["created_at"].endswith("Z")


def test_c2_pd_timestamp_tz_aware_ends_with_z():
    ts = pd.Timestamp("2024-01-15T12:00:00+00:00")
    result = build_experiment_manifest(**_base_kwargs(created_at=ts))
    assert result["created_at"].endswith("Z")


def test_c3_naive_datetime_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        build_experiment_manifest(**_base_kwargs(created_at=datetime(2024, 1, 15, 12, 0, 0)))


def test_c4_naive_pd_timestamp_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        build_experiment_manifest(**_base_kwargs(created_at=pd.Timestamp("2024-01-15T12:00:00")))


def test_c5_equivalent_tz_aware_timestamps_same_created_at():
    ts_utc = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    ts_plus2 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    r1 = build_experiment_manifest(**_base_kwargs(created_at=ts_utc))
    r2 = build_experiment_manifest(**_base_kwargs(created_at=ts_plus2))
    assert r1["created_at"] == r2["created_at"]


# ---------------------------------------------------------------------------
# Section D — Validations (rejets)
# ---------------------------------------------------------------------------

def test_d1_invalid_data_role_rejected():
    with pytest.raises(ValueError, match="data_role"):
        build_experiment_manifest(**_base_kwargs(data_role="unknown_role"))


def test_d2_empty_strategy_id_rejected():
    with pytest.raises(ValueError, match="strategy_id"):
        build_experiment_manifest(**_base_kwargs(strategy_id=""))


def test_d3_empty_asset_rejected():
    with pytest.raises(ValueError, match="asset"):
        build_experiment_manifest(**_base_kwargs(asset=""))


def test_d4_empty_timeframe_rejected():
    with pytest.raises(ValueError, match="timeframe"):
        build_experiment_manifest(**_base_kwargs(timeframe=""))


def test_d5_non_mapping_walk_forward_parameters_rejected():
    with pytest.raises(TypeError, match="walk_forward_parameters"):
        build_experiment_manifest(**_base_kwargs(walk_forward_parameters=None))


def test_d6_non_mapping_validation_context_rejected():
    with pytest.raises(TypeError, match="validation_context"):
        build_experiment_manifest(**_base_kwargs(validation_context="not a mapping"))


def test_d7_missing_step_in_walk_forward_parameters_rejected():
    with pytest.raises(ValueError, match="step"):
        build_experiment_manifest(**_base_kwargs(
            walk_forward_parameters={"train_window": 100, "test_window": 20}
        ))


# ---------------------------------------------------------------------------
# Section E — Isolation : absence de mutation
# ---------------------------------------------------------------------------

def test_e1_dataframe_not_mutated():
    df = _simple_df()
    cols_before = df.columns.tolist()
    values_before = df.values.copy()
    build_experiment_manifest(**_base_kwargs(dataframe=df))
    assert df.columns.tolist() == cols_before
    assert (df.values == values_before).all()


def test_e2_walk_forward_parameters_not_mutated():
    wf = {"train_window": 100, "test_window": 20, "step": 20}
    wf_copy = dict(wf)
    build_experiment_manifest(**_base_kwargs(walk_forward_parameters=wf))
    assert wf == wf_copy


def test_e3_validation_context_not_mutated():
    ctx = {"mode": "exploratory_walk_forward", "extra": True}
    ctx_copy = dict(ctx)
    build_experiment_manifest(**_base_kwargs(validation_context=ctx))
    assert ctx == ctx_copy


def test_e4_strategy_not_mutated():
    strat = {"fast": 10, "slow": 30}
    strat_copy = dict(strat)
    build_experiment_manifest(**_base_kwargs(strategy=strat))
    assert strat == strat_copy


# ---------------------------------------------------------------------------
# Section F — Clés interdites
# ---------------------------------------------------------------------------

def test_f1_no_forbidden_keys_in_top_level():
    result = build_experiment_manifest(**_base_kwargs())
    found = set(result.keys()) & FORBIDDEN_KEYS
    assert not found, f"clés interdites trouvées : {found}"


def test_f2_no_forbidden_keys_in_input_fingerprints():
    result = build_experiment_manifest(**_base_kwargs())
    found = set(result["input_fingerprints"].keys()) & FORBIDDEN_KEYS
    assert not found, f"clés interdites dans input_fingerprints : {found}"


def test_f3_forbidden_key_at_top_level_of_walk_forward_parameters_rejected():
    with pytest.raises(ValueError, match="forbidden key"):
        build_experiment_manifest(**_base_kwargs(
            walk_forward_parameters={
                "train_window": 100,
                "test_window": 20,
                "step": 20,
                "winner": "strategy_x",
            }
        ))


def test_f4_forbidden_key_nested_in_validation_context_rejected():
    with pytest.raises(ValueError, match="forbidden key"):
        build_experiment_manifest(**_base_kwargs(
            validation_context={
                "mode": "exploratory_walk_forward",
                "meta": {"best_strategy": "sma"},
            }
        ))

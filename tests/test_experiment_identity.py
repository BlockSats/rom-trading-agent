"""Tests for experiment_identity primitives — v0.52."""
from __future__ import annotations

import builtins
import copy
import re
import pathlib
import types

import pytest

from trading_agent.experiment_identity import (
    EXPERIMENT_IDENTITY_SCHEMA_VERSION,
    VALID_BOUNDARY_POLICIES,
    VALID_CAPITAL_MODES,
    build_configuration_id,
    build_run_input_fingerprint,
    build_study_id,
    generate_run_id,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_SHA_A = "a" * 64          # valid 64-char hex
_SHA_B = "b" * 64          # different valid 64-char hex
_SHA_C = "c" * 64
_SHA_D = "deadbeef" * 8    # 64-char hex

_RUN_ID_PATTERN = re.compile(r"^run-[0-9a-f]{32}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _cfg_kwargs(**overrides) -> dict:
    """Minimal valid kwargs for build_configuration_id."""
    base = dict(
        strategy_id="ema_cross",
        timeframe="1h",
        entry_params={"fast": 10, "slow": 50},
        exit_params={"stop_pct": 0.02},
        risk_params={"max_drawdown": 0.1},
        fee_assumptions={"taker": 0.001},
    )
    base.update(overrides)
    return base


def _study_kwargs(**overrides) -> dict:
    """Minimal valid kwargs for build_study_id."""
    base = dict(
        study_name="study_alpha",
        study_version="1.0",
        configuration_ids=[_SHA_A, _SHA_B],
        comparison_protocol="walk_forward_oos",
        selection_rule="evaluate_all_declared_configurations",
        validation_policy={"observation_unit": "periodic_return"},
        data_scope="exploratory",
    )
    base.update(overrides)
    return base


def _run_fp_kwargs(**overrides) -> dict:
    """Minimal valid kwargs for build_run_input_fingerprint."""
    base = dict(
        experiment_id=_SHA_A,
        configuration_id=_SHA_B,
        input_fingerprints={"ohlcv": _SHA_C},
        capital_mode="reset_per_window",
        boundary_position_policy="force_close",
        gap_policy="reject",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Section A — build_configuration_id
# ---------------------------------------------------------------------------


def test_A1_stability():
    assert build_configuration_id(**_cfg_kwargs()) == build_configuration_id(**_cfg_kwargs())


def test_A2_key_order_independence():
    id1 = build_configuration_id(**_cfg_kwargs(entry_params={"fast": 10, "slow": 50}))
    id2 = build_configuration_id(**_cfg_kwargs(entry_params={"slow": 50, "fast": 10}))
    assert id1 == id2


def test_A3_timeframe_change():
    id1 = build_configuration_id(**_cfg_kwargs(timeframe="1h"))
    id2 = build_configuration_id(**_cfg_kwargs(timeframe="4h"))
    assert id1 != id2


def test_A4_risk_param_change():
    id1 = build_configuration_id(**_cfg_kwargs(risk_params={"max_drawdown": 0.1}))
    id2 = build_configuration_id(**_cfg_kwargs(risk_params={"max_drawdown": 0.2}))
    assert id1 != id2


def test_A5_asset_in_extra_behavioral_params_rejected():
    with pytest.raises(ValueError, match="asset"):
        build_configuration_id(**_cfg_kwargs(extra_behavioral_params={"asset": "BTC"}))


def test_A6_asset_nested_in_entry_params_rejected():
    with pytest.raises(ValueError, match="asset"):
        build_configuration_id(**_cfg_kwargs(entry_params={"fast": 10, "asset": "ETH"}))


def test_A7_forbidden_key_nested_in_behavioral_mapping():
    with pytest.raises(ValueError, match="best_strategy"):
        build_configuration_id(**_cfg_kwargs(risk_params={"nested": {"best_strategy": 1}}))


def test_A8_non_string_key_in_behavioral_mapping():
    with pytest.raises((ValueError, TypeError)):
        build_configuration_id(**_cfg_kwargs(entry_params={1: "invalid"}))


def test_A9_entry_params_list_rejected():
    with pytest.raises(ValueError, match="entry_params.*Mapping"):
        build_configuration_id(**_cfg_kwargs(entry_params=[]))


def test_A9b_slippage_assumptions_list_rejected_even_empty():
    with pytest.raises(ValueError, match="slippage_assumptions.*Mapping"):
        build_configuration_id(**_cfg_kwargs(slippage_assumptions=[]))


def test_A9_no_mutation_of_inputs():
    entry = {"fast": 10, "slow": 50}
    exit_ = {"stop_pct": 0.02}
    risk = {"max_drawdown": 0.1}
    fee = {"taker": 0.001}
    entry_copy = copy.deepcopy(entry)
    exit_copy = copy.deepcopy(exit_)
    risk_copy = copy.deepcopy(risk)
    fee_copy = copy.deepcopy(fee)
    build_configuration_id(
        strategy_id="ema_cross",
        timeframe="1h",
        entry_params=entry,
        exit_params=exit_,
        risk_params=risk,
        fee_assumptions=fee,
    )
    assert entry == entry_copy
    assert exit_ == exit_copy
    assert risk == risk_copy
    assert fee == fee_copy


def test_A10_strategy_id_empty_raises():
    with pytest.raises(ValueError):
        build_configuration_id(**_cfg_kwargs(strategy_id=""))


def test_A10b_strategy_id_whitespace_raises():
    with pytest.raises(ValueError):
        build_configuration_id(**_cfg_kwargs(strategy_id="   "))


def test_A11_timeframe_empty_raises():
    with pytest.raises(ValueError):
        build_configuration_id(**_cfg_kwargs(timeframe=""))


def test_A11b_timeframe_whitespace_raises():
    with pytest.raises(ValueError):
        build_configuration_id(**_cfg_kwargs(timeframe="  "))


def test_A12_fee_forbidden_key():
    with pytest.raises(ValueError, match="winner"):
        build_configuration_id(**_cfg_kwargs(fee_assumptions={"winner": 0.001}))


def test_A13_result_is_sha256_hex():
    result = build_configuration_id(**_cfg_kwargs())
    assert _SHA256_PATTERN.match(result), f"Expected SHA-256 hex, got: {result!r}"


def test_A14_list_order_preserved():
    id1 = build_configuration_id(**_cfg_kwargs(entry_params={"signals": [1, 2, 3]}))
    id2 = build_configuration_id(**_cfg_kwargs(entry_params={"signals": [3, 2, 1]}))
    assert id1 != id2


# ---------------------------------------------------------------------------
# Section B — build_study_id
# ---------------------------------------------------------------------------


def test_B1_stability():
    assert build_study_id(**_study_kwargs()) == build_study_id(**_study_kwargs())


def test_B2_configuration_ids_order_independence():
    id1 = build_study_id(**_study_kwargs(configuration_ids=[_SHA_A, _SHA_B]))
    id2 = build_study_id(**_study_kwargs(configuration_ids=[_SHA_B, _SHA_A]))
    assert id1 == id2


def test_B3_adding_configuration_id_changes_result():
    id1 = build_study_id(**_study_kwargs(configuration_ids=[_SHA_A, _SHA_B]))
    id2 = build_study_id(**_study_kwargs(configuration_ids=[_SHA_A, _SHA_B, _SHA_C]))
    assert id1 != id2


def test_B4_removing_configuration_id_changes_result():
    id1 = build_study_id(**_study_kwargs(configuration_ids=[_SHA_A, _SHA_B]))
    id2 = build_study_id(**_study_kwargs(configuration_ids=[_SHA_A]))
    assert id1 != id2


def test_B5_duplicates_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        build_study_id(**_study_kwargs(configuration_ids=[_SHA_A, _SHA_A]))


def test_B6_empty_sequence_rejected():
    with pytest.raises(ValueError, match="empty"):
        build_study_id(**_study_kwargs(configuration_ids=[]))


def test_B7_comparison_protocol_change():
    id1 = build_study_id(**_study_kwargs(comparison_protocol="walk_forward_oos"))
    id2 = build_study_id(**_study_kwargs(comparison_protocol="holdout_only"))
    assert id1 != id2


def test_B8_validation_policy_change():
    id1 = build_study_id(**_study_kwargs(validation_policy={"observation_unit": "periodic_return"}))
    id2 = build_study_id(**_study_kwargs(validation_policy={"observation_unit": "per_trade"}))
    assert id1 != id2


def test_B9_configuration_ids_source_not_mutated():
    ids = [_SHA_B, _SHA_A]
    ids_copy = copy.deepcopy(ids)
    build_study_id(**_study_kwargs(configuration_ids=ids))
    assert ids == ids_copy


def test_B10_malformed_configuration_id_rejected():
    with pytest.raises(ValueError):
        build_study_id(**_study_kwargs(configuration_ids=["not-a-sha256", _SHA_B]))


def test_B11_forbidden_key_nested_in_validation_policy():
    with pytest.raises(ValueError, match="winner"):
        build_study_id(**_study_kwargs(
            validation_policy={"obs": "periodic_return", "nested": {"winner": True}}
        ))


def test_B12_non_string_key_in_validation_policy():
    with pytest.raises((ValueError, TypeError)):
        build_study_id(**_study_kwargs(validation_policy={42: "bad"}))


def test_B12b_validation_policy_string_rejected():
    with pytest.raises(ValueError, match="validation_policy.*Mapping"):
        build_study_id(**_study_kwargs(validation_policy="path.yaml"))


def test_B13_forbidden_key_in_extra_methodological_policies():
    with pytest.raises(ValueError, match="global_score"):
        build_study_id(**_study_kwargs(extra_methodological_policies={"global_score": 1.0}))


def test_B13b_extra_methodological_policies_list_rejected():
    with pytest.raises(ValueError, match="extra_methodological_policies.*Mapping"):
        build_study_id(**_study_kwargs(extra_methodological_policies=[]))


def test_B14_result_is_sha256_hex():
    result = build_study_id(**_study_kwargs())
    assert _SHA256_PATTERN.match(result), f"Expected SHA-256 hex, got: {result!r}"


# ---------------------------------------------------------------------------
# Section C — generate_run_id
# ---------------------------------------------------------------------------


def test_C1_format_valid():
    result = generate_run_id()
    assert _RUN_ID_PATTERN.match(result), f"Unexpected format: {result!r}"


def test_C2_uniqueness():
    id1 = generate_run_id()
    id2 = generate_run_id()
    assert id1 != id2


def test_C3_monkeypatch_deterministic(monkeypatch):
    import uuid as _uuid

    fixed_hex = "deadbeef" * 8  # 64 chars hex, but we only use 32
    fixed_hex_32 = "deadbeef" * 4  # 32 chars
    fake_uuid = types.SimpleNamespace(hex=fixed_hex_32)
    monkeypatch.setattr(_uuid, "uuid4", lambda: fake_uuid)

    result = generate_run_id()
    assert result == f"run-{fixed_hex_32}"
    assert _RUN_ID_PATTERN.match(result)


def test_C4_no_positional_argument_accepted():
    with pytest.raises(TypeError):
        generate_run_id("unexpected")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Section D — build_run_input_fingerprint
# ---------------------------------------------------------------------------


def test_D1_stability():
    assert build_run_input_fingerprint(**_run_fp_kwargs()) == build_run_input_fingerprint(**_run_fp_kwargs())


def test_D2_extra_run_params_key_order_independence():
    id1 = build_run_input_fingerprint(**_run_fp_kwargs(extra_run_params={"x": 1, "y": 2}))
    id2 = build_run_input_fingerprint(**_run_fp_kwargs(extra_run_params={"y": 2, "x": 1}))
    assert id1 == id2


def test_D3_input_fingerprints_change():
    id1 = build_run_input_fingerprint(**_run_fp_kwargs(input_fingerprints={"ohlcv": _SHA_C}))
    id2 = build_run_input_fingerprint(**_run_fp_kwargs(input_fingerprints={"ohlcv": _SHA_D}))
    assert id1 != id2


def test_D4_capital_mode_change():
    id1 = build_run_input_fingerprint(**_run_fp_kwargs(capital_mode="reset_per_window"))
    id2 = build_run_input_fingerprint(**_run_fp_kwargs(capital_mode="continuous"))
    assert id1 != id2


def test_D5_boundary_policy_change():
    id1 = build_run_input_fingerprint(**_run_fp_kwargs(boundary_position_policy="force_close"))
    id2 = build_run_input_fingerprint(**_run_fp_kwargs(boundary_position_policy="reject_open_position"))
    assert id1 != id2


def test_D6_gap_policy_change():
    id1 = build_run_input_fingerprint(**_run_fp_kwargs(gap_policy="reject"))
    id2 = build_run_input_fingerprint(**_run_fp_kwargs(gap_policy="segment"))
    assert id1 != id2


def test_D7_extra_run_params_not_mutated():
    params = {"alpha": 1, "beta": 2}
    params_copy = copy.deepcopy(params)
    build_run_input_fingerprint(**_run_fp_kwargs(extra_run_params=params))
    assert params == params_copy


def test_D8_window_definitions_not_mutated_and_order_preserved():
    w1 = {"index": 0, "train_start": "2020-01-01"}
    w2 = {"index": 1, "train_start": "2021-01-01"}
    windows = [w1, w2]
    windows_copy = copy.deepcopy(windows)
    build_run_input_fingerprint(**_run_fp_kwargs(window_definitions=windows))
    assert windows == windows_copy


def test_D9_window_order_is_semantic():
    w1 = {"index": 0, "train_start": "2020-01-01"}
    w2 = {"index": 1, "train_start": "2021-01-01"}
    id1 = build_run_input_fingerprint(**_run_fp_kwargs(window_definitions=[w1, w2]))
    id2 = build_run_input_fingerprint(**_run_fp_kwargs(window_definitions=[w2, w1]))
    assert id1 != id2


def test_D10_distinct_from_run_id():
    fp = build_run_input_fingerprint(**_run_fp_kwargs())
    run = generate_run_id()
    # run_input_fingerprint is a 64-char hex; run_id is 'run-<32 hex>'
    assert fp != run
    assert _SHA256_PATTERN.match(fp)
    assert _RUN_ID_PATTERN.match(run)


def test_D11_invalid_capital_mode():
    with pytest.raises(ValueError, match="capital_mode"):
        build_run_input_fingerprint(**_run_fp_kwargs(capital_mode="invalid_mode"))


def test_D12_invalid_boundary_policy():
    with pytest.raises(ValueError, match="boundary_position_policy"):
        build_run_input_fingerprint(**_run_fp_kwargs(boundary_position_policy="unknown"))


def test_D13_empty_input_fingerprints_rejected():
    with pytest.raises(ValueError, match="empty"):
        build_run_input_fingerprint(**_run_fp_kwargs(input_fingerprints={}))


def test_D13b_input_fingerprints_list_rejected():
    with pytest.raises(ValueError, match="input_fingerprints.*Mapping"):
        build_run_input_fingerprint(**_run_fp_kwargs(input_fingerprints=[]))


def test_D14_invalid_sha256_value_in_input_fingerprints():
    with pytest.raises(ValueError):
        build_run_input_fingerprint(**_run_fp_kwargs(input_fingerprints={"ohlcv": "not-a-hash"}))


def test_D15_empty_label_in_input_fingerprints():
    with pytest.raises(ValueError):
        build_run_input_fingerprint(**_run_fp_kwargs(input_fingerprints={"": _SHA_C}))


def test_D15b_whitespace_label_in_input_fingerprints():
    with pytest.raises(ValueError):
        build_run_input_fingerprint(**_run_fp_kwargs(input_fingerprints={"   ": _SHA_C}))


def test_D16_label_with_forward_slash_rejected():
    with pytest.raises(ValueError):
        build_run_input_fingerprint(**_run_fp_kwargs(input_fingerprints={"data/ohlcv": _SHA_C}))


def test_D17_label_with_backslash_rejected():
    with pytest.raises(ValueError):
        build_run_input_fingerprint(**_run_fp_kwargs(input_fingerprints={"data\\ohlcv": _SHA_C}))


def test_D18_malformed_experiment_id_rejected():
    with pytest.raises(ValueError):
        build_run_input_fingerprint(**_run_fp_kwargs(experiment_id="short"))


def test_D19_file_path_nested_in_window_definitions_rejected():
    with pytest.raises(ValueError, match="file_path"):
        build_run_input_fingerprint(**_run_fp_kwargs(
            window_definitions=[{"index": 0, "file_path": "/data/file.csv"}]
        ))


def test_D19b_window_definitions_non_mapping_element_rejected():
    with pytest.raises(ValueError, match="window_definitions\\[0\\].*Mapping"):
        build_run_input_fingerprint(**_run_fp_kwargs(window_definitions=["bad"]))


def test_D20_forbidden_concept_nested_in_extra_run_params():
    with pytest.raises(ValueError, match="best_strategy"):
        build_run_input_fingerprint(**_run_fp_kwargs(
            extra_run_params={"meta": {"best_strategy": "ema_cross"}}
        ))


def test_D20b_extra_run_params_list_rejected():
    with pytest.raises(ValueError, match="extra_run_params.*Mapping"):
        build_run_input_fingerprint(**_run_fp_kwargs(extra_run_params=[]))


def test_D21_result_is_sha256_hex():
    result = build_run_input_fingerprint(**_run_fp_kwargs())
    assert _SHA256_PATTERN.match(result), f"Expected SHA-256 hex, got: {result!r}"


def test_D22_multiple_input_fingerprints_key_order_independence():
    id1 = build_run_input_fingerprint(**_run_fp_kwargs(
        input_fingerprints={"ohlcv": _SHA_C, "features": _SHA_D}
    ))
    id2 = build_run_input_fingerprint(**_run_fp_kwargs(
        input_fingerprints={"features": _SHA_D, "ohlcv": _SHA_C}
    ))
    assert id1 == id2


# ---------------------------------------------------------------------------
# Section E — Anti-concepts interdits
# ---------------------------------------------------------------------------


def test_E1_best_strategy_nested_in_entry_params_rejected():
    with pytest.raises(ValueError, match="best_strategy"):
        build_configuration_id(**_cfg_kwargs(entry_params={"config": {"best_strategy": "x"}}))


def test_E2_winner_nested_in_validation_policy_rejected():
    with pytest.raises(ValueError, match="winner"):
        build_study_id(**_study_kwargs(validation_policy={"rules": {"winner": True}}))


def test_E3_selected_strategy_nested_in_risk_params_rejected():
    with pytest.raises(ValueError, match="selected_strategy"):
        build_configuration_id(**_cfg_kwargs(risk_params={"meta": {"selected_strategy": "rsi"}}))


def test_E4_ranking_in_list_in_extra_behavioral_params_rejected():
    with pytest.raises(ValueError, match="ranking"):
        build_configuration_id(**_cfg_kwargs(
            extra_behavioral_params={"options": [{"ranking": 1}]}
        ))


def test_E5_no_disk_writes(tmp_path, monkeypatch):
    original_open = builtins.open
    original_path_open = pathlib.Path.open
    original_write_text = pathlib.Path.write_text
    original_write_bytes = pathlib.Path.write_bytes
    original_touch = pathlib.Path.touch

    def guard_open(file, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "a", "x")):
            raise AssertionError(f"Unexpected file write: {file!r} mode={mode!r}")
        return original_open(file, mode, *args, **kwargs)

    def guard_path_open(self, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "a", "x")):
            raise AssertionError(f"Unexpected pathlib write: {self!r} mode={mode!r}")
        return original_path_open(self, mode, *args, **kwargs)

    def guard_write_text(self, *args, **kwargs):
        raise AssertionError(f"Unexpected pathlib write_text: {self!r}")

    def guard_write_bytes(self, *args, **kwargs):
        raise AssertionError(f"Unexpected pathlib write_bytes: {self!r}")

    def guard_touch(self, *args, **kwargs):
        raise AssertionError(f"Unexpected pathlib touch: {self!r}")

    monkeypatch.setattr(builtins, "open", guard_open)
    monkeypatch.setattr(pathlib.Path, "open", guard_path_open)
    monkeypatch.setattr(pathlib.Path, "write_text", guard_write_text)
    monkeypatch.setattr(pathlib.Path, "write_bytes", guard_write_bytes)
    monkeypatch.setattr(pathlib.Path, "touch", guard_touch)
    monkeypatch.chdir(tmp_path)

    build_configuration_id(**_cfg_kwargs())
    build_study_id(**_study_kwargs())
    generate_run_id()
    build_run_input_fingerprint(**_run_fp_kwargs())

    assert list(tmp_path.rglob("*")) == []


# ---------------------------------------------------------------------------
# Schema version sanity
# ---------------------------------------------------------------------------


def test_schema_version_is_integer():
    assert isinstance(EXPERIMENT_IDENTITY_SCHEMA_VERSION, int)
    assert EXPERIMENT_IDENTITY_SCHEMA_VERSION >= 1


def test_valid_capital_modes_non_empty():
    assert VALID_CAPITAL_MODES


def test_valid_boundary_policies_non_empty():
    assert VALID_BOUNDARY_POLICIES

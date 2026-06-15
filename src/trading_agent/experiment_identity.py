from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from trading_agent.fingerprint import fingerprint_payload

EXPERIMENT_IDENTITY_SCHEMA_VERSION = 1

_FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "best_strategy",
    "best_asset",
    "winner",
    "rank",
    "ranking",
    "global_score",
    "selected_strategy",
})

_FORBIDDEN_RUN_PARAM_KEYS: frozenset[str] = _FORBIDDEN_KEYS | frozenset({
    "created_at",
    "artifact_path",
    "file_path",
    "input_path",
    "output_path",
})

_RESERVED_FINGERPRINT_LABEL_KEYS: frozenset[str] = frozenset({
    "file_path",
    "input_path",
    "output_path",
    "artifact_path",
})

VALID_CAPITAL_MODES: frozenset[str] = frozenset({
    "reset_per_window",
    "continuous",
})

VALID_BOUNDARY_POLICIES: frozenset[str] = frozenset({
    "force_close",
    "carry",
    "reject_open_position",
})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_non_empty_str(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")


def _validate_sha256_hex(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise ValueError(
            f"{label} must be a 64-char lowercase hex SHA-256 string, got {value!r}"
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a Mapping, got {value!r}")
    return value


def _validate_mapping_tree(
    obj: Any,
    *,
    forbidden_keys: frozenset[str],
    context: str,
) -> None:
    """Recursively validate a structure: all mapping keys must be str and not forbidden."""
    if isinstance(obj, Mapping):
        for key in obj:
            if not isinstance(key, str):
                raise ValueError(
                    f"all mapping keys must be str in {context}; "
                    f"got key {key!r} of type {type(key).__name__!r}"
                )
            if key in forbidden_keys:
                raise ValueError(
                    f"forbidden key {key!r} found in {context}"
                )
            _validate_mapping_tree(obj[key], forbidden_keys=forbidden_keys, context=f"{context}[{key!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _validate_mapping_tree(item, forbidden_keys=forbidden_keys, context=f"{context}[{i}]")


def _validate_input_fingerprints_keys(keys: Iterable[str]) -> None:
    for label in keys:
        if not isinstance(label, str):
            raise ValueError(
                f"input_fingerprints keys must be str, got {label!r}"
            )
        if not label.strip():
            raise ValueError(
                f"input_fingerprints key must not be empty or whitespace-only, got {label!r}"
            )
        if "/" in label or "\\" in label:
            raise ValueError(
                f"input_fingerprints key must not contain '/' or '\\', got {label!r}"
            )
        if label in _RESERVED_FINGERPRINT_LABEL_KEYS:
            raise ValueError(
                f"input_fingerprints key {label!r} is reserved and cannot be used as a label"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_configuration_id(
    *,
    strategy_id: str,
    timeframe: str,
    entry_params: Mapping[str, Any],
    exit_params: Mapping[str, Any],
    risk_params: Mapping[str, Any],
    fee_assumptions: Mapping[str, Any],
    slippage_assumptions: Mapping[str, Any] | None = None,
    extra_behavioral_params: Mapping[str, Any] | None = None,
) -> str:
    """Return a deterministic SHA-256 identity for a behavioral strategy configuration.

    The asset is excluded by design and cannot be injected into any behavioral mapping.
    Cross-run stability: changing timeframe or any behavioral parameter changes the identity.
    """
    _validate_non_empty_str(strategy_id, "strategy_id")
    _validate_non_empty_str(timeframe, "timeframe")

    behavioral_forbidden = _FORBIDDEN_KEYS | {"asset"}
    _entry = _require_mapping(entry_params, "entry_params")
    _exit = _require_mapping(exit_params, "exit_params")
    _risk = _require_mapping(risk_params, "risk_params")
    _fee = _require_mapping(fee_assumptions, "fee_assumptions")
    _slippage = {} if slippage_assumptions is None else _require_mapping(slippage_assumptions, "slippage_assumptions")
    _extra = {} if extra_behavioral_params is None else _require_mapping(extra_behavioral_params, "extra_behavioral_params")

    for mapping, label in (
        (_entry, "entry_params"),
        (_exit, "exit_params"),
        (_risk, "risk_params"),
        (_fee, "fee_assumptions"),
        (_slippage, "slippage_assumptions"),
        (_extra, "extra_behavioral_params"),
    ):
        _validate_mapping_tree(mapping, forbidden_keys=behavioral_forbidden, context=label)

    payload = {
        "schema_version": EXPERIMENT_IDENTITY_SCHEMA_VERSION,
        "identity_type": "configuration_id",
        "strategy_id": strategy_id,
        "timeframe": timeframe,
        "entry_params": _entry,
        "exit_params": _exit,
        "risk_params": _risk,
        "fee_assumptions": _fee,
        "slippage_assumptions": _slippage,
        "extra_behavioral_params": _extra,
    }
    return fingerprint_payload(payload)


def build_study_id(
    *,
    study_name: str,
    study_version: str,
    configuration_ids: Sequence[str],
    comparison_protocol: str,
    selection_rule: str,
    validation_policy: Mapping[str, Any],
    data_scope: str,
    extra_methodological_policies: Mapping[str, Any] | None = None,
) -> str:
    """Return a deterministic SHA-256 identity for a pre-declared study scope.

    Adding, removing, or modifying a configuration_id changes the study_id.
    Input order of configuration_ids has no effect (sorted internally).
    Duplicate configuration_ids are rejected rather than silently deduplicated.
    """
    _validate_non_empty_str(study_name, "study_name")
    _validate_non_empty_str(study_version, "study_version")
    _validate_non_empty_str(comparison_protocol, "comparison_protocol")
    _validate_non_empty_str(selection_rule, "selection_rule")
    _validate_non_empty_str(data_scope, "data_scope")

    ids_list = list(configuration_ids)
    if not ids_list:
        raise ValueError("configuration_ids must not be empty")
    for cid in ids_list:
        _validate_sha256_hex(cid, "configuration_id in configuration_ids")
    if len(ids_list) != len(set(ids_list)):
        seen: set[str] = set()
        duplicates = [c for c in ids_list if c in seen or seen.add(c)]  # type: ignore[func-returns-value]
        raise ValueError(
            f"configuration_ids must not contain duplicates; found: {duplicates!r}"
        )

    _validation_policy = _require_mapping(validation_policy, "validation_policy")
    _extra = {} if extra_methodological_policies is None else _require_mapping(extra_methodological_policies, "extra_methodological_policies")
    _validate_mapping_tree(_validation_policy, forbidden_keys=_FORBIDDEN_KEYS, context="validation_policy")
    _validate_mapping_tree(_extra, forbidden_keys=_FORBIDDEN_KEYS, context="extra_methodological_policies")

    payload = {
        "schema_version": EXPERIMENT_IDENTITY_SCHEMA_VERSION,
        "identity_type": "study_id",
        "study_name": study_name,
        "study_version": study_version,
        "configuration_ids": sorted(ids_list),
        "comparison_protocol": comparison_protocol,
        "selection_rule": selection_rule,
        "validation_policy": dict(_validation_policy),
        "data_scope": data_scope,
        "extra_methodological_policies": dict(_extra),
    }
    return fingerprint_payload(payload)


def generate_run_id() -> str:
    """Return a unique run occurrence identifier.

    Format: 'run-<32 lowercase hex chars>' (UUID4-based).
    Not deterministic by design: each call produces a distinct value.
    Not derived from experiment content — use run_input_fingerprint for content identity.
    """
    return f"run-{uuid.uuid4().hex}"


def build_run_input_fingerprint(
    *,
    experiment_id: str,
    configuration_id: str,
    input_fingerprints: Mapping[str, str],
    capital_mode: str,
    boundary_position_policy: str,
    gap_policy: str,
    study_id: str | None = None,
    window_definitions: Sequence[Mapping[str, Any]] | None = None,
    extra_run_params: Mapping[str, Any] | None = None,
) -> str:
    """Return a deterministic SHA-256 fingerprint of declared run inputs.

    Conceptually distinct from run_id (occurrence), experiment_id (methodological definition),
    and configuration_id (behavioral definition).

    Same run_input_fingerprint across two runs indicates the same declared inputs were used.
    It does NOT automatically imply the same experiment_id, nor count as a new statistical trial.

    Note: cross-validation of capital_mode / boundary_position_policy combinations
    (e.g. reset_per_window + carry) is deferred to v0.53 per architecture contract §11.1.
    Only set membership is validated here.
    """
    _validate_sha256_hex(experiment_id, "experiment_id")
    _validate_sha256_hex(configuration_id, "configuration_id")
    if study_id is not None:
        _validate_sha256_hex(study_id, "study_id")

    _input_fingerprints = _require_mapping(input_fingerprints, "input_fingerprints")
    if not _input_fingerprints:
        raise ValueError("input_fingerprints must not be empty")
    _validate_input_fingerprints_keys(_input_fingerprints.keys())
    for label, fp in _input_fingerprints.items():
        _validate_sha256_hex(fp, f"input_fingerprints[{label!r}]")

    if capital_mode not in VALID_CAPITAL_MODES:
        raise ValueError(
            f"capital_mode {capital_mode!r} is not valid; "
            f"expected one of {sorted(VALID_CAPITAL_MODES)}"
        )
    if boundary_position_policy not in VALID_BOUNDARY_POLICIES:
        raise ValueError(
            f"boundary_position_policy {boundary_position_policy!r} is not valid; "
            f"expected one of {sorted(VALID_BOUNDARY_POLICIES)}"
        )

    _validate_non_empty_str(gap_policy, "gap_policy")

    _windows = [] if window_definitions is None else list(window_definitions)
    for i, w in enumerate(_windows):
        _require_mapping(w, f"window_definitions[{i}]")
        _validate_mapping_tree(w, forbidden_keys=_FORBIDDEN_RUN_PARAM_KEYS, context=f"window_definitions[{i}]")

    _extra = {} if extra_run_params is None else _require_mapping(extra_run_params, "extra_run_params")
    _validate_mapping_tree(_extra, forbidden_keys=_FORBIDDEN_RUN_PARAM_KEYS, context="extra_run_params")

    payload = {
        "schema_version": EXPERIMENT_IDENTITY_SCHEMA_VERSION,
        "identity_type": "run_input_fingerprint",
        "experiment_id": experiment_id,
        "configuration_id": configuration_id,
        "study_id": study_id,
        "input_fingerprints": dict(_input_fingerprints),
        "capital_mode": capital_mode,
        "boundary_position_policy": boundary_position_policy,
        "gap_policy": gap_policy,
        "window_definitions": [dict(w) for w in _windows],
        "extra_run_params": dict(_extra),
    }
    return fingerprint_payload(payload)

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_agent.fingerprint import (
    build_input_fingerprints,
    canonicalize_for_fingerprint,
    fingerprint_payload,
)

EXPERIMENT_MANIFEST_SCHEMA_VERSION = 1

VALID_DATA_ROLES: frozenset[str] = frozenset({
    "exploratory_data",
    "confirmatory_holdout",
    "prospective_holdout_candidate",
    "paper_forward",
})

_MANIFEST_FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "best_strategy",
    "best_asset",
    "winner",
    "rank",
    "ranking",
    "global_score",
    "selected_strategy",
})

_REQUIRED_WF_KEYS: frozenset[str] = frozenset({"train_window", "test_window", "step"})


def _normalize_created_at(created_at: datetime | pd.Timestamp) -> str:
    if isinstance(created_at, pd.Timestamp):
        if created_at.tzinfo is None or created_at.tzinfo.utcoffset(created_at) is None:
            raise ValueError("created_at must be timezone-aware")
        return created_at.tz_convert("UTC").isoformat().replace("+00:00", "Z")
    if isinstance(created_at, datetime):
        if created_at.tzinfo is None or created_at.tzinfo.utcoffset(created_at) is None:
            raise ValueError("created_at must be timezone-aware")
        return created_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    raise TypeError(
        f"created_at must be datetime or pd.Timestamp, got {type(created_at).__name__!r}"
    )


def _check_no_forbidden_keys(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            if k in _MANIFEST_FORBIDDEN_KEYS:
                raise ValueError(
                    f"forbidden key {k!r} found in mapping at {path!r}"
                )
            _check_no_forbidden_keys(v, path=f"{path}.{k}" if path else k)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_no_forbidden_keys(item, path=f"{path}[{i}]")


def build_experiment_manifest(
    *,
    dataframe: pd.DataFrame,
    strategy: Mapping[str, Any],
    goal: Mapping[str, Any],
    strategy_id: str,
    asset: str,
    timeframe: str,
    walk_forward_parameters: Mapping[str, Any],
    data_role: str,
    validation_context: Mapping[str, Any],
    created_at: datetime | pd.Timestamp,
    validation_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a pure, deterministic experiment manifest from in-memory objects.

    Does not read or write files, run backtests, collect git info, or make
    network calls. All inputs must be provided by the caller.

    Returns a JSON-compatible dict. The ``experiment_id`` is stable across
    calls with identical inputs regardless of ``created_at``.
    """
    # 1. Validate Mapping types before accessing their keys
    if not isinstance(walk_forward_parameters, Mapping):
        raise TypeError(
            f"walk_forward_parameters must be a Mapping, got {type(walk_forward_parameters).__name__!r}"
        )
    if not isinstance(validation_context, Mapping):
        raise TypeError(
            f"validation_context must be a Mapping, got {type(validation_context).__name__!r}"
        )

    # 2. Validate data_role
    if data_role not in VALID_DATA_ROLES:
        raise ValueError(
            f"data_role {data_role!r} is not valid; expected one of {sorted(VALID_DATA_ROLES)}"
        )

    # 3. Validate non-empty strings
    for name, value in [
        ("strategy_id", strategy_id),
        ("asset", asset),
        ("timeframe", timeframe),
    ]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string, got {value!r}")

    # 4. Validate required walk_forward_parameters keys
    missing = _REQUIRED_WF_KEYS - set(walk_forward_parameters)
    if missing:
        raise ValueError(
            f"walk_forward_parameters is missing required keys: {sorted(missing)}"
        )

    # 5. Recursive check for forbidden keys in caller-supplied mappings
    _check_no_forbidden_keys(walk_forward_parameters, path="walk_forward_parameters")
    _check_no_forbidden_keys(validation_context, path="validation_context")

    # 6. Normalize created_at → ISO 8601 UTC with Z suffix
    created_at_str = _normalize_created_at(created_at)

    # 7. Canonicalize mappings: JSON-compatible, detached from caller's objects
    wf_params_canonical = canonicalize_for_fingerprint(dict(walk_forward_parameters))
    val_ctx_canonical = canonicalize_for_fingerprint(dict(validation_context))

    # 8. Compute input fingerprints (reuses v0.48 primitives)
    input_fingerprints = build_input_fingerprints(
        dataframe=dataframe,
        strategy=strategy,
        goal=goal,
        validation_policy=validation_policy,
    )

    # 9. Compute experiment_id from identity payload (excludes created_at)
    identity_payload = {
        "manifest_schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "strategy_id": strategy_id,
        "asset": asset,
        "timeframe": timeframe,
        "data_role": data_role,
        "walk_forward_parameters": wf_params_canonical,
        "validation_context": val_ctx_canonical,
        "input_fingerprints": input_fingerprints,
    }
    experiment_id = fingerprint_payload(identity_payload)

    return {
        "manifest_schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "created_at": created_at_str,
        "strategy_id": strategy_id,
        "asset": asset,
        "timeframe": timeframe,
        "data_role": data_role,
        "walk_forward_parameters": wf_params_canonical,
        "validation_context": val_ctx_canonical,
        "input_fingerprints": input_fingerprints,
    }

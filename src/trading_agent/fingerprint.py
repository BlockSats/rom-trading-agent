from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FINGERPRINT_SCHEMA_VERSION = 1


def canonicalize_for_fingerprint(value: Any) -> Any:
    """Convert any supported Python value to a JSON-compatible, deterministic structure.

    Rules:
    - None, pd.NA, pd.NaT → null
    - numpy scalars → converted to Python native via .item() before dispatch
    - bool (before int, since bool is a subclass of int)
    - int, str → unchanged
    - float: NaN → {"__special__": "nan"}, +inf → {"__special__": "+inf"},
              -inf → {"__special__": "-inf"}, finite → unchanged
    - pathlib.Path → str
    - pd.Timestamp / datetime.datetime: tz-aware normalized to UTC ISO, naive as ISO
    - Mapping (dict, etc.): all keys must be str; keys sorted; values recursed
    - list / tuple → list (both treated identically, order preserved)
    - set → TypeError (non-deterministic order)
    - any other type → TypeError

    The original argument is never modified.
    """
    if value is None:
        return None

    # pandas NA and NaT singletons — check before numpy/bool/float
    try:
        if value is pd.NA:
            return None
    except Exception:
        pass
    try:
        if value is pd.NaT:
            return None
    except Exception:
        pass

    # numpy scalars — checked before Python primitives because np.float64 and
    # np.int64 are subclasses of float and int respectively
    if isinstance(value, np.generic):
        return canonicalize_for_fingerprint(value.item())

    # bool before int (bool is a subclass of int in Python)
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return {"__special__": "nan"}
        if math.isinf(value):
            return {"__special__": "+inf"} if value > 0 else {"__special__": "-inf"}
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.tz_convert("UTC").isoformat()
        return value.isoformat()

    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).isoformat()
        return value.isoformat()

    if isinstance(value, Mapping):
        items = list(value.items())
        for k, _ in items:
            if not isinstance(k, str):
                raise TypeError(
                    f"mapping keys must be str for deterministic fingerprinting, "
                    f"got {type(k).__name__!r}: {k!r}"
                )
        return {k: canonicalize_for_fingerprint(v) for k, v in sorted(items)}

    if isinstance(value, set):
        raise TypeError(
            "set is not supported: set order is non-deterministic. Use list or tuple."
        )

    if isinstance(value, (list, tuple)):
        return [canonicalize_for_fingerprint(v) for v in value]

    raise TypeError(
        f"unsupported type for fingerprinting: {type(value).__name__!r}"
    )


def _json_dumps_canonical(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def fingerprint_payload(value: Any) -> str:
    """Return the SHA-256 hex digest of the canonical JSON encoding of `value`."""
    canonical = canonicalize_for_fingerprint(value)
    return hashlib.sha256(_json_dumps_canonical(canonical)).hexdigest()


def fingerprint_dataframe(df: pd.DataFrame) -> str:
    """Return the SHA-256 hex digest of the canonical representation of `df`.

    Column order does not affect the hash (columns are sorted).
    Row order is significant (rows are not sorted).
    The pandas index is ignored.
    All column names must be str and unique.
    """
    columns = df.columns.tolist()

    non_string = [c for c in columns if not isinstance(c, str)]
    if non_string:
        raise TypeError(
            f"all DataFrame column names must be str; "
            f"non-string columns found: {non_string!r}"
        )

    if len(columns) != len(set(columns)):
        seen: set[str] = set()
        duplicates = [c for c in columns if c in seen or seen.add(c)]  # type: ignore[func-returns-value]
        raise ValueError(
            f"DataFrame column names must be unique; duplicates found: {duplicates!r}"
        )

    sorted_columns = sorted(columns)

    rows = []
    for i in range(len(df)):
        row = {col: canonicalize_for_fingerprint(df[col].iat[i]) for col in sorted_columns}
        rows.append(row)

    payload = {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "columns": sorted_columns,
        "dtypes": {col: str(df[col].dtype) for col in sorted_columns},
        "row_count": len(df),
        "rows": rows,
    }

    return hashlib.sha256(_json_dumps_canonical(payload)).hexdigest()


def build_input_fingerprints(
    *,
    dataframe: pd.DataFrame,
    strategy: Mapping[str, Any],
    goal: Mapping[str, Any],
    validation_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return SHA-256 fingerprints for the main inputs of an experiment.

    Returns a dict with keys:
        fingerprint_schema_version  int
        dataframe_sha256            str
        strategy_sha256             str
        goal_sha256                 str
        validation_policy_sha256    str | None
    """
    return {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "dataframe_sha256": fingerprint_dataframe(dataframe),
        "strategy_sha256": fingerprint_payload(dict(strategy)),
        "goal_sha256": fingerprint_payload(dict(goal)),
        "validation_policy_sha256": (
            fingerprint_payload(dict(validation_policy))
            if validation_policy is not None
            else None
        ),
    }

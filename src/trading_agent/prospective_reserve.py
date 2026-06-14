from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_agent.data import detect_time_gaps, load_ohlcv_csv
from trading_agent.storage import append_jsonl, read_jsonl


class ProspectiveReserveError(Exception):
    error_code: str = "prospective_reserve_error"


class InvalidMetadataError(ProspectiveReserveError):
    error_code = "invalid_metadata"


class DuplicateBatchError(ProspectiveReserveError):
    error_code = "duplicate_batch"


class ReserveFileExistsError(ProspectiveReserveError):
    error_code = "reserve_file_exists"


class InvalidOHLCVError(ProspectiveReserveError):
    error_code = "invalid_ohlcv"


def compute_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_text_metadata(asset: str, timeframe: str, source: str) -> None:
    for name, value in [("asset", asset), ("timeframe", timeframe), ("source", source)]:
        if not isinstance(value, str) or not value.strip():
            raise InvalidMetadataError(f"{name} must be a non-empty string")


def build_batch_record(
    *,
    batch_id: str,
    asset: str,
    timeframe: str,
    source: str,
    source_path: str,
    stored_path: str,
    file_sha256: str,
    rows: int,
    first_timestamp: str,
    last_timestamp: str,
    gaps: list[dict[str, Any]],
    registered_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_type": "prospective_batch_archived",
        "batch_id": batch_id,
        "data_role": "prospective_holdout_candidate",
        "status": "accumulating",
        "asset": asset,
        "timeframe": timeframe,
        "source": source,
        "source_path": source_path,
        "stored_path": stored_path,
        "file_sha256": file_sha256,
        "rows": rows,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "gaps_detected": len(gaps),
        "gaps": gaps,
        "registered_at": registered_at,
        "performance_analysis_performed": False,
        "independence_claimed": False,
    }


def archive_prospective_csv(
    path: str | Path,
    asset: str,
    timeframe: str,
    source: str,
    reserve_dir: str | Path = "data/prospective_reserve",
    registry_path: str | Path = "state/prospective_reserve.jsonl",
    registered_at: str | None = None,
) -> dict[str, Any]:
    validate_text_metadata(asset, timeframe, source)

    try:
        df = load_ohlcv_csv(path)
    except Exception as exc:
        raise InvalidOHLCVError(str(exc)) from exc

    gaps = detect_time_gaps(df)

    source_bytes = Path(path).read_bytes()
    sha256 = hashlib.sha256(source_bytes).hexdigest()
    batch_id = sha256

    existing = read_jsonl(registry_path)
    for entry in existing:
        if entry.get("batch_id") == batch_id or entry.get("file_sha256") == sha256:
            raise DuplicateBatchError(f"already registered: {batch_id}")

    reserve_path = Path(reserve_dir)
    dest_path = reserve_path / f"{sha256}.csv"

    if dest_path.exists():
        raise ReserveFileExistsError(f"reserve file already exists: {dest_path}")

    reserve_path.mkdir(parents=True, exist_ok=True)

    if registered_at is None:
        registered_at = datetime.now(timezone.utc).isoformat()

    first_ts = df["timestamp"].iloc[0]
    last_ts = df["timestamp"].iloc[-1]

    record = build_batch_record(
        batch_id=batch_id,
        asset=asset,
        timeframe=timeframe,
        source=source,
        source_path=str(path),
        stored_path=str(dest_path),
        file_sha256=sha256,
        rows=len(df),
        first_timestamp=first_ts.isoformat(),
        last_timestamp=last_ts.isoformat(),
        gaps=gaps,
        registered_at=registered_at,
    )

    dest_created = False
    try:
        try:
            with dest_path.open("xb") as f:
                dest_created = True
                f.write(source_bytes)
        except FileExistsError:
            raise ReserveFileExistsError(f"reserve file already exists: {dest_path}")

        copied_sha256 = hashlib.sha256(dest_path.read_bytes()).hexdigest()
        if copied_sha256 != sha256:
            raise RuntimeError(f"hash mismatch after copy: {copied_sha256} != {sha256}")

        append_jsonl(registry_path, record)

    except Exception:
        if dest_created and dest_path.exists():
            dest_path.unlink()
        raise

    return record

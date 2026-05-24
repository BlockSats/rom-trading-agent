from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("JSONL records must be objects")
            records.append(record)
    return records


def ensure_state_files() -> None:
    state_dir = Path("state")
    history_dir = state_dir / "history"
    state_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    for path in [state_dir / "trades.jsonl", state_dir / "hypotheses.jsonl"]:
        path.touch(exist_ok=True)

    heartbeat_path = state_dir / "heartbeat.json"
    if not heartbeat_path.exists():
        payload = {"created": True}
        heartbeat_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def save_strategy_version(
    strategy_path: str | Path = "config/strategy.yaml",
    history_dir: str | Path = "state/history",
) -> Path:
    strategy_file = Path(strategy_path)
    history_path = Path(history_dir)
    history_path.mkdir(parents=True, exist_ok=True)

    strategy = yaml.safe_load(strategy_file.read_text(encoding="utf-8"))
    if not isinstance(strategy, dict):
        raise ValueError("strategy YAML must be a mapping")

    version = str(strategy.get("version", "")).strip()
    if not version:
        raise ValueError("strategy version is required")

    versioned_path = history_path / f"strategy_{version}.yaml"
    if versioned_path.exists():
        raise FileExistsError(f"history file already exists: {versioned_path}")

    versioned_path.write_text(yaml.safe_dump(strategy, sort_keys=False), encoding="utf-8")
    return versioned_path


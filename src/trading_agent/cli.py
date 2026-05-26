from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml

from trading_agent.backtest import run_backtest, summarize_backtest
from trading_agent.config import load_goal, load_strategy, validate_goal, validate_strategy
from trading_agent.data import detect_time_gaps, load_ohlcv_csv
from trading_agent.loop import run_once as run_once_once
from trading_agent.reflection import apply_reflection_proposal, propose_one_change
from trading_agent.scoring import score_trades
from trading_agent.storage import (
    append_jsonl,
    ensure_state_files,
    read_jsonl,
    save_strategy_version,
    write_json,
    write_jsonl,
)

app = typer.Typer(add_completion=False)


def echo_json(payload: dict[str, Any]) -> None:
    """Print JSON consistently for all CLI commands."""
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def safe_count_jsonl(path: str) -> int:
    """Return the number of JSONL records, or 0 if the file is missing."""
    file_path = Path(path)
    if not file_path.exists():
        return 0
    return len(read_jsonl(path))


@app.command("check")
def check() -> None:
    """Validate configuration and ensure state files exist."""
    strategy = load_strategy()
    goal = load_goal()
    ensure_state_files()
    validate_strategy(strategy)
    validate_goal(goal)
    typer.echo("Configuration OK. State files ready.")


@app.command("status")
def status() -> None:
    """Show a quick operational status of the trading agent."""
    ensure_state_files()

    strategy = load_strategy()
    goal = load_goal()

    report = {
        "project": "trading-agent",
        "strategy": {
            "version": strategy.get("version"),
            "asset": strategy.get("asset"),
            "timeframe": strategy.get("timeframe"),
        },
        "goal": goal,
        "state": {
            "trades": safe_count_jsonl("state/trades.jsonl"),
            "hypotheses": safe_count_jsonl("state/hypotheses.jsonl"),
        },
        "outputs": {
            "backtest_report_exists": Path("outputs/backtest_report.json").exists(),
            "backtest_trades_exists": Path("outputs/backtest_trades.jsonl").exists(),
        },
    }

    echo_json(report)


@app.command("config")
def config(
    section: str = typer.Option(
        "all",
        "--section",
        "-s",
        help="Configuration section to display: all, strategy, or goal.",
    )
) -> None:
    """Display current strategy and/or goal configuration."""
    strategy = load_strategy()
    goal = load_goal()

    if section == "all":
        payload = {"strategy": strategy, "goal": goal}
    elif section == "strategy":
        payload = {"strategy": strategy}
    elif section == "goal":
        payload = {"goal": goal}
    else:
        typer.echo("Invalid section. Use: all, strategy, or goal.", err=True)
        raise typer.Exit(code=1)

    echo_json(payload)


@app.command("inspect-csv")
def inspect_csv(path: Path) -> None:
    """Inspect an OHLCV CSV without running a backtest."""
    ohlcv = load_ohlcv_csv(path)
    gaps = detect_time_gaps(ohlcv)

    report: dict[str, Any] = {
        "csv_path": str(path),
        "rows": int(len(ohlcv)),
        "columns": list(ohlcv.columns),
        "gaps_detected": int(len(gaps)),
        "gaps": gaps,
    }

    if len(ohlcv) > 0 and "timestamp" in ohlcv.columns:
        report["first_timestamp"] = ohlcv["timestamp"].iloc[0]
        report["last_timestamp"] = ohlcv["timestamp"].iloc[-1]

    echo_json(report)


@app.command("run-once")
def run_once() -> None:
    """Run one paper-trading decision cycle."""
    summary = run_once_once()
    echo_json(summary)


@app.command("score")
def score() -> None:
    """Score recorded trades against the configured goal."""
    trades = read_jsonl("state/trades.jsonl")
    goal = load_goal()
    result = score_trades(trades, goal)
    echo_json(result)


@app.command("backtest-csv")
def backtest_csv(path: Path) -> None:
    """Run a CSV backtest and write output reports."""
    strategy = load_strategy()
    goal = load_goal()
    ohlcv = load_ohlcv_csv(path)
    gaps = detect_time_gaps(ohlcv)

    backtest_result = run_backtest(ohlcv, strategy)
    score_result = score_trades(backtest_result["trades"], goal)
    summary = summarize_backtest(backtest_result["trades"], score_result)

    report = {
        "csv_path": str(path),
        "asset": strategy["asset"],
        "timeframe": strategy["timeframe"],
        "rows": int(len(ohlcv)),
        "gaps_detected": int(len(gaps)),
        "gaps": gaps,
        "initial_balance": backtest_result["initial_balance"],
        **summary,
    }

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    write_json(outputs_dir / "backtest_report.json", report)
    write_jsonl(outputs_dir / "backtest_trades.jsonl", backtest_result["trades"])

    echo_json(report)


@app.command("reflect")
def reflect() -> None:
    """Propose and apply one strategy improvement from trade scoring."""
    ensure_state_files()

    trades = read_jsonl("state/trades.jsonl")
    goal = load_goal()
    score_result = score_trades(trades, goal)

    strategy_path = Path("config/strategy.yaml")
    strategy = load_strategy(strategy_path)

    save_strategy_version(strategy_path=strategy_path)

    proposal = propose_one_change(strategy, score_result)
    updated_strategy = apply_reflection_proposal(strategy, proposal)

    strategy_path.write_text(
        yaml.safe_dump(updated_strategy, sort_keys=False),
        encoding="utf-8",
    )

    append_jsonl("state/hypotheses.jsonl", proposal)

    echo_json(
        {
            "proposal": proposal,
            "new_version": updated_strategy["version"],
        }
    )


if __name__ == "__main__":
    app()

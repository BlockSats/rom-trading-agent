from __future__ import annotations

import json
from pathlib import Path

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


@app.command("check")
def check() -> None:
    strategy = load_strategy()
    goal = load_goal()
    ensure_state_files()
    validate_strategy(strategy)
    validate_goal(goal)
    typer.echo("Configuration OK. State files ready.")


@app.command("run-once")
def run_once() -> None:
    summary = run_once_once()
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("score")
def score() -> None:
    trades = read_jsonl("state/trades.jsonl")
    goal = load_goal()
    result = score_trades(trades, goal)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("backtest-csv")
def backtest_csv(path: Path) -> None:
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
    typer.echo(json.dumps(report, indent=2, sort_keys=True, default=str))


@app.command("reflect")
def reflect() -> None:
    ensure_state_files()
    trades = read_jsonl("state/trades.jsonl")
    goal = load_goal()
    score_result = score_trades(trades, goal)
    strategy_path = Path("config/strategy.yaml")
    strategy = load_strategy(strategy_path)

    save_strategy_version(strategy_path=strategy_path)
    proposal = propose_one_change(strategy, score_result)
    updated_strategy = apply_reflection_proposal(strategy, proposal)

    strategy_path.write_text(yaml.safe_dump(updated_strategy, sort_keys=False), encoding="utf-8")
    append_jsonl("state/hypotheses.jsonl", proposal)
    typer.echo(json.dumps({"proposal": proposal, "new_version": updated_strategy["version"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()

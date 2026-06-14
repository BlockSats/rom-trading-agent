from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer
import yaml

from trading_agent.market_data.binance import fetch_binance_ohlcv, write_ohlcv_csv
from trading_agent.backtest import run_backtest, summarize_backtest
from trading_agent.config import (
    get_active_strategy_id,
    load_goal,
    load_research_policy,
    load_strategy,
    validate_goal,
    validate_strategy,
)
from trading_agent.data import detect_time_gaps, load_ohlcv_csv
from trading_agent.loop import run_once as run_once_once
from trading_agent.reflection import apply_reflection_proposal, propose_one_change
from trading_agent.research_reflection import build_research_reflection_report
from trading_agent.research_robustness import build_research_robustness_report
from trading_agent.scoring import score_trades
from trading_agent.prospective_reserve import (
    archive_prospective_csv as _do_archive_prospective_csv,
    DuplicateBatchError,
    InvalidMetadataError,
    InvalidOHLCVError,
    ReserveFileExistsError,
)
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

def validate_binance_limit(command: str, limit: int) -> None:
    """Fail clearly before network calls when Binance candle limit is invalid."""
    if limit < 1 or limit > 1000:
        echo_json(
            {
                "command": command,
                "status": "failed",
                "reason": "limit_must_be_between_1_and_1000",
                "limit": limit,
            }
        )
        raise typer.Exit(code=1)


def build_strategy_variant(base_strategy: dict[str, Any], strategy_id: str) -> dict[str, Any]:
    """Build an in-memory strategy variant without writing config files."""
    strategy = deepcopy(base_strategy)
    strategy["strategy_id"] = strategy_id

    if strategy_id == "ema_atr_trend":
        strategy["entry"] = {
            "indicator": "ema_atr",
            "fast_ema_period": 12,
            "slow_ema_period": 26,
            "direction": "long",
        }
        strategy["exit"] = {
            "atr_period": 14,
            "atr_stop_multiplier": 2.0,
        }
    elif strategy_id == "donchian_breakout":
        strategy["entry"] = {
            "indicator": "donchian",
            "donchian_period": 20,
            "direction": "long",
        }
        strategy["exit"] = {
            "atr_period": 14,
            "atr_stop_multiplier": 2.0,
        }
        # ATR risk mode: atr_period and atr_stop_multiplier match exit values exactly.
        strategy["risk"] = {
            **strategy["risk"],
            "mode": "atr",
            "risk_pct": 0.5,
            "atr_period": 14,
            "atr_stop_multiplier": 2.0,
        }
    elif strategy_id != "rsi_baseline":
        raise ValueError(f"unsupported strategy_id: {strategy_id}")

    return validate_strategy(strategy)


def summarize_strategy_on_dataframe(
    ohlcv: pd.DataFrame,
    strategy: dict[str, Any],
    goal: dict[str, Any],
) -> dict[str, Any]:
    strategy_id = get_active_strategy_id(strategy)
    backtest_result = run_backtest(ohlcv, strategy)
    score_result = score_trades(backtest_result["trades"], goal)
    summary = summarize_backtest(backtest_result["trades"], score_result)
    return {
        "strategy_id": strategy_id,
        "initial_balance": backtest_result["initial_balance"],
        **summary,
        "winrate": score_result["winrate"],
        "profit_factor": score_result["profit_factor"],
        "expectancy": score_result["expectancy"],
    }


def split_dataframe_windows(df: pd.DataFrame, windows: int) -> list[pd.DataFrame]:
    rows = len(df)
    base_size = rows // windows
    extra_rows = rows % windows
    result = []
    start = 0

    for window_index in range(windows):
        size = base_size + (1 if window_index < extra_rows else 0)
        end = start + size
        result.append(df.iloc[start:end].copy())
        start = end

    return result


def summarize_window_results_by_strategy(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for window_result in results:
        for strategy_result in window_result.get("strategies", []):
            strategy_id = str(strategy_result["strategy_id"])
            summary = grouped.setdefault(
                strategy_id,
                {
                    "strategy_id": strategy_id,
                    "windows": 0,
                    "windows_with_trades": 0,
                    "total_trades": 0,
                    "positive_expectancy_windows": 0,
                    "expectancy_values": [],
                    "profit_factor_values": [],
                    "max_drawdown_values": [],
                },
            )

            closed_trades = int(strategy_result.get("closed_trades", 0))
            expectancy = float(strategy_result.get("expectancy", 0.0))
            profit_factor = strategy_result.get("profit_factor")

            summary["windows"] += 1
            summary["total_trades"] += closed_trades
            if closed_trades > 0:
                summary["windows_with_trades"] += 1
            if expectancy > 0:
                summary["positive_expectancy_windows"] += 1

            summary["expectancy_values"].append(expectancy)
            if profit_factor is not None:
                summary["profit_factor_values"].append(float(profit_factor))
            if "max_drawdown" in strategy_result:
                summary["max_drawdown_values"].append(float(strategy_result["max_drawdown"]))

    strategy_summaries = []
    for summary in grouped.values():
        expectancy_values = summary.pop("expectancy_values")
        profit_factor_values = summary.pop("profit_factor_values")
        max_drawdown_values = summary.pop("max_drawdown_values")
        summary["average_expectancy"] = (
            sum(expectancy_values) / len(expectancy_values)
            if expectancy_values
            else None
        )
        summary["average_profit_factor"] = (
            sum(profit_factor_values) / len(profit_factor_values)
            if profit_factor_values
            else None
        )
        if max_drawdown_values:
            summary["worst_max_drawdown"] = max(max_drawdown_values)
        strategy_summaries.append(summary)

    return strategy_summaries


def _run_windows_for_asset(
    ohlcv: pd.DataFrame,
    windows: int,
    base_strategy: dict[str, Any],
    goal: dict[str, Any],
) -> dict[str, Any]:
    strategy_variants = [
        build_strategy_variant(base_strategy, "rsi_baseline"),
        build_strategy_variant(base_strategy, "ema_atr_trend"),
        build_strategy_variant(base_strategy, "donchian_breakout"),
    ]
    results = []
    start_index = 0
    for window_number, window_df in enumerate(split_dataframe_windows(ohlcv, windows), start=1):
        end_index = start_index + len(window_df) - 1
        results.append({
            "window": window_number,
            "start_index": int(start_index),
            "end_index": int(end_index),
            "rows": int(len(window_df)),
            "strategies": [
                summarize_strategy_on_dataframe(window_df, strategy, goal)
                for strategy in strategy_variants
            ],
        })
        start_index = end_index + 1
    return {
        "results": results,
        "summary_by_strategy": summarize_window_results_by_strategy(results),
    }


def classify_strategy_summary(summary: dict[str, Any], research_policy: dict[str, Any]) -> str:
    thresholds = research_policy["comparison_acceptance"]
    min_total_trades = int(thresholds["min_total_trades"])
    min_windows_with_trades = int(thresholds["min_windows_with_trades"])
    min_positive_expectancy_windows = int(thresholds["min_positive_expectancy_windows"])
    min_average_expectancy = float(thresholds["min_average_expectancy"])
    min_average_profit_factor = float(thresholds["min_average_profit_factor"])

    total_trades = int(summary.get("total_trades", 0))
    windows_with_trades = int(summary.get("windows_with_trades", 0))
    positive_expectancy_windows = int(summary.get("positive_expectancy_windows", 0))
    average_expectancy = summary.get("average_expectancy")
    average_profit_factor = summary.get("average_profit_factor")

    if total_trades < min_total_trades or windows_with_trades < min_windows_with_trades:
        return "insufficient_trades"

    if (
        average_expectancy is None
        or float(average_expectancy) <= min_average_expectancy
        or average_profit_factor is None
        or float(average_profit_factor) < min_average_profit_factor
    ):
        return "weak"

    if positive_expectancy_windows < min_positive_expectancy_windows:
        return "watchlist"

    return "candidate"


def get_classification_reasons(
    summary: dict[str, Any],
    research_policy: dict[str, Any],
    classification: str,
) -> list[str]:
    """Return diagnostic reasons explaining why a strategy received its classification.

    Reasons are purely diagnostic — they never change the classification,
    sort strategies, or select a best strategy.
    """
    thresholds = research_policy["comparison_acceptance"]
    min_total_trades = int(thresholds["min_total_trades"])
    min_windows_with_trades = int(thresholds["min_windows_with_trades"])
    min_positive_expectancy_windows = int(thresholds["min_positive_expectancy_windows"])
    min_average_expectancy = float(thresholds["min_average_expectancy"])
    min_average_profit_factor = float(thresholds["min_average_profit_factor"])

    total_trades = int(summary.get("total_trades", 0))
    windows_with_trades = int(summary.get("windows_with_trades", 0))
    positive_expectancy_windows = int(summary.get("positive_expectancy_windows", 0))
    average_expectancy = summary.get("average_expectancy")
    average_profit_factor = summary.get("average_profit_factor")

    reasons: list[str] = []

    if classification == "insufficient_trades":
        if total_trades < min_total_trades:
            reasons.append("insufficient_total_trades")
        if windows_with_trades < min_windows_with_trades:
            reasons.append("insufficient_windows_with_trades")
        return reasons

    if classification == "weak":
        if average_expectancy is None or float(average_expectancy) <= min_average_expectancy:
            reasons.append("average_expectancy_below_threshold")
        if average_profit_factor is None or float(average_profit_factor) < min_average_profit_factor:
            reasons.append("average_profit_factor_below_threshold")
        return reasons

    if classification == "watchlist":
        reasons.append("insufficient_positive_expectancy_windows")
        reasons.append("total_trades_above_threshold")
        reasons.append("windows_with_trades_above_threshold")
        return reasons

    # candidate
    reasons.append("total_trades_above_threshold")
    reasons.append("windows_with_trades_above_threshold")
    reasons.append("positive_expectancy_windows_above_threshold")
    reasons.append("average_expectancy_above_threshold")
    reasons.append("average_profit_factor_above_threshold")
    return reasons


def _get_strategy_window_results(
    report: dict[str, Any], strategy_id: str
) -> list[dict[str, Any]]:
    """Extract per-window results for one strategy from the report results list."""
    out = []
    for window in report.get("results", []):
        for sr in window.get("strategies", []):
            if str(sr.get("strategy_id")) == strategy_id:
                out.append(sr)
                break
    return out


def _get_asset_strategy_window_results(
    asset_data: dict[str, Any], strategy_id: str
) -> list[dict[str, Any]]:
    """Extract per-window results for one strategy from asset_data['results']."""
    out = []
    for window in asset_data.get("results", []):
        for sr in window.get("strategies", []):
            if str(sr.get("strategy_id")) == strategy_id:
                out.append(sr)
                break
    return out


def compute_window_robustness_diagnostics(
    summary: dict[str, Any],
    window_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute informational robustness diagnostics for one strategy.

    Diagnostics are purely informational — they never classify, sort, or select strategies.
    window_results is the list of per-window strategy results for this strategy.
    Pass an empty list when per-window data is unavailable (old reports).
    """
    windows = int(summary.get("windows", 0))
    windows_with_trades = int(summary.get("windows_with_trades", 0))
    total_trades = int(summary.get("total_trades", 0))
    positive_expectancy_windows = int(summary.get("positive_expectancy_windows", 0))

    zero_trade_windows = windows - windows_with_trades

    window_participation_rate = windows_with_trades / windows if windows > 0 else None

    average_trades_per_active_window = (
        total_trades / windows_with_trades if windows_with_trades > 0 else None
    )

    # Denominator is windows_with_trades: a zero-trade window cannot have positive expectancy.
    positive_expectancy_rate = (
        positive_expectancy_windows / windows_with_trades if windows_with_trades > 0 else None
    )

    negative_expectancy_windows = None
    expectancy_min = None
    expectancy_max = None
    profit_factor_min = None
    profit_factor_max = None

    if window_results:
        expectancy_values = [float(r.get("expectancy", 0.0)) for r in window_results]
        profit_factor_values = [
            float(r["profit_factor"])
            for r in window_results
            if r.get("profit_factor") is not None
        ]

        # expectancy == 0 is neutral: not counted as negative
        negative_expectancy_windows = sum(1 for e in expectancy_values if e < 0)

        if expectancy_values:
            expectancy_min = min(expectancy_values)
            expectancy_max = max(expectancy_values)

        if profit_factor_values:
            profit_factor_min = min(profit_factor_values)
            profit_factor_max = max(profit_factor_values)
        # if all profit_factor are None, profit_factor_min/max remain None

    return {
        "zero_trade_windows": zero_trade_windows,
        "negative_expectancy_windows": negative_expectancy_windows,
        "average_trades_per_active_window": average_trades_per_active_window,
        "expectancy_min": expectancy_min,
        "expectancy_max": expectancy_max,
        "profit_factor_min": profit_factor_min,
        "profit_factor_max": profit_factor_max,
        "window_participation_rate": window_participation_rate,
        "positive_expectancy_rate": positive_expectancy_rate,
    }


def _load_report_json(path: Path, not_found_msg: str, invalid_msg: str) -> dict[str, Any]:
    if not path.exists():
        typer.echo(not_found_msg, err=True)
        raise typer.Exit(code=1)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        typer.echo(invalid_msg, err=True)
        raise typer.Exit(code=1)
    if not isinstance(payload, dict):
        typer.echo(invalid_msg, err=True)
        raise typer.Exit(code=1)
    return payload


def _display_report_fields(payload: dict[str, Any], fields: list[tuple[str, str]]) -> None:
    for key, label in fields:
        if key in payload:
            typer.echo(f"  {label}: {payload[key]}")


def load_comparison_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        typer.echo("Comparison report not found", err=True)
        raise typer.Exit(code=1)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        typer.echo("Invalid comparison report JSON", err=True)
        raise typer.Exit(code=1)

    if not isinstance(payload, dict):
        typer.echo("Invalid comparison report JSON", err=True)
        raise typer.Exit(code=1)
    if "summary_by_strategy" not in payload:
        typer.echo("Comparison report missing summary_by_strategy", err=True)
        raise typer.Exit(code=1)
    if not isinstance(payload["summary_by_strategy"], list):
        typer.echo("Comparison report missing summary_by_strategy", err=True)
        raise typer.Exit(code=1)

    return payload


def load_assets_comparison_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        typer.echo("Assets comparison report not found", err=True)
        raise typer.Exit(code=1)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        typer.echo("Invalid assets comparison report JSON", err=True)
        raise typer.Exit(code=1)
    if not isinstance(payload, dict):
        typer.echo("Invalid assets comparison report JSON", err=True)
        raise typer.Exit(code=1)
    if "results_by_asset" not in payload:
        typer.echo("Assets comparison report missing results_by_asset", err=True)
        raise typer.Exit(code=1)
    if not isinstance(payload["results_by_asset"], dict):
        typer.echo("Assets comparison report missing results_by_asset", err=True)
        raise typer.Exit(code=1)
    return payload


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
    section: str | None = typer.Option(
        None,
        "--section",
        "-s",
        help="Configuration section to display: all, strategy, or goal.",
    )
) -> None:
    """Display current strategy and/or goal configuration."""
    strategy = load_strategy()
    goal = load_goal()

    if section is None:
        entry = strategy.get("entry", {})
        exit_ = strategy.get("exit", {})
        risk = strategy.get("risk", {})
        strategy_id = get_active_strategy_id(strategy)

        typer.echo("Strategy")
        typer.echo(f"  Strategy ID: {strategy_id}")
        typer.echo(f"  Version: {strategy.get('version')}")
        typer.echo(f"  Asset: {strategy.get('asset')}")
        typer.echo(f"  Timeframe: {strategy.get('timeframe')}")
        if strategy_id == "ema_atr_trend":
            typer.echo(f"  Entry: EMA {entry.get('fast_ema_period')}/{entry.get('slow_ema_period')} trend")
            typer.echo(f"  Exit: ATR {exit_.get('atr_period')} x {exit_.get('atr_stop_multiplier')}")
        else:
            typer.echo(f"  Entry: {entry.get('indicator')} <= {entry.get('threshold')}")
        typer.echo(f"  Stop loss: {risk.get('stop_loss_pct')}%")
        typer.echo(f"  Position size: {risk.get('position_size_pct')}%")
        typer.echo("")
        typer.echo("Goal")
        typer.echo(f"  Asset: {goal.get('asset')}")
        typer.echo(f"  Target return 30d: {goal.get('target_return_30d')}")
        typer.echo(f"  Max drawdown: {goal.get('max_drawdown')}")
        typer.echo(f"  Min Sharpe: {goal.get('min_sharpe')}")
        typer.echo(f"  Reflection every closed trades: {goal.get('reflection_every_closed_trades')}")
        return

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

@app.command("fetch-ohlcv")
def fetch_ohlcv(
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Binance symbol, e.g. BTCUSDT."),
    interval: str = typer.Option("1h", "--interval", help="Binance interval, e.g. 1m, 5m, 1h, 1d."),
    limit: int = typer.Option(500, "--limit", help="Number of candles to fetch. Binance max is 1000."),
    output: Path = typer.Option(
        Path("data/BTCUSDT_1h.csv"),
        "--output",
        "-o",
        help="Output CSV path.",
    ),
) -> None:
    """Fetch public OHLCV candles from Binance and save them as CSV."""
    validate_binance_limit("fetch-ohlcv", limit)
    rows = fetch_binance_ohlcv(symbol=symbol, interval=interval, limit=limit)
    output_path = write_ohlcv_csv(rows, output)

    echo_json(
        {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
            "rows": len(rows),
            "output": str(output_path),
        }
    )

@app.command("research-cycle")
def research_cycle(
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Binance symbol, e.g. BTCUSDT."),
    interval: str = typer.Option("1h", "--interval", help="Binance interval, e.g. 1m, 5m, 1h, 1d."),
    limit: int = typer.Option(500, "--limit", help="Number of candles to fetch. Binance max is 1000."),
    output: Path = typer.Option(
        Path("data/BTCUSDT_1h.csv"),
        "--output",
        "-o",
        help="Output CSV path.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output-dir",
        help="Directory where research cycle reports are written.",
    ),
    fail_on_gaps: bool = typer.Option(
        False,
        "--fail-on-gaps",
        help="Exit with code 1 if time gaps are detected.",
    ),
    reflect: bool = typer.Option(
        False,
        "--reflect",
        help="Run one controlled reflection hypothesis after the baseline cycle.",
    ),
) -> None:
    """Run a full research cycle: fetch, inspect, backtest, score, and optionally reflect."""
    validate_binance_limit("research-cycle", limit)

    strategy = load_strategy()
    goal = load_goal()

    rows = fetch_binance_ohlcv(symbol=symbol, interval=interval, limit=limit)
    output_path = write_ohlcv_csv(rows, output)

    ohlcv = load_ohlcv_csv(output_path)
    gaps = detect_time_gaps(ohlcv)

    if fail_on_gaps and gaps:
        echo_json(
            {
                "stage": "inspect",
                "status": "failed",
                "reason": "time_gaps_detected",
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": limit,
                "csv_path": str(output_path),
                "rows": int(len(ohlcv)),
                "gaps_detected": int(len(gaps)),
                "gaps": gaps,
            }
        )
        raise typer.Exit(code=1)

    backtest_result = run_backtest(ohlcv, strategy)
    score_result = score_trades(backtest_result["trades"], goal)
    summary = summarize_backtest(backtest_result["trades"], score_result)

    report = {
        "command": "research-cycle",
        "status": "completed",
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
        "csv_path": str(output_path),
        "asset": strategy["asset"],
        "timeframe": strategy["timeframe"],
        "rows": int(len(ohlcv)),
        "gaps_detected": int(len(gaps)),
        "gaps": gaps,
        "initial_balance": backtest_result["initial_balance"],
        **summary,
    }

    outputs_dir = output_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    research_report_path = outputs_dir / "research_cycle_report.json"
    backtest_report_path = outputs_dir / "backtest_report.json"
    trades_path = outputs_dir / "backtest_trades.jsonl"

    report["output_dir"] = str(outputs_dir)
    report["research_report_path"] = str(research_report_path)
    report["backtest_report_path"] = str(backtest_report_path)
    report["trades_path"] = str(trades_path)

    write_json(research_report_path, report)
    write_json(backtest_report_path, report)
    write_jsonl(trades_path, backtest_result["trades"])

    if reflect:
        proposal = propose_one_change(strategy, score_result)
        candidate_strategy = apply_reflection_proposal(strategy, proposal)

        candidate_backtest_result = run_backtest(ohlcv, candidate_strategy)
        candidate_score_result = score_trades(candidate_backtest_result["trades"], goal)
        candidate_summary = summarize_backtest(
            candidate_backtest_result["trades"],
            candidate_score_result,
        )

        candidate_report = {
            "command": "research-cycle",
            "status": "completed",
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
            "csv_path": str(output_path),
            "asset": candidate_strategy["asset"],
            "timeframe": candidate_strategy["timeframe"],
            "rows": int(len(ohlcv)),
            "gaps_detected": int(len(gaps)),
            "gaps": gaps,
            "initial_balance": candidate_backtest_result["initial_balance"],
            **candidate_summary,
        }

        reflection_report = build_research_reflection_report(
            symbol=symbol,
            interval=interval,
            limit=limit,
            baseline_report=report,
            proposal=proposal,
            candidate_report=candidate_report,
        )

        reflection_report_path = outputs_dir / "research_reflection_report.json"
        write_json(reflection_report_path, reflection_report)

        report["reflection"] = {
            "enabled": True,
            "report_path": str(reflection_report_path),
            "decision": reflection_report["decision"],
            "variable": reflection_report["hypothesis"]["variable"],
            "old_value": reflection_report["hypothesis"]["old_value"],
            "new_value": reflection_report["hypothesis"]["new_value"],
        }

    echo_json(report)

@app.command("research-robustness")
def research_robustness(
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Binance symbol, e.g. BTCUSDT."),
    interval: str = typer.Option("1h", "--interval", help="Binance interval, e.g. 1m, 5m, 1h, 1d."),
    limit: int = typer.Option(1000, "--limit", help="Number of candles to fetch. Binance max is 1000."),
    output: Path = typer.Option(
        Path("data/BTCUSDT_1h.csv"),
        "--output",
        "-o",
        help="Output CSV path.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output-dir",
        help="Directory where research robustness reports are written.",
    ),
    windows: int = typer.Option(
        4,
        "--windows",
        help="Number of chronological windows used for robustness validation.",
    ),
) -> None:
    """Run controlled reflection robustness validation across multiple windows."""
    if windows < 3:
        echo_json(
            {
                "command": "research-robustness",
                "status": "failed",
                "reason": "windows_must_be_at_least_3",
                "windows": windows,
            }
        )
        raise typer.Exit(code=1)

    validate_binance_limit("research-robustness", limit)

    strategy = load_strategy()
    goal = load_goal()

    rows = fetch_binance_ohlcv(symbol=symbol, interval=interval, limit=limit)
    output_path = write_ohlcv_csv(rows, output)

    ohlcv = load_ohlcv_csv(output_path)
    gaps = detect_time_gaps(ohlcv)

    baseline_backtest = run_backtest(ohlcv, strategy)
    baseline_score = score_trades(baseline_backtest["trades"], goal)

    proposal = propose_one_change(strategy, baseline_score)

    report = build_research_robustness_report(
        symbol=symbol,
        interval=interval,
        limit=limit,
        windows=windows,
        ohlcv=ohlcv,
        strategy=strategy,
        goal=goal,
        proposal=proposal,
    )

    report["csv_path"] = str(output_path)
    report["rows"] = int(len(ohlcv))
    report["gaps_detected"] = int(len(gaps))
    report["gaps"] = gaps

    outputs_dir = output_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    robustness_report_path = outputs_dir / "research_robustness_report.json"

    report["output_dir"] = str(outputs_dir)
    report["robustness_report_path"] = str(robustness_report_path)

    write_json(robustness_report_path, report)

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
def backtest_csv(
    path: Path,
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output-dir",
        help="Directory where backtest reports are written.",
    ),
) -> None:
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

    outputs_dir = output_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = outputs_dir / "backtest_report.json"
    trades_path = outputs_dir / "backtest_trades.jsonl"

    report["output_dir"] = str(outputs_dir)
    report["report_path"] = str(report_path)
    report["trades_path"] = str(trades_path)

    write_json(report_path, report)
    write_jsonl(trades_path, backtest_result["trades"])

    echo_json(report)


@app.command("compare-strategies-csv")
def compare_strategies_csv(path: Path) -> None:
    """Compare supported strategy candidates on the same local OHLCV CSV."""
    base_strategy = load_strategy()
    goal = load_goal()
    ohlcv = load_ohlcv_csv(path)
    gaps = detect_time_gaps(ohlcv)

    strategies = []
    for strategy_id in ["rsi_baseline", "ema_atr_trend", "donchian_breakout"]:
        strategy = build_strategy_variant(base_strategy, strategy_id)
        strategies.append(summarize_strategy_on_dataframe(ohlcv, strategy, goal))

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = outputs_dir / "strategy_comparison_report.json"
    report = {
        "command": "compare-strategies-csv",
        "csv_path": str(path),
        "rows": int(len(ohlcv)),
        "gaps_detected": int(len(gaps)),
        "gaps": gaps,
        "strategies": strategies,
        "output_dir": str(outputs_dir),
        "report_path": str(report_path),
    }

    write_json(report_path, report)
    echo_json(report)


@app.command("compare-strategies-windows-csv")
def compare_strategies_windows_csv(
    path: Path,
    windows: int = typer.Option(
        4,
        "--windows",
        help="Number of chronological windows used for strategy comparison.",
    ),
) -> None:
    """Compare strategy candidates across chronological windows from one local OHLCV CSV."""
    ohlcv = load_ohlcv_csv(path)

    if windows < 2:
        echo_json(
            {
                "command": "compare-strategies-windows-csv",
                "status": "failed",
                "reason": "windows_must_be_at_least_2",
                "windows": windows,
            }
        )
        raise typer.Exit(code=1)
    if windows > len(ohlcv):
        echo_json(
            {
                "command": "compare-strategies-windows-csv",
                "status": "failed",
                "reason": "windows_must_not_exceed_rows",
                "windows": windows,
                "rows": int(len(ohlcv)),
            }
        )
        raise typer.Exit(code=1)

    gaps = detect_time_gaps(ohlcv)
    base_strategy = load_strategy()
    goal = load_goal()
    strategy_variants = [
        build_strategy_variant(base_strategy, "rsi_baseline"),
        build_strategy_variant(base_strategy, "ema_atr_trend"),
        build_strategy_variant(base_strategy, "donchian_breakout"),
    ]

    results = []
    start_index = 0
    for window_number, window_df in enumerate(split_dataframe_windows(ohlcv, windows), start=1):
        end_index = start_index + len(window_df) - 1
        results.append(
            {
                "window": window_number,
                "start_index": int(start_index),
                "end_index": int(end_index),
                "rows": int(len(window_df)),
                "strategies": [
                    summarize_strategy_on_dataframe(window_df, strategy, goal)
                    for strategy in strategy_variants
                ],
            }
        )
        start_index = end_index + 1

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = outputs_dir / "strategy_windows_comparison_report.json"
    report = {
        "command": "compare-strategies-windows-csv",
        "csv_path": str(path),
        "rows": int(len(ohlcv)),
        "windows": windows,
        "gaps_detected": int(len(gaps)),
        "gaps": gaps,
        "results": results,
        "summary_by_strategy": summarize_window_results_by_strategy(results),
        "output_dir": str(outputs_dir),
        "report_path": str(report_path),
    }

    write_json(report_path, report)
    echo_json(report)


@app.command("compare-strategies-assets-csv")
def compare_strategies_assets_csv(
    asset: list[str] = typer.Option(
        [],
        "--asset",
        help="Asset to compare as SYMBOL:path/to/file.csv. Repeat for multiple assets.",
    ),
    windows: int = typer.Option(
        4,
        "--windows",
        help="Number of chronological windows used for strategy comparison.",
    ),
) -> None:
    """Compare strategy candidates across multiple assets from local OHLCV CSV files."""
    parsed_assets = []
    for asset_str in asset:
        if ":" not in asset_str:
            echo_json({
                "command": "compare-strategies-assets-csv",
                "status": "failed",
                "reason": "invalid_asset_format",
                "asset": asset_str,
                "expected_format": "SYMBOL:path/to/file.csv",
            })
            raise typer.Exit(code=1)
        symbol, csv_path_str = asset_str.split(":", 1)
        if not symbol:
            echo_json({
                "command": "compare-strategies-assets-csv",
                "status": "failed",
                "reason": "empty_symbol",
                "asset": asset_str,
            })
            raise typer.Exit(code=1)
        if not csv_path_str:
            echo_json({
                "command": "compare-strategies-assets-csv",
                "status": "failed",
                "reason": "empty_csv_path",
                "asset": asset_str,
            })
            raise typer.Exit(code=1)
        parsed_assets.append((symbol, csv_path_str))

    if windows < 2:
        echo_json({
            "command": "compare-strategies-assets-csv",
            "status": "failed",
            "reason": "windows_must_be_at_least_2",
            "windows": windows,
        })
        raise typer.Exit(code=1)

    base_strategy = load_strategy()
    goal = load_goal()

    loaded_assets = []
    for symbol, csv_path_str in parsed_assets:
        ohlcv = load_ohlcv_csv(Path(csv_path_str))
        if windows > len(ohlcv):
            echo_json({
                "command": "compare-strategies-assets-csv",
                "status": "failed",
                "reason": "windows_must_not_exceed_rows",
                "windows": windows,
                "symbol": symbol,
                "csv_path": csv_path_str,
                "rows": int(len(ohlcv)),
            })
            raise typer.Exit(code=1)
        loaded_assets.append((symbol, csv_path_str, ohlcv))

    results_by_asset: dict[str, Any] = {}
    for symbol, csv_path_str, ohlcv in loaded_assets:
        gaps = detect_time_gaps(ohlcv)
        window_output = _run_windows_for_asset(ohlcv, windows, base_strategy, goal)
        results_by_asset[symbol] = {
            "csv_path": csv_path_str,
            "rows": int(len(ohlcv)),
            "gaps_detected": int(len(gaps)),
            "gaps": gaps,
            "results": window_output["results"],
            "summary_by_strategy": window_output["summary_by_strategy"],
        }

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = outputs_dir / "strategy_assets_comparison_report.json"
    report = {
        "command": "compare-strategies-assets-csv",
        "windows": windows,
        "assets": [symbol for symbol, _, _ in loaded_assets],
        "strategies": ["rsi_baseline", "ema_atr_trend", "donchian_breakout"],
        "results_by_asset": results_by_asset,
        "output_dir": str(outputs_dir),
        "report_path": str(report_path),
    }

    write_json(report_path, report)
    echo_json(report)


@app.command("show-comparison-report")
def show_comparison_report(
    path: Path = typer.Option(
        Path("outputs/strategy_windows_comparison_report.json"),
        "--path",
        help="Path to a strategy windows comparison report JSON.",
    ),
) -> None:
    """Read and display a previously generated strategy comparison report."""
    report = load_comparison_report(path)
    research_policy = load_research_policy()
    thresholds = research_policy["comparison_acceptance"]

    typer.echo("Comparison report")
    if "command" in report:
        typer.echo(f"  Command: {report.get('command')}")
    typer.echo(f"  CSV path: {report.get('csv_path')}")
    typer.echo(f"  Rows: {report.get('rows')}")
    typer.echo(f"  Windows: {report.get('windows')}")
    typer.echo(f"  Gaps detected: {report.get('gaps_detected')}")
    if "report_path" in report:
        typer.echo(f"  Report path: {report.get('report_path')}")
    typer.echo("")
    typer.echo("Acceptance thresholds")
    typer.echo(f"  Min total trades: {thresholds['min_total_trades']}")
    typer.echo(f"  Min windows with trades: {thresholds['min_windows_with_trades']}")
    typer.echo(f"  Min positive expectancy windows: {thresholds['min_positive_expectancy_windows']}")
    typer.echo(f"  Min average expectancy: {thresholds['min_average_expectancy']}")
    typer.echo(f"  Min average profit factor: {thresholds['min_average_profit_factor']}")
    typer.echo("")
    typer.echo("Summary by strategy")

    for strategy_summary in report["summary_by_strategy"]:
        classification = classify_strategy_summary(strategy_summary, research_policy)
        reasons = get_classification_reasons(strategy_summary, research_policy, classification)
        strategy_id = str(strategy_summary.get("strategy_id", ""))
        window_results = _get_strategy_window_results(report, strategy_id)
        diagnostics = compute_window_robustness_diagnostics(strategy_summary, window_results)

        typer.echo(f"  {strategy_id}")
        typer.echo(f"    Research status: {classification}")
        typer.echo("    Reasons:")
        for reason in reasons:
            typer.echo(f"      - {reason}")
        typer.echo(f"    Windows: {strategy_summary.get('windows')}")
        typer.echo(f"    Windows with trades: {strategy_summary.get('windows_with_trades')}")
        typer.echo(f"    Total trades: {strategy_summary.get('total_trades')}")
        typer.echo(f"    Positive expectancy windows: {strategy_summary.get('positive_expectancy_windows')}")
        typer.echo(f"    Average expectancy: {strategy_summary.get('average_expectancy')}")
        typer.echo(f"    Average profit factor: {strategy_summary.get('average_profit_factor')}")
        typer.echo("    Window robustness diagnostics")
        typer.echo(f"      Zero trade windows: {diagnostics['zero_trade_windows']}")
        typer.echo(f"      Window participation rate: {diagnostics['window_participation_rate']}")
        typer.echo(f"      Average trades per active window: {diagnostics['average_trades_per_active_window']}")
        typer.echo(f"      Positive expectancy rate: {diagnostics['positive_expectancy_rate']}")
        if diagnostics["negative_expectancy_windows"] is not None:
            typer.echo(f"      Negative expectancy windows: {diagnostics['negative_expectancy_windows']}")
        if diagnostics["expectancy_min"] is not None:
            typer.echo(f"      Expectancy min: {diagnostics['expectancy_min']}")
        if diagnostics["expectancy_max"] is not None:
            typer.echo(f"      Expectancy max: {diagnostics['expectancy_max']}")
        if diagnostics["profit_factor_min"] is not None:
            typer.echo(f"      Profit factor min: {diagnostics['profit_factor_min']}")
        if diagnostics["profit_factor_max"] is not None:
            typer.echo(f"      Profit factor max: {diagnostics['profit_factor_max']}")
        typer.echo("")


@app.command("show-research-report")
def show_research_report(
    path: Path = typer.Option(
        Path("outputs/research_cycle_report.json"),
        "--path",
        help="Path to a research cycle report JSON.",
    ),
) -> None:
    """Read and display a previously generated research cycle report."""
    payload = _load_report_json(path, "Research report not found", "Invalid research report JSON")
    typer.echo("Research cycle report")
    _display_report_fields(payload, [
        ("command", "Command"),
        ("status", "Status"),
        ("symbol", "Symbol"),
        ("interval", "Interval"),
        ("limit", "Limit"),
        ("csv_path", "CSV path"),
        ("rows", "Rows"),
        ("gaps_detected", "Gaps detected"),
        ("asset", "Asset"),
        ("timeframe", "Timeframe"),
        ("initial_balance", "Initial balance"),
        ("total_trades", "Total trades"),
        ("final_balance", "Final balance"),
        ("net_pnl", "Net PnL"),
        ("winrate", "Winrate"),
        ("profit_factor", "Profit factor"),
        ("expectancy", "Expectancy"),
        ("score", "Score"),
        ("classification", "Classification"),
    ])


@app.command("show-backtest-report")
def show_backtest_report(
    path: Path = typer.Option(
        Path("outputs/backtest_report.json"),
        "--path",
        help="Path to a backtest report JSON.",
    ),
) -> None:
    """Read and display a previously generated backtest report."""
    payload = _load_report_json(path, "Backtest report not found", "Invalid backtest report JSON")
    typer.echo("Backtest report")
    _display_report_fields(payload, [
        ("csv_path", "CSV path"),
        ("asset", "Asset"),
        ("timeframe", "Timeframe"),
        ("rows", "Rows"),
        ("gaps_detected", "Gaps detected"),
        ("initial_balance", "Initial balance"),
        ("total_trades", "Total trades"),
        ("final_balance", "Final balance"),
        ("net_pnl", "Net PnL"),
        ("winrate", "Winrate"),
        ("profit_factor", "Profit factor"),
        ("expectancy", "Expectancy"),
        ("score", "Score"),
        ("classification", "Classification"),
    ])


@app.command("show-walk-forward-report")
def show_walk_forward_report(
    path: Path = typer.Option(
        Path("outputs/walk_forward_report.json"),
        "--path",
        help="Path to a walk-forward report JSON.",
    ),
) -> None:
    """Read and display a previously generated walk-forward report."""
    payload = _load_report_json(
        path,
        "Walk-forward report not found",
        "Invalid walk-forward report JSON",
    )

    typer.echo("Walk-forward report")

    wf = payload.get("walk_forward")
    if isinstance(wf, dict) and wf:
        typer.echo("  Walk-Forward")
        for key, label in [
            ("train_window", "Train window"),
            ("test_window", "Test window"),
            ("step", "Step"),
            ("windows", "Windows"),
        ]:
            if key in wf:
                typer.echo(f"    {label}: {wf[key]}")

    strat = payload.get("strategy")
    if isinstance(strat, dict) and strat:
        typer.echo("  Strategy")
        for key, label in [
            ("strategy_id", "Strategy ID"),
            ("asset", "Asset"),
            ("timeframe", "Timeframe"),
        ]:
            if key in strat:
                typer.echo(f"    {label}: {strat[key]}")

    sm = payload.get("summary")
    if isinstance(sm, dict) and sm:
        typer.echo("  Summary")
        for key, label in [
            ("windows", "Windows"),
            ("windows_with_trades", "Windows with trades"),
            ("total_trades", "Total trades"),
        ]:
            if key in sm:
                typer.echo(f"    {label}: {sm[key]}")

    vc = payload.get("validation_context")
    if isinstance(vc, dict) and vc:
        typer.echo("  Validation context")
        for key, label in [
            ("mode", "Mode"),
            ("data_role", "Data role"),
            ("confirmatory_holdout_used", "Confirmatory holdout used"),
            ("prospective_holdout_used", "Prospective holdout used"),
            ("paper_forward_used", "Paper-forward used"),
            ("parameter_optimization_performed", "Parameter optimization performed"),
            ("selection_performed", "Selection performed"),
        ]:
            if key in vc:
                typer.echo(f"    {label}: {vc[key]}")

    results = payload.get("results")
    if isinstance(results, list):
        typer.echo("  Results")
        for r in results:
            if not isinstance(r, dict):
                continue
            idx = r.get("window_index", "?")
            typer.echo(f"    Window {idx}")
            for key, label in [
                ("test_start", "Test start"),
                ("test_end", "Test end"),
                ("test_rows", "Test rows"),
            ]:
                if key in r:
                    typer.echo(f"      {label}: {r[key]}")
            rs = r.get("summary")
            if not isinstance(rs, dict):
                rs = {}
            for key, label in [
                ("closed_trades", "Closed trades"),
                ("total_net_pnl", "Total net PnL"),
                ("total_return", "Total return"),
                ("max_drawdown", "Max drawdown"),
                ("profit_factor", "Profit factor"),
                ("expectancy", "Expectancy"),
            ]:
                if key in rs:
                    typer.echo(f"      {label}: {rs[key]}")


@app.command("show-assets-comparison-report")
def show_assets_comparison_report(
    path: Path = typer.Option(
        Path("outputs/strategy_assets_comparison_report.json"),
        "--path",
        help="Path to a multi-asset strategy comparison report JSON.",
    ),
    assets: list[str] | None = typer.Option(
        None,
        "--asset",
        help="Filter display by asset symbol. Repeat for multiple assets.",
    ),
    strategies: list[str] | None = typer.Option(
        None,
        "--strategy",
        help="Filter display by strategy ID. Repeat for multiple strategies.",
    ),
    list_assets: bool = typer.Option(
        False,
        "--list-assets",
        help="List available assets in the report and exit.",
    ),
    list_strategies: bool = typer.Option(
        False,
        "--list-strategies",
        help="List available strategies in the report and exit.",
    ),
) -> None:
    """Read and display a previously generated multi-asset comparison report."""
    report = load_assets_comparison_report(path)
    research_policy = load_research_policy()

    asset_filter = assets or []
    strategy_filter = strategies or []

    if asset_filter:
        unknown = [a for a in asset_filter if a not in report["results_by_asset"]]
        if unknown:
            typer.echo(f"Error: asset(s) not found in report: {', '.join(unknown)}")
            raise typer.Exit(code=1)

    if strategy_filter:
        all_known: set[str] = set()
        for ad in report["results_by_asset"].values():
            for s in ad.get("summary_by_strategy", []):
                all_known.add(s.get("strategy_id", ""))
        unknown_s = [s for s in strategy_filter if s not in all_known]
        if unknown_s:
            typer.echo(f"Error: strategy(ies) not found in report: {', '.join(unknown_s)}")
            raise typer.Exit(code=1)

    if list_assets or list_strategies:
        if list_assets:
            typer.echo("Assets in report:")
            for asset, asset_data in report["results_by_asset"].items():
                if asset_filter and asset not in asset_filter:
                    continue
                if strategy_filter:
                    known = {s.get("strategy_id", "") for s in asset_data.get("summary_by_strategy", [])}
                    if not any(sf in known for sf in strategy_filter):
                        continue
                typer.echo(f"  {asset}")
        if list_strategies:
            if list_assets:
                typer.echo("")
            seen: set[str] = set()
            strategy_list: list[str] = []
            for asset, asset_data in report["results_by_asset"].items():
                if asset_filter and asset not in asset_filter:
                    continue
                for s in asset_data.get("summary_by_strategy", []):
                    sid = s.get("strategy_id", "")
                    if sid and (not strategy_filter or sid in strategy_filter) and sid not in seen:
                        seen.add(sid)
                        strategy_list.append(sid)
            if strategy_list:
                typer.echo("Strategies in report:")
                for sid in strategy_list:
                    typer.echo(f"  {sid}")
            else:
                typer.echo("No strategies found in report.")
        return

    typer.echo("Assets comparison report")
    _display_report_fields(report, [
        ("command", "Command"),
        ("windows", "Windows"),
        ("assets", "Assets"),
        ("strategies", "Strategies"),
        ("output_dir", "Output dir"),
        ("report_path", "Report path"),
    ])

    typer.echo("")
    typer.echo("Results by asset")

    for asset, asset_data in report["results_by_asset"].items():
        if asset_filter and asset not in asset_filter:
            continue
        typer.echo(f"  {asset}")
        if "csv_path" in asset_data:
            typer.echo(f"    CSV path: {asset_data['csv_path']}")
        if "rows" in asset_data:
            typer.echo(f"    Rows: {asset_data['rows']}")
        if "gaps_detected" in asset_data:
            typer.echo(f"    Gaps detected: {asset_data['gaps_detected']}")
        if "summary_by_strategy" in asset_data:
            typer.echo("    Summary by strategy")
            asset_has_results = bool(asset_data.get("results"))
            for s in asset_data["summary_by_strategy"]:
                strategy_id = s.get("strategy_id", "")
                if strategy_filter and strategy_id not in strategy_filter:
                    continue
                typer.echo(f"      {strategy_id}")
                for key, label in [
                    ("windows", "Windows"),
                    ("windows_with_trades", "Windows with trades"),
                    ("total_trades", "Total trades"),
                    ("positive_expectancy_windows", "Positive expectancy windows"),
                    ("average_expectancy", "Average expectancy"),
                    ("average_profit_factor", "Average profit factor"),
                ]:
                    if key in s:
                        typer.echo(f"        {label}: {s[key]}")
                classification = classify_strategy_summary(s, research_policy)
                typer.echo(f"        Research status: {classification}")
                reasons = get_classification_reasons(s, research_policy, classification)
                typer.echo(f"        Reasons: {', '.join(reasons)}")
                if asset_has_results:
                    window_results = _get_asset_strategy_window_results(asset_data, strategy_id)
                    diag = compute_window_robustness_diagnostics(s, window_results)
                    typer.echo("        Window robustness diagnostics")
                    for diag_key, diag_label in [
                        ("zero_trade_windows", "Zero trade windows"),
                        ("window_participation_rate", "Window participation rate"),
                        ("positive_expectancy_rate", "Positive expectancy rate"),
                        ("average_trades_per_active_window", "Average trades per active window"),
                        ("negative_expectancy_windows", "Negative expectancy windows"),
                        ("expectancy_min", "Expectancy min"),
                        ("expectancy_max", "Expectancy max"),
                        ("profit_factor_min", "Profit factor min"),
                        ("profit_factor_max", "Profit factor max"),
                    ]:
                        value = diag.get(diag_key)
                        if value is not None:
                            typer.echo(f"          {diag_label}: {value}")
        typer.echo("")


@app.command("research-policy")
def research_policy() -> None:
    """Display the current research policy configuration."""
    policy = load_research_policy()
    thresholds = policy["comparison_acceptance"]

    typer.echo("Research policy")
    typer.echo("")
    typer.echo("Comparison acceptance")
    typer.echo(f"  Min total trades: {thresholds['min_total_trades']}")
    typer.echo(f"  Min windows with trades: {thresholds['min_windows_with_trades']}")
    typer.echo(f"  Min positive expectancy windows: {thresholds['min_positive_expectancy_windows']}")
    typer.echo(f"  Min average expectancy: {thresholds['min_average_expectancy']}")
    typer.echo(f"  Min average profit factor: {thresholds['min_average_profit_factor']}")


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


@app.command("archive-prospective-csv")
def archive_prospective_csv(
    path: Path,
    asset: str = typer.Option(..., "--asset", help="Asset identifier, e.g. BTCUSDT."),
    timeframe: str = typer.Option(..., "--timeframe", help="Timeframe, e.g. 1h."),
    source: str = typer.Option(..., "--source", help="Data source identifier, e.g. binance-public."),
    reserve_dir: Path = typer.Option(
        Path("data/prospective_reserve"),
        "--reserve-dir",
        help="Directory where archived CSV files are stored.",
    ),
    registry_path: Path = typer.Option(
        Path("state/prospective_reserve.jsonl"),
        "--registry-path",
        help="Path to the prospective reserve JSONL registry.",
    ),
) -> None:
    """Archive a local OHLCV CSV to the prospective data reserve."""
    try:
        record = _do_archive_prospective_csv(
            path=path,
            asset=asset,
            timeframe=timeframe,
            source=source,
            reserve_dir=reserve_dir,
            registry_path=registry_path,
        )
    except InvalidMetadataError as exc:
        echo_json({
            "command": "archive-prospective-csv",
            "status": "failed",
            "reason": "invalid_metadata",
            "detail": str(exc),
        })
        raise typer.Exit(code=1)
    except DuplicateBatchError as exc:
        echo_json({
            "command": "archive-prospective-csv",
            "status": "failed",
            "reason": "duplicate_batch",
            "detail": str(exc),
        })
        raise typer.Exit(code=1)
    except ReserveFileExistsError as exc:
        echo_json({
            "command": "archive-prospective-csv",
            "status": "failed",
            "reason": "reserve_file_exists",
            "detail": str(exc),
        })
        raise typer.Exit(code=1)
    except InvalidOHLCVError as exc:
        echo_json({
            "command": "archive-prospective-csv",
            "status": "failed",
            "reason": "invalid_ohlcv",
            "detail": str(exc),
        })
        raise typer.Exit(code=1)

    echo_json({
        "command": "archive-prospective-csv",
        "status": "completed",
        "registry_path": str(registry_path),
        "stored_path": str(record["stored_path"]),
        "batch": record,
    })


if __name__ == "__main__":
    app()

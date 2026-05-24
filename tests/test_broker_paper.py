from __future__ import annotations

import pytest

from trading_agent.broker_paper import PaperBroker


def test_initial_broker_state() -> None:
    broker = PaperBroker(initial_balance=1000)
    assert broker.cash == 1000.0
    assert broker.position_size == 0.0
    assert broker.entry_price is None
    assert broker.is_open is False


def test_cannot_sell_without_open_position() -> None:
    broker = PaperBroker(initial_balance=1000)
    with pytest.raises(ValueError):
        broker.sell(100)


def test_cannot_buy_twice_while_position_is_open() -> None:
    broker = PaperBroker(initial_balance=1000)
    broker.buy(100, 10)
    with pytest.raises(ValueError):
        broker.buy(100, 10)


def test_buy_then_sell_returns_trade_dictionary() -> None:
    broker = PaperBroker(initial_balance=1000)
    broker.buy(100, 10, fee_pct=0.1, slippage_pct=0.05)
    trade = broker.sell(110, fee_pct=0.1, slippage_pct=0.05)
    assert isinstance(trade, dict)
    assert {"entry_price", "exit_price", "position_size", "gross_pnl", "net_pnl", "return_pct"} <= trade.keys()


def test_fees_and_slippage_affect_net_pnl() -> None:
    clean = PaperBroker(initial_balance=1000)
    clean.buy(100, 10)
    clean_trade = clean.sell(110)

    costly = PaperBroker(initial_balance=1000)
    costly.buy(100, 10, fee_pct=0.2, slippage_pct=0.2)
    costly_trade = costly.sell(110, fee_pct=0.2, slippage_pct=0.2)

    assert costly_trade["net_pnl"] < clean_trade["net_pnl"]


def test_equity_returns_numeric_value() -> None:
    broker = PaperBroker(initial_balance=1000)
    broker.buy(100, 10)
    value = broker.equity(105)
    assert isinstance(value, float)


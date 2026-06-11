from __future__ import annotations

import math

import pytest

from trading_agent.risk_model import compute_atr_position_size, compute_atr_stop_price


# --- compute_atr_position_size ---

def test_atr_position_size_basic() -> None:
    # stop_distance = 50 * 2.0 = 100
    # risk_amount = 10000 * 0.01 = 100
    # units = 100 / 100 = 1
    # position_value = 1 * 1000 = 1000
    # size_pct = 1000 / 10000 * 100 = 10.0
    result = compute_atr_position_size(
        equity=10000.0, price=1000.0, atr_value=50.0, risk_pct=1.0, atr_stop_multiplier=2.0
    )
    assert math.isclose(result, 10.0, rel_tol=1e-9)


def test_atr_position_size_capped_at_100() -> None:
    # Very small stop distance → enormous position → capped at 100%
    result = compute_atr_position_size(
        equity=1000.0, price=100.0, atr_value=1.0, risk_pct=50.0, atr_stop_multiplier=1.0
    )
    assert result == 100.0


def test_atr_position_size_rejects_zero_atr() -> None:
    with pytest.raises(ValueError, match="atr_value"):
        compute_atr_position_size(10000.0, 1000.0, 0.0, 1.0, 2.0)


def test_atr_position_size_rejects_negative_atr() -> None:
    with pytest.raises(ValueError, match="atr_value"):
        compute_atr_position_size(10000.0, 1000.0, -5.0, 1.0, 2.0)


def test_atr_position_size_rejects_nan_atr() -> None:
    with pytest.raises(ValueError, match="atr_value"):
        compute_atr_position_size(10000.0, 1000.0, float("nan"), 1.0, 2.0)


def test_atr_position_size_rejects_zero_price() -> None:
    with pytest.raises(ValueError, match="price"):
        compute_atr_position_size(10000.0, 0.0, 50.0, 1.0, 2.0)


def test_atr_position_size_rejects_negative_price() -> None:
    with pytest.raises(ValueError, match="price"):
        compute_atr_position_size(10000.0, -100.0, 50.0, 1.0, 2.0)


def test_atr_position_size_rejects_zero_equity() -> None:
    with pytest.raises(ValueError, match="equity"):
        compute_atr_position_size(0.0, 1000.0, 50.0, 1.0, 2.0)


def test_atr_position_size_rejects_negative_equity() -> None:
    with pytest.raises(ValueError, match="equity"):
        compute_atr_position_size(-100.0, 1000.0, 50.0, 1.0, 2.0)


def test_atr_position_size_rejects_zero_multiplier() -> None:
    with pytest.raises(ValueError, match="atr_stop_multiplier"):
        compute_atr_position_size(10000.0, 1000.0, 50.0, 1.0, 0.0)


def test_atr_position_size_rejects_zero_risk_pct() -> None:
    with pytest.raises(ValueError, match="risk_pct"):
        compute_atr_position_size(10000.0, 1000.0, 50.0, 0.0, 2.0)


# --- compute_atr_stop_price ---

def test_atr_stop_price_basic() -> None:
    # 1000 - 50 * 2.0 = 900
    result = compute_atr_stop_price(entry_price=1000.0, atr_value=50.0, atr_stop_multiplier=2.0)
    assert math.isclose(result, 900.0, rel_tol=1e-9)


def test_atr_stop_price_never_negative() -> None:
    # 10 - 5 * 3.0 = -5 → floored at 0
    result = compute_atr_stop_price(entry_price=10.0, atr_value=5.0, atr_stop_multiplier=3.0)
    assert result == 0.0


def test_atr_stop_price_rejects_zero_atr() -> None:
    with pytest.raises(ValueError, match="atr_value"):
        compute_atr_stop_price(1000.0, 0.0, 2.0)


def test_atr_stop_price_rejects_nan_atr() -> None:
    with pytest.raises(ValueError, match="atr_value"):
        compute_atr_stop_price(1000.0, float("nan"), 2.0)


def test_atr_stop_price_rejects_zero_entry_price() -> None:
    with pytest.raises(ValueError, match="entry_price"):
        compute_atr_stop_price(0.0, 50.0, 2.0)


def test_atr_stop_price_rejects_zero_multiplier() -> None:
    with pytest.raises(ValueError, match="atr_stop_multiplier"):
        compute_atr_stop_price(1000.0, 50.0, 0.0)

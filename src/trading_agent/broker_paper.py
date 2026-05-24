from __future__ import annotations

from dataclasses import dataclass, field


def _pct(value: float) -> float:
    return float(value) / 100.0


@dataclass
class PaperBroker:
    initial_balance: float = 0.0
    cash: float = field(init=False)
    position_size: float = field(default=0.0, init=False)
    entry_price: float | None = field(default=None, init=False)
    is_open: bool = field(default=False, init=False)
    _entry_fee: float = field(default=0.0, init=False, repr=False)
    _entry_notional: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_balance)
        self.initial_balance = float(self.initial_balance)

    def buy(self, price: float, size_pct: float, fee_pct: float = 0.0, slippage_pct: float = 0.0) -> None:
        if self.is_open:
            raise ValueError("position already open")
        if size_pct <= 0 or size_pct > 100:
            raise ValueError("size_pct must be between 0 and 100")

        fee_rate = _pct(fee_pct)
        slippage_rate = _pct(slippage_pct)
        execution_price = float(price) * (1 + slippage_rate)
        spendable_cash = self.cash / (1 + fee_rate)
        investment = spendable_cash * (float(size_pct) / 100.0)
        fee = investment * fee_rate
        total_cost = investment + fee

        self.cash -= total_cost
        self.position_size = investment / execution_price
        self.entry_price = execution_price
        self._entry_fee = fee
        self._entry_notional = investment
        self.is_open = True

    def sell(self, price: float, fee_pct: float = 0.0, slippage_pct: float = 0.0) -> dict[str, float]:
        if not self.is_open:
            raise ValueError("no open position")

        fee_rate = _pct(fee_pct)
        slippage_rate = _pct(slippage_pct)
        exit_price = float(price) * (1 - slippage_rate)
        gross_entry_value = self.position_size * float(self.entry_price or 0.0)
        gross_exit_value = self.position_size * exit_price
        exit_fee = gross_exit_value * fee_rate
        gross_pnl = gross_exit_value - gross_entry_value
        net_pnl = gross_exit_value - exit_fee - gross_entry_value - self._entry_fee
        return_pct = (net_pnl / gross_entry_value * 100.0) if gross_entry_value else 0.0

        self.cash += gross_exit_value - exit_fee
        trade = {
            "entry_price": float(self.entry_price or 0.0),
            "exit_price": exit_price,
            "position_size": float(self.position_size),
            "gross_pnl": float(gross_pnl),
            "net_pnl": float(net_pnl),
            "return_pct": float(return_pct),
        }

        self.position_size = 0.0
        self.entry_price = None
        self.is_open = False
        self._entry_fee = 0.0
        self._entry_notional = 0.0
        return trade

    def equity(self, current_price: float) -> float:
        position_value = self.position_size * float(current_price) if self.is_open else 0.0
        return float(self.cash + position_value)


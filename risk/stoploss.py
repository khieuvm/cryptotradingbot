"""Multi-phase ATR-based stoploss logic."""

from __future__ import annotations


class StoplossManager:
    """3-phase stoploss: initial → break-even → trail-lock.

    Phase 1 (initial): -sl_mult × ATR / entry_rate
    Phase 2 (break-even): lock at entry + fees when profit > 50% of TP
    Phase 3 (trail-lock): lock at 60% of TP distance when profit > TP
    """

    def calculate_stoploss(
        self,
        current_profit: float,
        sl_atr_mult: float,
        tp_atr_mult: float,
        atr: float,
        open_rate: float,
        current_rate: float,
        is_short: bool,
        fee: float = 0.001,
    ) -> float:
        """Calculate dynamic stoploss as a negative fraction.

        Returns:
            Stoploss value for freqtrade (negative float, e.g., -0.05 = 5% loss)
        """
        if atr <= 0 or open_rate <= 0:
            return -0.10

        sl_pct = sl_atr_mult * atr / open_rate
        tp_pct = tp_atr_mult * atr / open_rate

        # Phase 3: Trail-lock (profit >= TP target)
        if current_profit >= tp_pct:
            lock_fraction = 0.60
            if not is_short:
                lock_price = open_rate * (1 + tp_pct * lock_fraction)
                return max((lock_price / current_rate) - 1, -0.005)
            else:
                lock_price = open_rate * (1 - tp_pct * lock_fraction)
                return max(1 - (lock_price / current_rate), -0.005)

        # Phase 2: Break-even (profit >= 50% of TP)
        if current_profit >= tp_pct * 0.5:
            if not is_short:
                be_price = open_rate * (1 + fee)
                return max((be_price / current_rate) - 1, -sl_pct)
            else:
                be_price = open_rate * (1 - fee)
                return max(1 - (be_price / current_rate), -sl_pct)

        # Phase 1: Initial stoploss
        return max(-sl_pct, -0.20)

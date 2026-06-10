"""Multi-phase ATR-based stoploss logic with optional per-pair trailing."""

from __future__ import annotations


class StoplossManager:
    """3-phase stoploss: initial → break-even → trail-lock.

    Phase 1 (initial): Fixed SL at entry ± sl_mult × ATR
    Phase 2 (break-even): lock at entry + fees when profit > 50% of TP
    Phase 3 (trail-lock): lock at 60% of TP distance when profit > TP

    IMPORTANT: freqtrade divides the custom_stoploss return by leverage
    internally (adjust_stop_loss), so all returned values must be in
    account-level terms (price_pct × leverage).

    The return values are computed from desired SL *prices* so that
    freqtrade's adjust_stop_loss produces the exact SL level we want,
    regardless of current_rate.
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
        trailing_cfg: dict | None = None,
        leverage: float = 1.0,
        fixed_only: bool = False,
    ) -> float:
        """Calculate dynamic stoploss as a negative fraction.

        Returns:
            Stoploss value for freqtrade (negative float, account-level).
            Freqtrade divides by leverage to get price-level distance.
        """
        if atr <= 0 or open_rate <= 0 or current_rate <= 0:
            return -0.99

        sl_pct = sl_atr_mult * atr / open_rate
        tp_pct = tp_atr_mult * atr / open_rate
        tp_acct = tp_pct * leverage

        if trailing_cfg:
            return self._calculate_trailing(
                current_profit, sl_pct, tp_pct, atr, open_rate,
                current_rate, is_short, fee, trailing_cfg, leverage,
            )

        if fixed_only:
            sl_price = self._offset_price(open_rate, sl_pct, is_short)
            return self._bounded_sl(sl_price, current_rate, is_short, leverage,
                                    floor=-0.50)

        # Phase 3: Trail-lock (profit >= TP target)
        # Lock at 60% of TP distance from entry (in profit direction)
        if current_profit >= tp_acct:
            lock_price = self._offset_price(open_rate, tp_pct * 0.60, not is_short)
            return self._bounded_sl(lock_price, current_rate, is_short, leverage,
                                    floor=-0.005 * leverage)

        # Phase 2: Break-even (profit >= 50% of TP)
        if current_profit >= tp_acct * 0.5:
            be_price = open_rate * (1 + fee) if not is_short else open_rate * (1 - fee)
            return self._bounded_sl(be_price, current_rate, is_short, leverage,
                                    floor=-sl_pct * leverage)

        # Phase 1: Fixed SL at entry ± sl_distance
        sl_price = self._offset_price(open_rate, sl_pct, is_short)
        return self._bounded_sl(sl_price, current_rate, is_short, leverage,
                                floor=-0.50)

    def _calculate_trailing(
        self,
        current_profit: float,
        sl_pct: float,
        tp_pct: float,
        atr: float,
        open_rate: float,
        current_rate: float,
        is_short: bool,
        fee: float,
        cfg: dict,
        leverage: float = 1.0,
    ) -> float:
        """Trailing stoploss with configurable BE and trail triggers."""
        be_trigger = cfg.get("be_trigger", 0.4)
        be_offset = cfg.get("be_offset", 0.2)
        trail_trigger = cfg.get("trail_trigger", 0.7)
        trail_dist = cfg.get("trail_dist", 0.5)

        tp_acct = tp_pct * leverage
        trail_level = trail_trigger * tp_acct
        be_level = be_trigger * tp_acct
        trail_offset_pct = trail_dist * tp_pct

        # Trailing phase: constant distance behind current price
        if current_profit >= trail_level:
            return max(-trail_offset_pct * leverage, -sl_pct * leverage)

        # Break-even phase: lock at entry + small offset (in profit direction)
        if current_profit >= be_level:
            be_offset_pct = be_offset * atr / open_rate
            be_price = self._offset_price(open_rate, be_offset_pct, not is_short)
            return self._bounded_sl(be_price, current_rate, is_short, leverage,
                                    floor=-sl_pct * leverage)

        # Initial phase: fixed SL from entry
        # LONG: SL below entry (above=False → is_short for longs is False)
        # SHORT: SL above entry (above=True → is_short for shorts is True)
        sl_price = self._offset_price(open_rate, sl_pct, is_short)
        return self._bounded_sl(sl_price, current_rate, is_short, leverage,
                                floor=-0.50)

    @staticmethod
    def _offset_price(base: float, pct: float, above: bool) -> float:
        """Compute a price offset from base by pct."""
        return base * (1 + pct) if above else base * (1 - pct)

    @staticmethod
    def _price_to_sl(
        desired_price: float,
        current_rate: float,
        is_short: bool,
        leverage: float,
    ) -> float:
        """Convert a desired SL price to a custom_stoploss return value.

        Accounts for freqtrade's internal division by leverage in
        adjust_stop_loss().
        """
        if is_short:
            return -leverage * (desired_price / current_rate - 1)
        return -leverage * (1 - desired_price / current_rate)

    @classmethod
    def _bounded_sl(
        cls,
        desired_price: float,
        current_rate: float,
        is_short: bool,
        leverage: float,
        floor: float = -0.20,
    ) -> float:
        """Compute SL return value, clamped to a floor."""
        val = cls._price_to_sl(desired_price, current_rate, is_short, leverage)
        if val > 0:
            return -0.001
        return max(val, floor)

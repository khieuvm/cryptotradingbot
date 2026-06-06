"""Position sizing: ATR-risk-based, regime-aware, portfolio-heat-capped."""

from __future__ import annotations

from engine.config import AppConfig
from engine.events import Signal


class PositionSizer:
    """Determine position size based on signal, strategy config, and portfolio state."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._risk = config.get_risk_config()
        self._base_stake = config.data.get("stake_amount", 50)
        self._max_stake = config.data.get("max_stake_amount", 100)

    def calculate(
        self,
        signal: Signal,
        strategy_factor: float,
        current_exposure: float,
        wallet_balance: float,
        atr: float = 0.0,
        entry_price: float = 0.0,
    ) -> float:
        """Calculate stake amount.

        Args:
            signal: The entry signal
            strategy_factor: Multiplier from strategy config (e.g., trend_long=1.5)
            current_exposure: Current total exposure as fraction of wallet
            wallet_balance: Current wallet balance in USDT
            atr: Current ATR value (for risk-based sizing)
            entry_price: Expected entry price

        Returns:
            Stake amount in USDT
        """
        stake = self._base_stake * strategy_factor * signal.strength

        # ATR-risk-based adjustment: size inversely proportional to volatility
        if atr > 0 and entry_price > 0:
            risk_per_unit = atr / entry_price
            if risk_per_unit > 0.02:
                stake *= 0.02 / risk_per_unit

        # Portfolio heat cap
        max_heat = self._risk.get("max_portfolio_heat", 0.15)
        remaining_heat = max(0, max_heat - current_exposure)
        max_by_heat = remaining_heat * wallet_balance
        stake = min(stake, max_by_heat)

        # Hard cap
        stake = min(stake, self._max_stake)
        return max(stake, 0)

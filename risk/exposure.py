"""Exposure manager: portfolio heat and correlation limits."""

from __future__ import annotations

from engine.config import AppConfig


# Correlation groups: pairs that tend to move together
CORRELATION_GROUPS = {
    "crypto_major": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
    "crypto_alt": ["SOL/USDT:USDT"],
    "equity_token": ["SPX/USDT:USDT", "NVDA/USDT:USDT"],
}


class ExposureManager:
    """Portfolio-level exposure tracking and limits."""

    def __init__(self, config: AppConfig):
        self._risk = config.get_risk_config()
        self._max_trades = self._risk.get("max_open_trades", 6)
        self._max_heat = self._risk.get("max_portfolio_heat", 0.15)
        self._max_corr = self._risk.get("max_correlation_exposure", 0.6)
        self._open_positions: dict[str, dict] = {}

    def can_open_trade(
        self, pair: str, direction: str, stake: float, wallet_balance: float
    ) -> bool:
        """Check if a new trade is allowed given current exposure."""
        if len(self._open_positions) >= self._max_trades:
            return False

        current_heat = self._get_current_heat(wallet_balance)
        new_heat = stake / wallet_balance if wallet_balance > 0 else 1.0
        if current_heat + new_heat > self._max_heat:
            return False

        if self._would_exceed_correlation(pair, direction):
            return False

        return True

    def register_trade(
        self, trade_id: str, pair: str, direction: str, stake: float
    ) -> None:
        self._open_positions[trade_id] = {
            "pair": pair,
            "direction": direction,
            "stake": stake,
        }

    def unregister_trade(self, trade_id: str) -> None:
        self._open_positions.pop(trade_id, None)

    def get_current_exposure(self) -> float:
        """Total exposure in USDT."""
        return sum(pos["stake"] for pos in self._open_positions.values())

    def _get_current_heat(self, wallet_balance: float) -> float:
        if wallet_balance <= 0:
            return 1.0
        return self.get_current_exposure() / wallet_balance

    def _would_exceed_correlation(self, new_pair: str, new_direction: str) -> bool:
        """Check if opening this trade would create too much correlated exposure."""
        new_group = self._get_group(new_pair)
        if new_group is None:
            return False

        same_group_same_dir = sum(
            1
            for pos in self._open_positions.values()
            if self._get_group(pos["pair"]) == new_group
            and pos["direction"] == new_direction
        )
        return same_group_same_dir >= 2

    @staticmethod
    def _get_group(pair: str) -> str | None:
        for group_name, pairs in CORRELATION_GROUPS.items():
            if pair in pairs:
                return group_name
        return None

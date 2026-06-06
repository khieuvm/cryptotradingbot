"""Signal Tracker: monitors rolling win rate per combo/pair, auto-disables underperformers."""

from collections import deque
from datetime import datetime, timedelta

from src.strategy_config import get_signal_tracker_config


class SignalTracker:
    """Track trade outcomes and auto-disable combos that underperform."""

    def __init__(self):
        self._cfg = get_signal_tracker_config()
        self._trades: dict[str, deque] = {}
        self._disabled: dict[str, datetime] = {}

    def _key(self, combo_name: str, pair: str) -> str:
        return f"{combo_name}:{pair}"

    def record_trade(self, combo_name: str, pair: str, is_win: bool) -> None:
        key = self._key(combo_name, pair)
        window = self._cfg.get("window", 20)
        if key not in self._trades:
            self._trades[key] = deque(maxlen=window)
        self._trades[key].append(is_win)
        self._check_disable(combo_name, pair)

    def _check_disable(self, combo_name: str, pair: str) -> None:
        key = self._key(combo_name, pair)
        trades = self._trades.get(key, deque())

        disable_lookback = self._cfg.get("disable_lookback", 10)
        disable_wr = self._cfg.get("disable_threshold_wr", 0.40)
        duration_hours = self._cfg.get("disable_duration_hours", 24)

        if len(trades) >= disable_lookback:
            recent = list(trades)[-disable_lookback:]
            wr = sum(recent) / len(recent)
            if wr < disable_wr:
                self._disabled[key] = datetime.utcnow() + timedelta(hours=duration_hours)

    def is_disabled(self, combo_name: str, pair: str) -> bool:
        key = self._key(combo_name, pair)
        if key not in self._disabled:
            return False
        if datetime.utcnow() >= self._disabled[key]:
            del self._disabled[key]
            return False
        return True

    def get_win_rate(self, combo_name: str, pair: str) -> float | None:
        key = self._key(combo_name, pair)
        trades = self._trades.get(key, deque())
        if not trades:
            return None
        return sum(trades) / len(trades)

    def get_status(self) -> dict:
        """Return current status of all tracked combos."""
        status = {}
        for key, trades in self._trades.items():
            wr = sum(trades) / len(trades) if trades else 0
            disabled_until = self._disabled.get(key)
            status[key] = {
                "trades": len(trades),
                "win_rate": round(wr, 3),
                "disabled_until": disabled_until.isoformat() if disabled_until else None,
            }
        return status

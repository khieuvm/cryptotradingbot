"""Circuit breaker: portfolio-level protection with auto-disable."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any

from engine.config import AppConfig
from engine.event_bus import EventBus
from engine.events import AlertLevel, EventType, RiskAlert

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Portfolio-level protection that can halt trading.

    Monitors:
    - Per-strategy/pair win rate (auto-disable)
    - Daily/weekly drawdown (full halt)
    - Consecutive losses (strategy halt)
    """

    def __init__(self, config: AppConfig, event_bus: EventBus):
        self._event_bus = event_bus
        self._signal_cfg = config.get_signal_tracker_config()
        self._cb_cfg = config.get_circuit_breaker_config()

        # Per strategy:pair trade tracking
        self._trades: dict[str, deque] = {}
        self._disabled_until: dict[str, datetime] = {}

        # Portfolio-level
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._consecutive_losses: dict[str, int] = {}
        self._halted_until: datetime | None = None
        self._last_reset_day: int = -1
        self._last_reset_week: int = -1

    # ── Trade Recording ───────────────────────────────────────────────────────

    def record_trade(
        self, strategy_name: str, pair: str, profit_pct: float
    ) -> None:
        """Record a closed trade and check all thresholds."""
        key = f"{strategy_name}:{pair}"
        is_win = profit_pct > 0
        window = self._signal_cfg.get("window", 20)

        if key not in self._trades:
            self._trades[key] = deque(maxlen=window)
        self._trades[key].append(is_win)

        # Update portfolio PnL
        self._daily_pnl += profit_pct
        self._weekly_pnl += profit_pct

        # Update consecutive losses
        if not is_win:
            self._consecutive_losses[strategy_name] = (
                self._consecutive_losses.get(strategy_name, 0) + 1
            )
        else:
            self._consecutive_losses[strategy_name] = 0

        self._check_wr_disable(strategy_name, pair)
        self._check_drawdown()
        self._check_consecutive_losses(strategy_name)

    # ── Win Rate Auto-Disable ─────────────────────────────────────────────────

    def _check_wr_disable(self, strategy_name: str, pair: str) -> None:
        key = f"{strategy_name}:{pair}"
        trades = self._trades.get(key, deque())

        lookback = self._signal_cfg.get("disable_lookback", 10)
        threshold = self._signal_cfg.get("disable_threshold_wr", 0.40)
        duration = self._signal_cfg.get("disable_duration_hours", 24)

        if len(trades) >= lookback:
            recent = list(trades)[-lookback:]
            wr = sum(recent) / len(recent)
            if wr < threshold:
                until = datetime.utcnow() + timedelta(hours=duration)
                self._disabled_until[key] = until
                logger.warning(
                    f"Circuit breaker: disabled {key} until {until} (WR={wr:.1%})"
                )
                self._emit_alert(
                    AlertLevel.WARNING,
                    "signal_tracker",
                    f"{strategy_name} disabled on {pair}: WR {wr:.1%} < {threshold:.0%}",
                    [strategy_name],
                )

    # ── Drawdown Halt ─────────────────────────────────────────────────────────

    def _check_drawdown(self) -> None:
        max_daily = self._cb_cfg.get("max_daily_drawdown", 0.08)
        max_weekly = self._cb_cfg.get("max_weekly_drawdown", 0.12)
        cooldown = self._cb_cfg.get("cooldown_hours", 4)

        if self._daily_pnl < -max_daily:
            self._halted_until = datetime.utcnow() + timedelta(hours=cooldown)
            self._emit_alert(
                AlertLevel.HALT,
                "circuit_breaker",
                f"Daily drawdown {self._daily_pnl:.2%} exceeds {-max_daily:.1%}",
                [],
            )

        if self._weekly_pnl < -max_weekly:
            self._halted_until = datetime.utcnow() + timedelta(hours=cooldown * 2)
            self._emit_alert(
                AlertLevel.HALT,
                "circuit_breaker",
                f"Weekly drawdown {self._weekly_pnl:.2%} exceeds {-max_weekly:.1%}",
                [],
            )

    # ── Consecutive Losses ────────────────────────────────────────────────────

    def _check_consecutive_losses(self, strategy_name: str) -> None:
        max_consec = self._cb_cfg.get("consecutive_losses_halt", 5)
        if self._consecutive_losses.get(strategy_name, 0) >= max_consec:
            cooldown = self._cb_cfg.get("cooldown_hours", 4)
            key_prefix = f"{strategy_name}:"
            until = datetime.utcnow() + timedelta(hours=cooldown)
            for key in list(self._disabled_until.keys()):
                if key.startswith(key_prefix):
                    self._disabled_until[key] = max(self._disabled_until[key], until)
            # Also disable any pair not yet tracked
            self._disabled_until[f"{strategy_name}:*"] = until
            self._emit_alert(
                AlertLevel.CRITICAL,
                "circuit_breaker",
                f"{strategy_name}: {max_consec} consecutive losses → halted",
                [strategy_name],
            )

    # ── Query Methods ─────────────────────────────────────────────────────────

    def is_disabled(self, strategy_name: str, pair: str) -> bool:
        """Check if a strategy/pair combination is currently disabled."""
        if self.is_halted():
            return True

        now = datetime.utcnow()

        # Check strategy-wide disable
        wildcard_key = f"{strategy_name}:*"
        if wildcard_key in self._disabled_until:
            if now < self._disabled_until[wildcard_key]:
                return True
            del self._disabled_until[wildcard_key]

        # Check specific pair disable
        key = f"{strategy_name}:{pair}"
        if key in self._disabled_until:
            if now < self._disabled_until[key]:
                return True
            del self._disabled_until[key]

        return False

    def is_halted(self) -> bool:
        """Check if the entire portfolio is halted."""
        if self._halted_until is None:
            return False
        if datetime.utcnow() >= self._halted_until:
            self._halted_until = None
            return False
        return True

    def get_win_rate(self, strategy_name: str, pair: str) -> float | None:
        key = f"{strategy_name}:{pair}"
        trades = self._trades.get(key, deque())
        if not trades:
            return None
        return sum(trades) / len(trades)

    def get_status(self) -> dict[str, Any]:
        """Get full circuit breaker status."""
        status: dict[str, Any] = {
            "halted": self.is_halted(),
            "halted_until": self._halted_until.isoformat() if self._halted_until else None,
            "daily_pnl": round(self._daily_pnl, 4),
            "weekly_pnl": round(self._weekly_pnl, 4),
            "strategies": {},
        }
        for key, trades in self._trades.items():
            wr = sum(trades) / len(trades) if trades else 0
            disabled_until = self._disabled_until.get(key)
            status["strategies"][key] = {
                "trades": len(trades),
                "win_rate": round(wr, 3),
                "disabled_until": disabled_until.isoformat() if disabled_until else None,
            }
        return status

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0

    def reset_weekly(self) -> None:
        self._weekly_pnl = 0.0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit_alert(
        self, level: AlertLevel, source: str, message: str, affected: list[str]
    ) -> None:
        self._event_bus.emit(
            EventType.RISK_ALERT,
            RiskAlert(
                level=level,
                source=source,
                message=message,
                affected_strategies=affected,
                timestamp=datetime.utcnow(),
            ),
        )

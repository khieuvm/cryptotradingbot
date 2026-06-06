"""Enhanced BaseStrategy with full lifecycle hooks and event emission."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

from engine.config import StrategyConfig
from engine.event_bus import EventBus
from engine.events import (
    Direction,
    EventType,
    ExitRequest,
    Signal,
    Urgency,
)


class BaseStrategy(ABC):
    """Self-contained strategy unit with lifecycle hooks.

    Each strategy contains:
    - Indicator computation
    - Signal generation (entry + exit)
    - Risk parameters (SL/TP/sizing)
    - Internal state management

    Strategies emit events to the orchestrator via the event bus.
    """

    name: str = ""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self._event_bus: EventBus | None = None
        self._dp: Any = None
        self._state: dict[str, Any] = {}
        self._is_initialized = False

    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE HOOKS
    # ═══════════════════════════════════════════════════════════════════════════

    def on_init(self, dp: Any, event_bus: EventBus) -> None:
        """Called once when strategy is loaded by the orchestrator."""
        self._event_bus = event_bus
        self._dp = dp
        self._is_initialized = True
        self._on_init_hook()

    def _on_init_hook(self) -> None:
        """Override for custom initialization (subscribe to events, etc.)."""
        pass

    @abstractmethod
    def on_tick(self, dataframe: pd.DataFrame, pair: str, current_time: datetime) -> None:
        """Called every candle. Core logic: compute → detect → emit.

        The strategy should:
        1. Update indicators if needed
        2. Check for entry signals → call emit_signal()
        3. Check for exit conditions → call emit_exit()
        4. Update internal state
        """
        ...

    def on_entry(self, pair: str, side: str, rate: float, stake: float) -> None:
        """Called when a trade is actually opened (post-confirmation)."""
        pass

    def on_exit(self, pair: str, profit: float, exit_reason: str) -> None:
        """Called when a trade is closed. Update win/loss tracking."""
        pass

    def on_error(self, error: str) -> None:
        """Called on strategy error or risk halt."""
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL EMISSION
    # ═══════════════════════════════════════════════════════════════════════════

    def emit_signal(
        self,
        pair: str,
        direction: Direction,
        strength: float = 1.0,
        entry_price: float | None = None,
        tag: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit an entry signal to the event bus."""
        if self._event_bus is None:
            return
        signal = Signal(
            strategy_name=self.name,
            pair=pair,
            direction=direction,
            strength=strength,
            entry_price=entry_price,
            tag=tag or f"{self.name}_{direction.value}",
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )
        self._event_bus.emit(EventType.SIGNAL, signal)

    def emit_exit(
        self, pair: str, reason: str, urgency: Urgency = Urgency.NEXT_CANDLE
    ) -> None:
        """Emit an exit request to the event bus."""
        if self._event_bus is None:
            return
        self._event_bus.emit(
            EventType.EXIT_REQUEST,
            ExitRequest(
                strategy_name=self.name,
                pair=pair,
                reason=reason,
                urgency=urgency,
                timestamp=datetime.utcnow(),
            ),
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ABSTRACT: INDICATOR & SIGNAL METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Compute strategy-specific indicators on the dataframe.

        Called from the adapter's populate_indicators(). Should add columns
        prefixed with strategy abbreviation (e.g., 'ra_rsi', 'mr_bb_upper').
        """
        ...

    @abstractmethod
    def detect_entries(self, dataframe: pd.DataFrame, pair: str) -> list[Signal]:
        """Detect entry signals from current dataframe state.

        Returns list of Signal objects for the last row (current candle).
        Used by the adapter to set enter_long/enter_short columns in backtesting.
        """
        ...

    @abstractmethod
    def detect_exits(
        self, dataframe: pd.DataFrame, pair: str, trade_info: dict | None
    ) -> ExitRequest | None:
        """Check if current position should be exited.

        trade_info contains: entry_rate, entry_time, current_profit, enter_tag, is_short
        Returns ExitRequest or None.
        """
        ...

    # ═══════════════════════════════════════════════════════════════════════════
    # RISK PARAMETERS (strategy provides its own defaults)
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def sl_atr_mult(self) -> float:
        return self.config.exit.get("sl_atr_mult", 2.0)

    @property
    def tp_atr_mult(self) -> float:
        return self.config.exit.get("tp_atr_mult", 3.0)

    @property
    def entry_atr_fraction(self) -> float:
        return self.config.entry.get("entry_atr_fraction", 0.0)

    def get_position_size_factor(
        self, pair: str, direction: str, regime: str | None = None
    ) -> float:
        """Strategy-specific position sizing multiplier."""
        if regime:
            key = f"{regime}_{direction}"
        else:
            key = direction
        return self.config.stake.get(key, 1.0)

    def get_stoploss_distance(self, atr: float) -> float:
        """ATR-based stoploss distance (absolute price units)."""
        return self.sl_atr_mult * atr

    def get_tp_distance(self, atr: float) -> float:
        """ATR-based take-profit distance (absolute price units)."""
        return self.tp_atr_mult * atr

    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def grade(self) -> str:
        return self.config.grade

    @property
    def pairs(self) -> list[str]:
        return self.config.pairs

    @property
    def startup_candle_count(self) -> int:
        return self.config.startup_candle_count

    def is_pair_active(self, pair: str) -> bool:
        return pair in self.config.pairs

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} grade={self.grade}>"

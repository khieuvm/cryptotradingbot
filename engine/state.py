"""Global mutable state shared across the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from engine.events import Signal, ExitRequest


@dataclass
class TradeInfo:
    pair: str
    strategy_name: str
    direction: str
    entry_rate: float
    entry_time: datetime
    stake_amount: float
    enter_tag: str


@dataclass
class GlobalState:
    """Central state store for the engine."""

    pending_signals: dict[str, Signal] = field(default_factory=dict)
    pending_exits: dict[str, ExitRequest] = field(default_factory=dict)
    active_trades: dict[str, TradeInfo] = field(default_factory=dict)

    regime: dict[str, str] = field(default_factory=dict)
    btc_sentiment: float = 50.0
    funding_rates: dict[str, float] = field(default_factory=dict)

    strategy_health: dict[str, str] = field(default_factory=dict)
    last_heartbeat: dict[str, datetime] = field(default_factory=dict)

    def get_regime(self, pair: str) -> str:
        return self.regime.get(pair, "unknown")

    def get_funding(self, pair: str) -> float:
        return self.funding_rates.get(pair, 0.0)

    def is_strategy_active(self, strategy_name: str) -> bool:
        return self.strategy_health.get(strategy_name, "active") == "active"

    def set_strategy_disabled(self, strategy_name: str) -> None:
        self.strategy_health[strategy_name] = "disabled"

    def set_strategy_active(self, strategy_name: str) -> None:
        self.strategy_health[strategy_name] = "active"

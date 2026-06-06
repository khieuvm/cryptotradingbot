"""Typed event definitions for the strategy engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(Enum):
    SIGNAL = "signal"
    EXIT_REQUEST = "exit_request"
    SIZE_CHANGE = "size_change"
    RISK_ALERT = "risk_alert"
    STATE_CHANGE = "state_change"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


class Urgency(Enum):
    IMMEDIATE = "immediate"
    NEXT_CANDLE = "next_candle"
    CONDITIONAL = "conditional"


class AlertLevel(Enum):
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"


@dataclass(frozen=True)
class Signal:
    strategy_name: str
    pair: str
    direction: Direction
    strength: float
    tag: str
    timestamp: datetime
    entry_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExitRequest:
    strategy_name: str
    pair: str
    reason: str
    urgency: Urgency
    timestamp: datetime


@dataclass(frozen=True)
class SizeChange:
    strategy_name: str
    pair: str
    new_factor: float
    reason: str
    timestamp: datetime


@dataclass(frozen=True)
class RiskAlert:
    level: AlertLevel
    source: str
    message: str
    affected_strategies: list[str]
    timestamp: datetime

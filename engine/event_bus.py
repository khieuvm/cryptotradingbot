"""In-process pub/sub event bus with typed subscriptions."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable

from engine.events import EventType

logger = logging.getLogger(__name__)


class EventBus:
    """Thread-safe event bus for strategy-engine communication."""

    def __init__(self, max_log_size: int = 1000):
        self._subscribers: dict[EventType, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._event_log: list[tuple[datetime, EventType, Any]] = []
        self._max_log_size = max_log_size

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        with self._lock:
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        with self._lock:
            handlers = self._subscribers[event_type]
            if handler in handlers:
                handlers.remove(handler)

    def emit(self, event_type: EventType, payload: Any) -> None:
        with self._lock:
            handlers = list(self._subscribers[event_type])
            self._event_log.append((datetime.utcnow(), event_type, payload))
            if len(self._event_log) > self._max_log_size:
                self._event_log = self._event_log[-self._max_log_size:]

        for handler in handlers:
            try:
                handler(payload)
            except Exception as e:
                logger.error(f"Event handler error for {event_type.value}: {e}")

    def get_recent_events(
        self, event_type: EventType | None = None, limit: int = 50
    ) -> list[tuple[datetime, EventType, Any]]:
        with self._lock:
            events = self._event_log
            if event_type is not None:
                events = [e for e in events if e[1] == event_type]
            return events[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._subscribers.clear()
            self._event_log.clear()

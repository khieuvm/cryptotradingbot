"""Tests for the event bus and engine core."""

from datetime import datetime

from engine.event_bus import EventBus
from engine.events import (
    AlertLevel,
    Direction,
    EventType,
    ExitRequest,
    RiskAlert,
    Signal,
    Urgency,
)


def test_event_bus_subscribe_emit():
    bus = EventBus()
    received = []
    bus.subscribe(EventType.SIGNAL, lambda s: received.append(s))

    signal = Signal(
        strategy_name="test",
        pair="BTC/USDT:USDT",
        direction=Direction.LONG,
        strength=1.0,
        tag="test_long",
        timestamp=datetime.utcnow(),
    )
    bus.emit(EventType.SIGNAL, signal)
    assert len(received) == 1
    assert received[0].pair == "BTC/USDT:USDT"


def test_event_bus_multiple_subscribers():
    bus = EventBus()
    results_a = []
    results_b = []
    bus.subscribe(EventType.SIGNAL, lambda s: results_a.append(s))
    bus.subscribe(EventType.SIGNAL, lambda s: results_b.append(s))

    signal = Signal(
        strategy_name="test", pair="ETH/USDT:USDT",
        direction=Direction.SHORT, strength=0.5,
        tag="test_short", timestamp=datetime.utcnow(),
    )
    bus.emit(EventType.SIGNAL, signal)
    assert len(results_a) == 1
    assert len(results_b) == 1


def test_event_bus_different_types():
    bus = EventBus()
    signals = []
    exits = []
    bus.subscribe(EventType.SIGNAL, lambda s: signals.append(s))
    bus.subscribe(EventType.EXIT_REQUEST, lambda e: exits.append(e))

    bus.emit(EventType.SIGNAL, Signal(
        strategy_name="t", pair="BTC/USDT:USDT",
        direction=Direction.LONG, strength=1.0,
        tag="t", timestamp=datetime.utcnow(),
    ))
    bus.emit(EventType.EXIT_REQUEST, ExitRequest(
        strategy_name="t", pair="BTC/USDT:USDT",
        reason="tp", urgency=Urgency.IMMEDIATE,
        timestamp=datetime.utcnow(),
    ))
    assert len(signals) == 1
    assert len(exits) == 1


def test_event_bus_recent_events():
    bus = EventBus()
    for i in range(5):
        bus.emit(EventType.HEARTBEAT, {"tick": i})
    events = bus.get_recent_events(EventType.HEARTBEAT, limit=3)
    assert len(events) == 3


def test_event_bus_unsubscribe():
    bus = EventBus()
    received = []
    handler = lambda s: received.append(s)
    bus.subscribe(EventType.SIGNAL, handler)
    bus.unsubscribe(EventType.SIGNAL, handler)

    bus.emit(EventType.SIGNAL, Signal(
        strategy_name="t", pair="BTC/USDT:USDT",
        direction=Direction.LONG, strength=1.0,
        tag="t", timestamp=datetime.utcnow(),
    ))
    assert len(received) == 0

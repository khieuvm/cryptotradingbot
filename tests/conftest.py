"""Test fixtures for the trading engine."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.config import AppConfig
from engine.event_bus import EventBus
from engine.events import EventType


@pytest.fixture
def app_config():
    return AppConfig(env="backtest")


@pytest.fixture
def event_bus():
    return EventBus()

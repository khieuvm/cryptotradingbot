"""Tests for risk management modules."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from engine.config import AppConfig
from engine.event_bus import EventBus
from engine.events import Direction, Signal
from risk.circuit_breaker import CircuitBreaker
from risk.stoploss import StoplossManager


class TestStoplossManager:
    def setup_method(self):
        self.mgr = StoplossManager()

    def test_initial_stoploss(self):
        sl = self.mgr.calculate_stoploss(
            current_profit=-0.01,
            sl_atr_mult=2.0, tp_atr_mult=3.0,
            atr=100.0, open_rate=50000.0,
            current_rate=49500.0, is_short=False,
        )
        expected = -(2.0 * 100 / 50000)
        assert abs(sl - expected) < 0.001

    def test_breakeven_phase(self):
        # profit at 60% of TP (which is 3*atr/open = 0.006 → 60% = 0.0036)
        sl = self.mgr.calculate_stoploss(
            current_profit=0.004,
            sl_atr_mult=2.0, tp_atr_mult=3.0,
            atr=100.0, open_rate=50000.0,
            current_rate=50200.0, is_short=False,
        )
        # Should be close to break-even (slightly above initial SL)
        assert sl > -0.004

    def test_trail_lock_phase(self):
        # profit exceeds TP
        sl = self.mgr.calculate_stoploss(
            current_profit=0.01,
            sl_atr_mult=2.0, tp_atr_mult=3.0,
            atr=100.0, open_rate=50000.0,
            current_rate=50500.0, is_short=False,
        )
        assert sl > -0.01

    def test_invalid_atr_returns_default(self):
        sl = self.mgr.calculate_stoploss(
            current_profit=0.0,
            sl_atr_mult=2.0, tp_atr_mult=3.0,
            atr=0.0, open_rate=50000.0,
            current_rate=50000.0, is_short=False,
        )
        assert sl == -0.10


class TestCircuitBreaker:
    def setup_method(self):
        self.config = MagicMock(spec=AppConfig)
        self.config.get_signal_tracker_config.return_value = {
            "window": 20,
            "disable_threshold_wr": 0.40,
            "disable_lookback": 5,
            "disable_duration_hours": 24,
        }
        self.config.get_circuit_breaker_config.return_value = {
            "max_daily_drawdown": 0.08,
            "max_weekly_drawdown": 0.12,
            "consecutive_losses_halt": 3,
            "cooldown_hours": 4,
        }
        self.bus = EventBus()
        self.cb = CircuitBreaker(self.config, self.bus)

    def test_not_disabled_initially(self):
        assert not self.cb.is_disabled("regime_adaptive", "BTC/USDT:USDT")

    def test_disable_on_low_wr(self):
        # Record 5 losing trades → WR = 0% < 40%
        for _ in range(5):
            self.cb.record_trade("regime_adaptive", "BTC/USDT:USDT", -0.01)
        assert self.cb.is_disabled("regime_adaptive", "BTC/USDT:USDT")

    def test_not_disabled_with_wins(self):
        # Record 4 wins, 1 loss → WR = 80%
        for _ in range(4):
            self.cb.record_trade("regime_adaptive", "BTC/USDT:USDT", 0.01)
        self.cb.record_trade("regime_adaptive", "BTC/USDT:USDT", -0.01)
        assert not self.cb.is_disabled("regime_adaptive", "BTC/USDT:USDT")

    def test_halt_on_daily_drawdown(self):
        # Simulate losing > 8% in a day
        self.cb.record_trade("test", "BTC/USDT:USDT", -0.09)
        assert self.cb.is_halted()

    def test_consecutive_losses_halt(self):
        for _ in range(3):
            self.cb.record_trade("test_strat", "ETH/USDT:USDT", -0.01)
        assert self.cb.is_disabled("test_strat", "ETH/USDT:USDT")

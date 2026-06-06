"""Central orchestrator: manages strategy lifecycle, events, and risk."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from engine.config import AppConfig, StrategyConfig
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
from engine.state import GlobalState, TradeInfo
from risk.circuit_breaker import CircuitBreaker
from risk.exposure import ExposureManager
from risk.position_sizer import PositionSizer
from risk.stoploss import StoplossManager
from strategies import get_active_strategy_classes
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class Orchestrator:
    """Central engine that manages strategy lifecycle and event flow.

    Bridges between freqtrade's callback model and the event-driven strategies.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.event_bus = EventBus()
        self.state = GlobalState()

        self.circuit_breaker = CircuitBreaker(config, self.event_bus)
        self.exposure_mgr = ExposureManager(config)
        self.position_sizer = PositionSizer(config)
        self.stoploss_mgr = StoplossManager()

        self._strategies: dict[str, BaseStrategy] = {}
        self._pair_strategies: dict[str, list[BaseStrategy]] = {}
        self._entry_filters = config.data.get("entry_filters", {})
        self._dp: Any = None

    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def initialize(self, dp: Any) -> None:
        """Called once from bot_start(). Load strategies, subscribe events."""
        self._dp = dp
        for name, cls in get_active_strategy_classes(self.config):
            strat_cfg = self.config.get_strategy_config(name)
            strat = cls(strat_cfg)
            strat.on_init(dp=dp, event_bus=self.event_bus)
            self._strategies[name] = strat
            self.state.set_strategy_active(name)

            for pair in strat.pairs:
                if pair not in self._pair_strategies:
                    self._pair_strategies[pair] = []
                self._pair_strategies[pair].append(strat)

        self.event_bus.subscribe(EventType.SIGNAL, self._handle_signal)
        self.event_bus.subscribe(EventType.EXIT_REQUEST, self._handle_exit_request)
        self.event_bus.subscribe(EventType.RISK_ALERT, self._handle_risk_alert)

        logger.info(
            f"Orchestrator initialized with {len(self._strategies)} strategies: "
            f"{list(self._strategies.keys())}"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # FREQTRADE ADAPTER INTERFACE
    # ═══════════════════════════════════════════════════════════════════════════

    def get_informative_pairs(self) -> list[tuple]:
        """Aggregate informative pairs from all strategies."""
        pairs = set()
        pairs.add(("BTC/USDT:USDT", "1h"))
        from freqtrade.constants import CandleType
        for pair in self.config.get_pairs():
            pairs.add((pair, "1h", CandleType.FUNDING_RATE))
        return list(pairs)

    def get_max_startup_candles(self) -> int:
        return max(
            (s.startup_candle_count for s in self._strategies.values()),
            default=100,
        )

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """Delegate indicator computation to each strategy active on this pair."""
        pair = metadata["pair"]
        from indicators.market_data import add_btc_sentiment, add_funding_rate
        dataframe = add_btc_sentiment(dataframe, self._dp, self.config.get_timeframe())
        dataframe = add_funding_rate(dataframe, self._dp, pair, self.config.get_timeframe())

        for strat in self._pair_strategies.get(pair, []):
            dataframe = strat.compute_indicators(dataframe, metadata)

        return dataframe

    def get_pending_entries(
        self, pair: str, dataframe: pd.DataFrame
    ) -> list[Signal]:
        """Detect entry signals for backtesting (column-based path)."""
        signals: list[Signal] = []
        for strat in self._pair_strategies.get(pair, []):
            if strat.grade not in ("A", "B"):
                continue
            if not self.state.is_strategy_active(strat.name):
                continue
            strat_signals = strat.detect_entries(dataframe, pair)
            signals.extend(strat_signals)
        return signals

    def check_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
    ) -> str | None:
        """Check if current trade should be exited. Called from custom_exit."""
        enter_tag = getattr(trade, "enter_tag", "") or ""
        strategy_name = enter_tag.rsplit("_", 1)[0] if "_" in enter_tag else ""

        # Check pending exits from event bus
        if pair in self.state.pending_exits:
            exit_req = self.state.pending_exits.pop(pair)
            return exit_req.reason

        # Ask the strategy directly
        strat = self._strategies.get(strategy_name)
        if strat is None:
            return None

        df, _ = self._dp.get_analyzed_dataframe(pair, self.config.get_timeframe())
        if df.empty:
            return None

        trade_info = {
            "entry_rate": trade.open_rate,
            "entry_time": trade.open_date_utc,
            "current_profit": current_profit,
            "enter_tag": enter_tag,
            "is_short": trade.is_short,
            "current_time": current_time,
            "current_rate": current_rate,
        }
        exit_req = strat.detect_exits(df, pair, trade_info)
        if exit_req is not None:
            return exit_req.reason
        return None

    def get_stoploss(
        self,
        pair: str,
        trade: Any,
        current_rate: float,
        current_profit: float,
    ) -> float:
        """Calculate dynamic stoploss. Called from custom_stoploss."""
        enter_tag = getattr(trade, "enter_tag", "") or ""
        strategy_name = enter_tag.rsplit("_", 1)[0] if "_" in enter_tag else ""

        strat = self._strategies.get(strategy_name)
        if strat is None:
            return -0.10

        df, _ = self._dp.get_analyzed_dataframe(pair, self.config.get_timeframe())
        if df.empty:
            return -0.10

        last = df.iloc[-1]
        atr = self._get_atr(last, strategy_name)
        fee = getattr(trade, "fee_open", 0.0005) + getattr(trade, "fee_close", 0.0005)

        return self.stoploss_mgr.calculate_stoploss(
            current_profit=current_profit,
            sl_atr_mult=strat.sl_atr_mult,
            tp_atr_mult=strat.tp_atr_mult,
            atr=atr,
            open_rate=trade.open_rate,
            current_rate=current_rate,
            is_short=trade.is_short,
            fee=fee,
        )

    def calculate_stake(
        self,
        pair: str,
        entry_tag: str,
        side: str,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
    ) -> float:
        """Calculate position size. Called from custom_stake_amount."""
        strategy_name = entry_tag.rsplit("_", 1)[0] if "_" in entry_tag else ""
        strat = self._strategies.get(strategy_name)
        if strat is None:
            return proposed_stake

        regime = self.state.get_regime(pair)
        factor = strat.get_position_size_factor(pair, side, regime)
        stake = proposed_stake * factor

        if min_stake is not None:
            stake = max(stake, float(min_stake))
        return min(stake, max_stake)

    def confirm_entry(
        self,
        pair: str,
        entry_tag: str,
        side: str,
        rate: float,
        current_time: datetime,
    ) -> bool:
        """Pre-trade validation. Called from confirm_trade_entry."""
        strategy_name = entry_tag.rsplit("_", 1)[0] if "_" in entry_tag else ""

        # Circuit breaker check
        if self.circuit_breaker.is_disabled(strategy_name, pair):
            return False

        # Portfolio halt check
        if self.circuit_breaker.is_halted():
            return False

        df, _ = self._dp.get_analyzed_dataframe(pair, self.config.get_timeframe())
        if df.empty:
            return True
        last = df.iloc[-1]

        # Volume sanity
        vol_ratio = float(last.get("vol_ratio", last.get("ra_vol_ratio", 1.0)))
        if vol_ratio < self._entry_filters.get("min_volume_ratio", 0.4):
            return False

        # ATR spike protection
        atr = self._get_atr(last, strategy_name)
        close = float(last.get("close", 1))
        if atr > 0 and (atr / close) > self._entry_filters.get("atr_spike_max", 0.04):
            return False

        # BTC sentiment gate
        btc_rsi = float(last.get("btc_rsi_1h", 50))
        if side == "long" and btc_rsi < self._entry_filters.get("btc_rsi_long_min", 35):
            return False
        if side == "short" and btc_rsi > self._entry_filters.get("btc_rsi_short_max", 65):
            return False

        # Funding rate extreme filter
        funding = float(last.get("funding_rate", 0.0))
        if side == "long" and funding > self._entry_filters.get("funding_long_max", 0.00008):
            return False
        if side == "short" and funding < self._entry_filters.get("funding_short_min", -0.00007):
            return False

        return True

    def confirm_exit(
        self, pair: str, trade: Any, rate: float, exit_reason: str
    ) -> bool:
        """Post-trade recording. Called from confirm_trade_exit."""
        enter_tag = getattr(trade, "enter_tag", "") or ""
        strategy_name = enter_tag.rsplit("_", 1)[0] if "_" in enter_tag else ""
        is_win = (rate - trade.open_rate > 0) if not trade.is_short else (trade.open_rate - rate > 0)
        profit_pct = (rate - trade.open_rate) / trade.open_rate
        if trade.is_short:
            profit_pct = (trade.open_rate - rate) / trade.open_rate

        self.circuit_breaker.record_trade(strategy_name, pair, profit_pct)

        strat = self._strategies.get(strategy_name)
        if strat:
            strat.on_exit(pair, profit_pct, exit_reason)

        return True

    def get_leverage(self, pair: str, entry_tag: str) -> float:
        """Get leverage for a trade."""
        return float(self.config.get_leverage_config().get("default", 3))

    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _handle_signal(self, signal: Signal) -> None:
        self.state.pending_signals[signal.pair] = signal

    def _handle_exit_request(self, request: ExitRequest) -> None:
        self.state.pending_exits[request.pair] = request

    def _handle_risk_alert(self, alert: RiskAlert) -> None:
        if alert.level == AlertLevel.HALT:
            logger.critical(f"RISK HALT: {alert.message}")
            for name in alert.affected_strategies:
                strat = self._strategies.get(name)
                if strat:
                    strat.on_error(alert.message)
                self.state.set_strategy_disabled(name)
        elif alert.level == AlertLevel.CRITICAL:
            logger.error(f"RISK CRITICAL: {alert.message}")

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_atr(last_row: pd.Series, strategy_name: str) -> float:
        """Get ATR value from dataframe row, checking strategy-prefixed columns."""
        prefixes = ["ra_", "tc_", "mr_", ""]
        for p in prefixes:
            val = last_row.get(f"{p}atr", 0)
            if val and float(val) > 0:
                return float(val)
        return 0.0

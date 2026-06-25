"""Central orchestrator: manages strategy lifecycle, events, and risk."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Any

import numpy as np
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

_TG_TOKEN = ""
_TG_CHAT = ""


def _init_tg(config: AppConfig) -> None:
    global _TG_TOKEN, _TG_CHAT
    tg = config.data.get("telegram", {})
    _TG_TOKEN = tg.get("token", "")
    _TG_CHAT = str(tg.get("chat_id", ""))


def _send_tg(msg: str) -> None:
    if not _TG_TOKEN or not _TG_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{_TG_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": _TG_CHAT, "text": msg, "parse_mode": "Markdown",
        }).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=5)
    except Exception:
        pass


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
        self._filter_blocked_last: dict[str, str] = {}  # debounce: key -> candle_time

    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def initialize(self, dp: Any) -> None:
        """Called once from bot_start(). Load strategies, subscribe events."""
        self._dp = dp
        _init_tg(self.config)
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
        strategy_name = self._parse_strategy_name(enter_tag)

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
            "leverage": getattr(trade, "leverage", 1.0) or 1.0,
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
        current_time: datetime | None = None,
    ) -> float:
        """Calculate dynamic stoploss. Called from custom_stoploss."""
        enter_tag = getattr(trade, "enter_tag", "") or ""
        strategy_name = self._parse_strategy_name(enter_tag)

        # Grace period: don't tighten SL on the entry candle.
        # High-volatility strategies (volume_spike_rev, cb_adx_breakout) enter
        # on big-move candles where the next candle's bounce easily exceeds a
        # tight ATR-based SL.  Returning -1 tells freqtrade to use the initial
        # stoploss (-0.15) for the first candle, giving the trade room to breathe.
        if current_time is not None:
            open_date = getattr(trade, "open_date_utc", None)
            if open_date is not None:
                try:
                    duration_s = (current_time - open_date).total_seconds()
                except TypeError:
                    duration_s = 3600
                timeframe = self.config.get_timeframe()
                candle_s = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "1h": 3600}.get(
                    timeframe, 900
                )
                if duration_s < candle_s:
                    return -1

        strat = self._strategies.get(strategy_name)
        if strat is None:
            return -0.10

        df, _ = self._dp.get_analyzed_dataframe(pair, self.config.get_timeframe())
        if df.empty:
            return -0.10

        last = df.iloc[-1]
        atr = self._get_atr(last, strategy_name)
        fee = getattr(trade, "fee_open", 0.0005) + getattr(trade, "fee_close", 0.0005)

        trailing_cfg = strat.get_trailing_config(pair)

        leverage = getattr(trade, "leverage", 1.0) or 1.0

        return self.stoploss_mgr.calculate_stoploss(
            current_profit=current_profit,
            sl_atr_mult=strat.get_sl_atr_mult(pair),
            tp_atr_mult=strat.get_tp_atr_mult(pair),
            atr=atr,
            open_rate=trade.open_rate,
            current_rate=current_rate,
            is_short=trade.is_short,
            fee=fee,
            trailing_cfg=trailing_cfg,
            leverage=leverage,
            fixed_only=strat.is_fixed_sl_only(),
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
        strategy_name = self._parse_strategy_name(entry_tag)
        strat = self._strategies.get(strategy_name)
        if strat is None:
            return proposed_stake

        regime = self.state.get_regime(pair)
        factor = strat.get_position_size_factor(pair, side, regime)
        stake = proposed_stake * factor

        if min_stake is not None:
            stake = max(stake, float(min_stake))
        return min(stake, max_stake)

    def _should_log_filter(self, key: str, candle_time: Any) -> bool:
        """Debounce filter logs: only log once per candle per filter key."""
        ts = str(candle_time)
        if self._filter_blocked_last.get(key) == ts:
            return False
        self._filter_blocked_last[key] = ts
        return True

    def _detect_wave_trend(self, df: pd.DataFrame) -> str:
        """Detect wave context from price zigzag. Returns 'bullish'/'bearish'/'unclear'.
        Used to block entries against the current wave structure."""
        if len(df) < 60:
            return "unclear"

        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        n = len(highs)
        threshold = 0.01  # 1% zigzag threshold

        # Compute zigzag pivots
        pivots: list[tuple[int, float, str]] = []  # (idx, price, "H"/"L")
        last_h, last_h_idx = highs[0], 0
        last_l, last_l_idx = lows[0], 0
        direction = 0

        for i in range(1, n):
            if direction == 0:
                if highs[i] > last_h:
                    last_h, last_h_idx = highs[i], i
                if lows[i] < last_l:
                    last_l, last_l_idx = lows[i], i
                if last_l > 0 and (last_h - last_l) / last_l >= threshold:
                    if last_l_idx < last_h_idx:
                        pivots.append((last_l_idx, last_l, "L"))
                        direction = 1
                    else:
                        pivots.append((last_h_idx, last_h, "H"))
                        direction = -1
            elif direction == 1:
                if highs[i] > last_h:
                    last_h, last_h_idx = highs[i], i
                if last_h > 0 and (last_h - lows[i]) / last_h >= threshold:
                    pivots.append((last_h_idx, last_h, "H"))
                    direction = -1
                    last_l, last_l_idx = lows[i], i
            else:
                if lows[i] < last_l:
                    last_l, last_l_idx = lows[i], i
                if last_l > 0 and (highs[i] - last_l) / last_l >= threshold:
                    pivots.append((last_l_idx, last_l, "L"))
                    direction = 1
                    last_h, last_h_idx = highs[i], i

        if direction == 1:
            pivots.append((last_h_idx, last_h, "H"))
        elif direction == -1:
            pivots.append((last_l_idx, last_l, "L"))

        if len(pivots) < 4:
            return "unclear"

        # Check last 4 pivots for correction pattern
        p = pivots[-4:]
        prices = [pt[1] for pt in p]

        # Correction UP (L-H-L-H): bounce in downtrend
        if p[0][2] == "L" and p[1][2] == "H" and p[2][2] == "L" and p[3][2] == "H":
            a_up = prices[1] - prices[0]
            if a_up > 0 and prices[2] > prices[0] and prices[3] < prices[1]:
                b_ret = (prices[1] - prices[2]) / a_up
                if 0.2 <= b_ret <= 0.9:
                    return "bullish"  # correction up = bearish for shorts

        # Correction DOWN (H-L-H-L): pullback in uptrend
        if p[0][2] == "H" and p[1][2] == "L" and p[2][2] == "H" and p[3][2] == "L":
            a_down = prices[0] - prices[1]
            if a_down > 0 and prices[2] < prices[0] and prices[3] > prices[1]:
                b_ret = (prices[2] - prices[1]) / a_down
                if 0.2 <= b_ret <= 0.9:
                    return "bearish"  # correction down = bullish for longs

        # Check impulse: last 6 pivots
        if len(pivots) >= 6:
            p6 = pivots[-6:]
            prices6 = [pt[1] for pt in p6]
            # Impulse UP (L-H-L-H-L-H with higher highs)
            if p6[0][2] == "L" and prices6[5] > prices6[0]:
                w1 = prices6[1] - prices6[0]
                w3 = prices6[3] - prices6[2]
                w5 = prices6[5] - prices6[4]
                if w1 > 0 and w3 > 0 and w5 > 0:
                    if not (w3 < w1 and w3 < w5):  # W3 not shortest
                        return "bullish"
            # Impulse DOWN (H-L-H-L-H-L with lower lows)
            if p6[0][2] == "H" and prices6[5] < prices6[0]:
                w1 = prices6[0] - prices6[1]
                w3 = prices6[2] - prices6[3]
                w5 = prices6[4] - prices6[5]
                if w1 > 0 and w3 > 0 and w5 > 0:
                    if not (w3 < w1 and w3 < w5):
                        return "bearish"

        return "unclear"

    def confirm_entry(
        self,
        pair: str,
        entry_tag: str,
        side: str,
        rate: float,
        current_time: datetime,
    ) -> bool:
        """Pre-trade validation. Called from confirm_trade_entry."""
        strategy_name = self._parse_strategy_name(entry_tag)

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
        _candle_time = last.get("date", current_time)

        # Volume sanity
        vol_ratio = float(last.get("vol_ratio", last.get("ra_vol_ratio", 1.0)))
        if vol_ratio < self._entry_filters.get("min_volume_ratio", 0.4):
            return False

        # ATR spike protection
        atr = self._get_atr(last, strategy_name)
        close = float(last.get("close", 1))
        if atr > 0 and (atr / close) > self._entry_filters.get("atr_spike_max", 0.04):
            return False

        # Wave trend filter: block entries against the wave structure
        if self._entry_filters.get("wave_filter_enabled", True):
            wave_trend = self._detect_wave_trend(df)
            if side == "short" and wave_trend == "bullish":
                if self._should_log_filter(f"wave_short_{pair}", _candle_time):
                    logger.info(f"[FILTER] {entry_tag} SHORT blocked: wave=bullish")
                    _send_tg(f"\U0001f6ab **{entry_tag}** SHORT blocked\n"
                             f"Wave trend: bullish (correction up)")
                return False
            if side == "long" and wave_trend == "bearish":
                if self._should_log_filter(f"wave_long_{pair}", _candle_time):
                    logger.info(f"[FILTER] {entry_tag} LONG blocked: wave=bearish")
                    _send_tg(f"\U0001f6ab **{entry_tag}** LONG blocked\n"
                             f"Wave trend: bearish (correction down)")
                return False

        # BTC sentiment gate (RSI)
        btc_rsi = float(last.get("btc_rsi_1h", 50))
        if side == "long" and btc_rsi < self._entry_filters.get("btc_rsi_long_min", 35):
            return False
        if side == "short" and btc_rsi > self._entry_filters.get("btc_rsi_short_max", 65):
            return False

        # BTC 3-bar return canary
        btc_ret = self._get_btc_return_3bar()
        if btc_ret is not None:
            btc_threshold = self._entry_filters.get("btc_ret3_threshold", 0.005)
            if side == "long" and btc_ret < -btc_threshold:
                if self._should_log_filter(f"btc_ret3_long_{pair}", _candle_time):
                    msg = f"[FILTER] {entry_tag} LONG blocked: BTC ret3={btc_ret*100:.2f}%"
                    logger.info(msg)
                    _send_tg(f"\U0001f6ab **{entry_tag}** LONG blocked\nBTC 3bar: {btc_ret*100:+.2f}%")
                return False
            if side == "short" and btc_ret > btc_threshold:
                if self._should_log_filter(f"btc_ret3_short_{pair}", _candle_time):
                    msg = f"[FILTER] {entry_tag} SHORT blocked: BTC ret3={btc_ret*100:.2f}%"
                    logger.info(msg)
                    _send_tg(f"\U0001f6ab **{entry_tag}** SHORT blocked\nBTC 3bar: {btc_ret*100:+.2f}%")
                return False

        # Funding rate extreme filter
        funding = float(last.get("funding_rate", 0.0))
        if side == "long" and funding > self._entry_filters.get("funding_long_max", 0.0005):
            if self._should_log_filter(f"funding_long_{pair}", _candle_time):
                msg = f"[FILTER] {entry_tag} LONG blocked: funding={funding*100:.4f}%"
                logger.info(msg)
                _send_tg(f"\U0001f6ab **{entry_tag}** LONG blocked\nFunding: {funding*100:+.4f}%")
            return False
        if side == "short" and funding < self._entry_filters.get("funding_short_min", -0.0005):
            if self._should_log_filter(f"funding_short_{pair}", _candle_time):
                msg = f"[FILTER] {entry_tag} SHORT blocked: funding={funding*100:.4f}%"
                logger.info(msg)
                _send_tg(f"\U0001f6ab **{entry_tag}** SHORT blocked\nFunding: {funding*100:+.4f}%")
            return False

        # Taker volume filter (volume_spike_rev only)
        if strategy_name == "volume_spike_rev":
            taker_buy_pct = self._get_taker_buy_pct(pair)
            if taker_buy_pct is not None:
                if side == "short" and taker_buy_pct > 55:
                    if self._should_log_filter(f"taker_short_{pair}", _candle_time):
                        msg = f"[FILTER] vs_short blocked: taker_buy={taker_buy_pct:.1f}%"
                        logger.info(msg)
                        _send_tg(f"\U0001f6ab **vs_short** blocked\nTaker Buy: {taker_buy_pct:.0f}% (buyers dominant)")
                    return False
                if side == "long" and taker_buy_pct < 45:
                    if self._should_log_filter(f"taker_long_{pair}", _candle_time):
                        msg = f"[FILTER] vs_long blocked: taker_buy={taker_buy_pct:.1f}%"
                        logger.info(msg)
                        _send_tg(f"\U0001f6ab **vs_long** blocked\nTaker Buy: {taker_buy_pct:.0f}% (sellers dominant)")
                    return False

        return True

    def confirm_exit(
        self, pair: str, trade: Any, rate: float, exit_reason: str
    ) -> bool:
        """Post-trade recording. Called from confirm_trade_exit."""
        enter_tag = getattr(trade, "enter_tag", "") or ""
        strategy_name = self._parse_strategy_name(enter_tag)
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

    def _parse_strategy_name(self, tag: str) -> str:
        """Extract strategy name from enter_tag by matching registered strategies.

        Tags like 'regime_adaptive_trend_long' must match 'regime_adaptive',
        not 'regime_adaptive_trend' (which rsplit would produce).
        """
        if not tag:
            return ""
        for name in self._strategies:
            if tag == name or tag.startswith(name + "_"):
                return name
        return ""

    @staticmethod
    def _get_atr(last_row: pd.Series, strategy_name: str) -> float:
        """Get ATR value from dataframe row, checking strategy-prefixed columns."""
        prefix_map = {
            "regime_adaptive": "ra_",
            "trend_composite": "tc_",
            "meanrev_confluence": "mr_",
            "compression_breakout": "cb_",
            "cb_adx_breakout": "cba_",
            "volume_spike_rev": "vs_",
            "volatility_compression": "vc_",
            "funding_contrarian": "fc_",
            "micro_pullback": "mp_",
            "ml_scalping_sol_3m": "mls3_",
            "ml_scalping_enhanced_3m": "mle3_",
            "fast_scalper_3m": "fs3_",
            # 5m scalping strategies
            "squeeze_breakout_5m": "sq5_",
            "volume_climax_5m": "vc5_",
            "ema_pullback_5m": "ep5_",
            "asian_range_breakout_5m": "arb5_",
            "volume_impulse_15m": "vi_",
        }
        prefix = prefix_map.get(strategy_name, "")
        if prefix:
            for col_suffix in ["atr_14_raw", "atr_14", "atr"]:
                val = last_row.get(f"{prefix}{col_suffix}", 0)
                if val and float(val) > 0:
                    return float(val)
        for p in ["ra_", "cba_", "vs_", ""]:
            val = last_row.get(f"{p}atr", 0)
            if val and float(val) > 0:
                return float(val)
        return 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # OKX REAL-TIME DATA (for live entry filters)
    # ═══════════════════════════════════════════════════════════════════════════

    _OKX_INST_MAP = {
        "ETH/USDT:USDT": "ETH-USDT-SWAP",
        "BTC/USDT:USDT": "BTC-USDT-SWAP",
        "SOL/USDT:USDT": "SOL-USDT-SWAP",
    }
    _taker_cache: dict[str, tuple[float, float]] = {}  # pair -> (timestamp, buy_pct)
    _btc_ret_cache: tuple[float, float | None] = (0, None)  # (timestamp, ret_3bar)

    def _get_taker_buy_pct(self, pair: str) -> float | None:
        """Fetch latest taker buy/sell ratio from OKX. Cached 60s."""
        now = time.time()
        cached = self._taker_cache.get(pair)
        if cached and now - cached[0] < 60:
            return cached[1]

        inst = self._OKX_INST_MAP.get(pair)
        if not inst:
            return None
        try:
            url = (f"https://www.okx.com/api/v5/rubik/stat/taker-volume-contract"
                   f"?instId={inst}&period=15m&limit=1")
            req = urllib.request.Request(url, headers={"User-Agent": "freqtrade"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = json.loads(resp.read())
            rows = raw.get("data", [])
            if rows:
                buy = float(rows[0][1])
                sell = float(rows[0][2])
                total = buy + sell
                pct = buy / total * 100 if total > 0 else 50.0
                self._taker_cache[pair] = (now, pct)
                return pct
        except Exception as e:
            logger.debug(f"Taker fetch failed: {e}")
        return None

    def _get_btc_return_3bar(self) -> float | None:
        """Get BTC 3-bar (45min) return from analyzed dataframe."""
        now = time.time()
        if now - self._btc_ret_cache[0] < 60 and self._btc_ret_cache[1] is not None:
            return self._btc_ret_cache[1]

        try:
            btc_df, _ = self._dp.get_analyzed_dataframe("BTC/USDT:USDT", self.config.get_timeframe())
            if btc_df.empty or len(btc_df) < 4:
                return None
            close_now = float(btc_df["close"].iloc[-1])
            close_3ago = float(btc_df["close"].iloc[-4])
            ret = (close_now - close_3ago) / close_3ago
            self.__class__._btc_ret_cache = (now, ret)
            return ret
        except Exception:
            return None

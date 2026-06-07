"""NR Breakout 5m: Narrowest Range breakout adapted for crypto scalping.

NR4/NR7 concept: when the current bar's range is the smallest in N bars,
volatility expansion is imminent. Enter on breakout direction with volume
confirmation and ADX < 25 filter (no established trend = fresh breakout).

Optimized for 5m: tighter parameters, shorter hold time (15-60 min).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pandas_ta as ta

from engine.config import StrategyConfig
from engine.events import Direction, ExitRequest, Signal, Urgency
from strategies import register_strategy
from strategies.base import BaseStrategy


@register_strategy
class NRBreakout5mStrategy(BaseStrategy):
    name = "nr_breakout_5m"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"].astype(float)

        # Bar range
        dataframe["nr5_range"] = h - lo

        # ATR for reference
        atr_len = self.config.entry.get("atr_length", 14)
        dataframe["nr5_atr"] = ta.atr(h, lo, c, length=atr_len)

        # NR lookback — is current bar the narrowest in N bars?
        nr_lookback = self.config.entry.get("nr_lookback", 7)
        dataframe["nr5_min_range"] = dataframe["nr5_range"].rolling(
            window=nr_lookback, min_periods=nr_lookback
        ).min()
        dataframe["nr5_is_nr"] = dataframe["nr5_range"] <= dataframe["nr5_min_range"]

        # Range relative to ATR
        dataframe["nr5_range_atr_ratio"] = dataframe["nr5_range"] / (dataframe["nr5_atr"] + 1e-10)

        # ADX for trend filter
        adx_len = self.config.entry.get("adx_length", 14)
        adx_result = ta.adx(h, lo, c, length=adx_len)
        if adx_result is not None:
            dataframe["nr5_adx"] = adx_result.iloc[:, 0]
        else:
            dataframe["nr5_adx"] = 25.0

        # Volume
        vol_ema_len = self.config.entry.get("vol_ema_len", 20)
        dataframe["nr5_vol_ema"] = ta.ema(v, length=vol_ema_len)
        dataframe["nr5_vol_ratio"] = v / (dataframe["nr5_vol_ema"] + 1e-10)

        # RSI
        dataframe["nr5_rsi"] = ta.rsi(c, length=9)

        # Previous bar's high/low (breakout reference)
        dataframe["nr5_prev_high"] = h.shift(1)
        dataframe["nr5_prev_low"] = lo.shift(1)

        # Was previous bar an NR bar? (we enter on the NEXT bar after NR)
        dataframe["nr5_prev_is_nr"] = dataframe["nr5_is_nr"].shift(1).fillna(False)
        dataframe["nr5_prev_range_ratio"] = dataframe["nr5_range_atr_ratio"].shift(1).fillna(1.0)

        # Dedup: don't signal if we signaled within last N bars
        dedup_bars = self.config.entry.get("dedup_bars", 5)
        dataframe["nr5_signal_long"] = False
        dataframe["nr5_signal_short"] = False

        return dataframe

    def on_tick(self, dataframe: pd.DataFrame, pair: str, current_time: datetime) -> None:
        signals = self.detect_entries(dataframe, pair)
        for sig in signals:
            self.emit_signal(
                pair=sig.pair, direction=sig.direction,
                strength=sig.strength, tag=sig.tag, metadata=sig.metadata,
            )

    def detect_entries(self, dataframe: pd.DataFrame, pair: str) -> list[Signal]:
        signals: list[Signal] = []
        if len(dataframe) < 10:
            return signals

        last = dataframe.iloc[-1]

        # Previous bar must be NR bar
        if not bool(last.get("nr5_prev_is_nr", False)):
            return signals

        # Range must be tight relative to ATR
        range_atr_max = self.config.entry.get("range_atr_mult", 0.8)
        prev_ratio = float(last.get("nr5_prev_range_ratio", 1.0))
        if prev_ratio > range_atr_max:
            return signals

        # ADX must be low (no established trend)
        adx_max = self.config.entry.get("adx_max", 25)
        adx = float(last.get("nr5_adx", 30))
        if adx > adx_max:
            return signals

        # Volume must be present
        vol_min = self.config.entry.get("vol_min", 0.8)
        vol_ratio = float(last.get("nr5_vol_ratio", 0))
        if vol_ratio < vol_min:
            return signals

        close = float(last["close"])
        prev_high = float(last.get("nr5_prev_high", close))
        prev_low = float(last.get("nr5_prev_low", close))
        rsi = float(last.get("nr5_rsi", 50))

        rsi_max = self.config.entry.get("rsi_max", 72)
        rsi_min = self.config.entry.get("rsi_min", 28)

        # Long: close breaks above previous NR bar's high
        if close > prev_high and rsi < rsi_max:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=1.0,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"adx": adx, "vol": vol_ratio, "rsi": rsi, "nr_ratio": prev_ratio},
            ))

        # Short: close breaks below previous NR bar's low
        elif close < prev_low and rsi > rsi_min:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=1.0,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"adx": adx, "vol": vol_ratio, "rsi": rsi, "nr_ratio": prev_ratio},
            ))

        return signals

    def detect_exits(
        self, dataframe: pd.DataFrame, pair: str, trade_info: dict | None
    ) -> ExitRequest | None:
        if trade_info is None:
            return None

        last = dataframe.iloc[-1]
        current_profit = trade_info.get("current_profit", 0)
        current_time = trade_info.get("current_time", datetime.utcnow())
        entry_time = trade_info.get("entry_time", current_time)
        open_rate = trade_info.get("entry_rate", 0)

        atr = float(last.get("nr5_atr", 0))
        if atr <= 0 or open_rate <= 0:
            return None

        # TP hit
        tp_pct = self.tp_atr_mult * atr / open_rate
        if current_profit >= tp_pct:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="TP_HIT", urgency=Urgency.IMMEDIATE,
                timestamp=datetime.utcnow(),
            )

        # Time cuts (5m scalp)
        minutes = (current_time - entry_time).total_seconds() / 60
        exit_cfg = self.config.exit

        if minutes >= 30 and current_profit < exit_cfg.get("time_cut_30m", -0.005):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_30m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if minutes >= 60 and current_profit < exit_cfg.get("time_cut_60m", -0.003):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_60m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if minutes >= 90:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_90m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

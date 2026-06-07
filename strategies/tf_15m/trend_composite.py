"""Trend-composite strategy: EMA momentum trend-following.

Entry: EMA20 crosses EMA50 (within 3 bars) + ADX>25 + DI confirms + MACD + volume
Exit: ATR-TP, EMA reverse cross, cascading time-cuts
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
class TrendCompositeStrategy(BaseStrategy):
    name = "trend_composite"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"]

        ema_fast_p = self.config.entry.get("ema_fast", 20)
        ema_slow_p = self.config.entry.get("ema_slow", 50)

        dataframe["tc_ema_fast"] = ta.ema(c, length=ema_fast_p)
        dataframe["tc_ema_slow"] = ta.ema(c, length=ema_slow_p)
        dataframe["tc_atr"] = ta.atr(h, lo, c, length=14)

        adx_df = ta.adx(h, lo, c, length=14)
        if adx_df is not None:
            dataframe["tc_adx"] = adx_df.iloc[:, 0]
            dataframe["tc_plus_di"] = adx_df.iloc[:, 1]
            dataframe["tc_minus_di"] = adx_df.iloc[:, 2]
        else:
            dataframe["tc_adx"] = 0.0
            dataframe["tc_plus_di"] = 0.0
            dataframe["tc_minus_di"] = 0.0

        dataframe["tc_rsi"] = ta.rsi(c, length=14)

        dataframe["tc_vol_ema"] = ta.ema(v.astype(float), length=20)
        dataframe["tc_vol_ratio"] = v.astype(float) / (dataframe["tc_vol_ema"] + 1e-10)

        macd = ta.macd(c, fast=12, slow=26, signal=9)
        if macd is not None:
            dataframe["tc_macd_hist"] = macd.iloc[:, 1]
        else:
            dataframe["tc_macd_hist"] = 0.0

        dataframe["tc_cross_up"] = (
            (dataframe["tc_ema_fast"] > dataframe["tc_ema_slow"])
            & (dataframe["tc_ema_fast"].shift(1) <= dataframe["tc_ema_slow"].shift(1))
        )
        dataframe["tc_cross_down"] = (
            (dataframe["tc_ema_fast"] < dataframe["tc_ema_slow"])
            & (dataframe["tc_ema_fast"].shift(1) >= dataframe["tc_ema_slow"].shift(1))
        )

        lookback = self.config.entry.get("cross_lookback", 3)
        dataframe["tc_recent_cross_up"] = (
            dataframe["tc_cross_up"].rolling(lookback).max().fillna(0).astype(bool)
        )
        dataframe["tc_recent_cross_down"] = (
            dataframe["tc_cross_down"].rolling(lookback).max().fillna(0).astype(bool)
        )

        return dataframe

    def on_tick(self, dataframe: pd.DataFrame, pair: str, current_time: datetime) -> None:
        signals = self.detect_entries(dataframe, pair)
        for sig in signals:
            self.emit_signal(
                pair=sig.pair,
                direction=sig.direction,
                strength=sig.strength,
                tag=sig.tag,
            )

    def detect_entries(self, dataframe: pd.DataFrame, pair: str) -> list[Signal]:
        signals: list[Signal] = []
        last = dataframe.iloc[-1]

        adx_min = self.config.entry.get("adx_min", 25)
        vol_mult = self.config.entry.get("vol_mult", 1.2)
        rsi_low = self.config.entry.get("rsi_low", 48)
        rsi_high = self.config.entry.get("rsi_high", 65)

        adx = float(last.get("tc_adx", 0))
        vol_ratio = float(last.get("tc_vol_ratio", 0))

        if adx < adx_min or vol_ratio < vol_mult:
            return signals

        rsi = float(last.get("tc_rsi", 50))
        ema_fast = float(last.get("tc_ema_fast", 0))
        ema_slow = float(last.get("tc_ema_slow", 0))
        close = float(last.get("close", 0))
        macd_hist = float(last.get("tc_macd_hist", 0))
        plus_di = float(last.get("tc_plus_di", 0))
        minus_di = float(last.get("tc_minus_di", 0))

        # Long
        if (
            bool(last.get("tc_recent_cross_up", False))
            and ema_fast > ema_slow
            and plus_di > minus_di
            and rsi_low <= rsi <= rsi_high
            and macd_hist > 0
            and close > ema_fast
        ):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=1.0,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
            ))

        # Short (mirror RSI range)
        rsi_short_high = 100 - rsi_low
        rsi_short_low = 100 - rsi_high
        if (
            bool(last.get("tc_recent_cross_down", False))
            and ema_fast < ema_slow
            and minus_di > plus_di
            and rsi_short_low <= rsi <= rsi_short_high
            and macd_hist < 0
            and close < ema_fast
        ):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=1.0,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
            ))

        return signals

    def detect_exits(
        self, dataframe: pd.DataFrame, pair: str, trade_info: dict | None
    ) -> ExitRequest | None:
        if trade_info is None:
            return None

        last = dataframe.iloc[-1]
        current_profit = trade_info.get("current_profit", 0)
        is_short = trade_info.get("is_short", False)
        current_time = trade_info.get("current_time", datetime.utcnow())
        entry_time = trade_info.get("entry_time", current_time)
        open_rate = trade_info.get("entry_rate", 0)

        # ATR-based TP
        atr = float(last.get("tc_atr", 0))
        if atr > 0 and open_rate > 0:
            tp_pct = self.tp_atr_mult * atr / open_rate
            if current_profit >= tp_pct:
                return ExitRequest(
                    strategy_name=self.name, pair=pair,
                    reason="TP_HIT", urgency=Urgency.IMMEDIATE,
                    timestamp=datetime.utcnow(),
                )

        # EMA reversal exit
        is_long = not is_short
        if is_long and bool(last.get("tc_cross_down", False)):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="ema_reversal_exit", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if not is_long and bool(last.get("tc_cross_up", False)):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="ema_reversal_exit", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        # Cascading time-cuts
        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.exit
        time_cut_8h = exit_cfg.get("time_cut_8h", -0.008)
        time_cut_24h = exit_cfg.get("time_cut_24h", 0.0)
        time_cut_48h = exit_cfg.get("time_cut_48h", 0.005)

        if hours >= 8 and current_profit < time_cut_8h:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_8h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 24 and current_profit < time_cut_24h:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_24h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 48 and current_profit < time_cut_48h:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_48h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

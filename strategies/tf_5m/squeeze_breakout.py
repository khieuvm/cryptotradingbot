"""BB/KC Squeeze Breakout Scalp (5m): enter on volatility expansion after compression.

When Bollinger Bands contract inside Keltner Channels for N bars (squeeze),
the market is compressing energy. On breakout (BB expands past KC), enter in
the breakout direction with tight ATR-based SL/TP.

Optimized for 5m scalping: hold 15-60 minutes, tight risk.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta

from engine.config import StrategyConfig
from engine.events import Direction, ExitRequest, Signal, Urgency
from strategies import register_strategy
from strategies.base import BaseStrategy


@register_strategy
class SqueezeBreakout5mStrategy(BaseStrategy):
    name = "squeeze_breakout_5m"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"].astype(float)

        # ATR
        atr_len = self.config.entry.get("atr_length", 14)
        dataframe["sq5_atr"] = ta.atr(h, lo, c, length=atr_len)

        # Bollinger Bands
        bb_len = self.config.entry.get("bb_length", 20)
        bb_std = self.config.entry.get("bb_std", 2.0)
        bb = ta.bbands(c, length=bb_len, std=bb_std)
        if bb is not None:
            dataframe["sq5_bb_upper"] = bb.iloc[:, 0]
            dataframe["sq5_bb_mid"] = bb.iloc[:, 1]
            dataframe["sq5_bb_lower"] = bb.iloc[:, 2]
        else:
            dataframe["sq5_bb_upper"] = c
            dataframe["sq5_bb_mid"] = c
            dataframe["sq5_bb_lower"] = c

        # Keltner Channels
        kc_len = self.config.entry.get("kc_length", 20)
        kc_mult = self.config.entry.get("kc_mult", 1.5)
        kc_mid = ta.ema(c, length=kc_len)
        kc_atr = ta.atr(h, lo, c, length=kc_len)
        dataframe["sq5_kc_upper"] = kc_mid + kc_mult * kc_atr
        dataframe["sq5_kc_lower"] = kc_mid - kc_mult * kc_atr

        # Squeeze detection: BB inside KC
        dataframe["sq5_squeeze_on"] = (
            (dataframe["sq5_bb_upper"] < dataframe["sq5_kc_upper"])
            & (dataframe["sq5_bb_lower"] > dataframe["sq5_kc_lower"])
        )

        # Count consecutive squeeze bars
        squeeze_groups = (~dataframe["sq5_squeeze_on"]).cumsum()
        dataframe["sq5_squeeze_bars"] = dataframe.groupby(squeeze_groups).cumcount()
        dataframe.loc[~dataframe["sq5_squeeze_on"], "sq5_squeeze_bars"] = 0

        # Squeeze zone high/low = BB bands at the moment of fire (breakout level)
        dataframe["sq5_zone_high"] = dataframe["sq5_bb_upper"]
        dataframe["sq5_zone_low"] = dataframe["sq5_bb_lower"]

        # Volume
        vol_ema_len = self.config.entry.get("vol_ema_len", 20)
        dataframe["sq5_vol_ema"] = ta.ema(v, length=vol_ema_len)
        dataframe["sq5_vol_ratio"] = v / (dataframe["sq5_vol_ema"] + 1e-10)

        # RSI for direction filter
        rsi_len = self.config.entry.get("rsi_length", 9)
        dataframe["sq5_rsi"] = ta.rsi(c, length=rsi_len)

        # Squeeze was on previous bar but off now (squeeze fire)
        dataframe["sq5_squeeze_fire"] = (
            dataframe["sq5_squeeze_on"].shift(1).fillna(False)
            & ~dataframe["sq5_squeeze_on"]
        )

        # Previous squeeze duration (how many bars was the squeeze on before fire)
        dataframe["sq5_prev_squeeze_bars"] = dataframe["sq5_squeeze_bars"].shift(1).fillna(0)

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
        if len(dataframe) < 25:
            return signals

        last = dataframe.iloc[-1]

        # Must be a squeeze fire bar
        if not bool(last.get("sq5_squeeze_fire", False)):
            return signals

        # Minimum squeeze duration
        min_squeeze = self.config.entry.get("min_squeeze_bars", 4)
        prev_squeeze = int(last.get("sq5_prev_squeeze_bars", 0))
        if prev_squeeze < min_squeeze:
            return signals

        # Volume confirmation
        vol_min = self.config.entry.get("vol_min", 1.2)
        vol_ratio = float(last.get("sq5_vol_ratio", 0))
        if vol_ratio < vol_min:
            return signals

        close = float(last["close"])
        rsi = float(last.get("sq5_rsi", 50))
        zone_high = float(last.get("sq5_zone_high", close))
        zone_low = float(last.get("sq5_zone_low", close))

        rsi_max = self.config.entry.get("rsi_max", 72)
        rsi_min = self.config.entry.get("rsi_min", 28)

        # Long: close breaks above zone high, RSI not overbought
        if close > zone_high and rsi < rsi_max and rsi > 45:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=1.0,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"squeeze_bars": prev_squeeze, "vol": vol_ratio, "rsi": rsi},
            ))

        # Short: close breaks below zone low, RSI not oversold
        elif close < zone_low and rsi > rsi_min and rsi < 55:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=1.0,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"squeeze_bars": prev_squeeze, "vol": vol_ratio, "rsi": rsi},
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

        atr = float(last.get("sq5_atr", 0))
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

        # Time cuts (5m scalp — shorter durations)
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

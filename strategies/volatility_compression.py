"""Volatility Compression strategy: BB squeeze inside KC → breakout.

When Bollinger Bands contract inside Keltner Channel for 3+ bars,
volatility is compressed. The first bar where BB expands outside KC
signals imminent expansion. Entry is bidirectional based on price action.

Key insight from research: TTM squeeze predicts TIMING of volatility
expansion with high accuracy, but NOT direction. Therefore we use
price breakout direction (high/low triggers), not momentum histogram.
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
class VolatilityCompressionStrategy(BaseStrategy):
    name = "volatility_compression"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"]

        bb_len = self.config.entry.get("bb_length", 20)
        bb_std = self.config.entry.get("bb_std", 2.0)
        kc_len = self.config.entry.get("kc_length", 20)
        kc_mult = self.config.entry.get("kc_mult", 1.5)

        # Bollinger Bands
        bb = ta.bbands(c, length=bb_len, std=bb_std)
        if bb is not None:
            dataframe["vc_bb_upper"] = bb.iloc[:, 2]
            dataframe["vc_bb_lower"] = bb.iloc[:, 0]
        else:
            dataframe["vc_bb_upper"] = c + c * 0.02
            dataframe["vc_bb_lower"] = c - c * 0.02

        # Keltner Channel
        kc_mid = ta.ema(c, length=kc_len)
        kc_atr = ta.atr(h, lo, c, length=kc_len)
        dataframe["vc_kc_upper"] = kc_mid + kc_mult * kc_atr
        dataframe["vc_kc_lower"] = kc_mid - kc_mult * kc_atr

        # ATR for exits
        dataframe["vc_atr"] = ta.atr(h, lo, c, length=14)

        # Squeeze detection: BB inside KC
        dataframe["vc_squeeze"] = (
            (dataframe["vc_bb_lower"] > dataframe["vc_kc_lower"])
            & (dataframe["vc_bb_upper"] < dataframe["vc_kc_upper"])
        )

        # Squeeze duration (consecutive bars in squeeze)
        squeeze_groups = (~dataframe["vc_squeeze"]).cumsum()
        dataframe["vc_squeeze_bars"] = (
            dataframe["vc_squeeze"]
            .groupby(squeeze_groups)
            .cumcount()
        )
        dataframe.loc[~dataframe["vc_squeeze"], "vc_squeeze_bars"] = 0

        # Squeeze fire: transition from squeeze ON → OFF
        min_squeeze = self.config.entry.get("min_squeeze_bars", 3)
        dataframe["vc_squeeze_fire"] = (
            (~dataframe["vc_squeeze"])
            & (dataframe["vc_squeeze"].shift(1))
            & (dataframe["vc_squeeze_bars"].shift(1) >= min_squeeze)
        )

        # Volume
        dataframe["vc_vol_ema"] = ta.ema(v.astype(float), length=20)
        dataframe["vc_vol_ratio"] = v.astype(float) / (dataframe["vc_vol_ema"] + 1e-10)

        # Direction from price action (NOT from momentum histogram)
        # Use the compression zone high/low for breakout direction
        squeeze_high = h.rolling(min_squeeze + 1).max()
        squeeze_low = lo.rolling(min_squeeze + 1).min()
        dataframe["vc_zone_high"] = squeeze_high
        dataframe["vc_zone_low"] = squeeze_low

        dataframe["vc_break_up"] = (
            dataframe["vc_squeeze_fire"]
            & (c > dataframe["vc_zone_high"].shift(1))
        )
        dataframe["vc_break_down"] = (
            dataframe["vc_squeeze_fire"]
            & (c < dataframe["vc_zone_low"].shift(1))
        )

        # ADX for context
        adx_df = ta.adx(h, lo, c, length=14)
        if adx_df is not None:
            dataframe["vc_adx"] = adx_df.iloc[:, 0]
        else:
            dataframe["vc_adx"] = 20.0

        return dataframe

    def on_tick(self, dataframe: pd.DataFrame, pair: str, current_time: datetime) -> None:
        signals = self.detect_entries(dataframe, pair)
        for sig in signals:
            self.emit_signal(
                pair=sig.pair,
                direction=sig.direction,
                strength=sig.strength,
                tag=sig.tag,
                metadata=sig.metadata,
            )

    def detect_entries(self, dataframe: pd.DataFrame, pair: str) -> list[Signal]:
        signals: list[Signal] = []
        if len(dataframe) < 10:
            return signals

        last = dataframe.iloc[-1]
        vol_min = self.config.entry.get("vol_min", 1.0)
        vol_ratio = float(last.get("vc_vol_ratio", 0))

        if vol_ratio < vol_min:
            return signals

        adx = float(last.get("vc_adx", 20))
        adx_max = self.config.entry.get("adx_max", 25)
        if adx > adx_max:
            return signals

        squeeze_bars = int(dataframe.iloc[-2].get("vc_squeeze_bars", 0)) if len(dataframe) > 1 else 0
        strength = min(1.0, 0.7 + squeeze_bars * 0.05)

        # Long breakout from squeeze
        if bool(last.get("vc_break_up", False)):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=strength,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"squeeze_duration": squeeze_bars, "adx": adx},
            ))

        # Short breakout from squeeze
        if bool(last.get("vc_break_down", False)):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=strength,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"squeeze_duration": squeeze_bars, "adx": adx},
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

        atr = float(last.get("vc_atr", 0))
        if atr <= 0 or open_rate <= 0:
            return None

        # ATR-based TP (squeeze breakouts often move 2-3x ATR)
        tp_pct = self.tp_atr_mult * atr / open_rate
        if current_profit >= tp_pct:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="TP_HIT", urgency=Urgency.IMMEDIATE,
                timestamp=datetime.utcnow(),
            )

        # If price re-enters squeeze zone → failed breakout
        is_short = trade_info.get("is_short", False)
        close = float(last.get("close", 0))
        if bool(last.get("vc_squeeze", False)):
            if current_profit < 0:
                return ExitRequest(
                    strategy_name=self.name, pair=pair,
                    reason="squeeze_reentry", urgency=Urgency.NEXT_CANDLE,
                    timestamp=datetime.utcnow(),
                )

        # Time cuts (volatility expansion should happen fast)
        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.exit

        if hours >= 3 and current_profit < exit_cfg.get("time_cut_3h", -0.008):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_3h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 8 and current_profit < exit_cfg.get("time_cut_8h", -0.003):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_8h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 16:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_16h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

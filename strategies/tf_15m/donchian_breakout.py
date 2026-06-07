"""Donchian Channel Breakout strategy: 12-bar channel break trend-following.

Adapted from turtle trading rules for intraday crypto futures 15m timeframe.

CONCEPT: Donchian channel (N-bar highest high / lowest low) breakout is the
original systematic trend-following signal. When price breaks above the
12-bar high, a new uptrend is initiating; below the 12-bar low, downtrend.

Requires existing trend confirmation (ADX > threshold) to avoid ranging chop.
Uses Donchian midline as trailing exit target.

Entry: Close > 12-bar high (LONG), Close < 12-bar low (SHORT)
Exit: ATR-based TP/SL, Donchian mid trailing, cascading time cuts at 8h/24h/48h
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
class DonchianBreakoutStrategy(BaseStrategy):
    name = "donchian_breakout"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"]

        # Donchian Channel parameters
        dc_length = self.config.entry.get("dc_length", 12)

        # Donchian Channel: highest high / lowest low of N bars (excluding current)
        dataframe["dc_upper"] = h.shift(1).rolling(dc_length).max()
        dataframe["dc_lower"] = lo.shift(1).rolling(dc_length).min()
        dataframe["dc_mid"] = (dataframe["dc_upper"] + dataframe["dc_lower"]) / 2

        # ATR for exits
        dataframe["dc_atr"] = ta.atr(h, lo, c, length=14)

        # ADX for trend confirmation (only trade breakouts in trending regime)
        adx_df = ta.adx(h, lo, c, length=14)
        if adx_df is not None:
            dataframe["dc_adx"] = adx_df.iloc[:, 0]
            dataframe["dc_plus_di"] = adx_df.iloc[:, 1]
            dataframe["dc_minus_di"] = adx_df.iloc[:, 2]
        else:
            dataframe["dc_adx"] = 20.0
            dataframe["dc_plus_di"] = 0.0
            dataframe["dc_minus_di"] = 0.0

        # Volume filter
        dataframe["dc_vol_ema"] = ta.ema(v.astype(float), length=20)
        dataframe["dc_vol_ratio"] = v.astype(float) / (dataframe["dc_vol_ema"] + 1e-10)

        # Breakout detection
        adx_min = self.config.entry.get("adx_min", 20)
        vol_min = self.config.entry.get("vol_min", 0.8)

        dataframe["dc_break_up"] = (
            (c > dataframe["dc_upper"])
            & (dataframe["dc_adx"] > adx_min)
            & (dataframe["dc_vol_ratio"] > vol_min)
        )
        dataframe["dc_break_down"] = (
            (c < dataframe["dc_lower"])
            & (dataframe["dc_adx"] > adx_min)
            & (dataframe["dc_vol_ratio"] > vol_min)
        )

        # Dedup: no signal within N bars of previous signal
        dedup = self.config.entry.get("dedup_bars", 6)
        dataframe["dc_any_signal"] = dataframe["dc_break_up"] | dataframe["dc_break_down"]
        dataframe["dc_last_signal"] = (
            dataframe["dc_any_signal"].rolling(dedup).max().shift(1).fillna(0).astype(bool)
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
                metadata=sig.metadata,
            )

    def detect_entries(self, dataframe: pd.DataFrame, pair: str) -> list[Signal]:
        signals: list[Signal] = []
        if len(dataframe) < 14:
            return signals

        last = dataframe.iloc[-1]

        # Dedup filter
        if bool(last.get("dc_last_signal", False)):
            return signals

        adx = float(last.get("dc_adx", 0))
        vol_ratio = float(last.get("dc_vol_ratio", 0))
        plus_di = float(last.get("dc_plus_di", 0))
        minus_di = float(last.get("dc_minus_di", 0))

        # Strength scales with ADX (stronger trend = higher conviction)
        strength = min(1.0, 0.6 + (adx - 20) * 0.02) if adx > 20 else 0.6

        # Long breakout: close > Donchian upper + DI confirms
        if bool(last.get("dc_break_up", False)) and plus_di > minus_di:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=strength,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"adx": adx, "vol_ratio": vol_ratio, "dc_length": 12},
            ))

        # Short breakout: close < Donchian lower + DI confirms
        if bool(last.get("dc_break_down", False)) and minus_di > plus_di:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=strength,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"adx": adx, "vol_ratio": vol_ratio, "dc_length": 12},
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
        is_short = trade_info.get("is_short", False)

        atr = float(last.get("dc_atr", 0))
        if atr <= 0 or open_rate <= 0:
            return None

        # ATR-based TP
        tp_pct = self.tp_atr_mult * atr / open_rate
        if current_profit >= tp_pct:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="TP_HIT", urgency=Urgency.IMMEDIATE,
                timestamp=datetime.utcnow(),
            )

        # ATR-based SL
        sl_pct = self.sl_atr_mult * atr / open_rate
        if current_profit <= -sl_pct:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="SL_HIT", urgency=Urgency.IMMEDIATE,
                timestamp=datetime.utcnow(),
            )

        # Donchian mid trailing exit: if price reverts back through midline
        # after being profitable, trend has likely reversed
        close = float(last.get("close", 0))
        dc_mid = float(last.get("dc_mid", 0))
        if dc_mid > 0 and current_profit > 0:
            if not is_short and close < dc_mid:
                return ExitRequest(
                    strategy_name=self.name, pair=pair,
                    reason="dc_mid_trail", urgency=Urgency.NEXT_CANDLE,
                    timestamp=datetime.utcnow(),
                )
            if is_short and close > dc_mid:
                return ExitRequest(
                    strategy_name=self.name, pair=pair,
                    reason="dc_mid_trail", urgency=Urgency.NEXT_CANDLE,
                    timestamp=datetime.utcnow(),
                )

        # Cascading time cuts (trend-following: allow more time)
        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.exit

        if hours >= 8 and current_profit < exit_cfg.get("time_cut_8h", -0.008):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_8h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 24 and current_profit < exit_cfg.get("time_cut_24h", 0.0):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_24h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 48 and current_profit < exit_cfg.get("time_cut_48h", 0.005):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_48h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

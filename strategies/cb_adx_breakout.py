"""Compression Breakout + ADX < 20 filter: low-ADX compression breakout.

Variant of compression_breakout that adds an ADX filter requirement.
When ADX < 20 in at least 1 of the last 3 bars, there is no established
trend, so a breakout from compression is more likely to be the START of
a new trend rather than noise within an existing one.

Entry: Same 3-bar compression breakout as base, but only when ADX was
       recently below threshold (no established trend)
Exit: ATR-based TP, time cuts (same as compression_breakout)
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
class CbAdxBreakoutStrategy(BaseStrategy):
    name = "cb_adx_breakout"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"]

        # ATR for compression measurement
        dataframe["cba_atr"] = ta.atr(h, lo, c, length=14)
        dataframe["cba_atr_ma"] = ta.ema(dataframe["cba_atr"], length=50)

        # Per-bar range
        dataframe["cba_range"] = h - lo

        # 3-bar compression zone
        dataframe["cba_3bar_high"] = h.rolling(3).max()
        dataframe["cba_3bar_low"] = lo.rolling(3).min()
        dataframe["cba_3bar_range"] = dataframe["cba_3bar_high"] - dataframe["cba_3bar_low"]

        # Compression ratio (zone range vs ATR)
        compression_thr = self.config.entry.get("compression_thr", 0.7)
        dataframe["cba_compressed"] = (
            dataframe["cba_3bar_range"] < compression_thr * dataframe["cba_atr"]
        )

        # Volume for confirmation
        dataframe["cba_vol_ema"] = ta.ema(v.astype(float), length=20)
        dataframe["cba_vol_ratio"] = v.astype(float) / (dataframe["cba_vol_ema"] + 1e-10)

        # RSI for overbought/oversold filter
        dataframe["cba_rsi"] = ta.rsi(c, length=14)

        # ADX — the core additional filter for this variant
        adx_df = ta.adx(h, lo, c, length=14)
        if adx_df is not None:
            dataframe["cba_adx"] = adx_df.iloc[:, 0]
        else:
            dataframe["cba_adx"] = 25.0

        # ADX filter: ADX < adx_max in at least 1 of the last 3 bars
        adx_max = self.config.entry.get("adx_max", 20)
        adx_low = dataframe["cba_adx"] < adx_max
        dataframe["cba_adx_low_recent"] = adx_low.rolling(3).max().fillna(0).astype(bool)

        # Breakout detection: close breaks above/below compression zone
        dataframe["cba_break_up"] = (
            dataframe["cba_compressed"].shift(1)
            & (c > dataframe["cba_3bar_high"].shift(1))
            & (dataframe["cba_vol_ratio"] > self.config.entry.get("vol_min", 0.8))
            & dataframe["cba_adx_low_recent"]
        )
        dataframe["cba_break_down"] = (
            dataframe["cba_compressed"].shift(1)
            & (c < dataframe["cba_3bar_low"].shift(1))
            & (dataframe["cba_vol_ratio"] > self.config.entry.get("vol_min", 0.8))
            & dataframe["cba_adx_low_recent"]
        )

        # Dedup: no signal within N bars of previous signal
        dedup = self.config.entry.get("dedup_bars", 5)
        dataframe["cba_any_signal"] = dataframe["cba_break_up"] | dataframe["cba_break_down"]
        dataframe["cba_last_signal"] = (
            dataframe["cba_any_signal"].rolling(dedup).max().shift(1).fillna(0).astype(bool)
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
        if len(dataframe) < 5:
            return signals

        last = dataframe.iloc[-1]

        # Dedup filter
        if bool(last.get("cba_last_signal", False)):
            return signals

        # RSI filter (don't buy overbought, don't sell oversold)
        rsi = float(last.get("cba_rsi", 50))
        rsi_max = self.config.entry.get("rsi_max", 72)
        rsi_min = self.config.entry.get("rsi_min", 28)

        # ADX context for strength weighting
        adx = float(last.get("cba_adx", 25))
        strength = 1.0 if adx < 20 else 0.85  # Higher confidence when ADX is truly low

        # Long breakout
        if bool(last.get("cba_break_up", False)) and rsi < rsi_max:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=strength,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"adx": adx, "compression_bars": 3, "filter": "adx_low"},
            ))

        # Short breakout
        if bool(last.get("cba_break_down", False)) and rsi > rsi_min:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=strength,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"adx": adx, "compression_bars": 3, "filter": "adx_low"},
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

        atr = float(last.get("cba_atr", 0))
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

        # Time-based exits (compression trades should resolve quickly)
        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.exit

        if hours >= 2 and current_profit < exit_cfg.get("time_cut_2h", -0.01):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_2h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 6 and current_profit < exit_cfg.get("time_cut_6h", -0.005):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_6h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 12 and current_profit < exit_cfg.get("time_cut_12h", 0.002):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_12h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

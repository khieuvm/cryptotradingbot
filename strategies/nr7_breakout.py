"""NR7 Breakout strategy: Narrowest Range of 7 bars → bidirectional breakout.

Classic price-action pattern adapted for crypto futures 15m timeframe.

CONCEPT: When the current bar has the smallest high-low range of the last 7 bars,
AND the range is compressed relative to ATR, the market is in a consolidation phase.
A breakout on the next bar has high probability of continuation.

Entry: Bidirectional — close > previous high (LONG), close < previous low (SHORT)
Exit: ATR-based TP/SL, cascading time cuts at 4h/8h/16h
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
class NR7BreakoutStrategy(BaseStrategy):
    name = "nr7_breakout"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"]

        # ATR for compression measurement and exits
        dataframe["nr7_atr"] = ta.atr(h, lo, c, length=14)

        # Per-bar range
        dataframe["nr7_range"] = h - lo

        # Rolling minimum range over 7 bars
        dataframe["nr7_min_range_7"] = dataframe["nr7_range"].rolling(7).min()

        # NR7 detection: current bar has the narrowest range of last 7 bars
        # AND range is compressed relative to ATR
        atr_compress = self.config.entry.get("atr_compress", 0.8)
        dataframe["nr7_is_nr"] = (
            (dataframe["nr7_range"] == dataframe["nr7_min_range_7"])
            & (dataframe["nr7_range"] < atr_compress * dataframe["nr7_atr"])
        )

        # Breakout detection on NEXT bar after NR7:
        # Long: close > previous bar high (when previous bar was NR7)
        # Short: close < previous bar low (when previous bar was NR7)
        dataframe["nr7_prev_high"] = h.shift(1)
        dataframe["nr7_prev_low"] = lo.shift(1)

        # Volume filter
        dataframe["nr7_vol_ema"] = ta.ema(v.astype(float), length=20)
        dataframe["nr7_vol_ratio"] = v.astype(float) / (dataframe["nr7_vol_ema"] + 1e-10)

        vol_min = self.config.entry.get("vol_min", 0.8)
        dataframe["nr7_break_up"] = (
            dataframe["nr7_is_nr"].shift(1)
            & (c > dataframe["nr7_prev_high"])
            & (dataframe["nr7_vol_ratio"] > vol_min)
        )
        dataframe["nr7_break_down"] = (
            dataframe["nr7_is_nr"].shift(1)
            & (c < dataframe["nr7_prev_low"])
            & (dataframe["nr7_vol_ratio"] > vol_min)
        )

        # Dedup: no signal within N bars of previous signal
        dedup = self.config.entry.get("dedup_bars", 5)
        dataframe["nr7_any_signal"] = dataframe["nr7_break_up"] | dataframe["nr7_break_down"]
        dataframe["nr7_last_signal"] = (
            dataframe["nr7_any_signal"].rolling(dedup).max().shift(1).fillna(0).astype(bool)
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
        if len(dataframe) < 8:
            return signals

        last = dataframe.iloc[-1]

        # Dedup filter
        if bool(last.get("nr7_last_signal", False)):
            return signals

        atr = float(last.get("nr7_atr", 0))
        vol_ratio = float(last.get("nr7_vol_ratio", 0))

        # Strength based on how compressed the NR7 bar was vs ATR
        nr7_range = float(dataframe.iloc[-2].get("nr7_range", 0)) if len(dataframe) > 1 else 0
        if atr > 0 and nr7_range > 0:
            compression = 1.0 - (nr7_range / atr)
            strength = min(1.0, 0.6 + compression * 0.5)
        else:
            strength = 0.7

        # Long breakout
        if bool(last.get("nr7_break_up", False)):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=strength,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"vol_ratio": vol_ratio, "nr_bars": 7},
            ))

        # Short breakout
        if bool(last.get("nr7_break_down", False)):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=strength,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"vol_ratio": vol_ratio, "nr_bars": 7},
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

        atr = float(last.get("nr7_atr", 0))
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

        # Cascading time cuts (NR7 breakouts should resolve within hours)
        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.exit

        if hours >= 4 and current_profit < exit_cfg.get("time_cut_4h", -0.008):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_4h", urgency=Urgency.NEXT_CANDLE,
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

"""Micro Pullback strategy: enter continuation after brief pullback to EMA8 in trend.

Entry: ADX > threshold + EMA alignment + 2-bar pullback touching EMA8 + RSI filter
Exit: ATR-based TP/SL, EMA reversal exit, time cuts (4h/12h/24h)
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
class MicroPullbackStrategy(BaseStrategy):
    name = "micro_pullback"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"].astype(float)

        ema_fast_len = self.config.entry.get("ema_fast", 8)
        ema_slow_len = self.config.entry.get("ema_slow", 21)

        # EMAs
        dataframe["mp_ema8"] = ta.ema(c, length=ema_fast_len)
        dataframe["mp_ema21"] = ta.ema(c, length=ema_slow_len)

        # ADX
        adx_df = ta.adx(h, lo, c, length=14)
        if adx_df is not None:
            dataframe["mp_adx"] = adx_df.iloc[:, 0]
        else:
            dataframe["mp_adx"] = 20.0

        # RSI
        dataframe["mp_rsi"] = ta.rsi(c, length=14)

        # ATR
        dataframe["mp_atr"] = ta.atr(h, lo, c, length=14)

        # Volume filter
        dataframe["mp_vol_ema"] = ta.ema(v, length=20)
        dataframe["mp_vol_ratio"] = v / (dataframe["mp_vol_ema"] + 1e-10)

        # Previous close changes for pullback detection
        dataframe["mp_close_chg1"] = c.diff(1)
        dataframe["mp_close_chg2"] = c.diff(1).shift(1)

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
        if len(dataframe) < 5:
            return signals

        last = dataframe.iloc[-1]

        adx_min = self.config.entry.get("adx_min", 25)
        vol_min = self.config.entry.get("vol_min", 0.8)
        rsi_long_min = self.config.entry.get("rsi_long_min", 40)
        rsi_short_max = self.config.entry.get("rsi_short_max", 60)
        touch_tol = self.config.entry.get("touch_tol", 0.001)

        vol_ratio = float(last.get("mp_vol_ratio", 0))
        if vol_ratio < vol_min:
            return signals

        adx = float(last.get("mp_adx", 0))
        if adx < adx_min:
            return signals

        close = float(last.get("close", 0))
        ema8 = float(last.get("mp_ema8", 0))
        ema21 = float(last.get("mp_ema21", 0))
        rsi = float(last.get("mp_rsi", 50))
        close_chg1 = float(last.get("mp_close_chg1", 0))
        close_chg2 = float(last.get("mp_close_chg2", 0))

        # Long: Uptrend + 2-bar pullback (lower closes) + touching EMA8 from above
        if (
            ema8 > ema21
            and close <= ema8 * (1 + touch_tol)
            and close_chg1 < 0
            and close_chg2 < 0
            and rsi > rsi_long_min
        ):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=1.0,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"adx": adx, "rsi": rsi},
            ))

        # Short: Downtrend + 2-bar rally (higher closes) + touching EMA8 from below
        if (
            ema8 < ema21
            and close >= ema8 * (1 - touch_tol)
            and close_chg1 > 0
            and close_chg2 > 0
            and rsi < rsi_short_max
        ):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=1.0,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"adx": adx, "rsi": rsi},
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

        atr = float(last.get("mp_atr", 0))
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

        # EMA reversal exit: EMA8 crosses EMA21 against position
        ema8 = float(last.get("mp_ema8", 0))
        ema21 = float(last.get("mp_ema21", 0))
        is_long = not is_short

        if is_long and ema8 < ema21:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="ema_reversal", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if not is_long and ema8 > ema21:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="ema_reversal", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        # Cascading time cuts
        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.exit

        if hours >= 4 and current_profit < exit_cfg.get("time_cut_4h", -0.005):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_4h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 12 and current_profit < exit_cfg.get("time_cut_12h", 0.002):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_12h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 24:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_24h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

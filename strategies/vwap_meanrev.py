"""VWAP Mean Reversion strategy: fade extreme deviations from rolling VWAP.

Entry: Price deviates > 2 std from rolling VWAP + RSI confirmation
Exit: ATR-based TP/SL, VWAP touch target, time cuts (3h/6h/12h)
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
class VwapMeanRevStrategy(BaseStrategy):
    name = "vwap_meanrev"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"].astype(float)

        vwap_window = self.config.entry.get("vwap_window", 96)

        # Rolling VWAP (96-bar = 24h on 15m)
        typical_price = (h + lo + c) / 3.0
        tp_vol = typical_price * v
        dataframe["vw_vwap"] = (
            tp_vol.rolling(window=vwap_window, min_periods=1).sum()
            / v.rolling(window=vwap_window, min_periods=1).sum().replace(0, 1e-10)
        )

        # VWAP deviation (standard deviation of close - vwap over window)
        deviation = c - dataframe["vw_vwap"]
        dataframe["vw_std"] = deviation.rolling(window=vwap_window, min_periods=1).std()

        # VWAP bands
        std_mult = self.config.entry.get("std_mult", 2.0)
        dataframe["vw_upper"] = dataframe["vw_vwap"] + std_mult * dataframe["vw_std"]
        dataframe["vw_lower"] = dataframe["vw_vwap"] - std_mult * dataframe["vw_std"]

        # ATR
        dataframe["vw_atr"] = ta.atr(h, lo, c, length=14)

        # RSI
        dataframe["vw_rsi"] = ta.rsi(c, length=14)

        # Volume filter
        dataframe["vw_vol_ema"] = ta.ema(v, length=20)
        dataframe["vw_vol_ratio"] = v / (dataframe["vw_vol_ema"] + 1e-10)

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

        rsi_os_thr = self.config.entry.get("rsi_os_thr", 30)
        rsi_ob_thr = self.config.entry.get("rsi_ob_thr", 70)
        vol_min = self.config.entry.get("vol_min", 0.8)

        vol_ratio = float(last.get("vw_vol_ratio", 0))
        if vol_ratio < vol_min:
            return signals

        close = float(last.get("close", 0))
        rsi = float(last.get("vw_rsi", 50))
        vwap_lower = float(last.get("vw_lower", 0))
        vwap_upper = float(last.get("vw_upper", 0))

        # Long: price below lower VWAP band + RSI oversold
        if close < vwap_lower and rsi < rsi_os_thr:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=1.0,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
            ))

        # Short: price above upper VWAP band + RSI overbought
        if close > vwap_upper and rsi > rsi_ob_thr:
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

        atr = float(last.get("vw_atr", 0))
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

        # VWAP touch exit (reversion target reached while profitable)
        close = float(last.get("close", 0))
        vwap = float(last.get("vw_vwap", 0))
        is_long = not is_short

        if current_profit > 0:
            if is_long and close >= vwap:
                return ExitRequest(
                    strategy_name=self.name, pair=pair,
                    reason="vwap_touch", urgency=Urgency.NEXT_CANDLE,
                    timestamp=datetime.utcnow(),
                )
            if not is_long and close <= vwap:
                return ExitRequest(
                    strategy_name=self.name, pair=pair,
                    reason="vwap_touch", urgency=Urgency.NEXT_CANDLE,
                    timestamp=datetime.utcnow(),
                )

        # Cascading time cuts
        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.exit

        if hours >= 3 and current_profit < exit_cfg.get("time_cut_3h", -0.005):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_3h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 6 and current_profit < exit_cfg.get("time_cut_6h", 0.002):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_6h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 12:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_12h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

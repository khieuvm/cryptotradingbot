"""Volume Spike + Reversal strategy: exhaustion signal from volume spike + candle pattern.

Entry: Volume > 2x EMA + reversal candle (hammer/shooting star) + RSI confirmation
Exit: ATR-based TP/SL, time cuts (2h/6h/12h) — quick reversal trades
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
class VolumeSpikeRevStrategy(BaseStrategy):
    name = "volume_spike_rev"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        o = dataframe["open"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"].astype(float)

        # Volume EMA and spike detection
        vol_ema_len = self.config.entry.get("vol_ema_len", 20)
        dataframe["vs_vol_ema"] = ta.ema(v, length=vol_ema_len)
        dataframe["vs_vol_ratio"] = v / (dataframe["vs_vol_ema"] + 1e-10)

        # ATR
        dataframe["vs_atr"] = ta.atr(h, lo, c, length=14)

        # RSI
        dataframe["vs_rsi"] = ta.rsi(c, length=14)

        # Candle body and shadow calculations
        body = (c - o).abs()
        upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
        lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - lo

        # Hammer detection: long lower shadow > 2x body, small upper shadow
        shadow_mult = self.config.entry.get("shadow_mult", 2.0)
        dataframe["vs_hammer"] = (
            (lower_shadow > shadow_mult * body)
            & (upper_shadow < body * 0.5)
            & (body > 0)
        )

        # Shooting star detection: long upper shadow > 2x body, small lower shadow
        dataframe["vs_shooting_star"] = (
            (upper_shadow > shadow_mult * body)
            & (lower_shadow < body * 0.5)
            & (body > 0)
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

        spike_mult = self.config.entry.get("spike_mult", 2.0)
        vol_min = self.config.entry.get("vol_min", 0.8)
        rsi_os_thr = self.config.entry.get("rsi_os_thr", 35)
        rsi_ob_thr = self.config.entry.get("rsi_ob_thr", 65)

        vol_ratio = float(last.get("vs_vol_ratio", 0))
        if vol_ratio < vol_min:
            return signals

        # Volume spike required
        if vol_ratio < spike_mult:
            return signals

        rsi = float(last.get("vs_rsi", 50))

        # Long: Hammer pattern + volume spike + RSI oversold
        if bool(last.get("vs_hammer", False)) and rsi < rsi_os_thr:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=1.0,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"vol_ratio": vol_ratio, "rsi": rsi},
            ))

        # Short: Shooting star pattern + volume spike + RSI overbought
        if bool(last.get("vs_shooting_star", False)) and rsi > rsi_ob_thr:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=1.0,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"vol_ratio": vol_ratio, "rsi": rsi},
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

        atr = float(last.get("vs_atr", 0))
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

        # Cascading time cuts (quick reversal trades — shorter durations)
        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.exit

        if hours >= 2 and current_profit < exit_cfg.get("time_cut_2h", -0.008):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_2h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 6 and current_profit < exit_cfg.get("time_cut_6h", -0.003):
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

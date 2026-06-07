"""Volume Climax Reversal 5m: extreme volume + reversal candle = exhaustion.

Entry: Volume > 3x EMA + reversal pattern (hammer/shooting star/engulfing)
       + RSI at extreme + BB band touch. Multiple confluence = higher WR.
Exit: Tight ATR SL, quick TP (mean-reversion scalp, hold 10-45 min).

This is a 5m-native version — not a port from 15m volume_spike_rev.
Key differences: tighter thresholds, BB touch required, faster exits.
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
class VolumeClimax5mStrategy(BaseStrategy):
    name = "volume_climax_5m"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        o = dataframe["open"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"].astype(float)

        # ATR
        dataframe["vc5_atr"] = ta.atr(h, lo, c, length=14)

        # Volume analysis
        vol_ema_len = self.config.entry.get("vol_ema_len", 20)
        dataframe["vc5_vol_ema"] = ta.ema(v, length=vol_ema_len)
        dataframe["vc5_vol_ratio"] = v / (dataframe["vc5_vol_ema"] + 1e-10)

        # Bollinger Bands for extremity
        bb_len = self.config.entry.get("bb_length", 20)
        bb_std = self.config.entry.get("bb_std", 2.0)
        bb = ta.bbands(c, length=bb_len, std=bb_std)
        if bb is not None:
            dataframe["vc5_bb_upper"] = bb.iloc[:, 0]
            dataframe["vc5_bb_lower"] = bb.iloc[:, 2]
        else:
            dataframe["vc5_bb_upper"] = c + dataframe["vc5_atr"]
            dataframe["vc5_bb_lower"] = c - dataframe["vc5_atr"]

        # RSI
        dataframe["vc5_rsi"] = ta.rsi(c, length=9)

        # Candle analysis
        body = c - o
        body_abs = body.abs()
        upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
        lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - lo
        total_range = h - lo + 1e-10

        # Shadow-to-body ratio
        shadow_mult = self.config.entry.get("shadow_body_ratio", 2.0)

        # Hammer: bullish reversal (long lower shadow)
        dataframe["vc5_hammer"] = (
            (lower_shadow > shadow_mult * body_abs)
            & (upper_shadow < body_abs * 0.5)
            & (body_abs > 0)
        )

        # Shooting star: bearish reversal (long upper shadow)
        dataframe["vc5_shooting_star"] = (
            (upper_shadow > shadow_mult * body_abs)
            & (lower_shadow < body_abs * 0.5)
            & (body_abs > 0)
        )

        # Engulfing patterns
        prev_body = body.shift(1)
        dataframe["vc5_bull_engulf"] = (body > 0) & (prev_body < 0) & (body_abs > prev_body.abs())
        dataframe["vc5_bear_engulf"] = (body < 0) & (prev_body > 0) & (body_abs > prev_body.abs())

        # BB touch/breach
        dataframe["vc5_at_lower_bb"] = lo <= dataframe["vc5_bb_lower"]
        dataframe["vc5_at_upper_bb"] = h >= dataframe["vc5_bb_upper"]

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

        # Volume climax required (higher threshold than 15m version)
        spike_mult = self.config.entry.get("spike_mult", 3.0)
        vol_ratio = float(last.get("vc5_vol_ratio", 0))
        if vol_ratio < spike_mult:
            return signals

        rsi = float(last.get("vc5_rsi", 50))
        rsi_os = self.config.entry.get("rsi_os_thr", 25)
        rsi_ob = self.config.entry.get("rsi_ob_thr", 75)

        # === LONG: Volume climax at bottom ===
        # Requires: volume spike + (hammer OR bull engulfing) + RSI oversold + BB lower touch
        has_bull_pattern = bool(last.get("vc5_hammer", False)) or bool(last.get("vc5_bull_engulf", False))
        at_lower_bb = bool(last.get("vc5_at_lower_bb", False))

        if has_bull_pattern and rsi < rsi_os and at_lower_bb:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=1.0,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"vol_ratio": vol_ratio, "rsi": rsi},
            ))

        # === SHORT: Volume climax at top ===
        has_bear_pattern = bool(last.get("vc5_shooting_star", False)) or bool(last.get("vc5_bear_engulf", False))
        at_upper_bb = bool(last.get("vc5_at_upper_bb", False))

        if has_bear_pattern and rsi > rsi_ob and at_upper_bb:
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

        atr = float(last.get("vc5_atr", 0))
        if atr <= 0 or open_rate <= 0:
            return None

        # TP hit (mean-reversion: target BB mid)
        tp_pct = self.tp_atr_mult * atr / open_rate
        if current_profit >= tp_pct:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="TP_HIT", urgency=Urgency.IMMEDIATE,
                timestamp=datetime.utcnow(),
            )

        # Scalp time cuts (faster than 15m version)
        minutes = (current_time - entry_time).total_seconds() / 60
        exit_cfg = self.config.exit

        if minutes >= 20 and current_profit < exit_cfg.get("time_cut_20m", -0.004):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_20m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if minutes >= 45 and current_profit < exit_cfg.get("time_cut_45m", -0.002):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_45m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if minutes >= 60:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_60m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

"""Volume Spike Reversal: momentum short on volume climax + mean-reversion long on exhaustion.

SHORT signal (primary): Big red candle (body > 55% of range) with volume spike (>3x EMA)
  and RSI in 15-50 range (momentum, not yet bottomed).
LONG signal (secondary): Hammer candle with volume spike + RSI < 25 (capitulation buy).

Exit: ATR-based TP/SL, time cuts (4h/8h) to cut slow losers.

Walk-forward validated on 180d real OKX data (Jun 2026):
  Train 120d: +84.5%, OOS 60d: +21.5% (ETH+SOL+SPX), DOGE excluded.
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

        vol_ema_len = self.config.entry.get("vol_ema_len", 20)
        dataframe["vs_vol_ema"] = ta.ema(v, length=vol_ema_len)
        dataframe["vs_vol_ratio"] = v / (dataframe["vs_vol_ema"] + 1e-10)

        dataframe["vs_atr"] = ta.atr(h, lo, c, length=14)
        dataframe["vs_rsi"] = ta.rsi(c, length=14)

        body = (c - o).abs()
        candle_range = h - lo + 1e-10
        upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
        lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - lo

        dataframe["vs_body_ratio"] = body / candle_range
        dataframe["vs_is_red"] = c < o
        dataframe["vs_body"] = body

        shadow_mult = self.config.entry.get("shadow_body_ratio", 2.0)
        dataframe["vs_hammer"] = (
            (lower_shadow > shadow_mult * body)
            & (upper_shadow < body * 0.5)
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

        spike_mult = self.config.entry.get("spike_mult", 3.0)
        vol_min = self.config.entry.get("vol_min", 1.0)
        rsi_os_thr = self.config.entry.get("rsi_os_thr", 25)
        rsi_short_max = self.config.entry.get("rsi_short_max", 50)
        rsi_short_min = self.config.entry.get("rsi_short_min", 15)
        body_ratio_min = self.config.entry.get("body_ratio_min", 0.55)

        vol_ratio = float(last.get("vs_vol_ratio", 0))
        if vol_ratio < vol_min or vol_ratio < spike_mult:
            return signals

        rsi = float(last.get("vs_rsi", 50))
        atr = float(last.get("vs_atr", 0))
        body = float(last.get("vs_body", 0))
        br = float(last.get("vs_body_ratio", 0))
        is_red = bool(last.get("vs_is_red", False))

        # SHORT: big red candle + volume spike + RSI in momentum range
        if is_red and br > body_ratio_min and body > 0.3 * atr and rsi_short_min < rsi < rsi_short_max:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=1.0,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"vol_ratio": vol_ratio, "rsi": rsi, "body_ratio": br},
            ))

        # LONG: hammer candle + volume spike + extreme RSI (capitulation)
        if bool(last.get("vs_hammer", False)) and rsi < rsi_os_thr:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=1.0,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"vol_ratio": vol_ratio, "rsi": rsi},
            ))

        return signals

    def populate_entry_columns(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        spike_mult = self.config.entry.get("spike_mult", 3.0)
        rsi_os_thr = self.config.entry.get("rsi_os_thr", 25)
        rsi_short_max = self.config.entry.get("rsi_short_max", 50)
        rsi_short_min = self.config.entry.get("rsi_short_min", 15)
        body_ratio_min = self.config.entry.get("body_ratio_min", 0.55)

        startup = self.startup_candle_count
        df = dataframe

        vol_spike = df["vs_vol_ratio"] >= spike_mult
        atr_ok = df["vs_body"] > 0.3 * df["vs_atr"]

        enter_short = (
            vol_spike & df["vs_is_red"]
            & (df["vs_body_ratio"] > body_ratio_min)
            & atr_ok
            & (df["vs_rsi"] > rsi_short_min)
            & (df["vs_rsi"] < rsi_short_max)
        )
        enter_long = vol_spike & df["vs_hammer"] & (df["vs_rsi"] < rsi_os_thr)

        enter_long.iloc[:startup] = False
        enter_short.iloc[:startup] = False

        dataframe.loc[enter_long, "enter_long"] = 1
        dataframe.loc[enter_long, "enter_tag"] = f"{self.name}_long"
        dataframe.loc[enter_short, "enter_short"] = 1
        dataframe.loc[enter_short, "enter_tag"] = f"{self.name}_short"

        return dataframe

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

        tp_pct = self.tp_atr_mult * atr / open_rate
        if current_profit >= tp_pct:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="TP_HIT", urgency=Urgency.IMMEDIATE,
                timestamp=datetime.utcnow(),
            )

        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.exit

        if hours >= 4 and current_profit < exit_cfg.get("time_cut_4h", -0.01):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_4h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 8 and current_profit < exit_cfg.get("time_cut_8h", -0.005):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_8h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 12:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_12h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

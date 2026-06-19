"""Heikin-Ashi + Stochastic Reversal (HumbleTraders Strategy 3).

After 2+ consecutive bearish HA candles (red), Stochastic crosses into overbought (>70)
→ contrarian SHORT (momentum exhausted, price about to reverse).
After 2+ consecutive bullish HA candles (green), Stochastic crosses into oversold (<30)
→ contrarian LONG (momentum exhausted, price about to reverse).

Note: This is a contrarian/mean-reversion signal — we enter AGAINST the HA trend direction
after the momentum oscillator confirms exhaustion.

Edge: HA smoothing removes 15m noise. 2-consecutive-bar confirmation ensures the HA
  trend has "stuck" before we fade it. Stochastic OB/OS filters out entries when
  momentum still has room to run.

EXIT: ATR-based TP, time cuts (15m HA trades are short-lived).
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


def _compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin-Ashi OHLC. Sequential — no look-ahead."""
    ha_c = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_o = pd.Series(np.nan, index=df.index, dtype=float)
    ha_o.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_o.iloc[i] = (ha_o.iloc[i - 1] + ha_c.iloc[i - 1]) / 2
    ha_h = pd.concat([df["high"], ha_o, ha_c], axis=1).max(axis=1)
    ha_l = pd.concat([df["low"], ha_o, ha_c], axis=1).min(axis=1)
    return pd.DataFrame({"ha_o": ha_o, "ha_h": ha_h, "ha_l": ha_l, "ha_c": ha_c},
                        index=df.index)


@register_strategy
class HaStochReversalStrategy(BaseStrategy):
    name = "ha_stoch_rev"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata.get("pair", "")
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"].astype(float)

        dataframe["hsr_atr"] = ta.atr(h, lo, c, length=14)
        dataframe["hsr_rsi"] = ta.rsi(c, length=14)

        # ── Heikin-Ashi candles ───────────────────────────────────────────────
        ha = _compute_ha(dataframe)
        dataframe["hsr_ha_c"] = ha["ha_c"]
        dataframe["hsr_ha_o"] = ha["ha_o"]
        # HA color: 1 = green (bullish), -1 = red (bearish)
        ha_color = (ha["ha_c"] >= ha["ha_o"]).astype(int) * 2 - 1
        dataframe["hsr_ha_color"] = ha_color

        # Count consecutive same-color HA bars (run length)
        # A break (color change) resets the count
        min_run = self.config.entry.get("ha_run_min", 2)
        # Rolling sum: if all N recent = +1, sum=N; if all = -1, sum=-N
        ha_run = ha_color.rolling(min_run).sum()
        dataframe["hsr_ha_bull_run"] = ha_run >= min_run   # N+ consecutive green
        dataframe["hsr_ha_bear_run"] = ha_run <= -min_run  # N+ consecutive red

        # ── Stochastic (14, 7, 3) ─────────────────────────────────────────────
        k_period = self.config.entry.get("stoch_k", 14)
        d_period = self.config.entry.get("stoch_d", 3)
        smooth_k = self.config.entry.get("stoch_smooth", 7)

        stoch_df = ta.stoch(h, lo, c, k=k_period, d=d_period, smooth_k=smooth_k)
        if stoch_df is not None and not stoch_df.empty:
            k_col = next((col for col in stoch_df.columns if "STOCHk" in col), None)
            d_col = next((col for col in stoch_df.columns if "STOCHd" in col), None)
            dataframe["hsr_stoch_k"] = stoch_df[k_col] if k_col else 50.0
            dataframe["hsr_stoch_d"] = stoch_df[d_col] if d_col else 50.0
        else:
            dataframe["hsr_stoch_k"] = 50.0
            dataframe["hsr_stoch_d"] = 50.0

        entry_cfg = self.config.get_entry(pair)
        stoch_ob = entry_cfg.get("stoch_ob", 70)
        stoch_os = entry_cfg.get("stoch_os", 30)

        # ── Trend filter: EMA for macro direction ─────────────────────────────
        ema_trend = self.config.entry.get("ema_trend", 100)
        dataframe["hsr_ema_trend"] = ta.ema(c, length=ema_trend)

        # Volume confirmation
        dataframe["hsr_vol_ema"] = ta.ema(v, length=20)
        dataframe["hsr_vol_ratio"] = v / (dataframe["hsr_vol_ema"] + 1e-10)
        vol_min = entry_cfg.get("vol_min", 1.0)

        # ── Entry signals (contrarian — fade HA momentum) ─────────────────────
        # SHORT: 2+ green HA candles (bull run) + Stochastic > ob (overbought) + macro downtrend
        dataframe["hsr_enter_short"] = (
            dataframe["hsr_ha_bull_run"]
            & (dataframe["hsr_stoch_k"] > stoch_ob)
            & (dataframe["hsr_vol_ratio"] >= vol_min)
            & (c < dataframe["hsr_ema_trend"])  # macro downtrend
        )
        # LONG: 2+ red HA candles (bear run) + Stochastic < os (oversold) + macro uptrend
        allow_longs = entry_cfg.get("allow_longs", True)
        if allow_longs:
            dataframe["hsr_enter_long"] = (
                dataframe["hsr_ha_bear_run"]
                & (dataframe["hsr_stoch_k"] < stoch_os)
                & (dataframe["hsr_vol_ratio"] >= vol_min)
                & (c > dataframe["hsr_ema_trend"])  # macro uptrend
            )
        else:
            dataframe["hsr_enter_long"] = False

        dedup = self.config.entry.get("dedup_bars", 6)
        any_signal = dataframe["hsr_enter_long"] | dataframe["hsr_enter_short"]
        dataframe["hsr_last_signal"] = any_signal.rolling(dedup).max().shift(1).fillna(0).astype(bool)

        return dataframe

    def on_tick(self, dataframe: pd.DataFrame, pair: str, current_time: datetime) -> None:
        for sig in self.detect_entries(dataframe, pair):
            self.emit_signal(
                pair=sig.pair, direction=sig.direction, strength=sig.strength,
                tag=sig.tag, metadata=sig.metadata,
            )

    def detect_entries(self, dataframe: pd.DataFrame, pair: str) -> list[Signal]:
        signals: list[Signal] = []
        if len(dataframe) < self.startup_candle_count:
            return signals

        last = dataframe.iloc[-1]
        if bool(last.get("hsr_last_signal", False)):
            return signals
        if not bool(self.get_session_mask(dataframe).iloc[-1]):
            return signals

        stoch_k = float(last.get("hsr_stoch_k", 50))
        ha_color = int(last.get("hsr_ha_color", 0))

        if bool(last.get("hsr_enter_long", False)):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=1.0,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"stoch_k": stoch_k, "ha_color": ha_color},
            ))

        if bool(last.get("hsr_enter_short", False)):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=1.0,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"stoch_k": stoch_k, "ha_color": ha_color},
            ))

        return signals

    def populate_entry_columns(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        startup = self.startup_candle_count
        df = dataframe

        session = self.get_session_mask(dataframe)
        not_deduped = ~df["hsr_last_signal"]
        enter_long = df["hsr_enter_long"] & not_deduped & session
        enter_short = df["hsr_enter_short"] & not_deduped & session

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

        atr = float(last.get("hsr_atr", 0))
        if atr <= 0 or open_rate <= 0:
            return None

        tp_pct = self.get_tp_atr_mult(pair) * atr / open_rate
        if current_profit >= tp_pct:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="TP_HIT", urgency=Urgency.IMMEDIATE,
                timestamp=datetime.utcnow(),
            )

        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.get_exit(pair)

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
        if hours >= 16:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_16h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

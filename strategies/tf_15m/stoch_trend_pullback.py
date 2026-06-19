"""Stochastic Trend Pullback: fade counter-trend bounce in established trend.

SHORT: EMA18<EMA50 + ADX>18 + StochK>75 turning down + red candle + close<EMA18
LONG:  EMA18>EMA50 + ADX>18 + StochK<25 turning up + green candle + close>EMA18

Unlike ha_stoch_rev, does NOT require Heikin-Ashi bull run — just trend structure
+ stochastic reading. Works in normal volume (no vol spike needed).
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
class StochTrendPullbackStrategy(BaseStrategy):
    name = "stoch_trend_pullback"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata.get("pair", "")
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]

        dataframe["stp_atr"] = ta.atr(h, lo, c, length=14)
        dataframe["stp_ema18"] = ta.ema(c, length=18)
        dataframe["stp_ema50"] = ta.ema(c, length=50)

        adx_df = ta.adx(h, lo, c, length=14)
        if adx_df is not None:
            dataframe["stp_adx"] = adx_df.iloc[:, 0]
        else:
            dataframe["stp_adx"] = 0.0

        entry_cfg = self.config.get_entry(pair)
        k_period = entry_cfg.get("stoch_k", 14)
        d_period = entry_cfg.get("stoch_d", 3)
        smooth_k = entry_cfg.get("stoch_smooth", 7)

        stoch_df = ta.stoch(h, lo, c, k=k_period, d=d_period, smooth_k=smooth_k)
        if stoch_df is not None and len(stoch_df) > 0:
            k_col = next((col for col in stoch_df.columns if "STOCHk" in col), None)
            if k_col:
                dataframe["stp_stoch_k"] = stoch_df[k_col]
            else:
                dataframe["stp_stoch_k"] = 50.0
        else:
            dataframe["stp_stoch_k"] = 50.0

        dataframe["stp_stoch_k_prev"] = dataframe["stp_stoch_k"].shift(1)
        dataframe["stp_is_green"] = c > dataframe["open"]
        dataframe["stp_is_red"] = c < dataframe["open"]

        return dataframe

    def populate_entry_columns(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        entry_cfg = self.config.get_entry(pair)
        adx_min = entry_cfg.get("adx_min", 18)
        stoch_ob = entry_cfg.get("stoch_ob", 75)
        stoch_os = entry_cfg.get("stoch_os", 25)
        allow_longs = entry_cfg.get("allow_longs", False)

        startup = self.startup_candle_count
        session = self.get_session_mask(dataframe)

        trend_dn = dataframe["stp_ema18"] < dataframe["stp_ema50"]
        trend_up = dataframe["stp_ema18"] > dataframe["stp_ema50"]
        adx_ok = dataframe["stp_adx"] >= adx_min

        enter_short = (
            trend_dn & adx_ok & session
            & (dataframe["stp_stoch_k"] > stoch_ob)
            & (dataframe["stp_stoch_k"] < dataframe["stp_stoch_k_prev"])
            & dataframe["stp_is_red"]
            & (dataframe["close"] < dataframe["stp_ema18"])
        )

        if allow_longs:
            enter_long = (
                trend_up & adx_ok & session
                & (dataframe["stp_stoch_k"] < stoch_os)
                & (dataframe["stp_stoch_k"] > dataframe["stp_stoch_k_prev"])
                & dataframe["stp_is_green"]
                & (dataframe["close"] > dataframe["stp_ema18"])
            )
        else:
            enter_long = pd.Series(False, index=dataframe.index)

        enter_long.iloc[:startup] = False
        enter_short.iloc[:startup] = False

        dataframe.loc[enter_long, "enter_long"] = 1
        dataframe.loc[enter_long, "enter_tag"] = f"{self.name}_long"
        dataframe.loc[enter_short, "enter_short"] = 1
        dataframe.loc[enter_short, "enter_tag"] = f"{self.name}_short"

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
        if len(dataframe) < self.startup_candle_count:
            return signals

        last = dataframe.iloc[-1]
        if not bool(self.get_session_mask(dataframe).iloc[-1]):
            return signals

        entry_cfg = self.config.get_entry(pair)
        adx_min = entry_cfg.get("adx_min", 18)
        stoch_ob = entry_cfg.get("stoch_ob", 75)
        stoch_os = entry_cfg.get("stoch_os", 25)
        allow_longs = entry_cfg.get("allow_longs", False)

        adx = float(last.get("stp_adx", 0))
        stoch_k = float(last.get("stp_stoch_k", 50))
        stoch_k_prev = float(last.get("stp_stoch_k_prev", 50))
        ema18 = float(last.get("stp_ema18", 0))
        ema50 = float(last.get("stp_ema50", 0))
        close = float(last.get("close", 0))

        if adx < adx_min:
            return signals

        if ema18 < ema50 and stoch_k > stoch_ob and stoch_k < stoch_k_prev:
            if bool(last.get("stp_is_red", False)) and close < ema18:
                signals.append(Signal(
                    strategy_name=self.name, pair=pair,
                    direction=Direction.SHORT, strength=1.0,
                    tag=f"{self.name}_short", timestamp=datetime.utcnow(),
                    metadata={"stoch_k": stoch_k, "adx": adx},
                ))

        if allow_longs and ema18 > ema50 and stoch_k < stoch_os and stoch_k > stoch_k_prev:
            if bool(last.get("stp_is_green", False)) and close > ema18:
                signals.append(Signal(
                    strategy_name=self.name, pair=pair,
                    direction=Direction.LONG, strength=1.0,
                    tag=f"{self.name}_long", timestamp=datetime.utcnow(),
                    metadata={"stoch_k": stoch_k, "adx": adx},
                ))

        return signals

    def detect_exits(self, dataframe: pd.DataFrame, pair: str, trade_info: dict | None) -> ExitRequest | None:
        if trade_info is None:
            return None

        last = dataframe.iloc[-1]
        current_profit = trade_info.get("current_profit", 0)
        current_time = trade_info.get("current_time", datetime.utcnow())
        entry_time = trade_info.get("entry_time", current_time)
        open_rate = trade_info.get("entry_rate", 0)

        atr = float(last.get("stp_atr", 0))
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
        if hours >= 8 and current_profit < exit_cfg.get("time_cut_8h", 0.0):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_8h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

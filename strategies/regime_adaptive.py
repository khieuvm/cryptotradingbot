"""Regime-Adaptive strategy: detect market state, apply appropriate signals.

TRENDING (ADX > threshold): EMA cross + MACD + DI + SuperTrend + volume
RANGING (ADX <= threshold): RSI extreme + BB + OBV + candle pattern
EXIT: ATR-based TP, cascading time-loss-cut, signal reversal
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
class RegimeAdaptiveStrategy(BaseStrategy):
    name = "regime_adaptive"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"]

        ema_fast_p = self.config.entry.get("ema_fast", 18)
        ema_slow_p = self.config.entry.get("ema_slow", 50)

        dataframe["ra_ema_fast"] = ta.ema(c, length=ema_fast_p)
        dataframe["ra_ema_slow"] = ta.ema(c, length=ema_slow_p)
        dataframe["ra_ema200"] = ta.ema(c, length=200)

        dataframe["ra_atr"] = ta.atr(h, lo, c, length=14)
        dataframe["ra_atr_ma"] = ta.ema(dataframe["ra_atr"], length=50)
        dataframe["ra_atr_ratio"] = dataframe["ra_atr"] / (dataframe["ra_atr_ma"] + 1e-10)

        adx = ta.adx(h, lo, c, length=14)
        if adx is not None:
            dataframe["ra_adx"] = adx.iloc[:, 0]
            dataframe["ra_plus_di"] = adx.iloc[:, 1]
            dataframe["ra_minus_di"] = adx.iloc[:, 2]
        else:
            dataframe["ra_adx"] = dataframe["ra_plus_di"] = dataframe["ra_minus_di"] = 0.0

        dataframe["ra_rsi"] = ta.rsi(c, length=14)

        macd = ta.macd(c, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            dataframe["ra_macd_hist"] = macd.iloc[:, 1]
        else:
            dataframe["ra_macd_hist"] = 0.0

        bb = ta.bbands(c, length=20, std=2.0)
        if bb is not None:
            dataframe["ra_bb_upper"] = bb.iloc[:, 0]
            dataframe["ra_bb_mid"] = bb.iloc[:, 1]
            dataframe["ra_bb_lower"] = bb.iloc[:, 2]
        else:
            dataframe["ra_bb_upper"] = dataframe["ra_bb_lower"] = dataframe["ra_bb_mid"] = c

        dataframe["ra_vol_ema"] = ta.ema(v, length=20)
        dataframe["ra_vol_ratio"] = v / (dataframe["ra_vol_ema"] + 1e-10)

        dataframe["ra_obv"] = ta.obv(c, v)
        dataframe["ra_obv_ema"] = ta.ema(dataframe["ra_obv"], length=20)
        dataframe["ra_obv_rising"] = (dataframe["ra_obv"] > dataframe["ra_obv_ema"]).astype(int)

        try:
            st = ta.supertrend(h, lo, c, length=7, multiplier=3.0)
            if st is not None:
                st_dir_col = next((col for col in st.columns if "SUPERTd" in col), None)
                dataframe["ra_st_dir"] = st[st_dir_col].fillna(0) if st_dir_col else 0
            else:
                dataframe["ra_st_dir"] = 0
        except Exception:
            dataframe["ra_st_dir"] = 0

        dataframe["ra_is_bull"] = (c > dataframe["ra_ema200"]).astype(int)
        dataframe["ra_is_bear"] = (c < dataframe["ra_ema200"]).astype(int)

        # EMA cross freshness: bars since last cross
        cross_up = (dataframe["ra_ema_fast"] > dataframe["ra_ema_slow"]) & (
            dataframe["ra_ema_fast"].shift(1) <= dataframe["ra_ema_slow"].shift(1)
        )
        cross_down = (dataframe["ra_ema_fast"] < dataframe["ra_ema_slow"]) & (
            dataframe["ra_ema_fast"].shift(1) >= dataframe["ra_ema_slow"].shift(1)
        )
        cross_lookback = self.config.entry.get("cross_lookback", 8)
        dataframe["ra_cross_up_recent"] = cross_up.rolling(cross_lookback).max().fillna(0).astype(int)
        dataframe["ra_cross_down_recent"] = cross_down.rolling(cross_lookback).max().fillna(0).astype(int)

        # Exit signals
        dataframe["ra_trend_exit_long"] = (
            (dataframe["ra_ema_fast"] < dataframe["ra_ema_slow"])
            & (dataframe["ra_ema_fast"].shift(1) >= dataframe["ra_ema_slow"].shift(1))
            & (dataframe["ra_st_dir"] == -1)
        ).astype(int)

        dataframe["ra_trend_exit_short"] = (
            (dataframe["ra_ema_fast"] > dataframe["ra_ema_slow"])
            & (dataframe["ra_ema_fast"].shift(1) <= dataframe["ra_ema_slow"].shift(1))
            & (dataframe["ra_st_dir"] == 1)
        ).astype(int)

        dataframe["ra_range_exit_long"] = (
            (dataframe["ra_rsi"] > 52) & (c > dataframe["ra_bb_mid"])
        ).astype(int)

        dataframe["ra_range_exit_short"] = (
            (dataframe["ra_rsi"] < 48) & (c < dataframe["ra_bb_mid"])
        ).astype(int)

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
        last = dataframe.iloc[-1]

        adx_thr = self.config.entry.get("adx_trend_thr", 31)
        rsi_os = self.config.entry.get("rsi_os", 34)
        rsi_ob = self.config.entry.get("rsi_ob", 71)
        vol_min = self.config.entry.get("vol_min", 0.5)
        atr_spike = self.config.entry.get("atr_spike_thr", 2.2)

        if float(last.get("ra_atr_ratio", 0)) >= atr_spike:
            return signals
        if float(last.get("ra_vol_ratio", 0)) < vol_min:
            return signals
        if float(last.get("volume", 0)) <= 0:
            return signals

        is_trending = float(last.get("ra_adx", 0)) > adx_thr
        is_bull = int(last.get("ra_is_bull", 0)) == 1
        is_bear = int(last.get("ra_is_bear", 0)) == 1

        if is_trending:
            # Trend long — require EMA cross within lookback period
            if (
                is_bull
                and int(last.get("ra_cross_up_recent", 0)) == 1
                and float(last.get("ra_ema_fast", 0)) > float(last.get("ra_ema_slow", 0))
                and float(last.get("ra_macd_hist", 0)) > 0
                and float(last.get("ra_plus_di", 0)) > float(last.get("ra_minus_di", 0))
                and int(last.get("ra_st_dir", 0)) == 1
            ):
                signals.append(Signal(
                    strategy_name=self.name, pair=pair,
                    direction=Direction.LONG, strength=1.0,
                    tag=f"{self.name}_trend_long",
                    timestamp=datetime.utcnow(),
                    metadata={"regime": "trending"},
                ))

            # Trend short — require EMA cross within lookback period
            if (
                is_bear
                and int(last.get("ra_cross_down_recent", 0)) == 1
                and float(last.get("ra_ema_fast", 0)) < float(last.get("ra_ema_slow", 0))
                and float(last.get("ra_macd_hist", 0)) < 0
                and float(last.get("ra_minus_di", 0)) > float(last.get("ra_plus_di", 0))
                and int(last.get("ra_st_dir", 0)) == -1
            ):
                signals.append(Signal(
                    strategy_name=self.name, pair=pair,
                    direction=Direction.SHORT, strength=1.0,
                    tag=f"{self.name}_trend_short",
                    timestamp=datetime.utcnow(),
                    metadata={"regime": "trending"},
                ))
        else:
            # Range long
            rsi = float(last.get("ra_rsi", 50))
            prev_rsi = float(dataframe.iloc[-2].get("ra_rsi", 50)) if len(dataframe) > 1 else 50
            if (
                prev_rsi < rsi_os
                and rsi > prev_rsi
                and float(last.get("close", 0)) < float(last.get("ra_bb_lower", 0)) * 1.01
                and float(last.get("close", 0)) > float(last.get("open", 0))
                and int(last.get("ra_obv_rising", 0)) == 1
            ):
                signals.append(Signal(
                    strategy_name=self.name, pair=pair,
                    direction=Direction.LONG, strength=0.8,
                    tag=f"{self.name}_range_long",
                    timestamp=datetime.utcnow(),
                    metadata={"regime": "ranging"},
                ))

            # Range short
            if (
                prev_rsi > rsi_ob
                and rsi < prev_rsi
                and float(last.get("close", 0)) > float(last.get("ra_bb_upper", 0)) * 0.99
                and float(last.get("close", 0)) < float(last.get("open", 0))
                and int(last.get("ra_obv_rising", 0)) == 0
            ):
                signals.append(Signal(
                    strategy_name=self.name, pair=pair,
                    direction=Direction.SHORT, strength=0.8,
                    tag=f"{self.name}_range_short",
                    timestamp=datetime.utcnow(),
                    metadata={"regime": "ranging"},
                ))

        return signals

    def detect_exits(
        self, dataframe: pd.DataFrame, pair: str, trade_info: dict | None
    ) -> ExitRequest | None:
        if trade_info is None:
            return None

        last = dataframe.iloc[-1]
        enter_tag = trade_info.get("enter_tag", "")
        current_profit = trade_info.get("current_profit", 0)
        is_short = trade_info.get("is_short", False)
        current_time = trade_info.get("current_time", datetime.utcnow())
        entry_time = trade_info.get("entry_time", current_time)

        # ATR-based TP
        atr = float(last.get("ra_atr", 0))
        open_rate = trade_info.get("entry_rate", 0)
        if atr > 0 and open_rate > 0:
            tp_pct = self.tp_atr_mult * atr / open_rate
            if current_profit >= tp_pct:
                return ExitRequest(
                    strategy_name=self.name, pair=pair,
                    reason="TP_HIT", urgency=Urgency.IMMEDIATE,
                    timestamp=datetime.utcnow(),
                )

        # Signal-based exits
        is_long = not is_short
        if "trend" in enter_tag:
            sig_col = "ra_trend_exit_long" if is_long else "ra_trend_exit_short"
            if int(last.get(sig_col, 0)) == 1 and current_profit >= 0.0:
                return ExitRequest(
                    strategy_name=self.name, pair=pair,
                    reason="trend_signal_exit", urgency=Urgency.NEXT_CANDLE,
                    timestamp=datetime.utcnow(),
                )
        elif "range" in enter_tag:
            sig_col = "ra_range_exit_long" if is_long else "ra_range_exit_short"
            if int(last.get(sig_col, 0)) == 1:
                return ExitRequest(
                    strategy_name=self.name, pair=pair,
                    reason="range_target_exit", urgency=Urgency.NEXT_CANDLE,
                    timestamp=datetime.utcnow(),
                )

        # Cascading time-loss cuts
        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.exit
        time_cut_2h = exit_cfg.get("time_cut_2h", -0.015)
        time_cut_8h = exit_cfg.get("time_cut_8h", -0.008)
        time_cut_24h = exit_cfg.get("time_cut_24h", 0.0)
        time_cut_48h = exit_cfg.get("time_cut_48h", 0.005)

        if hours >= 2 and current_profit < time_cut_2h:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_2h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 8 and current_profit < time_cut_8h:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_8h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 24 and current_profit < time_cut_24h:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_24h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 48 and current_profit < time_cut_48h:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_48h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

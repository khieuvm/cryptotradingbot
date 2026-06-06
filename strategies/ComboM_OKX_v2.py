# -*- coding: utf-8 -*-
"""
ComboM_OKX_v2 - 15m EMA Momentum (simple & proven)
====================================================
After 3 failed attempts with complex signals (breakout, squeeze, pullback),
going with the simplest possible trend strategy:

Concept: EMA Cross + Momentum + Volume
- Entry LONG: EMA20 crosses above EMA50 + ADX>25 + Volume spike + RSI 50-70
- Entry SHORT: EMA20 crosses below EMA50 + ADX>25 + Volume spike + RSI 30-50
- Exit: ATR-based TP/SL + BB-mid as profit target
- Break-even at 1.5 ATR profit

Why this might work:
- EMA cross is a LAGGING signal → avoids fake breakout traps
- By the time EMA20 > EMA50, the trend is CONFIRMED
- ADX filter ensures we only trade in strong trends
- Volume filter confirms institutional participation
- Simple = robust = less overfitting
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter


class ComboM_OKX_v2(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"

    stoploss = -0.04
    minimal_roi = {"0": 0.06, "90": 0.02, "180": 0.01}

    can_short = True
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    trailing_stop = False

    protections = [
        {"method": "CooldownPeriod", "stop_duration_candles": 5},
        {"method": "StoplossGuard", "lookback_period_candles": 48,
         "trade_limit": 3, "stop_duration_candles": 24, "only_per_pair": True},
    ]

    # --- Risk ---
    SL_ATR_MULT = DecimalParameter(1.2, 2.5, default=1.8, decimals=1, space="sell", optimize=True)
    TP_ATR_MULT = DecimalParameter(1.5, 4.0, default=2.5, decimals=1, space="sell", optimize=True)
    BE_ATR_MULT = DecimalParameter(1.0, 2.0, default=1.5, decimals=1, space="sell", optimize=True)

    # --- Entry ---
    ADX_MIN = IntParameter(20, 35, default=25, space="buy", optimize=True)
    VOL_MULT = DecimalParameter(0.8, 2.0, default=1.2, decimals=1, space="buy", optimize=True)
    RSI_LOW = IntParameter(40, 55, default=48, space="buy", optimize=True)
    RSI_HIGH = IntParameter(55, 70, default=65, space="buy", optimize=True)
    EMA_FAST = IntParameter(10, 25, default=20, space="buy", optimize=False)
    EMA_SLOW = IntParameter(40, 60, default=50, space="buy", optimize=False)

    # ------------------------------------------------------------------ #
    #  INDICATORS                                                          #
    # ------------------------------------------------------------------ #
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # EMAs
        dataframe["ema_fast"] = ta.ema(dataframe["close"], length=self.EMA_FAST.value)
        dataframe["ema_slow"] = ta.ema(dataframe["close"], length=self.EMA_SLOW.value)

        # ATR
        dataframe["atr"] = ta.atr(
            dataframe["high"], dataframe["low"], dataframe["close"], length=14
        )

        # ADX + DI
        adx_df = ta.adx(dataframe["high"], dataframe["low"], dataframe["close"], length=14)
        if adx_df is not None:
            dataframe["adx"] = adx_df.iloc[:, 0]
            dataframe["plus_di"] = adx_df.iloc[:, 1]
            dataframe["minus_di"] = adx_df.iloc[:, 2]
        else:
            dataframe["adx"] = 0.0
            dataframe["plus_di"] = 0.0
            dataframe["minus_di"] = 0.0

        # RSI
        dataframe["rsi"] = ta.rsi(dataframe["close"], length=14)

        # Volume ratio
        dataframe["vol_ema"] = ta.ema(dataframe["volume"].astype(float), length=20)
        dataframe["vol_ratio"] = dataframe["volume"].astype(float) / (dataframe["vol_ema"] + 1e-10)

        # BB for exit
        bb = ta.bbands(dataframe["close"], length=20, std=2.0)
        if bb is not None:
            dataframe["bb_mid"] = bb.iloc[:, 1]
            dataframe["bb_upper"] = bb.iloc[:, 2]
            dataframe["bb_lower"] = bb.iloc[:, 0]
        else:
            dataframe["bb_mid"] = dataframe["close"]
            dataframe["bb_upper"] = dataframe["close"]
            dataframe["bb_lower"] = dataframe["close"]

        # MACD for momentum confirmation
        macd = ta.macd(dataframe["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            dataframe["macd_hist"] = macd.iloc[:, 1]
        else:
            dataframe["macd_hist"] = 0.0

        # EMA cross detection
        dataframe["ema_cross_up"] = (
            (dataframe["ema_fast"] > dataframe["ema_slow"]) &
            (dataframe["ema_fast"].shift(1) <= dataframe["ema_slow"].shift(1))
        )
        dataframe["ema_cross_down"] = (
            (dataframe["ema_fast"] < dataframe["ema_slow"]) &
            (dataframe["ema_fast"].shift(1) >= dataframe["ema_slow"].shift(1))
        )

        # "Recent" cross (within last 3 bars) — don't need to enter ON the exact cross bar
        dataframe["recent_cross_up"] = dataframe["ema_cross_up"].rolling(3).max().fillna(0).astype(bool)
        dataframe["recent_cross_down"] = dataframe["ema_cross_down"].rolling(3).max().fillna(0).astype(bool)

        return dataframe

    # ------------------------------------------------------------------ #
    #  ENTRY                                                               #
    # ------------------------------------------------------------------ #
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        adx_ok = dataframe["adx"] >= self.ADX_MIN.value
        vol_ok = dataframe["vol_ratio"] >= self.VOL_MULT.value

        # LONG: recent EMA cross up + trend confirmed + momentum + volume
        long_signal = (
            dataframe["recent_cross_up"] &
            (dataframe["ema_fast"] > dataframe["ema_slow"]) &  # still in uptrend
            (dataframe["plus_di"] > dataframe["minus_di"]) &  # DI confirms
            adx_ok &
            (dataframe["rsi"] >= self.RSI_LOW.value) &
            (dataframe["rsi"] <= self.RSI_HIGH.value) &
            (dataframe["macd_hist"] > 0) &
            vol_ok &
            (dataframe["close"] > dataframe["ema_fast"])  # price above fast EMA
        )

        # SHORT: recent EMA cross down + trend confirmed + momentum + volume
        rsi_short_high = 100 - self.RSI_LOW.value  # mirror
        rsi_short_low = 100 - self.RSI_HIGH.value
        short_signal = (
            dataframe["recent_cross_down"] &
            (dataframe["ema_fast"] < dataframe["ema_slow"]) &
            (dataframe["minus_di"] > dataframe["plus_di"]) &
            adx_ok &
            (dataframe["rsi"] >= rsi_short_low) &
            (dataframe["rsi"] <= rsi_short_high) &
            (dataframe["macd_hist"] < 0) &
            vol_ok &
            (dataframe["close"] < dataframe["ema_fast"])
        )

        dataframe.loc[long_signal, "enter_long"] = 1
        dataframe.loc[long_signal, "enter_tag"] = "ema_mom_long"
        dataframe.loc[short_signal, "enter_short"] = 1
        dataframe.loc[short_signal, "enter_tag"] = "ema_mom_short"

        return dataframe

    # ------------------------------------------------------------------ #
    #  EXIT (BB-mid target)                                                #
    # ------------------------------------------------------------------ #
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exit long: EMA fast crosses below slow (trend over)
        dataframe.loc[
            dataframe["ema_cross_down"],
            "exit_long"
        ] = 1

        # Exit short: EMA fast crosses above slow (trend over)
        dataframe.loc[
            dataframe["ema_cross_up"],
            "exit_short"
        ] = 1

        return dataframe

    # ------------------------------------------------------------------ #
    #  CUSTOM STOPLOSS                                                     #
    # ------------------------------------------------------------------ #
    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return self.stoploss

        last_atr = float(df["atr"].iat[-1])
        if last_atr <= 0 or current_rate <= 0:
            return self.stoploss

        # Break-even after BE_ATR profit
        be_distance = self.BE_ATR_MULT.value * last_atr / trade.open_rate
        if current_profit >= be_distance:
            return -0.001

        sl_distance = self.SL_ATR_MULT.value * last_atr / current_rate
        return max(-sl_distance, -0.04)

    # ------------------------------------------------------------------ #
    #  CUSTOM EXIT (TP)                                                    #
    # ------------------------------------------------------------------ #
    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return None

        last_atr = float(df["atr"].iat[-1])
        if last_atr <= 0 or trade.open_rate <= 0:
            return None

        tp_ratio = self.TP_ATR_MULT.value * last_atr / trade.open_rate
        if current_profit >= tp_ratio:
            return "TP_HIT"

        return None

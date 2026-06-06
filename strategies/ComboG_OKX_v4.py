# -*- coding: utf-8 -*-
"""
ComboG_OKX_v4 - Adaptive Pullback Strategy
============================================
Key insight: Don't fight the trend. Trade pullbacks WITH the trend direction.

Logic:
- 15m EMA trend determines direction (long only in uptrend, short only in downtrend)
- 5m RSI/BB oversold/overbought = pullback entry timing
- ATR-based TP/SL with realistic ratios
- Fast time-based exit (scalp approach)
- ADX filter: need SOME trend (not dead range) but not too strong (whipsaw)

This is "mean-reversion within trend" — the safest scalping approach.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta

from freqtrade.strategy import IStrategy, merge_informative_pair, DecimalParameter, IntParameter


class ComboG_OKX_v4(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    inf_timeframe = "15m"

    stoploss = -0.03
    minimal_roi = {"0": 0.08}

    can_short = True
    use_custom_stoploss = True

    protections = [
        {"method": "CooldownPeriod", "stop_duration_candles": 6},
        {"method": "StoplossGuard", "lookback_period_candles": 48,
         "trade_limit": 3, "stop_duration_candles": 48, "only_per_pair": True},
    ]

    use_exit_signal = False  # rely on TP/SL/time only
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # --- Risk ---
    SL_ATR_MULT = DecimalParameter(1.0, 3.0, default=1.8, decimals=1, space="sell", optimize=True)
    TP_ATR_MULT = DecimalParameter(1.5, 4.0, default=2.5, decimals=1, space="sell", optimize=True)

    # --- Entry ---
    RSI_BUY = IntParameter(20, 40, default=33, space="buy", optimize=True)
    RSI_SELL = IntParameter(60, 80, default=67, space="buy", optimize=True)
    ADX_MIN = IntParameter(12, 25, default=18, space="buy", optimize=True)
    ADX_MAX = IntParameter(35, 60, default=50, space="buy", optimize=True)
    VOL_MIN = DecimalParameter(0.6, 1.8, default=1.0, decimals=1, space="buy", optimize=True)
    EMA_FAST = IntParameter(8, 21, default=12, space="buy", optimize=False)
    EMA_SLOW = IntParameter(30, 60, default=50, space="buy", optimize=False)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        inf = [(pair, self.inf_timeframe) for pair in pairs]
        if ("BTC/USDT:USDT", "1h") not in inf:
            inf.append(("BTC/USDT:USDT", "1h"))
        return inf

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # --- 15m data for trend ---
        dataframe = self._add_informative(dataframe, metadata)

        # --- ATR ---
        dataframe["atr"] = ta.atr(dataframe["high"], dataframe["low"], dataframe["close"], length=14)

        # --- RSI ---
        dataframe["rsi"] = ta.rsi(dataframe["close"], length=14)

        # --- ADX ---
        adx_df = ta.adx(dataframe["high"], dataframe["low"], dataframe["close"], length=14)
        if adx_df is not None:
            dataframe["adx"] = adx_df.iloc[:, 0]
        else:
            dataframe["adx"] = 25.0

        # --- Bollinger Bands ---
        bb = ta.bbands(dataframe["close"], length=20, std=2.0)
        if bb is not None:
            dataframe["bb_upper"] = bb.iloc[:, 2]
            dataframe["bb_mid"] = bb.iloc[:, 1]
            dataframe["bb_lower"] = bb.iloc[:, 0]
        else:
            dataframe["bb_upper"] = dataframe["close"]
            dataframe["bb_mid"] = dataframe["close"]
            dataframe["bb_lower"] = dataframe["close"]

        # --- Volume ---
        dataframe["vol_ema"] = ta.ema(dataframe["volume"].astype(float), length=20)
        dataframe["vol_ratio"] = dataframe["volume"].astype(float) / (dataframe["vol_ema"] + 1e-10)

        # --- 5m EMA for micro-trend ---
        dataframe["ema_fast"] = ta.ema(dataframe["close"], length=self.EMA_FAST.value)
        dataframe["ema_slow"] = ta.ema(dataframe["close"], length=self.EMA_SLOW.value)

        # --- BTC sentiment ---
        dataframe = self._add_btc_sentiment(dataframe, metadata)

        return dataframe

    def _add_informative(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        inf_df = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_timeframe)
        if inf_df.empty:
            dataframe["ema_fast_15m"] = dataframe["close"]
            dataframe["ema_slow_15m"] = dataframe["close"]
            dataframe["rsi_15m"] = 50.0
            return dataframe

        inf_df["ema_fast_15m"] = ta.ema(inf_df["close"], length=12)
        inf_df["ema_slow_15m"] = ta.ema(inf_df["close"], length=50)
        inf_df["rsi_15m"] = ta.rsi(inf_df["close"], length=14)

        dataframe = merge_informative_pair(dataframe, inf_df, self.timeframe, self.inf_timeframe, ffill=True)
        return dataframe

    def _add_btc_sentiment(self, dataframe, metadata):
        if self.dp is None:
            dataframe["btc_rsi_1h"] = 50.0
            return dataframe
        btc_df = self.dp.get_pair_dataframe(pair="BTC/USDT:USDT", timeframe="1h")
        if btc_df.empty:
            dataframe["btc_rsi_1h"] = 50.0
            return dataframe
        btc_df = btc_df.copy()
        btc_df["btc_rsi"] = ta.rsi(btc_df["close"], length=14)
        btc_df = btc_df[["date", "btc_rsi"]].copy()
        dataframe = merge_informative_pair(dataframe, btc_df, self.timeframe, "1h", ffill=True)
        if "btc_rsi_1h" not in dataframe.columns:
            dataframe["btc_rsi_1h"] = 50.0
        return dataframe

    # ------------------------------------------------------------------ #
    #  ENTRY: Pullback-in-trend                                            #
    # ------------------------------------------------------------------ #
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # --- Trend direction from 15m EMA ---
        ema_f_15m = "ema_fast_15m_15m" if "ema_fast_15m_15m" in dataframe.columns else "ema_fast_15m"
        ema_s_15m = "ema_slow_15m_15m" if "ema_slow_15m_15m" in dataframe.columns else "ema_slow_15m"

        trend_up = dataframe[ema_f_15m] > dataframe[ema_s_15m]
        trend_down = dataframe[ema_f_15m] < dataframe[ema_s_15m]

        # --- ADX filter: moderate trend (not dead, not parabolic) ---
        adx_ok = (dataframe["adx"] >= self.ADX_MIN.value) & (dataframe["adx"] <= self.ADX_MAX.value)

        # --- Volume gate ---
        vol_ok = dataframe["vol_ratio"] >= self.VOL_MIN.value

        # --- LONG: Uptrend + RSI pullback to oversold + price near BB lower ---
        long_entry = (
            trend_up &
            (dataframe["rsi"] <= self.RSI_BUY.value) &
            (dataframe["low"] <= dataframe["bb_lower"]) &  # touched BB lower
            adx_ok &
            vol_ok &
            (dataframe["close"] > dataframe["low"])  # not closing at low (recovery candle)
        )

        dataframe.loc[long_entry, "enter_long"] = 1
        dataframe.loc[long_entry, "enter_tag"] = "pullback_long"

        # --- SHORT: Downtrend + RSI pullback to overbought + price near BB upper ---
        short_entry = (
            trend_down &
            (dataframe["rsi"] >= self.RSI_SELL.value) &
            (dataframe["high"] >= dataframe["bb_upper"]) &  # touched BB upper
            adx_ok &
            vol_ok &
            (dataframe["close"] < dataframe["high"])  # not closing at high (rejection candle)
        )

        dataframe.loc[short_entry, "enter_short"] = 1
        dataframe.loc[short_entry, "enter_tag"] = "pullback_short"

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    # ------------------------------------------------------------------ #
    #  STOPLOSS & EXIT                                                     #
    # ------------------------------------------------------------------ #
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return self.stoploss

        atr = float(df["atr"].iat[-1])
        if atr <= 0 or trade.open_rate <= 0:
            return self.stoploss

        sl_pct = self.SL_ATR_MULT.value * atr / trade.open_rate
        tp_pct = self.TP_ATR_MULT.value * atr / trade.open_rate

        # Trail: once at 60% of TP, move SL to break-even
        if current_profit >= tp_pct * 0.6:
            return max(-0.002, -(trade.fee_open + trade.fee_close))

        # Trail: once at 30% of TP, tighten SL to 50% of original
        if current_profit >= tp_pct * 0.3:
            return max(-sl_pct * 0.5, -0.02)

        return max(-sl_pct, -0.04)

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if not df.empty:
            atr = float(df["atr"].iat[-1])
            if atr > 0 and trade.open_rate > 0:
                tp_pct = self.TP_ATR_MULT.value * atr / trade.open_rate
                if current_profit >= tp_pct:
                    return "TP_HIT"

        # Time-based exits (pullback scalp)
        hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        if hours >= 2 and current_profit < -0.005:
            return "time_cut_2h"
        if hours >= 4 and current_profit < 0.002:
            return "time_cut_4h"
        if hours >= 8:
            return "time_cut_8h"

        return None

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return False
        last = df.iloc[-1]
        if float(last.get("atr", 0)) <= 0:
            return False

        # BTC sentiment: don't long when BTC crashing, don't short when pumping
        btc_rsi = float(last.get("btc_rsi_1h", 50))
        if side == "long" and btc_rsi < 30:
            return False
        if side == "short" and btc_rsi > 70:
            return False

        return True

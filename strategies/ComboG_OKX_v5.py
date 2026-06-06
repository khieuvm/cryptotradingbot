# -*- coding: utf-8 -*-
"""
ComboG_OKX_v5 - 15m Pullback Scalper (final redesign)
======================================================
Key insight from backtesting:
- 5m entries = too noisy, SL gets hit by random noise
- 15m entries = fewer but more reliable signals
- SHORTS work well in this market (clear downtrend)
- TP must be reachable within 4-8 candles (1-2 hours on 15m)

Strategy:
- Timeframe: 15m (not 5m!)
- Entry: RSI extreme + BB touch + volume spike in TREND direction
- Trend: EMA 20/50 on 15m (clear direction, less whipsaw)
- Exit: ATR-TP + BB-mid reversion + time-based
- BTC 1h RSI as macro filter
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta

from freqtrade.strategy import IStrategy, merge_informative_pair, DecimalParameter, IntParameter


class ComboG_OKX_v5(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"  # HIGHER timeframe = less noise

    stoploss = -0.03
    minimal_roi = {"0": 0.05, "60": 0.02, "120": 0.01}  # time-based ROI

    can_short = True
    use_custom_stoploss = True

    protections = [
        {"method": "CooldownPeriod", "stop_duration_candles": 4},
        {"method": "StoplossGuard", "lookback_period_candles": 24,
         "trade_limit": 2, "stop_duration_candles": 24, "only_per_pair": True},
    ]

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # --- Risk (in terms of 15m ATR — larger than 5m ATR) ---
    SL_ATR_MULT = DecimalParameter(1.0, 2.5, default=1.5, decimals=1, space="sell", optimize=True)
    TP_ATR_MULT = DecimalParameter(1.5, 3.5, default=2.2, decimals=1, space="sell", optimize=True)

    # --- Entry filters ---
    RSI_BUY = IntParameter(20, 38, default=30, space="buy", optimize=True)
    RSI_SELL = IntParameter(62, 80, default=70, space="buy", optimize=True)
    BB_PERIOD = IntParameter(14, 25, default=20, space="buy", optimize=False)
    EMA_FAST_LEN = IntParameter(10, 25, default=20, space="buy", optimize=False)
    EMA_SLOW_LEN = IntParameter(40, 60, default=50, space="buy", optimize=False)
    VOL_MULT = DecimalParameter(0.8, 2.0, default=1.2, decimals=1, space="buy", optimize=True)

    def informative_pairs(self):
        return [("BTC/USDT:USDT", "1h")]

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # --- 1h informative for BTC ---
        dataframe = self._add_btc_1h(dataframe, metadata)
        # --- 1h pair for macro trend ---
        dataframe = self._add_1h_trend(dataframe, metadata)

        # --- ATR (15m) ---
        dataframe["atr"] = ta.atr(dataframe["high"], dataframe["low"], dataframe["close"], length=14)

        # --- RSI ---
        dataframe["rsi"] = ta.rsi(dataframe["close"], length=14)

        # --- Bollinger Bands ---
        bb = ta.bbands(dataframe["close"], length=self.BB_PERIOD.value, std=2.0)
        if bb is not None:
            dataframe["bb_upper"] = bb.iloc[:, 2]
            dataframe["bb_mid"] = bb.iloc[:, 1]
            dataframe["bb_lower"] = bb.iloc[:, 0]
        else:
            dataframe["bb_upper"] = dataframe["close"]
            dataframe["bb_mid"] = dataframe["close"]
            dataframe["bb_lower"] = dataframe["close"]

        # --- EMA trend (on 15m directly) ---
        dataframe["ema_fast"] = ta.ema(dataframe["close"], length=self.EMA_FAST_LEN.value)
        dataframe["ema_slow"] = ta.ema(dataframe["close"], length=self.EMA_SLOW_LEN.value)

        # --- Volume ---
        dataframe["vol_ema"] = ta.ema(dataframe["volume"].astype(float), length=20)
        dataframe["vol_ratio"] = dataframe["volume"].astype(float) / (dataframe["vol_ema"] + 1e-10)

        # --- Stochastic RSI for timing ---
        stoch = ta.stoch(dataframe["high"], dataframe["low"], dataframe["close"], k=14, d=3)
        if stoch is not None:
            dataframe["stoch_k"] = stoch.iloc[:, 0]
        else:
            dataframe["stoch_k"] = 50.0

        return dataframe

    def _add_btc_1h(self, dataframe, metadata):
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

    def _add_1h_trend(self, dataframe, metadata):
        """Placeholder - using 15m EMA100 instead of 1h data."""
        return dataframe

    # ------------------------------------------------------------------ #
    #  ENTRY                                                               #
    # ------------------------------------------------------------------ #
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # --- Trend from 15m EMA ---
        trend_up_15m = dataframe["ema_fast"] > dataframe["ema_slow"]
        trend_down_15m = dataframe["ema_fast"] < dataframe["ema_slow"]

        # --- Use longer 15m EMA as macro direction (EMA50 vs EMA100) ---
        ema100 = ta.ema(dataframe["close"], length=100)
        macro_up = dataframe["ema_slow"] > ema100  # EMA50 > EMA100
        macro_down = dataframe["ema_slow"] < ema100

        # --- Volume spike ---
        vol_ok = dataframe["vol_ratio"] >= self.VOL_MULT.value

        # --- LONG: Uptrend on 15m, RSI oversold pullback, BB lower touch ---
        long_signal = (
            trend_up_15m &
            macro_up &
            (dataframe["rsi"] <= self.RSI_BUY.value) &
            (dataframe["low"] <= dataframe["bb_lower"]) &
            vol_ok
        )
        dataframe.loc[long_signal, "enter_long"] = 1
        dataframe.loc[long_signal, "enter_tag"] = "pb_long"

        # --- SHORT: Downtrend on 15m, RSI overbought bounce, BB upper touch ---
        short_signal = (
            trend_down_15m &
            macro_down &
            (dataframe["rsi"] >= self.RSI_SELL.value) &
            (dataframe["high"] >= dataframe["bb_upper"]) &
            vol_ok
        )
        dataframe.loc[short_signal, "enter_short"] = 1
        dataframe.loc[short_signal, "enter_tag"] = "pb_short"

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exit when price reaches BB mid (mean reversion target)
        dataframe["exit_long"] = (
            (dataframe["close"] >= dataframe["bb_mid"]) &
            (dataframe["rsi"] > 50)
        ).astype(int)

        dataframe["exit_short"] = (
            (dataframe["close"] <= dataframe["bb_mid"]) &
            (dataframe["rsi"] < 50)
        ).astype(int)

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

        # Break-even when at 50% of TP
        if current_profit >= tp_pct * 0.5:
            return max(-0.003, -(trade.fee_open + trade.fee_close + 0.001))

        return max(-sl_pct, -0.04)

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if not df.empty:
            atr = float(df["atr"].iat[-1])
            if atr > 0 and trade.open_rate > 0:
                tp_pct = self.TP_ATR_MULT.value * atr / trade.open_rate
                if current_profit >= tp_pct:
                    return "TP_HIT"

        # Time cuts (15m candles)
        hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        if hours >= 3 and current_profit < -0.005:
            return "time_cut_3h"
        if hours >= 6 and current_profit < 0.002:
            return "time_cut_6h"
        if hours >= 12:
            return "time_cut_12h"

        return None

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return False
        last = df.iloc[-1]
        if float(last.get("atr", 0)) <= 0:
            return False

        btc_rsi = float(last.get("btc_rsi_1h", 50))
        if side == "long" and btc_rsi < 30:
            return False
        if side == "short" and btc_rsi > 70:
            return False
        return True

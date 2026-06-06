# -*- coding: utf-8 -*-
"""
ComboG_OKX_v3 - BB Mean-Reversion Scalper (redesigned from scratch)
=====================================================================
Root cause fixes:
1. SIMPLE entry: BB touch + RSI extreme + volume spike (proven mean-rev combo)
2. REALISTIC TP: 1.5-2.5x ATR (reachable in 30-90 minutes)
3. TIGHT SL: 1.2-2.0x ATR
4. TIME FILTER: Only trade when market is ranging (ADX low)
5. NO paradoxical trend filter — mean-rev needs RANGE, not trend
6. 15m RSI divergence as quality gate (not as primary trigger)
7. Quick time-cuts: mean-rev either works in 1-2 hours or fails

Approach: Price touches BB band in low-ADX environment → scalp back to BB mid
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta

from freqtrade.strategy import IStrategy, merge_informative_pair, DecimalParameter, IntParameter


class ComboG_OKX_v3(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    inf_timeframe = "15m"

    stoploss = -0.03
    minimal_roi = {"0": 0.10}

    can_short = True
    use_custom_stoploss = True

    protections = [
        {"method": "CooldownPeriod", "stop_duration_candles": 8},
        {"method": "StoplossGuard", "lookback_period_candles": 48,
         "trade_limit": 3, "stop_duration_candles": 48, "only_per_pair": True},
        {"method": "MaxDrawdown", "lookback_period_candles": 96,
         "max_allowed_drawdown": 0.10, "stop_duration_candles": 96, "trade_limit": 5},
    ]

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # --- Risk params ---
    SL_ATR_MULT = DecimalParameter(1.0, 2.5, default=1.5, decimals=1, space="sell", optimize=True)
    TP_ATR_MULT = DecimalParameter(1.5, 3.5, default=2.2, decimals=1, space="sell", optimize=True)

    # --- Entry params ---
    RSI_OVERSOLD = IntParameter(15, 35, default=25, space="buy", optimize=True)
    RSI_OVERBOUGHT = IntParameter(65, 85, default=75, space="buy", optimize=True)
    ADX_MAX = IntParameter(20, 40, default=28, space="buy", optimize=True)
    VOL_SPIKE = DecimalParameter(1.0, 2.5, default=1.5, decimals=1, space="buy", optimize=True)
    BB_SQUEEZE_P = DecimalParameter(0.005, 0.03, default=0.015, decimals=3, space="buy", optimize=True)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        inf = [(pair, self.inf_timeframe) for pair in pairs]
        if ("BTC/USDT:USDT", "1h") not in inf:
            inf.append(("BTC/USDT:USDT", "1h"))
        return inf

    # ------------------------------------------------------------------ #
    #  INDICATORS                                                          #
    # ------------------------------------------------------------------ #
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # --- 15m informative ---
        dataframe = self._add_informative(dataframe, metadata)

        # --- ATR ---
        dataframe["atr"] = ta.atr(dataframe["high"], dataframe["low"], dataframe["close"], length=14)

        # --- RSI (14-period, standard) ---
        dataframe["rsi"] = ta.rsi(dataframe["close"], length=14)

        # --- ADX (trend strength - low = ranging) ---
        adx_df = ta.adx(dataframe["high"], dataframe["low"], dataframe["close"], length=14)
        if adx_df is not None:
            dataframe["adx"] = adx_df.iloc[:, 0]  # ADX
            dataframe["di_plus"] = adx_df.iloc[:, 1]  # +DI
            dataframe["di_minus"] = adx_df.iloc[:, 2]  # -DI
        else:
            dataframe["adx"] = 25.0
            dataframe["di_plus"] = 12.5
            dataframe["di_minus"] = 12.5

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

        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / (dataframe["bb_mid"] + 1e-10)

        # --- Volume ratio ---
        dataframe["vol_ema"] = ta.ema(dataframe["volume"].astype(float), length=20)
        dataframe["vol_ratio"] = dataframe["volume"].astype(float) / (dataframe["vol_ema"] + 1e-10)

        # --- Candle patterns (reversal confirmation) ---
        body = abs(dataframe["close"] - dataframe["open"])
        wick_lower = dataframe[["open", "close"]].min(axis=1) - dataframe["low"]
        wick_upper = dataframe["high"] - dataframe[["open", "close"]].max(axis=1)
        candle_range = dataframe["high"] - dataframe["low"] + 1e-10

        dataframe["hammer"] = ((wick_lower > 2 * body) & (body > 0)).astype(int)
        dataframe["inverted_hammer"] = ((wick_upper > 2 * body) & (body > 0)).astype(int)
        dataframe["bullish_candle"] = (dataframe["close"] > dataframe["open"]).astype(int)
        dataframe["bearish_candle"] = (dataframe["close"] < dataframe["open"]).astype(int)

        # --- EMA for BB-mid target ---
        dataframe["ema8"] = ta.ema(dataframe["close"], length=8)

        # --- BTC sentiment ---
        dataframe = self._add_btc_sentiment(dataframe, metadata)

        return dataframe

    def _add_informative(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        inf_df = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_timeframe)
        if inf_df.empty:
            dataframe["rsi_15m"] = 50.0
            dataframe["adx_15m"] = 25.0
            return dataframe

        inf_df["rsi_15m"] = ta.rsi(inf_df["close"], length=14)
        adx_15 = ta.adx(inf_df["high"], inf_df["low"], inf_df["close"], length=14)
        inf_df["adx_15m"] = adx_15.iloc[:, 0] if adx_15 is not None else 25.0

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
    #  ENTRY SIGNALS                                                       #
    # ------------------------------------------------------------------ #
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # --- Common gates ---
        range_market = dataframe["adx"] < self.ADX_MAX.value  # LOW ADX = range
        vol_ok = dataframe["vol_ratio"] >= self.VOL_SPIKE.value  # Volume spike = conviction

        # 15m ADX confirmation (not trending on higher TF either)
        adx_15m_col = "adx_15m_15m" if "adx_15m_15m" in dataframe.columns else "adx_15m"
        rsi_15m_col = "rsi_15m_15m" if "rsi_15m_15m" in dataframe.columns else "rsi_15m"
        adx_15m_ok = dataframe.get(adx_15m_col, pd.Series(25, index=dataframe.index)) < (self.ADX_MAX.value + 5)

        # --- LONG: Price touches lower BB + RSI oversold ---
        bb_touch_lower = dataframe["low"] <= dataframe["bb_lower"]
        rsi_oversold = dataframe["rsi"] <= self.RSI_OVERSOLD.value
        # Reversal confirmation: bullish candle OR hammer
        reversal_bull = (dataframe["bullish_candle"] == 1) | (dataframe["hammer"] == 1)
        # 15m RSI not also crashing (not catching falling knife)
        rsi_15m_not_crash = dataframe.get(rsi_15m_col, pd.Series(50, index=dataframe.index)) > 20

        long_sig = (
            bb_touch_lower &
            rsi_oversold &
            reversal_bull &
            range_market &
            vol_ok &
            adx_15m_ok &
            rsi_15m_not_crash
        )

        dataframe.loc[long_sig, "enter_long"] = 1
        dataframe.loc[long_sig, "enter_tag"] = "bb_rev_long"

        # --- SHORT: Price touches upper BB + RSI overbought ---
        bb_touch_upper = dataframe["high"] >= dataframe["bb_upper"]
        rsi_overbought = dataframe["rsi"] >= self.RSI_OVERBOUGHT.value
        reversal_bear = (dataframe["bearish_candle"] == 1) | (dataframe["inverted_hammer"] == 1)
        rsi_15m_not_pump = dataframe.get(rsi_15m_col, pd.Series(50, index=dataframe.index)) < 80

        short_sig = (
            bb_touch_upper &
            rsi_overbought &
            reversal_bear &
            range_market &
            vol_ok &
            adx_15m_ok &
            rsi_15m_not_pump
        )

        dataframe.loc[short_sig, "enter_short"] = 1
        dataframe.loc[short_sig, "enter_tag"] = "bb_rev_short"

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exit via BB-mid reversion + custom_exit
        # Signal-based exit: price crosses back to BB mid
        dataframe["exit_long"] = (
            (dataframe["close"] >= dataframe["bb_mid"]) &
            (dataframe["close"].shift(1) < dataframe["bb_mid"].shift(1))
        ).astype(int)

        dataframe["exit_short"] = (
            (dataframe["close"] <= dataframe["bb_mid"]) &
            (dataframe["close"].shift(1) > dataframe["bb_mid"].shift(1))
        ).astype(int)

        return dataframe

    # ------------------------------------------------------------------ #
    #  CUSTOM STOPLOSS                                                     #
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
            return max(-0.003, -(trade.fee_open + trade.fee_close))

        return max(-sl_pct, -0.04)

    # ------------------------------------------------------------------ #
    #  CUSTOM EXIT                                                         #
    # ------------------------------------------------------------------ #
    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if not df.empty:
            atr = float(df["atr"].iat[-1])
            if atr > 0 and trade.open_rate > 0:
                tp_pct = self.TP_ATR_MULT.value * atr / trade.open_rate
                if current_profit >= tp_pct:
                    return "TP_HIT"

        # Mean-reversion must work FAST — aggressive time cuts
        hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        if hours >= 1.5 and current_profit < -0.005:
            return "time_cut_90m"
        if hours >= 3 and current_profit < 0.001:
            return "time_cut_3h"
        if hours >= 6:
            return "time_cut_6h"  # force close — mean-rev failed

        return None

    # ------------------------------------------------------------------ #
    #  CONFIRM ENTRY                                                       #
    # ------------------------------------------------------------------ #
    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return False

        last = df.iloc[-1]
        if float(last.get("atr", 0)) <= 0:
            return False

        # BTC sentiment
        btc_rsi = float(last.get("btc_rsi_1h", 50))
        if side == "long" and btc_rsi < 25:
            return False
        if side == "short" and btc_rsi > 75:
            return False

        return True

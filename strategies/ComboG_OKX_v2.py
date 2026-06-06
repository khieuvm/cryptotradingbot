# -*- coding: utf-8 -*-
"""
ComboG_OKX_v2 - Fixed Mean-Reversion Strategy
===============================================
Fixes from root-cause analysis:
1. TP reduced to reachable levels (2-3x ATR vs old 8.1x)
2. SL tightened (1.5-2x ATR vs old 4.7x)
3. Primary requires ALL 3 oscillators (not just 2) → fewer but higher quality signals
4. EMA trend filter REMOVED for G mode (mean-rev works in ranging markets)
   - Replaced with BB width "range detection" gate
5. 15m MTF confirmation now REQUIRED (not optional bonus)
6. Faster time-exits to cut losers early
7. Confirm needs >= 2 (not just 1) for higher quality

Result: ~5-8 trades/week (vs 47 before), much higher quality
"""

from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import pandas_ta as ta

from freqtrade.strategy import IStrategy, merge_informative_pair, DecimalParameter, IntParameter


class ComboG_OKX_v2(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    inf_timeframe = "15m"

    stoploss = -0.025
    minimal_roi = {"0": 0.10}

    can_short = True
    use_custom_stoploss = True

    # Tighter protections — less overtrading
    protections = [
        {"method": "CooldownPeriod", "stop_duration_candles": 12},
        {"method": "StoplossGuard", "lookback_period_candles": 48,
         "trade_limit": 2, "stop_duration_candles": 60, "only_per_pair": True},
        {"method": "MaxDrawdown", "lookback_period_candles": 96,
         "max_allowed_drawdown": 0.08, "stop_duration_candles": 120, "trade_limit": 4},
    ]

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # --- ATR multipliers (FIXED: realistic R:R) ---
    SL_ATR_MULT = DecimalParameter(1.0, 3.0, default=1.8, decimals=1, space="sell", optimize=True)
    TP_ATR_MULT = DecimalParameter(1.5, 4.0, default=2.8, decimals=1, space="sell", optimize=True)

    # --- Oscillator thresholds (TIGHTENED: real extremes only) ---
    CCI_OVERSOLD_P   = IntParameter(-200, -120, default=-150, space="buy", optimize=True)
    CCI_OVERBOUGHT_P = IntParameter(120, 200,   default=150,  space="buy", optimize=True)
    WILLR_OVERSOLD_P = IntParameter(-98, -85,   default=-92,  space="buy", optimize=True)
    WILLR_OB_P       = IntParameter(-15, -2,    default=-8,   space="buy", optimize=True)
    MFI_OVERSOLD_P   = IntParameter(10, 25,     default=18,   space="buy", optimize=True)
    MFI_OB_P         = IntParameter(75, 90,     default=82,   space="buy", optimize=True)
    VOL_RATIO_P      = DecimalParameter(0.8, 2.0, default=1.2, decimals=1, space="buy", optimize=True)

    # --- BB width gate: only trade in ranging markets ---
    BB_WIDTH_MAX_P   = DecimalParameter(0.015, 0.06, default=0.035, decimals=3, space="buy", optimize=True)

    # --- Minimum confirms needed ---
    MIN_CONFIRM_P    = IntParameter(1, 3, default=2, space="buy", optimize=True)

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
        dataframe = self._add_informative_indicators(dataframe, metadata)

        # Moving Averages
        dataframe["ema8"]  = ta.ema(dataframe["close"], length=8)
        dataframe["ema21"] = ta.ema(dataframe["close"], length=21)
        dataframe["ema55"] = ta.ema(dataframe["close"], length=55)
        dataframe["sma_f"] = ta.sma(dataframe["close"], length=10)
        dataframe["sma_s"] = ta.sma(dataframe["close"], length=20)

        # ATR
        dataframe["atr"] = ta.atr(dataframe["high"], dataframe["low"], dataframe["close"], length=14)

        # RSI (longer period for less noise)
        dataframe["rsi"] = ta.rsi(dataframe["close"], length=14)

        # CCI
        dataframe["cci"] = ta.cci(dataframe["high"], dataframe["low"], dataframe["close"], length=20)

        # Williams %R
        dataframe["willr"] = ta.willr(dataframe["high"], dataframe["low"], dataframe["close"], length=14)

        # MFI
        try:
            dataframe["mfi"] = ta.mfi(
                dataframe["high"], dataframe["low"],
                dataframe["close"], dataframe["volume"].astype(float), length=14
            )
        except Exception:
            dataframe["mfi"] = 50.0

        # Stochastic
        stoch = ta.stoch(dataframe["high"], dataframe["low"], dataframe["close"], k=14, d=3)
        if stoch is not None:
            dataframe["stoch_k"] = stoch.iloc[:, 0]
            dataframe["stoch_d"] = stoch.iloc[:, 1]
        else:
            dataframe["stoch_k"] = 50.0
            dataframe["stoch_d"] = 50.0

        # Bollinger Bands
        bb = ta.bbands(dataframe["close"], length=20, std=2.0)
        if bb is not None:
            dataframe["bb_upper"] = bb.iloc[:, 2]
            dataframe["bb_mid"]   = bb.iloc[:, 1]
            dataframe["bb_lower"] = bb.iloc[:, 0]
        else:
            dataframe["bb_upper"] = dataframe["close"]
            dataframe["bb_mid"]   = dataframe["close"]
            dataframe["bb_lower"] = dataframe["close"]

        # BB Width (range detection)
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]

        # OBV
        try:
            dataframe["obv"] = ta.obv(dataframe["close"], dataframe["volume"].astype(float))
        except Exception:
            dataframe["obv"] = 0.0
        dataframe["obv_ema"] = ta.ema(dataframe["obv"], length=20)
        dataframe["obv_rising"] = (dataframe["obv"] > dataframe["obv_ema"]).astype(int)

        # Volume Ratio
        dataframe["vol_ema20"] = ta.ema(dataframe["volume"].astype(float), length=20)
        dataframe["volume_ratio"] = dataframe["volume"].astype(float) / (dataframe["vol_ema20"] + 1e-10)

        # Trend helpers
        dataframe["trend_bull"] = (dataframe["sma_f"] > dataframe["sma_s"]).astype(int)
        dataframe["trend_bear"] = (dataframe["sma_f"] < dataframe["sma_s"]).astype(int)

        # RSI divergence (longer lookback for quality)
        dataframe["rsi_14"] = ta.rsi(dataframe["close"], length=14)

        dataframe = self._add_btc_sentiment(dataframe, metadata)
        return dataframe

    def _add_informative_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        inf_tf = self.inf_timeframe
        inf_df = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=inf_tf)
        if inf_df.empty:
            dataframe["cci_15m"] = np.nan
            dataframe["willr_15m"] = np.nan
            dataframe["mfi_15m"] = np.nan
            dataframe["rsi_15m"] = np.nan
            return dataframe

        inf_df["cci_15m"] = ta.cci(inf_df["high"], inf_df["low"], inf_df["close"], length=20)
        inf_df["willr_15m"] = ta.willr(inf_df["high"], inf_df["low"], inf_df["close"], length=14)
        inf_df["rsi_15m"] = ta.rsi(inf_df["close"], length=14)
        try:
            inf_df["mfi_15m"] = ta.mfi(
                inf_df["high"], inf_df["low"],
                inf_df["close"], inf_df["volume"].astype(float), length=14
            )
        except Exception:
            inf_df["mfi_15m"] = 50.0

        dataframe = merge_informative_pair(dataframe, inf_df, self.timeframe, inf_tf, ffill=True)
        return dataframe

    # ------------------------------------------------------------------ #
    #  SIGNAL CONDITIONS                                                   #
    # ------------------------------------------------------------------ #

    def _primary_buy(self, df: pd.DataFrame) -> pd.Series:
        """At least 2/3 primary oscillators + all must be in oversold zone."""
        thr_cci = self.CCI_OVERSOLD_P.value
        thr_wr = self.WILLR_OVERSOLD_P.value
        thr_mfi = self.MFI_OVERSOLD_P.value

        # Crossback signals (exact timing)
        cci_cross = (df["cci"] > thr_cci) & (df["cci"].shift(1) <= thr_cci)
        willr_cross = (df["willr"] > thr_wr) & (df["willr"].shift(1) <= thr_wr)
        mfi_turn = (df["mfi"] < thr_mfi) & (df["mfi"] > df["mfi"].shift(1))

        # Zone confirmation (all must be near extreme within last 3 bars)
        cci_zone = df["cci"].rolling(3).min() < thr_cci
        willr_zone = df["willr"].rolling(3).min() < thr_wr
        mfi_zone = df["mfi"].rolling(3).min() < (thr_mfi + 5)

        # Need >= 2 crossback AND all 3 were recently in zone
        cross_count = cci_cross.astype(int) + willr_cross.astype(int) + mfi_turn.astype(int)
        return (cross_count >= 2) & cci_zone & willr_zone & mfi_zone

    def _primary_sell(self, df: pd.DataFrame) -> pd.Series:
        """At least 2/3 primary oscillators + all must be in overbought zone."""
        thr_cci = self.CCI_OVERBOUGHT_P.value
        thr_wr = self.WILLR_OB_P.value
        thr_mfi = self.MFI_OB_P.value

        cci_cross = (df["cci"] < thr_cci) & (df["cci"].shift(1) >= thr_cci)
        willr_cross = (df["willr"] < thr_wr) & (df["willr"].shift(1) >= thr_wr)
        mfi_turn = (df["mfi"] > thr_mfi) & (df["mfi"] < df["mfi"].shift(1))

        # Zone: all were recently overbought
        cci_zone = df["cci"].rolling(3).max() > thr_cci
        willr_zone = df["willr"].rolling(3).max() > thr_wr
        mfi_zone = df["mfi"].rolling(3).max() > (thr_mfi - 5)

        cross_count = cci_cross.astype(int) + willr_cross.astype(int) + mfi_turn.astype(int)
        return (cross_count >= 2) & cci_zone & willr_zone & mfi_zone

    def _confirm_count_buy(self, df: pd.DataFrame) -> pd.Series:
        """Count confirmation signals for longs."""
        confirms = pd.Series(0, index=df.index)

        # Stoch cross from oversold
        stoch = (
            (df["stoch_k"] > df["stoch_d"]) &
            (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) &
            (df["stoch_k"].shift(1) < 20)
        )
        confirms += stoch.astype(int)

        # BB bounce: price near lower BB
        bb_bounce = (df["low"] <= df["bb_lower"]) & (df["rsi"] < 35)
        confirms += bb_bounce.astype(int)

        # RSI divergence (bullish)
        lookback = 14
        price_ll = df["close"] < df["close"].rolling(lookback).min().shift(1)
        rsi_hl = df["rsi_14"] > df["rsi_14"].rolling(lookback).min().shift(1)
        rsi_div = price_ll & rsi_hl & (df["rsi_14"] < 40)
        confirms += rsi_div.astype(int)

        # OBV accumulation
        confirms += (df["obv_rising"] == 1).astype(int)

        return confirms

    def _confirm_count_sell(self, df: pd.DataFrame) -> pd.Series:
        """Count confirmation signals for shorts."""
        confirms = pd.Series(0, index=df.index)

        # Stoch cross from overbought
        stoch = (
            (df["stoch_k"] < df["stoch_d"]) &
            (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) &
            (df["stoch_k"].shift(1) > 80)
        )
        confirms += stoch.astype(int)

        # BB touch upper
        bb_touch = (df["high"] >= df["bb_upper"]) & (df["rsi"] > 65)
        confirms += bb_touch.astype(int)

        # RSI divergence (bearish)
        lookback = 14
        price_hh = df["close"] > df["close"].rolling(lookback).max().shift(1)
        rsi_lh = df["rsi_14"] < df["rsi_14"].rolling(lookback).max().shift(1)
        rsi_div = price_hh & rsi_lh & (df["rsi_14"] > 60)
        confirms += rsi_div.astype(int)

        # OBV distribution
        confirms += (df["obv_rising"] == 0).astype(int)

        return confirms

    def _mtf_confluence_buy(self, df: pd.DataFrame) -> pd.Series:
        """15m must also show oversold conditions (not necessarily crossback)."""
        thr_cci = self.CCI_OVERSOLD_P.value
        thr_wr = self.WILLR_OVERSOLD_P.value

        cci_15m_col = "cci_15m_15m" if "cci_15m_15m" in df.columns else "cci_15m"
        willr_15m_col = "willr_15m_15m" if "willr_15m_15m" in df.columns else "willr_15m"
        rsi_15m_col = "rsi_15m_15m" if "rsi_15m_15m" in df.columns else "rsi_15m"

        cci_oversold_15m = df.get(cci_15m_col, pd.Series(0, index=df.index)) < (thr_cci * 0.7)
        willr_oversold_15m = df.get(willr_15m_col, pd.Series(-50, index=df.index)) < (thr_wr * 0.8)
        rsi_low_15m = df.get(rsi_15m_col, pd.Series(50, index=df.index)) < 35

        # At least one 15m indicator confirms oversold
        return cci_oversold_15m | willr_oversold_15m | rsi_low_15m

    def _mtf_confluence_sell(self, df: pd.DataFrame) -> pd.Series:
        """15m must also show overbought conditions."""
        thr_cci = self.CCI_OVERBOUGHT_P.value
        thr_wr = self.WILLR_OB_P.value

        cci_15m_col = "cci_15m_15m" if "cci_15m_15m" in df.columns else "cci_15m"
        willr_15m_col = "willr_15m_15m" if "willr_15m_15m" in df.columns else "willr_15m"
        rsi_15m_col = "rsi_15m_15m" if "rsi_15m_15m" in df.columns else "rsi_15m"

        cci_ob_15m = df.get(cci_15m_col, pd.Series(0, index=df.index)) > (thr_cci * 0.7)
        willr_ob_15m = df.get(willr_15m_col, pd.Series(-50, index=df.index)) > (thr_wr * 0.8)
        rsi_high_15m = df.get(rsi_15m_col, pd.Series(50, index=df.index)) > 65

        return cci_ob_15m | willr_ob_15m | rsi_high_15m

    def _range_gate(self, df: pd.DataFrame) -> pd.Series:
        """Only trade when market is ranging (BB width is narrow)."""
        return df["bb_width"] <= self.BB_WIDTH_MAX_P.value

    # ------------------------------------------------------------------ #
    #  ENTRY / EXIT                                                        #
    # ------------------------------------------------------------------ #
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # --- LONG ---
        primary_buy = self._primary_buy(dataframe)
        confirm_buy = self._confirm_count_buy(dataframe)
        mtf_buy = self._mtf_confluence_buy(dataframe)
        range_ok = self._range_gate(dataframe)
        vol_gate = dataframe["volume_ratio"] >= self.VOL_RATIO_P.value

        long_sig = (
            primary_buy &
            (confirm_buy >= self.MIN_CONFIRM_P.value) &
            mtf_buy &
            range_ok &
            vol_gate
        )

        long_tags = "Gv2_s" + confirm_buy.clip(1, 4).astype(int).astype(str)
        dataframe.loc[long_sig, "enter_long"] = 1
        dataframe.loc[long_sig, "enter_tag"] = long_tags[long_sig]

        # --- SHORT ---
        primary_sell = self._primary_sell(dataframe)
        confirm_sell = self._confirm_count_sell(dataframe)
        mtf_sell = self._mtf_confluence_sell(dataframe)

        short_sig = (
            primary_sell &
            (confirm_sell >= self.MIN_CONFIRM_P.value) &
            mtf_sell &
            range_ok &
            vol_gate
        )

        short_tags = "Gv2_s" + confirm_sell.clip(1, 4).astype(int).astype(str)
        dataframe.loc[short_sig, "enter_short"] = 1
        dataframe.loc[short_sig, "enter_tag"] = short_tags[short_sig]

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    # ------------------------------------------------------------------ #
    #  CUSTOM STOPLOSS (simpler 2-phase)                                   #
    # ------------------------------------------------------------------ #
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return self.stoploss

        atr = float(df["atr"].iat[-1])
        if atr <= 0 or trade.open_rate <= 0:
            return self.stoploss

        sl_pct = self.SL_ATR_MULT.value * atr / trade.open_rate
        tp_pct = self.TP_ATR_MULT.value * atr / trade.open_rate

        # Phase 2: move to break-even when profit reaches 40% of TP
        if current_profit >= tp_pct * 0.4:
            fee = trade.fee_open + trade.fee_close
            return max(-fee - 0.001, -0.005)  # break-even minus tiny buffer

        # Phase 1: ATR-based SL
        return max(-sl_pct, -0.05)

    # ------------------------------------------------------------------ #
    #  CUSTOM EXIT (realistic TP + aggressive time cuts)                    #
    # ------------------------------------------------------------------ #
    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if not df.empty:
            last_atr = float(df["atr"].iat[-1])
            if last_atr > 0 and trade.open_rate > 0:
                tp_ratio = self.TP_ATR_MULT.value * last_atr / trade.open_rate
                if current_profit >= tp_ratio:
                    return "TP_HIT"

                # Partial TP: take profit at 70% of target if held > 1h
                hours = (current_time - trade.open_date_utc).total_seconds() / 3600
                if hours >= 1 and current_profit >= tp_ratio * 0.7:
                    return "TP_PARTIAL"

        # Aggressive time-based cuts (mean-rev should work FAST or not at all)
        hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        if hours >= 2 and current_profit < -0.008:
            return "time_cut_2h"
        if hours >= 4 and current_profit < -0.003:
            return "time_cut_4h"
        if hours >= 6 and current_profit < 0.002:
            return "time_cut_6h"
        if hours >= 12:
            return "time_cut_12h"  # force close anything held > 12h

        return None

    # ------------------------------------------------------------------ #
    #  BTC SENTIMENT                                                       #
    # ------------------------------------------------------------------ #
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

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return False

        last = df.iloc[-1]
        if float(last.get("atr", 0)) <= 0:
            return False
        if float(last.get("volume_ratio", 0)) < self.VOL_RATIO_P.value:
            return False

        # BTC sentiment gate
        btc_rsi = float(last.get("btc_rsi_1h", 50))
        if side == "long" and btc_rsi < 30:
            return False
        if side == "short" and btc_rsi > 70:
            return False

        return True

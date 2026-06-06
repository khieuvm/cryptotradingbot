import numpy as np
import pandas as pd
import pandas_ta as ta

from combos.base import BaseCryptoCombo


class RegimeAdaptiveCombo(BaseCryptoCombo):
    """Regime-adaptive strategy: detect market state, apply appropriate signals.

    TRENDING (ADX > threshold, price aligned with EMA200):
      Long:  EMA cross up + MACD + DI + SuperTrend + volume
      Short: EMA cross down + MACD + DI + SuperTrend + volume

    RANGING (ADX <= threshold):
      Long:  RSI oversold + BB lower + OBV rising + bullish candle
      Short: RSI overbought + BB upper + OBV falling + bearish candle

    EXIT: ATR-based TP, cascading time-loss-cut (2h/8h/24h/48h)
    """

    name = "regime_adaptive"
    timeframe = "15m"

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"]

        ema_fast_p = self.entry_cfg.get("ema_fast", 18)
        ema_slow_p = self.entry_cfg.get("ema_slow", 50)

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
            dataframe["ra_bb_width"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / (bb.iloc[:, 1] + 1e-10)
        else:
            dataframe["ra_bb_upper"] = dataframe["ra_bb_lower"] = dataframe["ra_bb_mid"] = c
            dataframe["ra_bb_width"] = 0.02
        dataframe["ra_bb_width_sma"] = ta.sma(dataframe["ra_bb_width"], length=50)

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

        # Exit signal columns
        dataframe["ra_sig_trend_exit_long"] = (
            (dataframe["ra_ema_fast"] < dataframe["ra_ema_slow"]) &
            (dataframe["ra_ema_fast"].shift(1) >= dataframe["ra_ema_slow"].shift(1)) &
            (dataframe["ra_st_dir"] == -1)
        ).astype(int)

        dataframe["ra_sig_trend_exit_short"] = (
            (dataframe["ra_ema_fast"] > dataframe["ra_ema_slow"]) &
            (dataframe["ra_ema_fast"].shift(1) <= dataframe["ra_ema_slow"].shift(1)) &
            (dataframe["ra_st_dir"] == 1)
        ).astype(int)

        dataframe["ra_sig_range_exit_long"] = (
            (dataframe["ra_rsi"] > 52) &
            (c > dataframe["ra_bb_mid"])
        ).astype(int)

        dataframe["ra_sig_range_exit_short"] = (
            (dataframe["ra_rsi"] < 48) &
            (c < dataframe["ra_bb_mid"])
        ).astype(int)

        return dataframe

    def detect_long(self, dataframe: pd.DataFrame, metadata: dict) -> pd.Series:
        adx_thr = self.entry_cfg.get("adx_trend_thr", 31)
        rsi_os = self.entry_cfg.get("rsi_os", 34)
        vol_min = self.entry_cfg.get("vol_min", 0.5)
        atr_spike = self.entry_cfg.get("atr_spike_thr", 2.2)

        is_trending = dataframe["ra_adx"] > adx_thr
        is_bull = dataframe["ra_is_bull"] == 1
        atr_ok = dataframe["ra_atr_ratio"] < atr_spike

        # EMA cross within last 3 candles
        cross_up_recent = (
            (dataframe["ra_ema_fast"].shift(1) <= dataframe["ra_ema_slow"].shift(1)) |
            (dataframe["ra_ema_fast"].shift(2) <= dataframe["ra_ema_slow"].shift(2)) |
            (dataframe["ra_ema_fast"].shift(3) <= dataframe["ra_ema_slow"].shift(3))
        )

        # S1: Trending long
        trend_long = (
            is_trending & is_bull &
            (dataframe["ra_ema_fast"] > dataframe["ra_ema_slow"]) &
            cross_up_recent &
            (dataframe["ra_macd_hist"] > 0) &
            (dataframe["ra_plus_di"] > dataframe["ra_minus_di"]) &
            (dataframe["ra_st_dir"] == 1) &
            (dataframe["ra_vol_ratio"] > vol_min) &
            atr_ok & (dataframe["volume"] > 0)
        )

        # S2: Trend continuation long
        trend_cont_long = (
            is_trending & is_bull &
            (dataframe["ra_ema_fast"] > dataframe["ra_ema_slow"]) &
            (dataframe["ra_rsi"].shift(1) < 48) &
            (dataframe["ra_rsi"] > dataframe["ra_rsi"].shift(1)) &
            (dataframe["ra_macd_hist"] > 0) &
            (dataframe["ra_st_dir"] == 1) &
            (dataframe["ra_plus_di"] > dataframe["ra_minus_di"]) &
            (dataframe["ra_vol_ratio"] > vol_min) &
            atr_ok & (dataframe["volume"] > 0) &
            (~trend_long)
        )

        # S3: Ranging long
        range_long = (
            (~is_trending) &
            (dataframe["ra_rsi"].shift(1) < rsi_os) &
            (dataframe["ra_rsi"] > dataframe["ra_rsi"].shift(1)) &
            (dataframe["close"] < dataframe["ra_bb_lower"] * 1.01) &
            (dataframe["close"] > dataframe["open"]) &
            (dataframe["ra_obv_rising"] == 1) &
            (dataframe["ra_vol_ratio"] > vol_min) &
            atr_ok & (dataframe["volume"] > 0)
        )

        return trend_long | trend_cont_long | range_long

    def detect_short(self, dataframe: pd.DataFrame, metadata: dict) -> pd.Series:
        adx_thr = self.entry_cfg.get("adx_trend_thr", 31)
        rsi_ob = self.entry_cfg.get("rsi_ob", 71)
        vol_min = self.entry_cfg.get("vol_min", 0.5)
        atr_spike = self.entry_cfg.get("atr_spike_thr", 2.2)

        is_trending = dataframe["ra_adx"] > adx_thr
        is_bear = dataframe["ra_is_bear"] == 1
        atr_ok = dataframe["ra_atr_ratio"] < atr_spike

        cross_dn_recent = (
            (dataframe["ra_ema_fast"].shift(1) >= dataframe["ra_ema_slow"].shift(1)) |
            (dataframe["ra_ema_fast"].shift(2) >= dataframe["ra_ema_slow"].shift(2)) |
            (dataframe["ra_ema_fast"].shift(3) >= dataframe["ra_ema_slow"].shift(3))
        )

        # S1: Trending short
        trend_short = (
            is_trending & is_bear &
            (dataframe["ra_ema_fast"] < dataframe["ra_ema_slow"]) &
            cross_dn_recent &
            (dataframe["ra_macd_hist"] < 0) &
            (dataframe["ra_minus_di"] > dataframe["ra_plus_di"]) &
            (dataframe["ra_st_dir"] == -1) &
            (dataframe["ra_vol_ratio"] > vol_min) &
            atr_ok & (dataframe["volume"] > 0)
        )

        # S2: Trend continuation short
        trend_cont_short = (
            is_trending & is_bear &
            (dataframe["ra_ema_fast"] < dataframe["ra_ema_slow"]) &
            (dataframe["ra_rsi"].shift(1) > 52) &
            (dataframe["ra_rsi"] < dataframe["ra_rsi"].shift(1)) &
            (dataframe["ra_macd_hist"] < 0) &
            (dataframe["ra_st_dir"] == -1) &
            (dataframe["ra_minus_di"] > dataframe["ra_plus_di"]) &
            (dataframe["ra_vol_ratio"] > vol_min) &
            atr_ok & (dataframe["volume"] > 0) &
            (~trend_short)
        )

        # S3: Ranging short
        range_short = (
            (~is_trending) &
            (dataframe["ra_rsi"].shift(1) > rsi_ob) &
            (dataframe["ra_rsi"] < dataframe["ra_rsi"].shift(1)) &
            (dataframe["close"] > dataframe["ra_bb_upper"] * 0.99) &
            (dataframe["close"] < dataframe["open"]) &
            (dataframe["ra_obv_rising"] == 0) &
            (dataframe["ra_vol_ratio"] > vol_min) &
            atr_ok & (dataframe["volume"] > 0)
        )

        return trend_short | trend_cont_short | range_short

    def get_entry_tag(self, dataframe: pd.DataFrame, metadata: dict, is_long: bool) -> pd.Series:
        """Return entry tag Series for signal classification."""
        adx_thr = self.entry_cfg.get("adx_trend_thr", 31)
        is_trending = dataframe["ra_adx"] > adx_thr

        if is_long:
            tag = pd.Series("ra_range_long", index=dataframe.index)
            tag[is_trending] = "ra_trend_long"
        else:
            tag = pd.Series("ra_range_short", index=dataframe.index)
            tag[is_trending] = "ra_trend_short"
        return tag

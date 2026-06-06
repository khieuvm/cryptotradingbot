import pandas as pd
import pandas_ta as ta

from combos.base import BaseCryptoCombo


class TrendCompositeCombo(BaseCryptoCombo):
    """EMA momentum trend-following strategy.

    Entry: EMA20 crosses EMA50 (within 3 bars) + ADX>25 + DI confirms + MACD + volume
    Exit: ATR-TP, EMA reverse cross, break-even at BE_ATR_MULT
    """

    name = "trend_composite"
    timeframe = "15m"

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"]

        ema_fast_p = self.entry_cfg.get("ema_fast", 20)
        ema_slow_p = self.entry_cfg.get("ema_slow", 50)

        dataframe["tc_ema_fast"] = ta.ema(c, length=ema_fast_p)
        dataframe["tc_ema_slow"] = ta.ema(c, length=ema_slow_p)
        dataframe["tc_atr"] = ta.atr(h, lo, c, length=14)

        adx_df = ta.adx(h, lo, c, length=14)
        if adx_df is not None:
            dataframe["tc_adx"] = adx_df.iloc[:, 0]
            dataframe["tc_plus_di"] = adx_df.iloc[:, 1]
            dataframe["tc_minus_di"] = adx_df.iloc[:, 2]
        else:
            dataframe["tc_adx"] = 0.0
            dataframe["tc_plus_di"] = 0.0
            dataframe["tc_minus_di"] = 0.0

        dataframe["tc_rsi"] = ta.rsi(c, length=14)

        dataframe["tc_vol_ema"] = ta.ema(v.astype(float), length=20)
        dataframe["tc_vol_ratio"] = v.astype(float) / (dataframe["tc_vol_ema"] + 1e-10)

        macd = ta.macd(c, fast=12, slow=26, signal=9)
        if macd is not None:
            dataframe["tc_macd_hist"] = macd.iloc[:, 1]
        else:
            dataframe["tc_macd_hist"] = 0.0

        bb = ta.bbands(c, length=20, std=2.0)
        if bb is not None:
            dataframe["tc_bb_mid"] = bb.iloc[:, 1]
        else:
            dataframe["tc_bb_mid"] = c

        dataframe["tc_cross_up"] = (
            (dataframe["tc_ema_fast"] > dataframe["tc_ema_slow"]) &
            (dataframe["tc_ema_fast"].shift(1) <= dataframe["tc_ema_slow"].shift(1))
        )
        dataframe["tc_cross_down"] = (
            (dataframe["tc_ema_fast"] < dataframe["tc_ema_slow"]) &
            (dataframe["tc_ema_fast"].shift(1) >= dataframe["tc_ema_slow"].shift(1))
        )
        dataframe["tc_recent_cross_up"] = dataframe["tc_cross_up"].rolling(3).max().fillna(0).astype(bool)
        dataframe["tc_recent_cross_down"] = dataframe["tc_cross_down"].rolling(3).max().fillna(0).astype(bool)

        return dataframe

    def detect_long(self, dataframe: pd.DataFrame, metadata: dict) -> pd.Series:
        adx_min = self.entry_cfg.get("adx_min", 25)
        vol_mult = self.entry_cfg.get("vol_mult", 1.2)
        rsi_low = self.entry_cfg.get("rsi_low", 48)
        rsi_high = self.entry_cfg.get("rsi_high", 65)

        return (
            dataframe["tc_recent_cross_up"] &
            (dataframe["tc_ema_fast"] > dataframe["tc_ema_slow"]) &
            (dataframe["tc_plus_di"] > dataframe["tc_minus_di"]) &
            (dataframe["tc_adx"] >= adx_min) &
            (dataframe["tc_rsi"] >= rsi_low) &
            (dataframe["tc_rsi"] <= rsi_high) &
            (dataframe["tc_macd_hist"] > 0) &
            (dataframe["tc_vol_ratio"] >= vol_mult) &
            (dataframe["close"] > dataframe["tc_ema_fast"])
        )

    def detect_short(self, dataframe: pd.DataFrame, metadata: dict) -> pd.Series:
        adx_min = self.entry_cfg.get("adx_min", 25)
        vol_mult = self.entry_cfg.get("vol_mult", 1.2)
        rsi_low = self.entry_cfg.get("rsi_low", 48)
        rsi_high = self.entry_cfg.get("rsi_high", 65)
        rsi_short_high = 100 - rsi_low
        rsi_short_low = 100 - rsi_high

        return (
            dataframe["tc_recent_cross_down"] &
            (dataframe["tc_ema_fast"] < dataframe["tc_ema_slow"]) &
            (dataframe["tc_minus_di"] > dataframe["tc_plus_di"]) &
            (dataframe["tc_adx"] >= adx_min) &
            (dataframe["tc_rsi"] >= rsi_short_low) &
            (dataframe["tc_rsi"] <= rsi_short_high) &
            (dataframe["tc_macd_hist"] < 0) &
            (dataframe["tc_vol_ratio"] >= vol_mult) &
            (dataframe["close"] < dataframe["tc_ema_fast"])
        )

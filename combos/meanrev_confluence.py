import pandas as pd
import pandas_ta as ta

from combos.base import BaseCryptoCombo


class MeanRevConfluenceCombo(BaseCryptoCombo):
    """Mean-reversion pullback scalper on 15m.

    Entry: RSI extreme + BB touch + volume spike in TREND direction
    Trend: EMA fast/slow + macro EMA100
    Exit: BB-mid reversion target + ATR-based TP/SL + time cuts
    """

    name = "meanrev_confluence"
    timeframe = "15m"

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"]

        ema_fast_len = self.entry_cfg.get("ema_fast", 20)
        ema_slow_len = self.entry_cfg.get("ema_slow", 50)
        bb_period = self.entry_cfg.get("bb_period", 20)

        dataframe["mr_atr"] = ta.atr(h, lo, c, length=14)
        dataframe["mr_rsi"] = ta.rsi(c, length=14)

        bb = ta.bbands(c, length=bb_period, std=2.0)
        if bb is not None:
            dataframe["mr_bb_upper"] = bb.iloc[:, 2]
            dataframe["mr_bb_mid"] = bb.iloc[:, 1]
            dataframe["mr_bb_lower"] = bb.iloc[:, 0]
        else:
            dataframe["mr_bb_upper"] = c
            dataframe["mr_bb_mid"] = c
            dataframe["mr_bb_lower"] = c

        dataframe["mr_ema_fast"] = ta.ema(c, length=ema_fast_len)
        dataframe["mr_ema_slow"] = ta.ema(c, length=ema_slow_len)
        dataframe["mr_ema100"] = ta.ema(c, length=100)

        dataframe["mr_vol_ema"] = ta.ema(v.astype(float), length=20)
        dataframe["mr_vol_ratio"] = v.astype(float) / (dataframe["mr_vol_ema"] + 1e-10)

        return dataframe

    def detect_long(self, dataframe: pd.DataFrame, metadata: dict) -> pd.Series:
        rsi_buy = self.entry_cfg.get("rsi_buy", 30)
        vol_mult = self.entry_cfg.get("vol_mult", 1.2)

        trend_up = dataframe["mr_ema_fast"] > dataframe["mr_ema_slow"]
        macro_up = dataframe["mr_ema_slow"] > dataframe["mr_ema100"]
        vol_ok = dataframe["mr_vol_ratio"] >= vol_mult

        return (
            trend_up & macro_up &
            (dataframe["mr_rsi"] <= rsi_buy) &
            (dataframe["low"] <= dataframe["mr_bb_lower"]) &
            vol_ok
        )

    def detect_short(self, dataframe: pd.DataFrame, metadata: dict) -> pd.Series:
        rsi_sell = self.entry_cfg.get("rsi_sell", 70)
        vol_mult = self.entry_cfg.get("vol_mult", 1.2)

        trend_down = dataframe["mr_ema_fast"] < dataframe["mr_ema_slow"]
        macro_down = dataframe["mr_ema_slow"] < dataframe["mr_ema100"]
        vol_ok = dataframe["mr_vol_ratio"] >= vol_mult

        return (
            trend_down & macro_down &
            (dataframe["mr_rsi"] >= rsi_sell) &
            (dataframe["high"] >= dataframe["mr_bb_upper"]) &
            vol_ok
        )

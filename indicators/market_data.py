"""Market data indicators: funding rate, BTC sentiment, OI."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandas_ta as ta

from freqtrade.constants import CandleType
from freqtrade.strategy import merge_informative_pair


def add_btc_sentiment(
    df: pd.DataFrame, dp: Any, timeframe: str = "15m"
) -> pd.DataFrame:
    """Add BTC 1h RSI as market-wide sentiment filter."""
    if dp is None:
        df["btc_rsi_1h"] = 50.0
        return df
    btc_df = dp.get_pair_dataframe(pair="BTC/USDT:USDT", timeframe="1h")
    if btc_df.empty:
        df["btc_rsi_1h"] = 50.0
        return df
    btc_df = btc_df.copy()
    btc_df["btc_rsi"] = ta.rsi(btc_df["close"], length=14)
    btc_df = btc_df[["date", "btc_rsi"]].copy()
    df = merge_informative_pair(df, btc_df, timeframe, "1h", ffill=True)
    if "btc_rsi_1h" not in df.columns:
        df["btc_rsi_1h"] = 50.0
    return df


def add_funding_rate(
    df: pd.DataFrame, dp: Any, pair: str, timeframe: str = "15m"
) -> pd.DataFrame:
    """Add funding rate from informative pair data."""
    if dp is None:
        df["funding_rate"] = 0.0
        return df
    try:
        fr_df = dp.get_pair_dataframe(
            pair=pair, timeframe="1h", candle_type=CandleType.FUNDING_RATE
        )
        if fr_df.empty:
            df["funding_rate"] = 0.0
            return df
        fr_df = fr_df[["date", "open"]].copy().rename(columns={"open": "fr"})
        df = merge_informative_pair(df, fr_df, timeframe, "1h", ffill=True)
        df["funding_rate"] = df.get("fr_1h", 0.0)
    except Exception:
        df["funding_rate"] = 0.0
    return df

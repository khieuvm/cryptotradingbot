"""Volume indicators: OBV, volume ratio, VWAP."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def add_obv(df: pd.DataFrame, col_prefix: str = "") -> pd.DataFrame:
    df[f"{col_prefix}obv"] = ta.obv(df["close"], df["volume"])
    return df


def add_volume_ratio(df: pd.DataFrame, length: int = 20, col_prefix: str = "") -> pd.DataFrame:
    """Volume relative to its SMA (ratio > 1 = above average)."""
    p = col_prefix
    vol_sma = df["volume"].rolling(length).mean()
    df[f"{p}vol_ratio"] = df["volume"] / vol_sma
    return df


def add_vwap(df: pd.DataFrame, col_prefix: str = "") -> pd.DataFrame:
    """Session VWAP (resets each day for crypto 24h trading)."""
    p = col_prefix
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical * df["volume"]).cumsum()
    df[f"{p}vwap"] = cum_tp_vol / cum_vol
    return df

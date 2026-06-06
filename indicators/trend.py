"""Trend indicators: EMA, ADX, DI, SuperTrend, MACD."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def add_ema(df: pd.DataFrame, length: int, col_prefix: str = "") -> pd.DataFrame:
    col = f"{col_prefix}ema_{length}" if col_prefix else f"ema_{length}"
    df[col] = ta.ema(df["close"], length=length)
    return df


def add_ema_pair(
    df: pd.DataFrame, fast: int, slow: int, col_prefix: str = ""
) -> pd.DataFrame:
    p = col_prefix
    df[f"{p}ema_fast"] = ta.ema(df["close"], length=fast)
    df[f"{p}ema_slow"] = ta.ema(df["close"], length=slow)
    df[f"{p}ema_cross_up"] = (
        (df[f"{p}ema_fast"] > df[f"{p}ema_slow"])
        & (df[f"{p}ema_fast"].shift(1) <= df[f"{p}ema_slow"].shift(1))
    )
    df[f"{p}ema_cross_down"] = (
        (df[f"{p}ema_fast"] < df[f"{p}ema_slow"])
        & (df[f"{p}ema_fast"].shift(1) >= df[f"{p}ema_slow"].shift(1))
    )
    return df


def add_ema_cross_recent(
    df: pd.DataFrame, fast: int, slow: int, lookback: int = 3, col_prefix: str = ""
) -> pd.DataFrame:
    """Detect if EMA cross happened within the last `lookback` bars."""
    p = col_prefix
    if f"{p}ema_fast" not in df.columns:
        df = add_ema_pair(df, fast, slow, col_prefix=p)
    df[f"{p}cross_up_recent"] = df[f"{p}ema_cross_up"].rolling(lookback).max().astype(bool)
    df[f"{p}cross_down_recent"] = df[f"{p}ema_cross_down"].rolling(lookback).max().astype(bool)
    return df


def add_adx(df: pd.DataFrame, length: int = 14, col_prefix: str = "") -> pd.DataFrame:
    p = col_prefix
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=length)
    df[f"{p}adx"] = adx_df[f"ADX_{length}"]
    df[f"{p}di_plus"] = adx_df[f"DMP_{length}"]
    df[f"{p}di_minus"] = adx_df[f"DMN_{length}"]
    return df


def add_supertrend(
    df: pd.DataFrame, length: int = 10, multiplier: float = 3.0, col_prefix: str = ""
) -> pd.DataFrame:
    p = col_prefix
    st = ta.supertrend(df["high"], df["low"], df["close"], length=length, multiplier=multiplier)
    st_col = f"SUPERT_{length}_{multiplier}"
    st_dir_col = f"SUPERTd_{length}_{multiplier}"
    df[f"{p}supertrend"] = st[st_col]
    df[f"{p}supertrend_dir"] = st[st_dir_col]
    return df


def add_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, col_prefix: str = ""
) -> pd.DataFrame:
    p = col_prefix
    macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    df[f"{p}macd"] = macd_df[f"MACD_{fast}_{slow}_{signal}"]
    df[f"{p}macd_signal"] = macd_df[f"MACDs_{fast}_{slow}_{signal}"]
    df[f"{p}macd_hist"] = macd_df[f"MACDh_{fast}_{slow}_{signal}"]
    return df

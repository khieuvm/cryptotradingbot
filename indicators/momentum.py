"""Momentum indicators: RSI, Stochastic, CCI."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def add_rsi(df: pd.DataFrame, length: int = 14, col_prefix: str = "") -> pd.DataFrame:
    df[f"{col_prefix}rsi"] = ta.rsi(df["close"], length=length)
    return df


def add_stochastic(
    df: pd.DataFrame, k: int = 14, d: int = 3, smooth_k: int = 3, col_prefix: str = ""
) -> pd.DataFrame:
    p = col_prefix
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=k, d=d, smooth_k=smooth_k)
    df[f"{p}stoch_k"] = stoch[f"STOCHk_{k}_{d}_{smooth_k}"]
    df[f"{p}stoch_d"] = stoch[f"STOCHd_{k}_{d}_{smooth_k}"]
    return df


def add_cci(df: pd.DataFrame, length: int = 20, col_prefix: str = "") -> pd.DataFrame:
    df[f"{col_prefix}cci"] = ta.cci(df["high"], df["low"], df["close"], length=length)
    return df

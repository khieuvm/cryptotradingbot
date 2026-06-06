"""Volatility indicators: ATR, Bollinger Bands, Keltner Channel, BB Squeeze."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def add_atr(df: pd.DataFrame, length: int = 14, col_prefix: str = "") -> pd.DataFrame:
    col = f"{col_prefix}atr"
    df[col] = ta.atr(df["high"], df["low"], df["close"], length=length)
    return df


def add_bollinger(
    df: pd.DataFrame, length: int = 20, std: float = 2.0, col_prefix: str = ""
) -> pd.DataFrame:
    p = col_prefix
    bb = ta.bbands(df["close"], length=length, std=std)
    df[f"{p}bb_upper"] = bb[f"BBU_{length}_{std}"]
    df[f"{p}bb_mid"] = bb[f"BBM_{length}_{std}"]
    df[f"{p}bb_lower"] = bb[f"BBL_{length}_{std}"]
    df[f"{p}bb_width"] = (df[f"{p}bb_upper"] - df[f"{p}bb_lower"]) / df[f"{p}bb_mid"]
    return df


def add_keltner(
    df: pd.DataFrame, length: int = 20, multiplier: float = 1.5, col_prefix: str = ""
) -> pd.DataFrame:
    p = col_prefix
    mid = ta.ema(df["close"], length=length)
    atr = ta.atr(df["high"], df["low"], df["close"], length=length)
    df[f"{p}kc_upper"] = mid + multiplier * atr
    df[f"{p}kc_mid"] = mid
    df[f"{p}kc_lower"] = mid - multiplier * atr
    return df


def compute_bb_squeeze(df: pd.DataFrame, col_prefix: str = "") -> pd.DataFrame:
    """BB inside KC = squeeze (volatility compression)."""
    p = col_prefix
    required = [f"{p}bb_upper", f"{p}bb_lower", f"{p}kc_upper", f"{p}kc_lower"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column {col} required. Call add_bollinger + add_keltner first.")
    df[f"{p}squeeze"] = (
        (df[f"{p}bb_lower"] > df[f"{p}kc_lower"])
        & (df[f"{p}bb_upper"] < df[f"{p}kc_upper"])
    )
    df[f"{p}squeeze_bars"] = (
        df[f"{p}squeeze"]
        .groupby((~df[f"{p}squeeze"]).cumsum())
        .cumcount()
    )
    return df

"""Analysis: Comprehensive Signal Opportunity Analysis
All pairs (ETH/SOL/BTC) at 1m, 5m, 15m timeframes.
Measures how many profit opportunities exist vs what current strategies capture.

Data coverage:
  15m: 2026-01-15 to 2026-06-08  (143 days)
  5m : 2026-04-08 to 2026-06-09  (~62 days)
  1m : 2026-04-08 to 2026-06-07  (~60 days)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore

warnings.filterwarnings("ignore")

DATA_DIR = Path("e:/freqtrade/user_data/data/okx/futures")
# UTC-aware timestamps to match feather file timezone
START_15M = pd.Timestamp("2026-01-15", tz="UTC")
END_15M = pd.Timestamp("2026-06-08", tz="UTC")
DAYS_15M = 143

# 5m/1m data only available from 2026-04-08
START_SHORT = pd.Timestamp("2026-04-08", tz="UTC")
END_SHORT = pd.Timestamp("2026-06-08", tz="UTC")
DAYS_SHORT = 61   # ~April 8 to June 8

PAIRS = ["ETH_USDT_USDT", "SOL_USDT_USDT", "BTC_USDT_USDT"]
PAIR_LABELS = {"ETH_USDT_USDT": "ETH", "SOL_USDT_USDT": "SOL", "BTC_USDT_USDT": "BTC"}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_pair(pair_slug: str, timeframe: str,
              start: pd.Timestamp | None = None,
              end: pd.Timestamp | None = None) -> pd.DataFrame:
    fpath = DATA_DIR / f"{pair_slug}-{timeframe}-futures.feather"
    if not fpath.exists():
        raise FileNotFoundError(f"No data: {fpath}")
    df = pd.read_feather(fpath)
    # Ensure UTC-aware datetime
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    df = df.reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ha(df: pd.DataFrame):
    """Heikin-Ashi open & close. Sequential, no look-ahead (matches strategy)."""
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    ha_c = (open_ + high + low + close) / 4.0
    ha_o = np.empty(len(df), dtype=float)
    ha_o[0] = (open_[0] + close[0]) / 2.0
    for i in range(1, len(ha_o)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    return pd.Series(ha_o, index=df.index), pd.Series(ha_c, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def forward_returns(df: pd.DataFrame, signal: pd.Series, direction: int, n: int) -> pd.Series:
    fwd = (df["close"].shift(-n) / df["close"] - 1.0) * direction
    return fwd[signal].dropna()


def profit_factor(ret: pd.Series) -> float:
    pos = ret[ret > 0].sum()
    neg = abs(ret[ret < 0].sum())
    if neg == 0:
        return float("inf") if pos > 0 else 1.0
    return float(pos / neg)


def win_rate(ret: pd.Series) -> float:
    return float((ret > 0).mean()) if len(ret) > 0 else 0.0


def random_baseline(df: pd.DataFrame, n: int, direction: int, n_bars: int = 5, seed: int = 42) -> pd.Series:
    if n == 0:
        return pd.Series(dtype=float)
    rng = np.random.default_rng(seed)
    valid_idx = np.arange(len(df) - n_bars)
    if len(valid_idx) == 0:
        return pd.Series(dtype=float)
    chosen = rng.choice(valid_idx, size=min(n, len(valid_idx)), replace=False)
    fwd = (df["close"].shift(-n_bars) / df["close"] - 1.0) * direction
    return fwd.iloc[chosen].dropna()


def row_result(
    tf: str, pair: str, strat: str, direction: str,
    raw: int, taken: int, fwd5, fwd10, rand5, days: float,
) -> dict:
    gap_wr = (win_rate(fwd5) - win_rate(rand5)) * 100
    return {
        "timeframe": tf,
        "pair": pair,
        "strategy": strat,
        "direction": direction,
        "raw_signals": raw,
        "taken_signals": taken,
        "utilization": taken / raw if raw > 0 else 0.0,
        "wr_5bar": win_rate(fwd5),
        "pf_5bar": profit_factor(fwd5),
        "wr_10bar": win_rate(fwd10),
        "pf_10bar": profit_factor(fwd10),
        "avg_fwd5_pct": fwd5.mean() * 100 if len(fwd5) > 0 else 0.0,
        "rand_wr5": win_rate(rand5),
        "gap_wr_pp": gap_wr,
        "signals_per_day": taken / days if days > 0 else 0.0,
        "days": days,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

results: list[dict] = []

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — 15m Signals
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("Loading 15m data (2026-01-15 to 2026-06-08, 143 days)...")
print("=" * 70)

for pair in PAIRS:
    lbl = PAIR_LABELS[pair]
    try:
        df = load_pair(pair, "15m", start=START_15M, end=END_15M)
    except FileNotFoundError as e:
        print(f"  SKIP {lbl}: {e}")
        continue

    actual_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days + 1
    print(f"  {lbl} 15m: {len(df)} bars  "
          f"({df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}, {actual_days}d)")

    c = df["close"]
    o = df["open"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"].astype(float)

    # Shared indicators
    rsi14 = ta.rsi(c, length=14)
    atr14 = ta.atr(h, lo, c, length=14)
    vol_ema20 = ta.ema(v, length=20)
    vol_ratio = v / (vol_ema20 + 1e-10)

    # ── ha_stoch_rev ─────────────────────────────────────────────────────────
    ha_o_s, ha_c_s = compute_ha(df)
    ha_color = pd.Series(
        np.where(ha_c_s.values >= ha_o_s.values, 1, -1), index=df.index
    )
    # Exactly as in the strategy: rolling(ha_run_min=2).sum()
    ha_run = ha_color.rolling(2).sum()
    ha_bull_run = ha_run >= 2    # 2+ consecutive green  → SHORT entry (contrarian)
    ha_bear_run = ha_run <= -2   # 2+ consecutive red    → LONG entry (contrarian)

    stoch_df = ta.stoch(h, lo, c, k=14, d=3, smooth_k=7)
    k_col = next(col for col in stoch_df.columns if "STOCHk" in col)
    stoch_k = stoch_df[k_col]

    ema100 = ta.ema(c, length=100)

    hsr_short_raw = (
        ha_bull_run & (stoch_k > 70) & (c < ema100) & (vol_ratio >= 1.0)
    )
    hsr_long_raw = (
        ha_bear_run & (stoch_k < 30) & (c > ema100) & (vol_ratio >= 1.0)
    )

    # Dedup: exactly as in populate_entry_columns (dedup_bars=6)
    any_sig = hsr_short_raw | hsr_long_raw
    last_sig = any_sig.rolling(6).max().shift(1).fillna(0).astype(bool)
    not_deduped = ~last_sig

    hsr_short_taken = hsr_short_raw & not_deduped
    hsr_long_taken = hsr_long_raw & not_deduped

    for direction, raw_sig, taken_sig in [
        ("SHORT", hsr_short_raw, hsr_short_taken),
        ("LONG", hsr_long_raw, hsr_long_taken),
    ]:
        dv = -1 if direction == "SHORT" else 1
        raw_n = int(raw_sig.sum())
        taken_n = int(taken_sig.sum())
        fwd5 = forward_returns(df, taken_sig, dv, 5)
        fwd10 = forward_returns(df, taken_sig, dv, 10)
        rand5 = random_baseline(df, len(fwd5), dv, 5)
        results.append(row_result("15m", lbl, "ha_stoch_rev", direction,
                                  raw_n, taken_n, fwd5, fwd10, rand5, actual_days))

    # ── volume_spike_rev ─────────────────────────────────────────────────────
    body = abs(c - o)
    candle_range = h - lo + 1e-9
    body_ratio = body / candle_range
    is_red = c < o
    lower_shadow = df[["open", "close"]].min(axis=1) - lo
    upper_shadow = h - df[["open", "close"]].max(axis=1)
    is_hammer = (lower_shadow > 2.0 * body) & (upper_shadow < body * 0.5) & (body > 0)

    spike = vol_ratio >= 3.0
    atr_ok = body > 0.3 * atr14

    vsr_short_raw = (
        spike & is_red & (body_ratio > 0.55) & atr_ok
        & (rsi14 > 15) & (rsi14 < 50)
    )
    vsr_long_raw = spike & is_hammer & (rsi14 < 25)

    for direction, raw_sig in [("SHORT", vsr_short_raw), ("LONG", vsr_long_raw)]:
        dv = -1 if direction == "SHORT" else 1
        raw_n = int(raw_sig.sum())
        fwd5 = forward_returns(df, raw_sig, dv, 5)
        fwd10 = forward_returns(df, raw_sig, dv, 10)
        rand5 = random_baseline(df, raw_n, dv, 5)
        results.append(row_result("15m", lbl, "vol_spike_rev", direction,
                                  raw_n, raw_n, fwd5, fwd10, rand5, actual_days))

    # ── regime_adaptive ──────────────────────────────────────────────────────
    ema18 = ta.ema(c, length=18)
    ema50 = ta.ema(c, length=50)
    ema200 = ta.ema(c, length=200)

    adx_df = ta.adx(h, lo, c, length=14)
    adx_v = adx_df.iloc[:, 0]
    plus_di = adx_df.iloc[:, 1]
    minus_di = adx_df.iloc[:, 2]

    macd_df = ta.macd(c, fast=12, slow=26, signal=9)
    macd_hist = macd_df.iloc[:, 1]

    try:
        st = ta.supertrend(h, lo, c, length=7, multiplier=3.0)
        st_dir_col = next(col for col in st.columns if "SUPERTd" in col)
        st_dir = st[st_dir_col].fillna(0)
    except Exception:
        st_dir = pd.Series(0, index=df.index, dtype=float)

    cross_up = (ema18 > ema50) & (ema18.shift(1) <= ema50.shift(1))
    cross_dn = (ema18 < ema50) & (ema18.shift(1) >= ema50.shift(1))
    cross_up_rec = cross_up.rolling(5).max().fillna(0).astype(int)
    cross_dn_rec = cross_dn.rolling(5).max().fillna(0).astype(int)

    atr_ma = ta.ema(atr14, length=50)
    atr_ratio = atr14 / (atr_ma + 1e-10)
    valid = (atr_ratio < 2.2) & (vol_ratio >= 1.5) & (v > 0)

    is_trending = adx_v > 31
    is_bull = c > ema200
    is_bear = c < ema200

    trend_long = (
        valid & is_trending & is_bull
        & (cross_up_rec == 1)
        & (ema18 > ema50)
        & (macd_hist > 0)
        & (plus_di > minus_di)
        & (st_dir == 1)
    )
    trend_short = (
        valid & is_trending & is_bear
        & (cross_dn_rec == 1)
        & (ema18 < ema50)
        & (macd_hist < 0)
        & (minus_di > plus_di)
        & (st_dir == -1)
    )

    bb = ta.bbands(c, length=20, std=2.0)
    bb_upper = bb.iloc[:, 0]
    bb_lower = bb.iloc[:, 2]

    obv = ta.obv(c, v)
    obv_ema = ta.ema(obv, length=20)
    obv_rising = (obv > obv_ema).astype(int)

    prev_rsi = rsi14.shift(1)
    range_long = (
        valid & ~is_trending
        & (prev_rsi < 34) & (rsi14 > prev_rsi)
        & (c < bb_lower * 1.01)
        & (c > o)
        & (obv_rising == 1)
    )
    range_short = (
        valid & ~is_trending
        & (prev_rsi > 71) & (rsi14 < prev_rsi)
        & (c > bb_upper * 0.99)
        & (c < o)
        & (obv_rising == 0)
    )

    ra_long = trend_long | range_long
    ra_short = trend_short | range_short

    for direction, sig in [("LONG", ra_long), ("SHORT", ra_short)]:
        dv = 1 if direction == "LONG" else -1
        n = int(sig.sum())
        fwd5 = forward_returns(df, sig, dv, 5)
        fwd10 = forward_returns(df, sig, dv, 10)
        rand5 = random_baseline(df, n, dv, 5)
        results.append(row_result("15m", lbl, "regime_adaptive", direction,
                                  n, n, fwd5, fwd10, rand5, actual_days))

    print(f"    ha_stoch_rev:    SHORT raw={int(hsr_short_raw.sum()):>4} taken={int(hsr_short_taken.sum()):>4}  "
          f"LONG raw={int(hsr_long_raw.sum()):>4} taken={int(hsr_long_taken.sum()):>4}")
    print(f"    vol_spike_rev:   SHORT={int(vsr_short_raw.sum()):>4}  LONG={int(vsr_long_raw.sum()):>4}")
    print(f"    regime_adaptive: LONG={int(ra_long.sum()):>4}  SHORT={int(ra_short.sum()):>4}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — 5m Signals  (data covers ~2026-04-08 to 2026-06-09, ~62 days)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Loading 5m data (2026-04-08 to 2026-06-08, ~61 days)...")
print("=" * 70)

for pair in PAIRS:
    lbl = PAIR_LABELS[pair]
    try:
        df5 = load_pair(pair, "5m", start=START_SHORT, end=END_SHORT)
        df1h = load_pair(pair, "1h", start=START_SHORT, end=END_SHORT)
        df15 = load_pair(pair, "15m", start=START_SHORT, end=END_SHORT)
    except FileNotFoundError as e:
        print(f"  SKIP {lbl}: {e}")
        continue

    actual_days_5m = (df5["date"].iloc[-1] - df5["date"].iloc[0]).days + 1
    print(f"  {lbl} 5m: {len(df5)} bars  "
          f"({df5['date'].iloc[0].date()} to {df5['date'].iloc[-1].date()}, {actual_days_5m}d)")

    # ── 1h HTF trend gate ────────────────────────────────────────────────────
    ema21_1h = ta.ema(df1h["close"], length=21)
    ema50_1h = ta.ema(df1h["close"], length=50)
    df1h = df1h.copy()
    df1h["htf_trend"] = np.where(ema21_1h > ema50_1h, 1, -1)

    df5 = pd.merge_asof(
        df5.sort_values("date"),
        df1h[["date", "htf_trend"]].sort_values("date"),
        on="date", direction="backward",
    )
    df5["htf_trend"] = df5["htf_trend"].fillna(0).astype(int)

    # ── 15m ADX gate ─────────────────────────────────────────────────────────
    adx15_df = ta.adx(df15["high"], df15["low"], df15["close"], length=14)
    df15 = df15.copy()
    df15["adx"] = adx15_df.iloc[:, 0]

    df5 = pd.merge_asof(
        df5.sort_values("date"),
        df15[["date", "adx"]].sort_values("date"),
        on="date", direction="backward",
    )
    df5["adx"] = df5["adx"].fillna(0)
    df5 = df5.reset_index(drop=True)

    # ── 5m EMAs ──────────────────────────────────────────────────────────────
    ema8 = ta.ema(df5["close"], length=8)
    ema21 = ta.ema(df5["close"], length=21)
    c5 = df5["close"]

    pb_short = (
        (df5["htf_trend"] == -1)
        & (df5["adx"] > 20)
        & (ema8 < ema21)
        & (c5.shift(1) > ema21.shift(1))  # was above
        & (c5 <= ema21)                   # now at or below
    )
    pb_long = (
        (df5["htf_trend"] == 1)
        & (df5["adx"] > 20)
        & (ema8 > ema21)
        & (c5.shift(1) < ema21.shift(1))  # was below
        & (c5 >= ema21)                   # now at or above
    )

    for direction, sig in [("SHORT", pb_short), ("LONG", pb_long)]:
        dv = -1 if direction == "SHORT" else 1
        n = int(sig.sum())
        fwd5 = forward_returns(df5, sig, dv, 5)
        fwd10 = forward_returns(df5, sig, dv, 10)
        rand5 = random_baseline(df5, n, dv, 5)
        results.append(row_result("5m", lbl, "ema_pullback", direction,
                                  n, n, fwd5, fwd10, rand5, actual_days_5m))

    # ── Asian Range Breakout ──────────────────────────────────────────────────
    df5 = df5.copy()
    df5["hour"] = df5["date"].dt.hour
    df5["minute"] = df5["date"].dt.minute
    df5["date_only"] = df5["date"].dt.date

    asian = (
        df5[df5["hour"] < 8]
        .groupby("date_only")
        .agg(asian_high=("high", "max"), asian_low=("low", "min"))
        .reset_index()
    )
    asian["asian_range_pct"] = (
        (asian["asian_high"] - asian["asian_low"]) / asian["asian_low"]
    )

    df5 = df5.merge(asian, on="date_only", how="left")

    london_window = (df5["hour"] == 8) | (
        (df5["hour"] == 9) & (df5["minute"] <= 30)
    )
    range_ok = (
        (df5["asian_range_pct"] >= 0.0015)
        & (df5["asian_range_pct"] <= 0.015)
    )

    aso_short = (
        london_window
        & range_ok
        & (df5["close"] < df5["asian_low"] * 0.999)
        & (df5["close"].shift(1) >= df5["asian_low"].shift(1))
        & (df5["htf_trend"] == -1)
    )
    aso_long = (
        london_window
        & range_ok
        & (df5["close"] > df5["asian_high"] * 1.001)
        & (df5["close"].shift(1) <= df5["asian_high"].shift(1))
        & (df5["htf_trend"] == 1)
    )

    for direction, sig in [("SHORT", aso_short), ("LONG", aso_long)]:
        dv = -1 if direction == "SHORT" else 1
        n = int(sig.sum())
        fwd5 = forward_returns(df5, sig, dv, 5)
        fwd10 = forward_returns(df5, sig, dv, 10)
        rand5 = random_baseline(df5, n, dv, 5)
        results.append(row_result("5m", lbl, "asian_range", direction,
                                  n, n, fwd5, fwd10, rand5, actual_days_5m))

    print(f"    ema_pullback:  SHORT={int(pb_short.sum()):>4}  LONG={int(pb_long.sum()):>4}")
    print(f"    asian_range:   SHORT={int(aso_short.sum()):>4}  LONG={int(aso_long.sum()):>4}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — 1m Signals  (data covers ~2026-04-08 to 2026-06-07, ~60 days)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Loading 1m data (2026-04-08 to 2026-06-07, ~60 days)...")
print("=" * 70)

for pair in PAIRS:
    lbl = PAIR_LABELS[pair]
    try:
        df1 = load_pair(pair, "1m", start=START_SHORT, end=END_SHORT)
    except FileNotFoundError as e:
        print(f"  SKIP {lbl}: {e}")
        continue

    actual_days_1m = (df1["date"].iloc[-1] - df1["date"].iloc[0]).days + 1
    print(f"  {lbl} 1m: {len(df1)} bars  "
          f"({df1['date'].iloc[0].date()} to {df1['date'].iloc[-1].date()}, {actual_days_1m}d)")

    rsi7 = ta.rsi(df1["close"], length=7)

    hook_short = (rsi7.shift(1) > 65) & (rsi7 <= 65)
    hook_long = (rsi7.shift(1) < 35) & (rsi7 >= 35)

    for direction, sig in [("SHORT", hook_short), ("LONG", hook_long)]:
        dv = -1 if direction == "SHORT" else 1
        n = int(sig.sum())
        fwd5 = forward_returns(df1, sig, dv, 5)
        fwd10 = forward_returns(df1, sig, dv, 10)
        rand5 = random_baseline(df1, n, dv, 5)
        results.append(row_result("1m", lbl, "rsi_hook", direction,
                                  n, n, fwd5, fwd10, rand5, actual_days_1m))

    print(f"    rsi_hook:      SHORT={int(hook_short.sum()):>5}  LONG={int(hook_long.sum()):>5}")


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

df_r = pd.DataFrame(results)

sep = "=" * 95
sep2 = "-" * 95
hdr_main = (
    f"{'Pair':<6} {'TF':<4} {'Strategy':<18} {'Dir':<6}"
    f" {'Raw':>6} {'Taken':>6} {'Util%':>6}"
    f" {'WR5':>7} {'PF5':>7} {'WR10':>7} {'PF10':>7}"
    f" {'RandWR5':>8} {'Gap':>7} {'Per/day':>7}"
)

print("\n\n" + sep)
print("  ALL-PAIR SIGNAL OPPORTUNITY ANALYSIS")
print("  15m: 2026-01-15 to 2026-06-08 (143d) | 5m/1m: 2026-04-08 to 2026-06-08 (~61d)")
print(sep)

print("""
CURRENT SYSTEM (latest backtest 143d Jan15-Jun7):
  ha_stoch_rev    ETH SHORT only:         64 trades, 67.2% WR
  volume_spike_rev ETH+SOL mostly SHORT:  72 trades, 62.6% WR
  regime_adaptive  ETH+SOL both dirs:     16 trades, 43.8% WR
  cb_adx_breakout  ETH+SOL both dirs:    318 trades,   49% WR
  Total (system-wide):                   413 trades
""")

print(sep)
print("PART 1 — Signal Utilization: raw vs taken for the 3 measured strategies (15m)")
print(sep2)
print(hdr_main)
print(sep2)

for _, row in df_r[
    (df_r["timeframe"] == "15m") &
    (df_r["strategy"].isin(["ha_stoch_rev", "vol_spike_rev", "regime_adaptive"]))
].sort_values(["strategy", "pair", "direction"]).iterrows():
    print(
        f"{row['pair']:<6} {row['timeframe']:<4} {row['strategy']:<18} {row['direction']:<6}"
        f" {row['raw_signals']:>6} {row['taken_signals']:>6} {row['utilization']*100:>5.0f}%"
        f" {row['wr_5bar']*100:>6.1f}% {row['pf_5bar']:>7.2f}"
        f" {row['wr_10bar']*100:>6.1f}% {row['pf_10bar']:>7.2f}"
        f" {row['rand_wr5']*100:>7.1f}% {row['gap_wr_pp']:>+6.1f}pp {row['signals_per_day']:>7.2f}"
    )

print()
print(sep)
print("PART 2A — RAW 15m OPPORTUNITIES (all pairs x all strategies)")
print(sep2)
print(hdr_main)
print(sep2)

for _, row in df_r[df_r["timeframe"] == "15m"].sort_values(
    ["strategy", "pair", "direction"]
).iterrows():
    print(
        f"{row['pair']:<6} {row['timeframe']:<4} {row['strategy']:<18} {row['direction']:<6}"
        f" {row['raw_signals']:>6} {row['taken_signals']:>6} {row['utilization']*100:>5.0f}%"
        f" {row['wr_5bar']*100:>6.1f}% {row['pf_5bar']:>7.2f}"
        f" {row['wr_10bar']*100:>6.1f}% {row['pf_10bar']:>7.2f}"
        f" {row['rand_wr5']*100:>7.1f}% {row['gap_wr_pp']:>+6.1f}pp {row['signals_per_day']:>7.2f}"
    )

print()
print(sep)
print("PART 2B — 5m OPPORTUNITIES (new strategy candidates, ~61d of data)")
print(sep2)
print(hdr_main)
print(sep2)

for _, row in df_r[df_r["timeframe"] == "5m"].sort_values(
    ["strategy", "pair", "direction"]
).iterrows():
    print(
        f"{row['pair']:<6} {row['timeframe']:<4} {row['strategy']:<18} {row['direction']:<6}"
        f" {row['raw_signals']:>6} {row['taken_signals']:>6} {row['utilization']*100:>5.0f}%"
        f" {row['wr_5bar']*100:>6.1f}% {row['pf_5bar']:>7.2f}"
        f" {row['wr_10bar']*100:>6.1f}% {row['pf_10bar']:>7.2f}"
        f" {row['rand_wr5']*100:>7.1f}% {row['gap_wr_pp']:>+6.1f}pp {row['signals_per_day']:>7.2f}"
    )

print()
print(sep)
print("PART 2C — 1m ENTRY TIMING (RSI-7 hook signals, ~60d of data)")
print(sep2)
print(hdr_main)
print(sep2)

for _, row in df_r[df_r["timeframe"] == "1m"].sort_values(
    ["pair", "direction"]
).iterrows():
    print(
        f"{row['pair']:<6} {row['timeframe']:<4} {row['strategy']:<18} {row['direction']:<6}"
        f" {row['raw_signals']:>6} {row['taken_signals']:>6} {row['utilization']*100:>5.0f}%"
        f" {row['wr_5bar']*100:>6.1f}% {row['pf_5bar']:>7.2f}"
        f" {row['wr_10bar']*100:>6.1f}% {row['pf_10bar']:>7.2f}"
        f" {row['rand_wr5']*100:>7.1f}% {row['gap_wr_pp']:>+6.1f}pp {row['signals_per_day']:>7.2f}"
    )

# ── Grand totals ──────────────────────────────────────────────────────────────
total_15m_all = int(df_r[df_r["timeframe"] == "15m"]["taken_signals"].sum())
total_5m_all = int(df_r[df_r["timeframe"] == "5m"]["taken_signals"].sum())
total_1m_all = int(df_r[df_r["timeframe"] == "1m"]["taken_signals"].sum())

current_15m_sigs = int(
    df_r[
        (df_r["timeframe"] == "15m") &
        (df_r["strategy"].isin(["ha_stoch_rev", "vol_spike_rev", "regime_adaptive"]))
    ]["taken_signals"].sum()
)

# Extrapolate 5m/1m signals to 143-day basis for comparison
ratio_5m = DAYS_15M / DAYS_SHORT
total_5m_ext = int(total_5m_all * ratio_5m)
total_1m_ext = int(total_1m_all * ratio_5m)
grand_total_ext = total_15m_all + total_5m_ext + total_1m_ext

promising = df_r[
    (df_r["pf_5bar"] > 1.3) &
    (df_r["wr_5bar"] > 0.52) &
    (df_r["gap_wr_pp"] > 3.0) &
    (df_r["taken_signals"] >= 15)
]
marginal = df_r[
    (df_r["pf_5bar"] > 1.1) &
    (df_r["gap_wr_pp"] > 1.0) &
    (df_r["taken_signals"] >= 15) &
    ~(df_r.index.isin(promising.index))
]

print()
print(sep)
print("GRAND SUMMARY")
print(sep2)
print(f"  15m measured strategies (ha+vsr+ra):    {current_15m_sigs:>6} raw signals  "
      f"(system took ~152 of these per backtest)")
print(f"  15m total (all 3 strats x 3 pairs):     {total_15m_all:>6}")
print(f"  5m new candidates ({DAYS_SHORT}d actual):        {total_5m_all:>6}  "
      f"(~{total_5m_ext} extrapolated to 143d)")
print(f"  1m RSI hooks ({DAYS_SHORT}d actual):              {total_1m_all:>6}  "
      f"(~{total_1m_ext} extrapolated to 143d)")
print(f"  Grand total (all, extrapolated to 143d): {grand_total_ext:>6}")
print(f"  Current system trades (143d backtest):  {413:>6}")
print(f"  15m utilization (vs raw 3-strat sigs):  {413/current_15m_sigs*100:.1f}%")
print()
print(f"  Signal types PROMISING (PF>1.3 WR>52% gap>3pp n>=15):  {len(promising)}")
print(f"  Signal types MARGINAL  (PF>1.1 gap>1pp n>=15):          {len(marginal)}")

print()
print(sep)
print("VERDICT TABLE (sorted by PF desc, n >= 15 taken signals)")
print(sep2)
print(
    f"{'Pair':<6} {'TF':<4} {'Strategy':<18} {'Dir':<6}"
    f" {'N':>5} {'WR5':>7} {'PF5':>7} {'AvgFwd5%':>9} {'vs Rand':>8}  VERDICT"
)
print(sep2)

for _, row in df_r.sort_values("pf_5bar", ascending=False).iterrows():
    n = row["taken_signals"]
    if n < 15:
        continue
    pf = row["pf_5bar"]
    wr5 = row["wr_5bar"]
    gap = row["gap_wr_pp"]
    if pf > 1.3 and wr5 > 0.52 and gap > 3.0:
        verdict = "PROMISING"
    elif pf > 1.1 and gap > 1.0:
        verdict = "MARGINAL"
    else:
        verdict = "NO EDGE"
    print(
        f"{row['pair']:<6} {row['timeframe']:<4} {row['strategy']:<18} {row['direction']:<6}"
        f" {n:>5} {wr5*100:>6.1f}% {pf:>7.2f} {row['avg_fwd5_pct']:>+8.3f}%"
        f" {gap:>+7.1f}pp  {verdict}"
    )

print(sep)
print("Analysis complete.")

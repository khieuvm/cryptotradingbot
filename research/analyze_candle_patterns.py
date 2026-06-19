"""
Analysis: Candlestick Pattern Statistical Edge Measurement
Pairs: ETH/USDT:USDT, SOL/USDT:USDT
Timeframe: 15m
Purpose: Determine which patterns have real predictive power vs random baseline
         for use as filters in existing strategies.
"""
import sys
from pathlib import Path

ROOT = Path("e:/freqtrade")
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

DATA_DIR = ROOT / "data/okx/futures"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pair(pair_slug: str, timeframe: str = "15m") -> pd.DataFrame:
    pattern = f"{pair_slug}-{timeframe}*.feather"
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No data for {pattern} in {DATA_DIR}")
    df = pd.read_feather(files[-1])
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize("UTC")
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_volume_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"].ewm(span=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Forward return calculation
# ---------------------------------------------------------------------------

def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close forward returns at N+1, N+3, N+8."""
    df = df.copy()
    df["fwd_1"] = df["close"].pct_change().shift(-1) * 100
    df["fwd_3"] = (df["close"].shift(-3) / df["close"] - 1) * 100
    df["fwd_8"] = (df["close"].shift(-8) / df["close"] - 1) * 100
    return df


# ---------------------------------------------------------------------------
# Pattern detection functions
# ---------------------------------------------------------------------------

def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    body_curr = df["close"] - df["open"]
    body_prev = df["close"].shift(1) - df["open"].shift(1)
    cond = (
        (body_curr < 0) &                              # current red
        (body_prev > 0) &                              # prev green
        (df["close"] < df["open"].shift(1)) &          # close below prev open
        (df["open"] > df["close"].shift(1)) &          # open above prev close
        (abs(body_curr) > abs(body_prev))              # body fully engulfs
    )
    return cond


def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    body_curr = df["close"] - df["open"]
    body_prev = df["close"].shift(1) - df["open"].shift(1)
    cond = (
        (body_curr > 0) &                              # current green
        (body_prev < 0) &                              # prev red
        (df["close"] > df["open"].shift(1)) &          # close above prev open
        (df["open"] < df["close"].shift(1)) &          # open below prev close
        (abs(body_curr) > abs(body_prev))              # body fully engulfs
    )
    return cond


def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    body = abs(df["close"] - df["open"])
    candle_range = df["high"] - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    body_top = df[["open", "close"]].max(axis=1)
    # upper wick > 2x body, lower wick < 0.3x body, body in lower 30% of range
    cond = (
        (candle_range > 0) &
        (upper_wick > 2 * body) &
        (lower_wick < 0.3 * body.replace(0, np.nan).fillna(candle_range * 0.01)) &
        ((body_top - df["low"]) / candle_range.replace(0, np.nan) < 0.35)
    )
    return cond.fillna(False)


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    body = abs(df["close"] - df["open"])
    candle_range = df["high"] - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    body_bottom = df[["open", "close"]].min(axis=1)
    # lower wick > 2x body, upper wick < 0.3x body, body in upper 30% of range
    cond = (
        (candle_range > 0) &
        (lower_wick > 2 * body) &
        (upper_wick < 0.3 * body.replace(0, np.nan).fillna(candle_range * 0.01)) &
        ((df["high"] - body_bottom) / candle_range.replace(0, np.nan) < 0.35)
    )
    return cond.fillna(False)


def detect_doji_bearish(df: pd.DataFrame, rsi: pd.Series) -> pd.Series:
    """Doji at RSI > 65 (bearish reversal signal)."""
    body = abs(df["close"] - df["open"])
    candle_range = df["high"] - df["low"]
    cond = (
        (candle_range > 0) &
        (body < 0.1 * candle_range) &
        (rsi > 65)
    )
    return cond.fillna(False)


def detect_doji_bullish(df: pd.DataFrame, rsi: pd.Series) -> pd.Series:
    """Doji at RSI < 35 (bullish reversal signal)."""
    body = abs(df["close"] - df["open"])
    candle_range = df["high"] - df["low"]
    cond = (
        (candle_range > 0) &
        (body < 0.1 * candle_range) &
        (rsi < 35)
    )
    return cond.fillna(False)


def detect_evening_star(df: pd.DataFrame) -> pd.Series:
    """
    3-candle bearish reversal:
    bar-2: strong green (body > 60% range)
    bar-1: small body (doji-like, body < 35% range)
    bar-0: strong red (body > 60% range), closes below midpoint of bar-2
    """
    body_0 = df["close"] - df["open"]
    body_1 = (df["close"] - df["open"]).shift(1)
    body_2 = (df["close"] - df["open"]).shift(2)
    range_0 = df["high"] - df["low"]
    range_1 = (df["high"] - df["low"]).shift(1)
    range_2 = (df["high"] - df["low"]).shift(2)
    mid_2 = (df["open"].shift(2) + df["close"].shift(2)) / 2
    cond = (
        (body_2 > 0.6 * range_2) &                    # bar-2 strong green
        (abs(body_1) < 0.35 * range_1) &              # bar-1 small/doji
        (body_0 < -0.6 * range_0) &                   # bar-0 strong red
        (df["close"] < mid_2)                          # closes below bar-2 midpoint
    )
    return cond.fillna(False)


def detect_morning_star(df: pd.DataFrame) -> pd.Series:
    """
    3-candle bullish reversal:
    bar-2: strong red (body > 60% range)
    bar-1: small body (doji-like, body < 35% range)
    bar-0: strong green (body > 60% range), closes above midpoint of bar-2
    """
    body_0 = df["close"] - df["open"]
    body_1 = (df["close"] - df["open"]).shift(1)
    body_2 = (df["close"] - df["open"]).shift(2)
    range_0 = df["high"] - df["low"]
    range_1 = (df["high"] - df["low"]).shift(1)
    range_2 = (df["high"] - df["low"]).shift(2)
    mid_2 = (df["open"].shift(2) + df["close"].shift(2)) / 2
    cond = (
        (body_2 < -0.6 * range_2) &                   # bar-2 strong red
        (abs(body_1) < 0.35 * range_1) &              # bar-1 small/doji
        (body_0 > 0.6 * range_0) &                    # bar-0 strong green
        (df["close"] > mid_2)                          # closes above bar-2 midpoint
    )
    return cond.fillna(False)


def detect_bullish_marubozu(df: pd.DataFrame) -> pd.Series:
    """Body > 85% of range, green candle."""
    body = df["close"] - df["open"]
    candle_range = df["high"] - df["low"]
    cond = (
        (candle_range > 0) &
        (body > 0) &
        (body / candle_range > 0.85)
    )
    return cond.fillna(False)


def detect_bearish_marubozu(df: pd.DataFrame) -> pd.Series:
    """Body > 85% of range, red candle."""
    body = df["close"] - df["open"]
    candle_range = df["high"] - df["low"]
    cond = (
        (candle_range > 0) &
        (body < 0) &
        (abs(body) / candle_range > 0.85)
    )
    return cond.fillna(False)


# ---------------------------------------------------------------------------
# Statistical measurement engine
# ---------------------------------------------------------------------------

def grade_z(z: float, n: int) -> str:
    if z > 2.5 and n > 50:
        return "A"
    elif z > 1.96 and n > 30:
        return "B"
    elif z > 1.0:
        return "C"
    else:
        return "F"


def measure_pattern(
    df: pd.DataFrame,
    signal_mask: pd.Series,
    direction: int,       # +1 for bullish, -1 for bearish
    pattern_name: str,
    pair_name: str,
    vol_ema: pd.Series,
):
    """
    For signals in signal_mask:
    - direction * forward_return gives the "signed" return in the predicted direction
    - Compare to random baseline
    """
    # Drop last 8 rows where fwd returns are NaN
    valid = df.index[signal_mask & df["fwd_8"].notna()]
    n = len(valid)

    # Random baseline: all bars with valid fwd returns
    all_valid = df.index[df["fwd_8"].notna()]
    rand_fwd1 = df.loc[all_valid, "fwd_1"] * direction
    rand_fwd3 = df.loc[all_valid, "fwd_3"] * direction
    rand_fwd8 = df.loc[all_valid, "fwd_8"] * direction

    rand_mean1 = rand_fwd1.mean()
    rand_mean3 = rand_fwd3.mean()
    rand_mean8 = rand_fwd8.mean()
    rand_std1  = rand_fwd1.std()
    rand_std3  = rand_fwd3.std()
    rand_std8  = rand_fwd8.std()

    result = {
        "pattern": pattern_name,
        "pair": pair_name,
        "direction": "LONG" if direction == 1 else "SHORT",
        "n": n,
    }

    if n < 5:
        result["skip"] = True
        return result

    sig_fwd1 = df.loc[valid, "fwd_1"] * direction
    sig_fwd3 = df.loc[valid, "fwd_3"] * direction
    sig_fwd8 = df.loc[valid, "fwd_8"] * direction

    for horizon, sig, rand_mean, rand_std in [
        ("N+1", sig_fwd1, rand_mean1, rand_std1),
        ("N+3", sig_fwd3, rand_mean3, rand_std3),
        ("N+8", sig_fwd8, rand_mean8, rand_std8),
    ]:
        mean_r = sig.mean()
        wr = (sig > 0).mean() * 100
        z = (mean_r - rand_mean) / (rand_std / np.sqrt(n)) if rand_std > 0 else 0
        result[f"{horizon}_avg"] = round(mean_r, 4)
        result[f"{horizon}_wr"] = round(wr, 1)
        result[f"{horizon}_z"] = round(z, 2)
        result[f"{horizon}_grade"] = grade_z(z, n)

    # Volume-filtered subset
    vol_mask = signal_mask & (df["volume"] > 1.5 * vol_ema)
    vol_valid = df.index[vol_mask & df["fwd_8"].notna()]
    nv = len(vol_valid)
    result["n_vol"] = nv
    if nv >= 5:
        sv_fwd1 = df.loc[vol_valid, "fwd_1"] * direction
        sv_fwd3 = df.loc[vol_valid, "fwd_3"] * direction
        sv_fwd8 = df.loc[vol_valid, "fwd_8"] * direction
        for horizon, sig, rand_mean, rand_std in [
            ("N+1", sv_fwd1, rand_mean1, rand_std1),
            ("N+3", sv_fwd3, rand_mean3, rand_std3),
            ("N+8", sv_fwd8, rand_mean8, rand_std8),
        ]:
            mean_r = sig.mean()
            wr = (sig > 0).mean() * 100
            z = (mean_r - rand_mean) / (rand_std / np.sqrt(nv)) if rand_std > 0 else 0
            result[f"vol_{horizon}_avg"] = round(mean_r, 4)
            result[f"vol_{horizon}_wr"] = round(wr, 1)
            result[f"vol_{horizon}_z"] = round(z, 2)
            result[f"vol_{horizon}_grade"] = grade_z(z, nv)

    return result


def print_pattern_result(r: dict):
    if r.get("skip"):
        print(f"Pattern: {r['pattern']} ({r['pair']})  N={r['n']}  [INSUFFICIENT DATA - skipped]")
        return

    print(f"Pattern: {r['pattern']} ({r['pair']})  N={r['n']}  Direction={r['direction']}")
    for horizon in ["N+1", "N+3", "N+8"]:
        avg = r.get(f"{horizon}_avg", "n/a")
        wr  = r.get(f"{horizon}_wr",  "n/a")
        z   = r.get(f"{horizon}_z",   "n/a")
        g   = r.get(f"{horizon}_grade","?")
        sign = "+" if isinstance(avg, float) and avg > 0 else ""
        print(f"  {horizon} return:  avg={sign}{avg}%  WR={wr}%  Z={z}  [{g}]")

    nv = r.get("n_vol", 0)
    if nv and nv >= 5:
        avg_v = r.get("vol_N+3_avg", "n/a")
        z_v   = r.get("vol_N+3_z", "n/a")
        g_v   = r.get("vol_N+3_grade", "?")
        sign = "+" if isinstance(avg_v, float) and avg_v > 0 else ""
        improvement = ""
        base_z = r.get("N+3_z", 0)
        if isinstance(z_v, float) and isinstance(base_z, float) and z_v > base_z:
            improvement = "  <- vol improvement"
        print(f"  With vol>1.5x:  N={nv}  N+3 avg={sign}{avg_v}%  Z={z_v}  [{g_v}]{improvement}")
    print()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_pair(pair_slug: str, pair_name: str) -> list:
    print(f"\n{'='*70}")
    print(f"  PAIR: {pair_name}")
    print(f"{'='*70}\n")

    df = load_pair(pair_slug, "15m")
    df = compute_forward_returns(df)
    rsi = compute_rsi(df["close"], period=14)
    vol_ema = compute_volume_ema(df, period=20)

    print(f"Data range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total bars: {len(df)}")
    random_mean_fwd3 = (df["fwd_3"].dropna()).mean()
    random_std_fwd3  = (df["fwd_3"].dropna()).std()
    print(f"Random baseline N+3: mean={random_mean_fwd3:.4f}%  std={random_std_fwd3:.4f}%\n")

    patterns_cfg = [
        # (name, mask_series, direction)
        ("bearish_engulfing",  detect_bearish_engulfing(df),         -1),
        ("bullish_engulfing",  detect_bullish_engulfing(df),         +1),
        ("shooting_star",      detect_shooting_star(df),             -1),
        ("hammer",             detect_hammer(df),                    +1),
        ("doji_bearish",       detect_doji_bearish(df, rsi),         -1),
        ("doji_bullish",       detect_doji_bullish(df, rsi),         +1),
        ("evening_star",       detect_evening_star(df),              -1),
        ("morning_star",       detect_morning_star(df),              +1),
        ("bullish_marubozu",   detect_bullish_marubozu(df),          +1),
        ("bearish_marubozu",   detect_bearish_marubozu(df),          -1),
    ]

    results = []
    for name, mask, direction in patterns_cfg:
        r = measure_pattern(df, mask, direction, name, pair_name, vol_ema)
        print_pattern_result(r)
        results.append(r)

    return results


def print_ranking_table(all_results: list, horizon: str = "N+3"):
    print(f"\n{'='*70}")
    print(f"  RANKING TABLE — Top patterns by Z-score at {horizon}")
    print(f"{'='*70}")
    z_key  = f"{horizon}_z"
    avg_key = f"{horizon}_avg"
    wr_key  = f"{horizon}_wr"
    g_key   = f"{horizon}_grade"

    rows = [r for r in all_results if not r.get("skip") and z_key in r]
    rows_sorted = sorted(rows, key=lambda r: r.get(z_key, -999), reverse=True)

    print(f"\n{'Pattern':<25} {'Pair':<6} {'Dir':<6} {'N':>5}  {'Avg%':>7}  {'WR%':>6}  {'Z':>6}  {'Grade'}")
    print("-" * 75)
    for r in rows_sorted:
        avg = r[avg_key]
        sign = "+" if avg > 0 else ""
        print(
            f"{r['pattern']:<25} {r['pair']:<6} {r['direction']:<6} {r['n']:>5}  "
            f"{sign}{avg:>6.3f}%  {r[wr_key]:>5.1f}%  {r[z_key]:>6.2f}  [{r[g_key]}]"
        )

    # Also show vol-filtered top performers
    print(f"\n--- With volume > 1.5x EMA filter ---")
    vol_z_key  = f"vol_{horizon}_z"
    vol_avg_key = f"vol_{horizon}_avg"
    vol_wr_key  = f"vol_{horizon}_wr"
    vol_g_key   = f"vol_{horizon}_grade"
    vol_rows = [r for r in all_results if not r.get("skip") and vol_z_key in r]
    vol_rows_sorted = sorted(vol_rows, key=lambda r: r.get(vol_z_key, -999), reverse=True)
    print(f"\n{'Pattern':<25} {'Pair':<6} {'Dir':<6} {'N_vol':>6}  {'Avg%':>7}  {'WR%':>6}  {'Z':>6}  {'Grade'}")
    print("-" * 75)
    for r in vol_rows_sorted[:15]:
        avg = r[vol_avg_key]
        sign = "+" if avg > 0 else ""
        print(
            f"{r['pattern']:<25} {r['pair']:<6} {r['direction']:<6} {r['n_vol']:>6}  "
            f"{sign}{avg:>6.3f}%  {r[vol_wr_key]:>5.1f}%  {r[vol_z_key]:>6.2f}  [{r[vol_g_key]}]"
        )


def print_strategy_recommendations(all_results: list):
    print(f"\n{'='*70}")
    print("  STRATEGY FILTER RECOMMENDATIONS")
    print(f"{'='*70}\n")

    z_key = "N+3_z"
    g_key = "N+3_grade"
    avg_key = "N+3_avg"

    # Build lookup by pattern+pair
    lookup = {(r["pattern"], r["pair"]): r for r in all_results if not r.get("skip")}

    def best_for(patterns: list, direction: str, note: str):
        print(f"  {note}")
        candidates = [
            r for r in all_results
            if not r.get("skip")
            and r["direction"] == direction
            and r["pattern"] in patterns
            and r.get(g_key, "F") in ("A", "B", "C")
        ]
        candidates.sort(key=lambda r: r.get(z_key, -999), reverse=True)
        if not candidates:
            print("    No patterns with C+ grade found.")
        for r in candidates[:5]:
            print(
                f"    {r['pattern']} ({r['pair']}): "
                f"Z={r.get(z_key,'n/a')} [{r.get(g_key,'?')}]  "
                f"avg={r.get(avg_key,'n/a')}%  N={r['n']}"
            )
            # Vol-filtered if better
            vz = r.get("vol_N+3_z")
            if vz and isinstance(vz, float) and vz > r.get(z_key, 0):
                print(
                    f"      -> With vol>1.5x: Z={vz} [{r.get('vol_N+3_grade','?')}]  "
                    f"avg={r.get('vol_N+3_avg','n/a')}%  N={r.get('n_vol')}"
                )
        print()

    reversal_patterns = [
        "bearish_engulfing", "bullish_engulfing", "shooting_star",
        "hammer", "doji_bearish", "doji_bullish", "evening_star", "morning_star"
    ]
    momentum_patterns = ["bullish_marubozu", "bearish_marubozu"]
    all_short_patterns = ["bearish_engulfing", "shooting_star", "doji_bearish",
                          "evening_star", "bearish_marubozu"]
    all_long_patterns  = ["bullish_engulfing", "hammer", "doji_bullish",
                          "morning_star", "bullish_marubozu"]

    print("1. volume_spike_rev — SHORT reversal confirmation")
    print("   (Big red body + volume spike. Looking for additional bearish pattern confirmation)")
    best_for(all_short_patterns, "SHORT",
             "Best SHORT reversal filters for volume_spike_rev:")

    print("2. cb_adx_breakout — Momentum continuation confirmation")
    print("   (Bollinger compression breakout. Looking for momentum candles on breakout direction)")
    best_for(momentum_patterns + reversal_patterns, "LONG",
             "Best LONG momentum filters for cb_adx_breakout (upside breakout):")
    best_for(momentum_patterns + reversal_patterns, "SHORT",
             "Best SHORT momentum filters for cb_adx_breakout (downside breakout):")

    print("3. ha_stoch_rev — Exhaustion/reversal after HA run")
    print("   (Reversal after Heikin-Ashi trend. Looking for exhaustion patterns at extremes)")
    best_for(all_short_patterns, "SHORT",
             "Best SHORT exhaustion filters for ha_stoch_rev:")
    best_for(all_long_patterns,  "LONG",
             "Best LONG exhaustion filters for ha_stoch_rev:")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  CANDLESTICK PATTERN STATISTICAL EDGE — 15m ETH & SOL")
    print("=" * 70)

    all_results = []
    for slug, name in [("ETH_USDT_USDT", "ETH"), ("SOL_USDT_USDT", "SOL")]:
        results = analyze_pair(slug, name)
        all_results.extend(results)

    print_ranking_table(all_results, horizon="N+3")
    print_ranking_table(all_results, horizon="N+1")
    print_strategy_recommendations(all_results)

    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*70}")

"""Backtest 10 candidate 5m scalping strategies on real data.

Tests each strategy across BTC, ETH, SOL, SPX with forward-scan exits.
Outputs: trades, win rate, profit factor, total profit for each.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
TAKER = 0.0005
PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT"]


def load(pair):
    fp = DATA_DIR / f"{pair}-5m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp)
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


def scan_trades(entries, close, high, low, atr, sl_mult, tp_mult, is_long_arr, max_bars=18):
    """Forward-scan from each entry signal to find SL/TP/time-cut exit."""
    n = len(close)
    trades = []
    i = 0
    while i < n:
        if entries[i] == 0:
            i += 1
            continue
        entry = close[i]
        a = atr[i]
        if a <= 0 or entry <= 0:
            i += 1
            continue

        is_long = is_long_arr[i] > 0

        if is_long:
            sl_p = entry - sl_mult * a
            tp_p = entry + tp_mult * a
        else:
            sl_p = entry + sl_mult * a
            tp_p = entry - tp_mult * a

        exited = False
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if is_long:
                if low[j] <= sl_p:
                    trades.append(-sl_mult * a / entry - 2 * TAKER)
                    exited = True
                    i = j + 1
                    break
                if high[j] >= tp_p:
                    trades.append(tp_mult * a / entry - 2 * TAKER)
                    exited = True
                    i = j + 1
                    break
            else:
                if high[j] >= sl_p:
                    trades.append(-sl_mult * a / entry - 2 * TAKER)
                    exited = True
                    i = j + 1
                    break
                if low[j] <= tp_p:
                    trades.append(tp_mult * a / entry - 2 * TAKER)
                    exited = True
                    i = j + 1
                    break

        if not exited:
            exit_idx = min(i + max_bars, n - 1)
            if is_long:
                trades.append((close[exit_idx] - entry) / entry - 2 * TAKER)
            else:
                trades.append((entry - close[exit_idx]) / entry - 2 * TAKER)
            i = exit_idx + 1

    return trades


def compute_base(df):
    """Pre-compute common indicators."""
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    o = df["open"].values
    v = df["volume"].values.astype(float)

    atr14 = ta.atr(df["high"], df["low"], df["close"], length=14).values
    rsi9 = ta.rsi(df["close"], length=9).values
    rsi14 = ta.rsi(df["close"], length=14).values

    ema8 = ta.ema(df["close"], length=8).values
    ema21 = ta.ema(df["close"], length=21).values
    ema50 = ta.ema(df["close"], length=50).values

    adx_r = ta.adx(df["high"], df["low"], df["close"], length=14)
    adx = adx_r.iloc[:, 0].values if adx_r is not None else np.full(len(df), 25.0)

    vol_ema = ta.ema(df["volume"].astype(float), length=20).values
    vol_ratio = v / (vol_ema + 1e-10)

    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is not None:
        bb_upper = bb.iloc[:, 0].values
        bb_mid = bb.iloc[:, 1].values
        bb_lower = bb.iloc[:, 2].values
    else:
        bb_upper = c + atr14
        bb_mid = c
        bb_lower = c - atr14

    # Hour of day (UTC)
    hours = pd.to_datetime(df["date"]).dt.hour.values

    # Body and shadows
    body = c - o
    body_abs = np.abs(body)
    bar_range = h - lo

    return {
        "close": c, "high": h, "low": lo, "open": o, "volume": v,
        "atr": atr14, "rsi9": rsi9, "rsi14": rsi14,
        "ema8": ema8, "ema21": ema21, "ema50": ema50,
        "adx": adx, "vol_ratio": vol_ratio,
        "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
        "hours": hours, "body": body, "body_abs": body_abs, "bar_range": bar_range,
    }


# =============================================================================
# STRATEGY SIGNAL GENERATORS
# Each returns (entries, is_long) arrays
# =============================================================================

def s1_session_bias(d, pair):
    """S1: Session time bias - long at US open (14:00-15:00), short at Asia dead (05:00-06:00)."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(50, n):
        h = d["hours"][i]
        # Long: US session open (strong momentum)
        if h in (14, 15) and d["vol_ratio"][i] > 1.0:
            entries[i] = 1
            is_long[i] = 1
        # Short: Asia dead zone (mean-reversion after US close)
        elif h in (5, 6) and d["vol_ratio"][i] > 0.8:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def s2_rsi_extreme_volume(d, pair):
    """S2: RSI extreme + volume spike reversal."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(20, n):
        if d["vol_ratio"][i] < 2.0:
            continue
        # Long: RSI oversold
        if d["rsi9"][i] < 20:
            entries[i] = 1
            is_long[i] = 1
        # Short: RSI overbought
        elif d["rsi9"][i] > 80:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def s3_ema_pullback(d, pair):
    """S3: EMA pullback in strong trend (ADX>30, price touches EMA8)."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(50, n):
        if d["adx"][i] < 30:
            continue
        # Uptrend: EMA8 > EMA21 > EMA50, price touched EMA8
        if d["ema8"][i] > d["ema21"][i] > d["ema50"][i]:
            if d["low"][i] <= d["ema8"][i] * 1.001 and d["close"][i] > d["ema8"][i]:
                entries[i] = 1
                is_long[i] = 1
        # Downtrend
        elif d["ema8"][i] < d["ema21"][i] < d["ema50"][i]:
            if d["high"][i] >= d["ema8"][i] * 0.999 and d["close"][i] < d["ema8"][i]:
                entries[i] = 1
                is_long[i] = -1

    return entries, is_long


def s4_bb_bounce(d, pair):
    """S4: Bollinger Band bounce with RSI confirmation."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(25, n):
        # Long: close below lower BB + RSI < 30
        if d["close"][i] <= d["bb_lower"][i] and d["rsi14"][i] < 30:
            entries[i] = 1
            is_long[i] = 1
        # Short: close above upper BB + RSI > 70
        elif d["close"][i] >= d["bb_upper"][i] and d["rsi14"][i] > 70:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def s5_momentum_burst(d, pair):
    """S5: 3 consecutive same-direction bars + increasing volume."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(53, n):
        # 3 green bars with increasing volume
        if (d["body"][i-2] > 0 and d["body"][i-1] > 0 and d["body"][i] > 0
            and d["volume"][i] > d["volume"][i-1] > d["volume"][i-2]
            and d["vol_ratio"][i] > 1.5):
            entries[i] = 1
            is_long[i] = 1
        # 3 red bars with increasing volume
        elif (d["body"][i-2] < 0 and d["body"][i-1] < 0 and d["body"][i] < 0
              and d["volume"][i] > d["volume"][i-1] > d["volume"][i-2]
              and d["vol_ratio"][i] > 1.5):
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def s6_vwap_reversion(d, pair):
    """S6: VWAP deviation reversion (>1.5% from rolling VWAP)."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    # Simple VWAP approximation: cumulative(price*volume) / cumulative(volume) over 96 bars
    window = 96
    for i in range(window, n):
        pv = np.sum(d["close"][i-window:i] * d["volume"][i-window:i])
        vol_sum = np.sum(d["volume"][i-window:i])
        if vol_sum <= 0:
            continue
        vwap = pv / vol_sum
        dev = (d["close"][i] - vwap) / vwap

        # Long: price far below VWAP
        if dev < -0.015 and d["rsi14"][i] < 35:
            entries[i] = 1
            is_long[i] = 1
        # Short: price far above VWAP
        elif dev > 0.015 and d["rsi14"][i] > 65:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def s7_large_candle_fade(d, pair):
    """S7: Fade large candles (>2.5x ATR range) on next bar."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(15, n):
        prev_range = d["bar_range"][i-1]
        if d["atr"][i-1] <= 0:
            continue
        if prev_range < 2.5 * d["atr"][i-1]:
            continue

        # Previous was a big green candle -> fade short
        if d["body"][i-1] > 0 and d["body"][i-1] > 0.6 * prev_range:
            entries[i] = 1
            is_long[i] = -1
        # Previous was a big red candle -> fade long
        elif d["body"][i-1] < 0 and abs(d["body"][i-1]) > 0.6 * prev_range:
            entries[i] = 1
            is_long[i] = 1

    return entries, is_long


def s8_atr_contraction_breakout(d, pair):
    """S8: ATR contracts to 50% of 20-bar avg then expands — enter breakout direction."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    atr_sma = pd.Series(d["atr"]).rolling(20, min_periods=20).mean().values

    for i in range(25, n):
        if np.isnan(atr_sma[i-1]) or atr_sma[i-1] <= 0:
            continue
        # Previous bar: ATR was contracted
        prev_ratio = d["atr"][i-1] / atr_sma[i-1]
        curr_ratio = d["atr"][i] / atr_sma[i] if atr_sma[i] > 0 else 1.0

        if prev_ratio < 0.6 and curr_ratio > 0.9 and d["vol_ratio"][i] > 1.2:
            # Direction from the expansion bar
            if d["body"][i] > 0:
                entries[i] = 1
                is_long[i] = 1
            elif d["body"][i] < 0:
                entries[i] = 1
                is_long[i] = -1

    return entries, is_long


def s9_rsi_divergence(d, pair):
    """S9: RSI bullish/bearish divergence (simple: price lower low, RSI higher low)."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    lookback = 20
    for i in range(lookback + 5, n):
        # Find lowest price in lookback
        window_lo = d["low"][i-lookback:i+1]
        window_rsi = d["rsi14"][i-lookback:i+1]

        # Current bar near local low
        if d["low"][i] > np.min(window_lo) * 1.002:
            # Check for bearish divergence (price near high, RSI lower)
            window_hi = d["high"][i-lookback:i+1]
            if d["high"][i] >= np.max(window_hi) * 0.998:
                # Price at high, check if RSI is lower than previous high's RSI
                prev_high_idx = np.argmax(window_hi[:-5])  # previous high
                if window_rsi[-1] < window_rsi[prev_high_idx] - 5 and d["rsi14"][i] > 60:
                    entries[i] = 1
                    is_long[i] = -1
            continue

        # Bullish divergence: price at new low, RSI higher
        if d["low"][i] <= np.min(window_lo) * 1.002:
            prev_low_idx = np.argmin(window_lo[:-5])
            if len(window_rsi) > prev_low_idx and prev_low_idx < lookback - 4:
                if window_rsi[-1] > window_rsi[prev_low_idx] + 5 and d["rsi14"][i] < 40:
                    entries[i] = 1
                    is_long[i] = 1

    return entries, is_long


def s10_opening_hour_momentum(d, pair):
    """S10: First 5m bar of major session (13:00 UTC = US pre-market) sets direction if > 1.5x ATR."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    # Key session openings: 00:00 (Asia), 08:00 (Europe), 13:00 (US)
    session_hours = {0, 8, 13}

    for i in range(50, n):
        h = d["hours"][i]
        if h not in session_hours:
            continue
        if d["atr"][i] <= 0:
            continue
        candle_size = d["bar_range"][i]
        if candle_size < 1.5 * d["atr"][i]:
            continue

        # Strong opening candle sets direction
        if d["body"][i] > 0.5 * candle_size:
            entries[i] = 1
            is_long[i] = 1
        elif d["body"][i] < -0.5 * candle_size:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


# =============================================================================
# MAIN
# =============================================================================

STRATEGIES = [
    ("S1_Session_Bias", s1_session_bias, 1.5, 2.5, 18),
    ("S2_RSI_Extreme_Vol", s2_rsi_extreme_volume, 1.2, 2.0, 12),
    ("S3_EMA_Pullback", s3_ema_pullback, 1.5, 3.0, 18),
    ("S4_BB_Bounce", s4_bb_bounce, 1.2, 2.0, 12),
    ("S5_Momentum_Burst", s5_momentum_burst, 1.5, 2.5, 12),
    ("S6_VWAP_Reversion", s6_vwap_reversion, 1.5, 2.5, 18),
    ("S7_Large_Candle_Fade", s7_large_candle_fade, 1.5, 2.0, 12),
    ("S8_ATR_Contract_BO", s8_atr_contraction_breakout, 1.5, 2.5, 18),
    ("S9_RSI_Divergence", s9_rsi_divergence, 1.5, 2.5, 18),
    ("S10_Session_Open_Mom", s10_opening_hour_momentum, 1.5, 2.5, 12),
]


def main():
    print("=" * 80)
    print("10-STRATEGY 5m SCALPING CANDIDATE BACKTEST")
    print("Pairs:", ", ".join(PAIRS))
    print("Period: 2026-01-01 to 2026-05-21 (142 days)")
    print("Costs: 0.05% taker per side (0.10% round trip)")
    print("=" * 80)

    # Load and precompute
    print("\nLoading data and computing indicators...")
    pair_data = {}
    for p in PAIRS:
        df = load(p)
        if df.empty:
            print(f"  {p}: NO DATA")
            continue
        d = compute_base(df)
        pair_data[p] = d
        print(f"  {p}: {len(df)} bars")

    # Run each strategy
    print("\n" + "-" * 80)
    print(f"{'Strategy':<25} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Profit%':>9} {'Avg%':>8} {'MaxDD%':>8}")
    print("-" * 80)

    results = []

    for name, func, sl, tp, max_bars in STRATEGIES:
        total_trades = []

        for p, d in pair_data.items():
            entries, is_long = func(d, p)
            trades = scan_trades(entries, d["close"], d["high"], d["low"],
                                 d["atr"], sl, tp, is_long, max_bars)
            total_trades.extend(trades)

        if not total_trades:
            print(f"{name:<25} {'0':>7} {'--':>7} {'--':>7} {'--':>9} {'--':>8} {'--':>8}")
            results.append({"name": name, "trades": 0, "profit": 0})
            continue

        arr = np.array(total_trades)
        n_trades = len(arr)
        wins = arr[arr > 0]
        losses = arr[arr <= 0]
        wr = len(wins) / n_trades * 100
        pf = wins.sum() / max(abs(losses.sum()), 0.0001) if len(losses) > 0 else 99.0
        total_pnl = arr.sum() * 100
        avg_pnl = arr.mean() * 100

        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        max_dd = abs(np.min(cum - peak)) * 100

        print(f"{name:<25} {n_trades:>7} {wr:>6.1f}% {pf:>7.2f} {total_pnl:>+8.2f}% {avg_pnl:>+7.3f}% {max_dd:>7.2f}%")

        results.append({
            "name": name, "trades": n_trades, "wr": wr, "pf": pf,
            "profit": total_pnl, "avg": avg_pnl, "max_dd": max_dd,
        })

    # Per-pair breakdown for top strategies
    print("\n" + "=" * 80)
    print("PER-PAIR BREAKDOWN (Top strategies)")
    print("=" * 80)

    # Sort by profit
    profitable = [r for r in results if r.get("profit", 0) > 0]
    profitable.sort(key=lambda x: x["profit"], reverse=True)

    if not profitable:
        print("\nNo profitable strategies found. All candidates lose money on 5m.")
        print("Consider:")
        print("  1. Wider parameter ranges")
        print("  2. Different structural edges (funding, session-specific)")
        print("  3. 5m scalping may not have viable edges for these signal types")
    else:
        print(f"\n{'Strategy':<25} {'Pair':<15} {'Trades':>7} {'WR%':>7} {'Profit%':>9}")
        print("-" * 70)
        for r in profitable[:5]:
            name = r["name"]
            # Find the function
            for sname, func, sl, tp, max_bars in STRATEGIES:
                if sname == name:
                    for p, d in pair_data.items():
                        entries, is_long = func(d, p)
                        trades = scan_trades(entries, d["close"], d["high"], d["low"],
                                             d["atr"], sl, tp, is_long, max_bars)
                        if trades:
                            arr = np.array(trades)
                            wr = len(arr[arr > 0]) / len(arr) * 100
                            pnl = arr.sum() * 100
                            print(f"{name:<25} {p:<15} {len(arr):>7} {wr:>6.1f}% {pnl:>+8.2f}%")
                    break

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    viable = [r for r in results if r.get("trades", 0) >= 30 and r.get("profit", 0) > 0]
    print(f"  Total strategies tested: {len(results)}")
    print(f"  Profitable (any trades): {len(profitable)}")
    print(f"  Viable (>30 trades + profitable): {len(viable)}")

    if viable:
        print("\n  VIABLE for further development:")
        for r in viable:
            print(f"    {r['name']}: {r['profit']:+.2f}% | {r['trades']} trades | WR {r['wr']:.1f}% | PF {r['pf']:.2f}")
    else:
        print("\n  No strategy meets viability threshold (>30 trades + profitable).")

    # Comparison to random baseline
    print("\n  RANDOM BASELINE (buy random, same SL/TP):")
    for sl, tp in [(1.5, 2.5), (1.2, 2.0)]:
        all_random = []
        for p, d in pair_data.items():
            n = len(d["close"])
            np.random.seed(42)
            random_entries = np.zeros(n)
            random_long = np.zeros(n)
            idxs = np.random.choice(range(50, n-20), size=200, replace=False)
            random_entries[idxs] = 1
            random_long[idxs] = np.random.choice([1, -1], size=200)
            trades = scan_trades(random_entries, d["close"], d["high"], d["low"],
                                 d["atr"], sl, tp, random_long, 18)
            all_random.extend(trades)
        arr = np.array(all_random)
        wr = len(arr[arr > 0]) / len(arr) * 100 if len(arr) > 0 else 0
        pnl = arr.sum() * 100
        print(f"    SL={sl} TP={tp}: {len(arr)} trades, WR={wr:.1f}%, profit={pnl:+.2f}%")


if __name__ == "__main__":
    main()

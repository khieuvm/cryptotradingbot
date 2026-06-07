"""Round 2: Structural edge strategies for 5m scalping.

Focuses on what actually worked in prior analysis:
- Session timing (SPX US hours)
- Funding pre-settlement drift
- Tight scalp exits (1:1 to 1:1.5 RR with high WR)
- Pair-specific behavior
- Microstructure patterns (spread, volume profile)
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


def scan_trades_fixed_pct(entries, close, is_long_arr, sl_pct, tp_pct, max_bars=12):
    """Fixed percentage SL/TP instead of ATR-based."""
    n = len(close)
    trades = []
    i = 0
    while i < n:
        if entries[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        is_long = is_long_arr[i] > 0

        if is_long:
            sl_p = entry * (1 - sl_pct)
            tp_p = entry * (1 + tp_pct)
        else:
            sl_p = entry * (1 + sl_pct)
            tp_p = entry * (1 - tp_pct)

        exited = False
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if is_long:
                if close[j] <= sl_p:  # Use close for simpler model
                    trades.append(-sl_pct - 2 * TAKER)
                    exited = True
                    i = j + 1
                    break
                if close[j] >= tp_p:
                    trades.append(tp_pct - 2 * TAKER)
                    exited = True
                    i = j + 1
                    break
            else:
                if close[j] >= sl_p:
                    trades.append(-sl_pct - 2 * TAKER)
                    exited = True
                    i = j + 1
                    break
                if close[j] <= tp_p:
                    trades.append(tp_pct - 2 * TAKER)
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


def scan_trades_atr(entries, close, high, low, atr, sl_mult, tp_mult, is_long_arr, max_bars=12):
    """ATR-based SL/TP with HL for stop check."""
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
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    o = df["open"].values
    v = df["volume"].values.astype(float)

    atr14 = ta.atr(df["high"], df["low"], df["close"], length=14).values
    atr5 = ta.atr(df["high"], df["low"], df["close"], length=5).values
    rsi9 = ta.rsi(df["close"], length=9).values
    rsi3 = ta.rsi(df["close"], length=3).values

    ema8 = ta.ema(df["close"], length=8).values
    ema21 = ta.ema(df["close"], length=21).values

    adx_r = ta.adx(df["high"], df["low"], df["close"], length=14)
    adx = adx_r.iloc[:, 0].values if adx_r is not None else np.full(len(df), 25.0)
    plus_di = adx_r.iloc[:, 1].values if adx_r is not None else np.full(len(df), 25.0)
    minus_di = adx_r.iloc[:, 2].values if adx_r is not None else np.full(len(df), 25.0)

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

    hours = pd.to_datetime(df["date"]).dt.hour.values
    minutes = pd.to_datetime(df["date"]).dt.minute.values
    body = c - o
    body_abs = np.abs(body)
    bar_range = h - lo

    # Stochastic RSI
    stoch_rsi = ta.stochrsi(df["close"], length=14)
    if stoch_rsi is not None:
        stoch_k = stoch_rsi.iloc[:, 0].values
        stoch_d = stoch_rsi.iloc[:, 1].values
    else:
        stoch_k = np.full(len(df), 50.0)
        stoch_d = np.full(len(df), 50.0)

    return {
        "close": c, "high": h, "low": lo, "open": o, "volume": v,
        "atr14": atr14, "atr5": atr5, "rsi9": rsi9, "rsi3": rsi3,
        "ema8": ema8, "ema21": ema21,
        "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
        "vol_ratio": vol_ratio,
        "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
        "hours": hours, "minutes": minutes,
        "body": body, "body_abs": body_abs, "bar_range": bar_range,
        "stoch_k": stoch_k, "stoch_d": stoch_d,
    }


# =============================================================================
# STRUCTURAL EDGE STRATEGIES
# =============================================================================

def s1_spx_us_session_long(d, pair):
    """SPX-only: Long bias during US market hours (13:30-20:00 UTC)."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)
    if "SPX" not in pair:
        return entries, is_long

    for i in range(50, n):
        h = d["hours"][i]
        # Only during US session (13-20 UTC)
        if 13 <= h <= 19:
            # Pullback entry: price below EMA8 in uptrend session
            if d["close"][i] < d["ema8"][i] and d["rsi3"][i] < 30:
                entries[i] = 1
                is_long[i] = 1
        # Short during dead zone (21-07 UTC) on rejection
        elif h >= 21 or h <= 6:
            if d["close"][i] > d["ema8"][i] and d["rsi3"][i] > 70:
                entries[i] = 1
                is_long[i] = -1

    return entries, is_long


def s2_funding_presettlement(d, pair):
    """Pre-funding settlement drift: enter 30-60 min before 00:00/08:00/16:00 UTC."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    # Settlement at 00:00, 08:00, 16:00 UTC
    # Pre-settlement window: 30-60 min before (negative funding = long, positive = short)
    # Since we don't have funding data, use price drift: if price trending down pre-settlement -> long
    for i in range(60, n):
        h = d["hours"][i]
        m = d["minutes"][i]

        # 30-60 min before settlement
        is_presettlement = False
        if (h == 23 and m >= 0) or (h == 7 and m >= 0) or (h == 15 and m >= 0):
            is_presettlement = True

        if not is_presettlement:
            continue

        # Contrarian: if RSI oversold pre-settlement, likely short squeeze
        if d["rsi9"][i] < 35 and d["vol_ratio"][i] > 0.8:
            entries[i] = 1
            is_long[i] = 1
        elif d["rsi9"][i] > 65 and d["vol_ratio"][i] > 0.8:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def s3_micro_mean_rev(d, pair):
    """Ultra-tight mean reversion: RSI3 < 10 or > 90, target BB mid, tiny SL."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(25, n):
        # Extreme RSI3 = very short-term oversold/overbought
        if d["rsi3"][i] < 10 and d["close"][i] < d["bb_lower"][i]:
            entries[i] = 1
            is_long[i] = 1
        elif d["rsi3"][i] > 90 and d["close"][i] > d["bb_upper"][i]:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def s4_stoch_cross_momentum(d, pair):
    """Stochastic RSI cross from extreme + volume + trend alignment."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(20, n):
        if np.isnan(d["stoch_k"][i]) or np.isnan(d["stoch_d"][i]):
            continue
        if np.isnan(d["stoch_k"][i-1]) or np.isnan(d["stoch_d"][i-1]):
            continue

        # Bullish: StochK crosses above StochD from oversold (<20)
        if (d["stoch_k"][i-1] < d["stoch_d"][i-1] and d["stoch_k"][i] > d["stoch_d"][i]
            and d["stoch_k"][i-1] < 20 and d["vol_ratio"][i] > 1.2):
            if d["ema8"][i] > d["ema21"][i]:  # With trend
                entries[i] = 1
                is_long[i] = 1

        # Bearish: StochK crosses below StochD from overbought (>80)
        elif (d["stoch_k"][i-1] > d["stoch_d"][i-1] and d["stoch_k"][i] < d["stoch_d"][i]
              and d["stoch_k"][i-1] > 80 and d["vol_ratio"][i] > 1.2):
            if d["ema8"][i] < d["ema21"][i]:
                entries[i] = 1
                is_long[i] = -1

    return entries, is_long


def s5_volume_imbalance(d, pair):
    """Volume imbalance: 3 bars with >2x volume all same direction = continuation."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(55, n):
        # 3 bars high volume same direction (accumulation/distribution)
        if (d["vol_ratio"][i] > 2.0 and d["vol_ratio"][i-1] > 1.5
            and d["body"][i] > 0 and d["body"][i-1] > 0 and d["body"][i-2] > 0):
            # Accumulation -> continuation long
            if d["adx"][i] > 20 and d["plus_di"][i] > d["minus_di"][i]:
                entries[i] = 1
                is_long[i] = 1

        elif (d["vol_ratio"][i] > 2.0 and d["vol_ratio"][i-1] > 1.5
              and d["body"][i] < 0 and d["body"][i-1] < 0 and d["body"][i-2] < 0):
            if d["adx"][i] > 20 and d["minus_di"][i] > d["plus_di"][i]:
                entries[i] = 1
                is_long[i] = -1

    return entries, is_long


def s6_range_breakout_tight(d, pair):
    """5-bar consolidation then breakout with tight 1:1 exits (high WR target)."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(10, n):
        # Check if last 5 bars were in tight range (range < 0.5 * ATR)
        last5_high = np.max(d["high"][i-5:i])
        last5_low = np.min(d["low"][i-5:i])
        consolidation_range = last5_high - last5_low

        if d["atr14"][i] <= 0:
            continue
        if consolidation_range > 0.8 * d["atr14"][i]:
            continue

        # Breakout: current bar closes outside range
        if d["close"][i] > last5_high and d["vol_ratio"][i] > 1.3:
            entries[i] = 1
            is_long[i] = 1
        elif d["close"][i] < last5_low and d["vol_ratio"][i] > 1.3:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def s7_wick_rejection(d, pair):
    """Long wick rejection from key level (EMA21) with volume."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(25, n):
        if d["bar_range"][i] <= 0 or d["body_abs"][i] <= 0:
            continue

        lower_shadow = min(d["close"][i], d["open"][i]) - d["low"][i]
        upper_shadow = d["high"][i] - max(d["close"][i], d["open"][i])

        # Bullish wick: long lower shadow > 2x body, near EMA21
        if (lower_shadow > 2.0 * d["body_abs"][i]
            and d["low"][i] <= d["ema21"][i] * 1.002
            and d["close"][i] > d["open"][i]
            and d["vol_ratio"][i] > 1.3):
            entries[i] = 1
            is_long[i] = 1

        # Bearish wick: long upper shadow > 2x body, near EMA21
        elif (upper_shadow > 2.0 * d["body_abs"][i]
              and d["high"][i] >= d["ema21"][i] * 0.998
              and d["close"][i] < d["open"][i]
              and d["vol_ratio"][i] > 1.3):
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def s8_multi_tf_momentum(d, pair):
    """Aligned momentum: RSI9 + ADX + DI cross + volume - all confirming same direction."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(50, n):
        # All momentum aligned for LONG
        if (d["rsi9"][i] > 55 and d["rsi9"][i] < 75
            and d["adx"][i] > 25
            and d["plus_di"][i] > d["minus_di"][i]
            and d["ema8"][i] > d["ema21"][i]
            and d["vol_ratio"][i] > 1.2
            and d["close"][i] > d["ema8"][i]
            and d["body"][i] > 0):  # Green bar
            entries[i] = 1
            is_long[i] = 1

        # All momentum aligned for SHORT
        elif (d["rsi9"][i] < 45 and d["rsi9"][i] > 25
              and d["adx"][i] > 25
              and d["minus_di"][i] > d["plus_di"][i]
              and d["ema8"][i] < d["ema21"][i]
              and d["vol_ratio"][i] > 1.2
              and d["close"][i] < d["ema8"][i]
              and d["body"][i] < 0):  # Red bar
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def s9_gap_fill(d, pair):
    """Gap between bars (open != prev close by >0.3%) tends to fill on 5m."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(1, n):
        prev_close = d["close"][i-1]
        if prev_close <= 0:
            continue
        gap = (d["open"][i] - prev_close) / prev_close

        # Gap up > 0.3% -> fade (expect fill)
        if gap > 0.003 and d["rsi9"][i] > 60:
            entries[i] = 1
            is_long[i] = -1
        # Gap down > 0.3% -> fade
        elif gap < -0.003 and d["rsi9"][i] < 40:
            entries[i] = 1
            is_long[i] = 1

    return entries, is_long


def s10_double_bottom_top(d, pair):
    """Quick double bottom/top within 10 bars (W/M pattern)."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(15, n):
        # Look for double bottom: two lows within 1% of each other in last 10 bars
        window_low = d["low"][i-10:i+1]
        window_high = d["high"][i-10:i+1]

        min_val = np.min(window_low)
        max_val = np.max(window_high)

        # Current bar near the low and bouncing (potential double bottom)
        if (d["low"][i] <= min_val * 1.001
            and d["close"][i] > d["open"][i]  # Green candle
            and d["rsi3"][i] < 25):
            # Check there was a prior low within 1% that's at least 3 bars ago
            for k in range(i-10, i-3):
                if d["low"][k] <= min_val * 1.005:
                    entries[i] = 1
                    is_long[i] = 1
                    break

        # Double top
        elif (d["high"][i] >= max_val * 0.999
              and d["close"][i] < d["open"][i]  # Red candle
              and d["rsi3"][i] > 75):
            for k in range(i-10, i-3):
                if d["high"][k] >= max_val * 0.995:
                    entries[i] = 1
                    is_long[i] = -1
                    break

    return entries, is_long


# =============================================================================
# MAIN - Test with multiple SL/TP configurations
# =============================================================================

STRATEGIES = [
    ("S1_SPX_Session", s1_spx_us_session_long),
    ("S2_Funding_Presett", s2_funding_presettlement),
    ("S3_Micro_MeanRev", s3_micro_mean_rev),
    ("S4_Stoch_Cross", s4_stoch_cross_momentum),
    ("S5_Vol_Imbalance", s5_volume_imbalance),
    ("S6_Range_BO_Tight", s6_range_breakout_tight),
    ("S7_Wick_Rejection", s7_wick_rejection),
    ("S8_Multi_Momentum", s8_multi_tf_momentum),
    ("S9_Gap_Fill", s9_gap_fill),
    ("S10_Double_BotTop", s10_double_bottom_top),
]

# Test different risk:reward configs (tighter = more scalp-friendly)
CONFIGS = [
    ("Tight_1:1", 0.8, 1.0, 6),    # Very tight scalp
    ("Scalp_1:1.5", 1.0, 1.5, 9),   # Standard scalp
    ("Swing_1:2", 1.5, 2.5, 18),    # Mini-swing
]


def evaluate(pair_data, strategy_func, sl, tp, max_bars):
    """Run strategy across all pairs with given SL/TP."""
    all_trades = []
    per_pair = {}

    for p, d in pair_data.items():
        entries, is_long = strategy_func(d, p)
        trades = scan_trades_atr(entries, d["close"], d["high"], d["low"],
                                  d["atr14"], sl, tp, is_long, max_bars)
        all_trades.extend(trades)
        per_pair[p] = trades

    return all_trades, per_pair


def main():
    print("=" * 90)
    print("ROUND 2: STRUCTURAL EDGE 5m STRATEGIES")
    print("Focus: session timing, funding, microstructure, tight scalp exits")
    print("=" * 90)

    # Load data
    print("\nLoading data...")
    pair_data = {}
    for p in PAIRS:
        df = load(p)
        if df.empty:
            continue
        d = compute_base(df)
        pair_data[p] = d
        print(f"  {p}: {len(df)} bars")

    # Test each strategy with each config
    for config_name, sl, tp, max_bars in CONFIGS:
        print(f"\n{'=' * 90}")
        print(f"CONFIG: {config_name} (SL={sl}x ATR, TP={tp}x ATR, max_bars={max_bars})")
        print(f"{'=' * 90}")
        print(f"{'Strategy':<22} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Profit%':>9} {'Avg%':>8} {'MaxDD%':>8}")
        print("-" * 75)

        for name, func in STRATEGIES:
            all_trades, per_pair = evaluate(pair_data, func, sl, tp, max_bars)

            if not all_trades:
                print(f"{name:<22} {'0':>7} {'--':>7} {'--':>7} {'--':>9} {'--':>8} {'--':>8}")
                continue

            arr = np.array(all_trades)
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

            marker = " ***" if total_pnl > 0 else ""
            print(f"{name:<22} {n_trades:>7} {wr:>6.1f}% {pf:>7.2f} {total_pnl:>+8.2f}% {avg_pnl:>+7.3f}% {max_dd:>7.2f}%{marker}")

    # Deep dive on any profitable combos
    print(f"\n{'=' * 90}")
    print("DEEP DIVE: Best performing combinations per-pair")
    print(f"{'=' * 90}")

    best_results = []
    for name, func in STRATEGIES:
        for config_name, sl, tp, max_bars in CONFIGS:
            all_trades, per_pair = evaluate(pair_data, func, sl, tp, max_bars)
            if all_trades:
                arr = np.array(all_trades)
                profit = arr.sum()
                if profit > 0 and len(arr) >= 20:
                    wr = len(arr[arr > 0]) / len(arr) * 100
                    best_results.append({
                        "name": name, "config": config_name,
                        "sl": sl, "tp": tp, "max_bars": max_bars,
                        "trades": len(arr), "wr": wr, "profit": profit * 100,
                        "per_pair": per_pair,
                    })

    if best_results:
        best_results.sort(key=lambda x: x["profit"], reverse=True)
        print(f"\n{'Strategy':<22} {'Config':<15} {'Trades':>7} {'WR%':>7} {'Profit%':>9}")
        print("-" * 70)
        for r in best_results[:10]:
            print(f"{r['name']:<22} {r['config']:<15} {r['trades']:>7} {r['wr']:>6.1f}% {r['profit']:>+8.2f}%")

        # Per-pair for top 3
        print("\nPer-pair breakdown (top 3):")
        for r in best_results[:3]:
            print(f"\n  {r['name']} ({r['config']}):")
            for p, trades in r["per_pair"].items():
                if trades:
                    arr = np.array(trades)
                    wr = len(arr[arr > 0]) / len(arr) * 100
                    pnl = arr.sum() * 100
                    print(f"    {p:<18} {len(arr):>4} trades | WR={wr:>5.1f}% | PnL={pnl:>+7.2f}%")
    else:
        print("\n  NO profitable strategy/config combination found.")
        print("\n  CONCLUSION: Standard 5m scalping strategies cannot overcome")
        print("  the 0.10% round-trip cost with these signal types on crypto futures.")
        print("\n  Viable paths forward:")
        print("    1. Market-making approach (limit orders, maker fees 0.02%)")
        print("    2. Higher timeframe (15m proven to work)")
        print("    3. Event-driven only (news, liquidation cascades, funding spikes)")
        print("    4. Pair-specific structural edges (SPX session, funding pre-settlement)")
        print("    5. ML-based feature combinations (beyond simple rules)")


if __name__ == "__main__":
    main()

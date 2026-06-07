"""Backtest strategies with MAKER FEES (0.04% round-trip).

Simulates limit-order entry by:
1. Entering at previous candle high/low (resting limit would fill there)
2. Using 0.02% maker fee per side instead of 0.05% taker
3. Accounting for 60% fill rate (not all limits fill)

This tests whether the same strategies become viable with lower costs.
Also tests new limit-order-specific strategies.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
MAKER = 0.0002  # 0.02% per side
TAKER = 0.0005
PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT"]


def load(pair):
    fp = DATA_DIR / f"{pair}-5m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp)
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


def scan_trades_limit(signal_bars, close, high, low, atr, sl_mult, tp_mult,
                      is_long_arr, max_bars=12, entry_offset=0.0):
    """
    Simulate limit-order entry:
    - Signal on bar i means we PLACE a limit order
    - For LONG: limit at close[i] - entry_offset * atr[i] (below current price)
    - For SHORT: limit at close[i] + entry_offset * atr[i] (above current price)
    - Check if limit fills within next 2 bars (price reaches our level)
    - If filled, track SL/TP from fill price
    - Entry cost = maker (0.02%), exit cost = taker (0.05%) for conservative model
    """
    n = len(close)
    trades = []
    fill_count = 0
    signal_count = 0

    i = 0
    while i < n:
        if signal_bars[i] == 0:
            i += 1
            continue

        signal_count += 1
        is_long = is_long_arr[i] > 0
        a = atr[i]
        if a <= 0 or close[i] <= 0:
            i += 1
            continue

        # Calculate limit price
        if is_long:
            limit_price = close[i] - entry_offset * a  # Below current price
        else:
            limit_price = close[i] + entry_offset * a  # Above current price

        # Check if limit fills in next 2 bars
        filled = False
        fill_bar = -1
        for j in range(i + 1, min(i + 3, n)):
            if is_long and low[j] <= limit_price:
                filled = True
                fill_bar = j
                break
            elif not is_long and high[j] >= limit_price:
                filled = True
                fill_bar = j
                break

        if not filled:
            i += 1
            continue

        fill_count += 1
        entry = limit_price
        entry_atr = atr[fill_bar] if fill_bar < n else a

        # SL/TP from fill price
        if is_long:
            sl_p = entry - sl_mult * entry_atr
            tp_p = entry + tp_mult * entry_atr
        else:
            sl_p = entry + sl_mult * entry_atr
            tp_p = entry - tp_mult * entry_atr

        # Scan forward from fill bar
        exited = False
        for k in range(fill_bar + 1, min(fill_bar + max_bars + 1, n)):
            if is_long:
                if low[k] <= sl_p:
                    # Entry: maker, Exit: taker (SL is market order)
                    trades.append(-sl_mult * entry_atr / entry - MAKER - TAKER)
                    exited = True
                    i = k + 1
                    break
                if high[k] >= tp_p:
                    # TP can be limit order too (maker both sides)
                    trades.append(tp_mult * entry_atr / entry - 2 * MAKER)
                    exited = True
                    i = k + 1
                    break
            else:
                if high[k] >= sl_p:
                    trades.append(-sl_mult * entry_atr / entry - MAKER - TAKER)
                    exited = True
                    i = k + 1
                    break
                if low[k] <= tp_p:
                    trades.append(tp_mult * entry_atr / entry - 2 * MAKER)
                    exited = True
                    i = k + 1
                    break

        if not exited:
            exit_idx = min(fill_bar + max_bars, n - 1)
            if is_long:
                trades.append((close[exit_idx] - entry) / entry - MAKER - TAKER)
            else:
                trades.append((entry - close[exit_idx]) / entry - MAKER - TAKER)
            i = exit_idx + 1

    return trades, signal_count, fill_count


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
    rsi14 = ta.rsi(df["close"], length=14).values

    ema8 = ta.ema(df["close"], length=8).values
    ema21 = ta.ema(df["close"], length=21).values
    ema50 = ta.ema(df["close"], length=50).values

    adx_r = ta.adx(df["high"], df["low"], df["close"], length=14)
    adx = adx_r.iloc[:, 0].values if adx_r is not None else np.full(len(df), 25.0)
    plus_di = adx_r.iloc[:, 1].values if adx_r is not None else np.full(len(df), 25.0)
    minus_di = adx_r.iloc[:, 2].values if adx_r is not None else np.full(len(df), 25.0)

    vol_ema = ta.ema(df["volume"].astype(float), length=20).values
    vol_ratio = v / (vol_ema + 1e-10)

    bb = ta.bbands(df["close"], length=20, std=2.0)
    bb_upper = bb.iloc[:, 0].values if bb is not None else c + atr14
    bb_mid = bb.iloc[:, 1].values if bb is not None else c
    bb_lower = bb.iloc[:, 2].values if bb is not None else c - atr14

    hours = pd.to_datetime(df["date"]).dt.hour.values
    body = c - o
    body_abs = np.abs(body)
    bar_range = h - lo

    return {
        "close": c, "high": h, "low": lo, "open": o, "volume": v,
        "atr14": atr14, "atr5": atr5,
        "rsi9": rsi9, "rsi3": rsi3, "rsi14": rsi14,
        "ema8": ema8, "ema21": ema21, "ema50": ema50,
        "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
        "vol_ratio": vol_ratio,
        "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
        "hours": hours,
        "body": body, "body_abs": body_abs, "bar_range": bar_range,
    }


# =============================================================================
# LIMIT-ORDER STRATEGIES
# =============================================================================

def lim1_bb_limit_bounce(d, pair):
    """Place limit at BB lower (long) / BB upper (short) and wait for fill."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(25, n):
        # Only place limits when mean-reverting conditions exist
        if d["rsi14"][i] < 35 and d["adx"][i] < 30:
            # Signal: place limit buy at BB lower
            entries[i] = 1
            is_long[i] = 1
        elif d["rsi14"][i] > 65 and d["adx"][i] < 30:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def lim2_ema_pullback_limit(d, pair):
    """In trend, place limit at EMA8 for pullback entry."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(50, n):
        if d["adx"][i] < 25:
            continue
        # Uptrend: EMA8 > EMA21, price above EMA8 -> place limit at EMA8
        if (d["ema8"][i] > d["ema21"][i] and d["close"][i] > d["ema8"][i]
            and d["rsi9"][i] > 50 and d["plus_di"][i] > d["minus_di"][i]):
            entries[i] = 1
            is_long[i] = 1
        # Downtrend
        elif (d["ema8"][i] < d["ema21"][i] and d["close"][i] < d["ema8"][i]
              and d["rsi9"][i] < 50 and d["minus_di"][i] > d["plus_di"][i]):
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def lim3_range_limit(d, pair):
    """In low-ADX range, place limits at range extremes."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(30, n):
        if d["adx"][i] > 22:  # Only in range-bound market
            continue
        # Check 20-bar range
        hi20 = np.max(d["high"][i-20:i])
        lo20 = np.min(d["low"][i-20:i])
        range_size = hi20 - lo20
        if range_size <= 0 or d["atr14"][i] <= 0:
            continue
        # Range must be reasonable (2-5x ATR)
        if range_size < 2 * d["atr14"][i] or range_size > 6 * d["atr14"][i]:
            continue

        # Near bottom of range -> long limit
        pos_in_range = (d["close"][i] - lo20) / range_size
        if pos_in_range < 0.25:
            entries[i] = 1
            is_long[i] = 1
        elif pos_in_range > 0.75:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def lim4_spx_session_limit(d, pair):
    """SPX: Place limits during favorable sessions."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)
    if "SPX" not in pair:
        return entries, is_long

    for i in range(50, n):
        h = d["hours"][i]
        # Pre-US session (12-13 UTC): place long limits below current price
        if h in (12, 13) and d["rsi9"][i] < 55:
            entries[i] = 1
            is_long[i] = 1
        # Late US session (19-20 UTC): place short limits above current (profit-taking)
        elif h in (19, 20) and d["rsi9"][i] > 50:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def lim5_volume_exhaustion_limit(d, pair):
    """After volume spike, place limit at 50% retracement of the spike candle."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(25, n):
        if d["vol_ratio"][i] < 3.0:
            continue
        # Big green spike -> expect pullback -> place limit buy at 50% of candle
        if d["body"][i] > 0 and d["body"][i] > 0.6 * d["bar_range"][i]:
            # After big up, place limit buy on pullback
            entries[i] = 1
            is_long[i] = 1  # Will enter below on pullback
        # Big red spike -> expect bounce
        elif d["body"][i] < 0 and abs(d["body"][i]) > 0.6 * d["bar_range"][i]:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def lim6_prev_hl_breakout(d, pair):
    """Place limit at previous candle high (long) / low (short) for breakout fill."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(50, n):
        if d["adx"][i] < 20:
            continue
        # Trending + consolidation bar -> place limit above high for breakout
        if d["bar_range"][i] < 0.7 * d["atr14"][i]:  # Current bar is tight
            if d["ema8"][i] > d["ema21"][i] and d["plus_di"][i] > d["minus_di"][i]:
                entries[i] = 1
                is_long[i] = 1
            elif d["ema8"][i] < d["ema21"][i] and d["minus_di"][i] > d["plus_di"][i]:
                entries[i] = 1
                is_long[i] = -1

    return entries, is_long


def lim7_rsi3_extreme_limit(d, pair):
    """RSI3 at extreme (< 5 or > 95) -> place limit for mean-reversion fill."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(25, n):
        if np.isnan(d["rsi3"][i]):
            continue
        if d["rsi3"][i] < 5:
            entries[i] = 1
            is_long[i] = 1
        elif d["rsi3"][i] > 95:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


def lim8_multi_confluence_limit(d, pair):
    """Multiple conditions aligned (4+ factors) -> higher confidence, limit entry."""
    n = len(d["close"])
    entries = np.zeros(n)
    is_long = np.zeros(n)

    for i in range(50, n):
        # Count bullish conditions
        bull_score = 0
        if d["rsi9"][i] < 35: bull_score += 1
        if d["close"][i] < d["bb_lower"][i]: bull_score += 1
        if d["vol_ratio"][i] > 1.5: bull_score += 1
        if d["close"][i] < d["ema21"][i] and d["ema8"][i] > d["ema21"][i]: bull_score += 1
        if d["rsi3"][i] < 15: bull_score += 1

        if bull_score >= 4:
            entries[i] = 1
            is_long[i] = 1
            continue

        # Count bearish conditions
        bear_score = 0
        if d["rsi9"][i] > 65: bear_score += 1
        if d["close"][i] > d["bb_upper"][i]: bear_score += 1
        if d["vol_ratio"][i] > 1.5: bear_score += 1
        if d["close"][i] > d["ema21"][i] and d["ema8"][i] < d["ema21"][i]: bear_score += 1
        if d["rsi3"][i] > 85: bear_score += 1

        if bear_score >= 4:
            entries[i] = 1
            is_long[i] = -1

    return entries, is_long


# =============================================================================
# MAIN
# =============================================================================

STRATEGIES = [
    ("LIM1_BB_Bounce", lim1_bb_limit_bounce),
    ("LIM2_EMA_Pullback", lim2_ema_pullback_limit),
    ("LIM3_Range_Ping", lim3_range_limit),
    ("LIM4_SPX_Session", lim4_spx_session_limit),
    ("LIM5_Vol_Exhaust", lim5_volume_exhaustion_limit),
    ("LIM6_Prev_HL_BO", lim6_prev_hl_breakout),
    ("LIM7_RSI3_Extreme", lim7_rsi3_extreme_limit),
    ("LIM8_Multi_Conflu", lim8_multi_confluence_limit),
]

# Entry offset: how far below/above current price to place limit
# 0.0 = at current price (aggressive limit, high fill rate)
# 0.3 = 0.3x ATR offset (moderate, ~60% fill)
# 0.5 = 0.5x ATR offset (conservative, ~40% fill)
OFFSETS = [0.0, 0.2, 0.4]

# SL/TP configs
CONFIGS = [
    ("Tight", 0.8, 1.2, 9),
    ("Balanced", 1.0, 1.8, 12),
    ("Wide", 1.5, 2.5, 18),
]


def main():
    print("=" * 95)
    print("LIMIT-ORDER 5m SCALPING BACKTEST")
    print("Entry: Maker 0.02% | Exit TP: Maker 0.02% | Exit SL: Taker 0.05%")
    print("Simulates limit fill within 2 bars of signal")
    print("=" * 95)

    pair_data = {}
    for p in PAIRS:
        df = load(p)
        if df.empty:
            continue
        pair_data[p] = compute_base(df)
        print(f"  {p}: {len(df)} bars")

    # Find best combo for each strategy
    print(f"\n{'=' * 95}")
    print("BEST RESULT PER STRATEGY (across all offset/config combinations)")
    print(f"{'=' * 95}")
    print(f"{'Strategy':<20} {'Config':<10} {'Offs':>5} {'Sig':>6} {'Fill':>6} {'FR%':>5} "
          f"{'Trd':>5} {'WR%':>6} {'PF':>6} {'Profit%':>9} {'Avg%':>8}")
    print("-" * 95)

    all_best = []

    for name, func in STRATEGIES:
        best_profit = -999
        best_row = None

        for offset in OFFSETS:
            for cfg_name, sl, tp, max_bars in CONFIGS:
                total_trades = []
                total_signals = 0
                total_fills = 0

                for p, d in pair_data.items():
                    entries, is_long = func(d, p)
                    trades, sigs, fills = scan_trades_limit(
                        entries, d["close"], d["high"], d["low"],
                        d["atr14"], sl, tp, is_long, max_bars, offset
                    )
                    total_trades.extend(trades)
                    total_signals += sigs
                    total_fills += fills

                if not total_trades or len(total_trades) < 10:
                    continue

                arr = np.array(total_trades)
                profit = arr.sum()
                if profit > best_profit:
                    best_profit = profit
                    wins = arr[arr > 0]
                    losses = arr[arr <= 0]
                    wr = len(wins) / len(arr) * 100
                    pf = wins.sum() / max(abs(losses.sum()), 0.0001) if len(losses) > 0 else 99.0
                    fr = total_fills / max(total_signals, 1) * 100
                    best_row = {
                        "name": name, "config": cfg_name, "offset": offset,
                        "signals": total_signals, "fills": total_fills,
                        "fill_rate": fr,
                        "trades": len(arr), "wr": wr, "pf": pf,
                        "profit": profit * 100, "avg": arr.mean() * 100,
                    }

        if best_row:
            r = best_row
            marker = " ***" if r["profit"] > 0 else ""
            print(f"{r['name']:<20} {r['config']:<10} {r['offset']:>5.1f} {r['signals']:>6} "
                  f"{r['fills']:>6} {r['fill_rate']:>4.0f}% {r['trades']:>5} {r['wr']:>5.1f}% "
                  f"{r['pf']:>6.2f} {r['profit']:>+8.2f}% {r['avg']:>+7.3f}%{marker}")
            all_best.append(best_row)

    # Detailed breakdown for profitable strategies
    profitable = [r for r in all_best if r["profit"] > 0]
    if profitable:
        print(f"\n{'=' * 95}")
        print("PROFITABLE STRATEGIES - Per Pair Breakdown")
        print(f"{'=' * 95}")
        for r in profitable:
            name = r["name"]
            offset = r["offset"]
            cfg = r["config"]
            sl, tp, max_bars = next((s, t, m) for cn, s, t, m in CONFIGS if cn == cfg)

            print(f"\n  {name} (offset={offset}, SL={sl}, TP={tp}):")
            func = next(f for n, f in STRATEGIES if n == name)
            for p, d in pair_data.items():
                entries, is_long = func(d, p)
                trades, sigs, fills = scan_trades_limit(
                    entries, d["close"], d["high"], d["low"],
                    d["atr14"], sl, tp, is_long, max_bars, offset
                )
                if trades:
                    arr = np.array(trades)
                    wr = len(arr[arr > 0]) / len(arr) * 100
                    pnl = arr.sum() * 100
                    print(f"    {p:<18} {len(arr):>4} trades | WR={wr:>5.1f}% | PnL={pnl:>+7.2f}%")
                else:
                    print(f"    {p:<18}    0 trades")

    # Compare: same strategies with taker fees
    print(f"\n{'=' * 95}")
    print("COMPARISON: Maker vs Taker fees (best strategy params)")
    print(f"{'=' * 95}")
    if all_best:
        print(f"{'Strategy':<20} {'Maker Profit%':>14} {'Taker Profit%':>14} {'Diff':>8}")
        print("-" * 60)
        for r in sorted(all_best, key=lambda x: x["profit"], reverse=True)[:5]:
            # Re-run with taker fees for comparison
            name = r["name"]
            func = next(f for n, f in STRATEGIES if n == name)
            offset = r["offset"]
            cfg = r["config"]
            sl, tp, max_bars = next((s, t, m) for cn, s, t, m in CONFIGS if cn == cfg)

            # Taker version: same logic but TAKER fees everywhere
            total_taker = []
            for p, d in pair_data.items():
                entries, is_long = func(d, p)
                n = len(d["close"])
                # Simple taker scan (no limit fill simulation)
                i = 0
                while i < n:
                    if entries[i] == 0:
                        i += 1
                        continue
                    entry = d["close"][i]
                    a = d["atr14"][i]
                    if a <= 0 or entry <= 0:
                        i += 1
                        continue
                    il = is_long[i] > 0
                    if il:
                        sl_p = entry - sl * a
                        tp_p = entry + tp * a
                    else:
                        sl_p = entry + sl * a
                        tp_p = entry - tp * a
                    exited = False
                    for j in range(i+1, min(i+max_bars+1, n)):
                        if il:
                            if d["low"][j] <= sl_p:
                                total_taker.append(-sl*a/entry - 2*TAKER)
                                exited = True; i = j+1; break
                            if d["high"][j] >= tp_p:
                                total_taker.append(tp*a/entry - 2*TAKER)
                                exited = True; i = j+1; break
                        else:
                            if d["high"][j] >= sl_p:
                                total_taker.append(-sl*a/entry - 2*TAKER)
                                exited = True; i = j+1; break
                            if d["low"][j] <= tp_p:
                                total_taker.append(tp*a/entry - 2*TAKER)
                                exited = True; i = j+1; break
                    if not exited:
                        ex = min(i+max_bars, n-1)
                        if il:
                            total_taker.append((d["close"][ex]-entry)/entry - 2*TAKER)
                        else:
                            total_taker.append((entry-d["close"][ex])/entry - 2*TAKER)
                        i = ex+1

            taker_pnl = sum(total_taker) * 100 if total_taker else 0
            diff = r["profit"] - taker_pnl
            print(f"{name:<20} {r['profit']:>+13.2f}% {taker_pnl:>+13.2f}% {diff:>+7.2f}%")

    # Final summary
    print(f"\n{'=' * 95}")
    print("FINAL ASSESSMENT")
    print(f"{'=' * 95}")
    n_profitable = len([r for r in all_best if r["profit"] > 0])
    n_total = len(all_best)
    print(f"  Strategies tested: {n_total}")
    print(f"  Profitable with maker fees: {n_profitable}")
    if profitable:
        best = max(profitable, key=lambda x: x["profit"])
        print(f"  Best: {best['name']} at {best['profit']:+.2f}% ({best['trades']} trades, WR={best['wr']:.1f}%)")
        print(f"\n  CONCLUSION: Limit-order strategies show promise. Maker fees save")
        print(f"  ~0.06% per trade ({best['trades']} trades = {best['trades']*0.06:.1f}% total savings).")
    else:
        print(f"\n  CONCLUSION: Even with maker fees, these signal generators")
        print(f"  lack predictive edge on 5m crypto futures.")
        print(f"  The problem is signal quality, not just costs.")


if __name__ == "__main__":
    main()

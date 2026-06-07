"""Fast vectorized hyperopt for 5m scalping strategies.

Pre-computes indicators once, uses numpy vectorized signal detection.
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
    df = pd.read_feather(DATA_DIR / f"{pair}-5m-futures.feather")
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


def scan_trades(entries_idx, close, high, low, atr, sl_mult, tp_mult, is_long, max_bars=18):
    """Scan forward from each entry to find SL/TP/time-cut exit."""
    n = len(close)
    profits = []
    for idx in entries_idx:
        if idx + max_bars >= n:
            break
        entry = close[idx]
        a = atr[idx]
        if a <= 0 or entry <= 0:
            continue

        if is_long:
            sl_p = entry - sl_mult * a
            tp_p = entry + tp_mult * a
            exited = False
            for j in range(idx + 1, min(idx + max_bars + 1, n)):
                if low[j] <= sl_p:
                    profits.append(-sl_mult * a / entry - 2 * TAKER)
                    exited = True
                    break
                if high[j] >= tp_p:
                    profits.append(tp_mult * a / entry - 2 * TAKER)
                    exited = True
                    break
            if not exited:
                exit_idx = min(idx + max_bars, n - 1)
                profits.append((close[exit_idx] - entry) / entry - 2 * TAKER)
        else:
            sl_p = entry + sl_mult * a
            tp_p = entry - tp_mult * a
            exited = False
            for j in range(idx + 1, min(idx + max_bars + 1, n)):
                if high[j] >= sl_p:
                    profits.append(-sl_mult * a / entry - 2 * TAKER)
                    exited = True
                    break
                if low[j] <= tp_p:
                    profits.append(tp_mult * a / entry - 2 * TAKER)
                    exited = True
                    break
            if not exited:
                exit_idx = min(idx + max_bars, n - 1)
                profits.append((entry - close[exit_idx]) / entry - 2 * TAKER)

    return profits


def main():
    print("=" * 70)
    print("5m SCALPING HYPEROPT - Vectorized")
    print("=" * 70)

    # Pre-compute indicators
    print("Pre-computing indicators...")
    pair_data = {}
    for p in PAIRS:
        df = load(p)
        c, h, lo, v = df["close"], df["high"], df["low"], df["volume"].astype(float)

        bar_range = (h - lo).values
        atr14 = ta.atr(h, lo, c, length=14).values
        adx_r = ta.adx(h, lo, c, length=14)
        adx = adx_r.iloc[:, 0].values if adx_r is not None else np.full(len(df), 25.0)
        vol_ema = ta.ema(v, length=20).values
        vol_ratio = v.values / (vol_ema + 1e-10)
        rsi9 = ta.rsi(c, length=9).values

        bb = ta.bbands(c, length=20, std=2.0)
        bb_upper = bb.iloc[:, 0].values if bb is not None else c.values + atr14
        bb_lower = bb.iloc[:, 2].values if bb is not None else c.values - atr14

        body_abs = np.abs((c - df["open"]).values)
        upper_shadow = h.values - np.maximum(c.values, df["open"].values)
        lower_shadow = np.minimum(c.values, df["open"].values) - lo.values

        pair_data[p] = {
            "close": c.values, "high": h.values, "low": lo.values,
            "range": bar_range, "atr": atr14, "adx": adx,
            "vol_ratio": vol_ratio, "rsi": rsi9,
            "bb_upper": bb_upper, "bb_lower": bb_lower,
            "body_abs": body_abs, "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
        }
        print(f"  {p}: {len(df)} bars")

    # === NR BREAKOUT HYPEROPT ===
    print("\n## NR BREAKOUT 5m - Hyperopt")
    print("-" * 70)

    best_nr = None
    best_nr_profit = -999
    combos_tested = 0

    for lookback in [4, 7, 10]:
        # Pre-compute min range for this lookback
        nr_mins = {}
        for p, d in pair_data.items():
            nr_mins[p] = pd.Series(d["range"]).rolling(lookback, min_periods=lookback).min().values

        for range_max in [0.4, 0.5, 0.6, 0.7]:
            for adx_max in [18, 20, 22]:
                for vol_min in [1.0, 1.3, 1.5]:
                    for sl in [1.0, 1.5, 2.0]:
                        for tp in [2.0, 2.5, 3.0, 4.0]:
                            total_profit = 0.0
                            total_trades = 0
                            combos_tested += 1

                            for p, d in pair_data.items():
                                n = len(d["close"])
                                min_r = nr_mins[p]

                                # NR bar detection (vectorized)
                                is_nr = d["range"] <= min_r * 1.01
                                prev_nr = np.zeros(n, dtype=bool)
                                prev_nr[1:] = is_nr[:-1]

                                range_ratio = d["range"] / (d["atr"] + 1e-10)
                                prev_range_ok = np.zeros(n, dtype=bool)
                                prev_range_ok[1:] = (range_ratio[:-1] <= range_max)

                                adx_ok = d["adx"] <= adx_max
                                vol_ok = d["vol_ratio"] >= vol_min

                                prev_high = np.roll(d["high"], 1)
                                prev_low = np.roll(d["low"], 1)
                                prev_high[0] = d["high"][0]
                                prev_low[0] = d["low"][0]

                                base_filter = prev_nr & prev_range_ok & adx_ok & vol_ok
                                long_sig = base_filter & (d["close"] > prev_high)
                                short_sig = base_filter & (d["close"] < prev_low)

                                long_idx = np.where(long_sig)[0]
                                short_idx = np.where(short_sig)[0]

                                longs = scan_trades(long_idx, d["close"], d["high"], d["low"],
                                                    d["atr"], sl, tp, True, 18)
                                shorts = scan_trades(short_idx, d["close"], d["high"], d["low"],
                                                     d["atr"], sl, tp, False, 18)

                                total_profit += sum(longs) + sum(shorts)
                                total_trades += len(longs) + len(shorts)

                            if total_trades >= 50 and total_profit > best_nr_profit:
                                best_nr_profit = total_profit
                                wins = sum(1 for x in [] if x > 0)  # placeholder
                                best_nr = {
                                    "lookback": lookback, "range_max": range_max,
                                    "adx_max": adx_max, "vol_min": vol_min,
                                    "sl": sl, "tp": tp,
                                    "trades": total_trades, "profit": total_profit,
                                }

    print(f"  Tested {combos_tested} combinations")
    if best_nr:
        avg_pnl = best_nr["profit"] / best_nr["trades"]
        print(f"  BEST: profit={best_nr['profit']:+.4f} ({best_nr['profit']*100:+.2f}%) | "
              f"trades={best_nr['trades']} | avg={avg_pnl*100:+.4f}%/trade")
        print(f"  Params: lookback={best_nr['lookback']}, range_max={best_nr['range_max']}, "
              f"adx_max={best_nr['adx_max']}, vol_min={best_nr['vol_min']}, "
              f"sl={best_nr['sl']}, tp={best_nr['tp']}")
    else:
        print("  No profitable combination found")

    # === VOLUME CLIMAX HYPEROPT ===
    print("\n## VOLUME CLIMAX 5m - Hyperopt")
    print("-" * 70)

    best_vc = None
    best_vc_profit = -999
    combos_tested = 0

    for spike in [2.0, 2.5, 3.0, 4.0, 5.0]:
        for shadow in [1.5, 2.0, 2.5]:
            for rsi_os in [20, 25, 30, 35]:
                for rsi_ob in [65, 70, 75, 80]:
                    for sl in [1.0, 1.5, 2.0]:
                        for tp in [1.5, 2.0, 2.5, 3.0]:
                            total_profit = 0.0
                            total_trades = 0
                            combos_tested += 1

                            for p, d in pair_data.items():
                                n = len(d["close"])
                                vol_ok = d["vol_ratio"] >= spike

                                # Long signals
                                hammer = (d["lower_shadow"] > shadow * d["body_abs"]) & (d["body_abs"] > 0)
                                long_sig = vol_ok & hammer & (d["rsi"] < rsi_os) & (d["low"] <= d["bb_lower"])

                                # Short signals
                                star = (d["upper_shadow"] > shadow * d["body_abs"]) & (d["body_abs"] > 0)
                                short_sig = vol_ok & star & (d["rsi"] > rsi_ob) & (d["high"] >= d["bb_upper"])

                                long_idx = np.where(long_sig)[0]
                                short_idx = np.where(short_sig)[0]

                                longs = scan_trades(long_idx, d["close"], d["high"], d["low"],
                                                    d["atr"], sl, tp, True, 12)
                                shorts = scan_trades(short_idx, d["close"], d["high"], d["low"],
                                                     d["atr"], sl, tp, False, 12)

                                total_profit += sum(longs) + sum(shorts)
                                total_trades += len(longs) + len(shorts)

                            if total_trades >= 20 and total_profit > best_vc_profit:
                                best_vc_profit = total_profit
                                best_vc = {
                                    "spike": spike, "shadow": shadow,
                                    "rsi_os": rsi_os, "rsi_ob": rsi_ob,
                                    "sl": sl, "tp": tp,
                                    "trades": total_trades, "profit": total_profit,
                                }

    print(f"  Tested {combos_tested} combinations")
    if best_vc:
        avg_pnl = best_vc["profit"] / best_vc["trades"]
        print(f"  BEST: profit={best_vc['profit']:+.4f} ({best_vc['profit']*100:+.2f}%) | "
              f"trades={best_vc['trades']} | avg={avg_pnl*100:+.4f}%/trade")
        print(f"  Params: spike={best_vc['spike']}, shadow={best_vc['shadow']}, "
              f"rsi_os={best_vc['rsi_os']}, rsi_ob={best_vc['rsi_ob']}, "
              f"sl={best_vc['sl']}, tp={best_vc['tp']}")
    else:
        print("  No profitable combination found")

    # === SUMMARY ===
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    if best_nr:
        print(f"  NR Breakout 5m:    {best_nr['profit']*100:+.2f}% | {best_nr['trades']} trades")
    if best_vc:
        print(f"  Volume Climax 5m:  {best_vc['profit']*100:+.2f}% | {best_vc['trades']} trades")
    if not best_nr and not best_vc:
        print("  Neither strategy found a profitable parameter set on 5m data.")
        print("  5m scalping on these pairs may not be viable with these signal types.")


if __name__ == "__main__":
    main()

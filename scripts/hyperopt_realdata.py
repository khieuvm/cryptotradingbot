"""Hyperopt on REAL OKX data — VECTORIZED for speed.

Walk-Forward: 120d train / 60d validate on 7 pairs.
All signal generation is fully vectorized (numpy), no per-bar loops.
"""

import sys
from pathlib import Path
import time
from itertools import product

import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, str(Path(__file__).parent.parent))

TAKER_FEE = 0.0005
DATA_DIR = Path(__file__).parent.parent / "user_data" / "data" / "okx" / "futures"
PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT",
         "DOGE_USDT_USDT", "SUI_USDT_USDT", "LINK_USDT_USDT"]
BARS_PER_DAY = 96
TRAIN_BARS = 120 * BARS_PER_DAY  # ~11520
STARTUP = 60


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_feather(str(fp)).sort_values("date").reset_index(drop=True)


def backtest_vec(signals, high, low, close, atr, sl_mult, tp_mult, max_bars=96):
    """Vectorized-ish backtest (still needs loop for trade sequencing)."""
    fee = 2 * TAKER_FEE
    n = len(close)
    entries = np.where(signals != 0)[0]
    if len(entries) == 0:
        return 0.0, 0, 0.0, 0.0, 0.0

    trades = []
    next_allowed = 0
    for idx in entries:
        if idx < next_allowed:
            continue
        ep = close[idx]
        ea = atr[idx]
        if ea <= 0 or ep <= 0 or np.isnan(ea):
            continue
        is_long = signals[idx] > 0
        sl_d = sl_mult * ea
        tp_d = tp_mult * ea
        end_bar = min(idx + max_bars, n - 1)

        exited = False
        for j in range(idx + 1, end_bar + 1):
            if is_long:
                if low[j] <= ep - sl_d:
                    trades.append(-sl_mult * ea / ep - fee)
                    next_allowed = j + 1
                    exited = True
                    break
                if high[j] >= ep + tp_d:
                    trades.append(tp_mult * ea / ep - fee)
                    next_allowed = j + 1
                    exited = True
                    break
            else:
                if high[j] >= ep + sl_d:
                    trades.append(-sl_mult * ea / ep - fee)
                    next_allowed = j + 1
                    exited = True
                    break
                if low[j] <= ep - tp_d:
                    trades.append(tp_mult * ea / ep - fee)
                    next_allowed = j + 1
                    exited = True
                    break
        if not exited:
            pnl = ((close[end_bar] - ep) / ep if is_long else (ep - close[end_bar]) / ep) - fee
            trades.append(pnl)
            next_allowed = end_bar + 1

    if not trades:
        return 0.0, 0, 0.0, 0.0, 0.0
    arr = np.array(trades)
    total = arr.sum()
    wr = (arr > 0).sum() / len(arr)
    cum = np.cumsum(arr)
    dd = abs(np.min(cum - np.maximum.accumulate(cum))) if len(cum) > 1 else 0
    wins = arr[arr > 0].sum() if (arr > 0).any() else 0
    losses = abs(arr[arr < 0].sum()) if (arr < 0).any() else 1e-10
    pf = wins / losses
    return total, len(arr), wr, dd, pf


def main():
    t0 = time.time()
    print("=" * 80)
    print("  HYPEROPT ON REAL OKX DATA (Dec 2025 - Jun 2026)")
    print("  Walk-Forward: 120d train -> 60d validate | 7 pairs")
    print("=" * 80)

    # Load and precompute
    data = {}
    for pair in PAIRS:
        df = load_15m(pair)
        if df.empty:
            continue
        c = df["close"].values.astype(float)
        h = df["high"].values.astype(float)
        lo = df["low"].values.astype(float)
        o = df["open"].values.astype(float)
        v = df["volume"].values.astype(float)

        ind = {"close": c, "high": h, "low": lo, "open": o, "volume": v, "n": len(df)}

        adx_r = ta.adx(df["high"], df["low"], df["close"], length=14)
        ind["adx"] = adx_r.iloc[:, 0].values if adx_r is not None else np.full(len(df), 25.0)
        ind["di_plus"] = adx_r.iloc[:, 1].values if adx_r is not None else np.full(len(df), 12.5)
        ind["di_minus"] = adx_r.iloc[:, 2].values if adx_r is not None else np.full(len(df), 12.5)
        ind["rsi"] = ta.rsi(df["close"], length=14).values
        ind["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14).values

        ve = ta.ema(df["volume"].astype(float), length=20).values
        ind["vol_ratio"] = v / (ve + 1e-10)

        for p in [14, 18, 21, 50, 60]:
            ind[f"ema{p}"] = ta.ema(df["close"], length=p).values

        body = c - o
        ind["body_abs"] = np.abs(body)
        ind["upper_shadow"] = h - np.maximum(c, o)
        ind["lower_shadow"] = np.minimum(c, o) - lo

        # Precompute EMA cross signals
        for ef_name, es_name in [("ema14", "ema50"), ("ema18", "ema50"), ("ema21", "ema60")]:
            ef = ind[ef_name]; es = ind[es_name]
            cross_up = np.zeros(len(df), dtype=bool)
            cross_down = np.zeros(len(df), dtype=bool)
            cross_up[1:] = (ef[1:] > es[1:]) & (ef[:-1] <= es[:-1])
            cross_down[1:] = (ef[1:] < es[1:]) & (ef[:-1] >= es[:-1])
            ind[f"cross_up_{ef_name}_{es_name}"] = cross_up
            ind[f"cross_down_{ef_name}_{es_name}"] = cross_down

        # 3-bar range for compression
        h3_max = np.maximum(np.maximum(np.roll(h, 1), np.roll(h, 2)), h)
        l3_min = np.minimum(np.minimum(np.roll(lo, 1), np.roll(lo, 2)), lo)
        ind["range_3bar"] = h3_max - l3_min
        ind["prev_h2"] = np.maximum(np.roll(h, 1), np.roll(h, 2))
        ind["prev_l2"] = np.minimum(np.roll(lo, 1), np.roll(lo, 2))

        data[pair] = ind
        print(f"  {pair}: {len(df)} bars")

    print(f"\n  Loaded {len(data)} pairs, split at bar {TRAIN_BARS}")

    # ═══════════════════════════════════════════════════════════════════════
    # REGIME ADAPTIVE — vectorized
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  REGIME ADAPTIVE")
    print(f"{'='*80}")

    best_ra = {"profit": -999}
    count = 0

    ra_grid = list(product(
        [25, 28, 31, 35],           # adx_thr
        [28, 30, 33, 35],           # rsi_os
        [65, 68, 70, 73],           # rsi_ob
        [0.5, 0.8, 1.0],           # vol_min
        [("ema14", "ema50"), ("ema18", "ema50"), ("ema21", "ema60")],  # ema pair
        [5, 8, 12],                 # cross_lb
        [(5.0, 8.0), (5.9, 9.3), (6.0, 10.0), (7.0, 11.0), (5.0, 9.0)],  # sl, tp
    ))

    for adx_thr, rsi_os, rsi_ob, vol_min, (ema_f, ema_s), cross_lb, (sl, tp) in ra_grid:
        total_profit = 0
        total_trades = 0

        for pair, ind in data.items():
            n = TRAIN_BARS
            adx = ind["adx"][:n]
            di_p = ind["di_plus"][:n]
            di_m = ind["di_minus"][:n]
            rsi = ind["rsi"][:n]
            vr = ind["vol_ratio"][:n]

            cross_up = ind[f"cross_up_{ema_f}_{ema_s}"][:n]
            cross_down = ind[f"cross_down_{ema_f}_{ema_s}"][:n]

            # Vectorized: recent cross within lookback
            from scipy.ndimage import maximum_filter1d
            recent_cross_up = maximum_filter1d(cross_up.astype(float), size=cross_lb, origin=cross_lb//2-1)
            recent_cross_down = maximum_filter1d(cross_down.astype(float), size=cross_lb, origin=cross_lb//2-1)

            # Trend signals
            trend_long = (
                (adx >= adx_thr) & (vr >= vol_min)
                & (recent_cross_up > 0) & (di_p > di_m)
            )
            trend_short = (
                (adx >= adx_thr) & (vr >= vol_min)
                & (recent_cross_down > 0) & (di_m > di_p)
            )

            # Range signals
            range_long = (~(adx >= adx_thr)) & (vr >= vol_min) & (rsi < rsi_os)
            range_short = (~(adx >= adx_thr)) & (vr >= vol_min) & (rsi > rsi_ob)

            signals = np.zeros(n)
            signals[trend_long | range_long] = 1
            signals[trend_short | range_short] = -1
            signals[:STARTUP] = 0

            p, t, w, d, pf = backtest_vec(
                signals, ind["high"][:n], ind["low"][:n],
                ind["close"][:n], ind["atr"][:n], sl, tp, 192)
            total_profit += p
            total_trades += t

        count += 1
        if total_trades >= 30 and total_profit > best_ra["profit"]:
            best_ra = {"profit": total_profit, "trades": total_trades,
                       "adx_thr": adx_thr, "rsi_os": rsi_os, "rsi_ob": rsi_ob,
                       "vol_min": vol_min, "ema_f": ema_f, "ema_s": ema_s,
                       "cross_lb": cross_lb, "sl": sl, "tp": tp}

    print(f"  Tested {count} combos in {time.time()-t0:.0f}s")
    print(f"  BEST TRAIN: profit={best_ra.get('profit',0):+.2%} | trades={best_ra.get('trades',0)}")
    print(f"    adx={best_ra.get('adx_thr')}, rsi={best_ra.get('rsi_os')}/{best_ra.get('rsi_ob')}, vol={best_ra.get('vol_min')}")
    print(f"    ema={best_ra.get('ema_f')}/{best_ra.get('ema_s')}, lb={best_ra.get('cross_lb')}, sl/tp={best_ra.get('sl')}/{best_ra.get('tp')}")

    # Validate
    print(f"\n  VALIDATION (last 60 days):")
    ra_val = 0
    for pair, ind in data.items():
        n = ind["n"]
        adx = ind["adx"][TRAIN_BARS:n]
        di_p = ind["di_plus"][TRAIN_BARS:n]
        di_m = ind["di_minus"][TRAIN_BARS:n]
        rsi = ind["rsi"][TRAIN_BARS:n]
        vr = ind["vol_ratio"][TRAIN_BARS:n]
        cross_up = ind[f"cross_up_{best_ra['ema_f']}_{best_ra['ema_s']}"][TRAIN_BARS:n]
        cross_down = ind[f"cross_down_{best_ra['ema_f']}_{best_ra['ema_s']}"][TRAIN_BARS:n]
        recent_up = maximum_filter1d(cross_up.astype(float), size=best_ra["cross_lb"], origin=best_ra["cross_lb"]//2-1)
        recent_down = maximum_filter1d(cross_down.astype(float), size=best_ra["cross_lb"], origin=best_ra["cross_lb"]//2-1)

        trend_l = (adx >= best_ra["adx_thr"]) & (vr >= best_ra["vol_min"]) & (recent_up > 0) & (di_p > di_m)
        trend_s = (adx >= best_ra["adx_thr"]) & (vr >= best_ra["vol_min"]) & (recent_down > 0) & (di_m > di_p)
        range_l = (~(adx >= best_ra["adx_thr"])) & (vr >= best_ra["vol_min"]) & (rsi < best_ra["rsi_os"])
        range_s = (~(adx >= best_ra["adx_thr"])) & (vr >= best_ra["vol_min"]) & (rsi > best_ra["rsi_ob"])

        signals = np.zeros(len(adx))
        signals[trend_l | range_l] = 1
        signals[trend_s | range_s] = -1

        p, t, w, d, pf = backtest_vec(
            signals, ind["high"][TRAIN_BARS:n], ind["low"][TRAIN_BARS:n],
            ind["close"][TRAIN_BARS:n], ind["atr"][TRAIN_BARS:n],
            best_ra["sl"], best_ra["tp"], 192)
        ra_val += p
        if t > 0:
            print(f"    {pair}: {p:+.2%} ({t} trades, WR {w*100:.1f}%, PF {pf:.2f})")
    print(f"    TOTAL VAL: {ra_val:+.2%}")

    # ═══════════════════════════════════════════════════════════════════════
    # VOLUME SPIKE REVERSAL — vectorized
    # ═══════════════════════════════════════════════════════════════════════
    t1 = time.time()
    print(f"\n{'='*80}")
    print("  VOLUME SPIKE REVERSAL")
    print(f"{'='*80}")

    best_vs = {"profit": -999}
    count = 0

    vs_grid = list(product(
        [1.8, 2.0, 2.5, 3.0, 3.5],   # spike
        [1.5, 2.0, 2.5, 3.0],         # shadow
        [25, 28, 30, 33, 35],          # rsi_os
        [65, 67, 70, 72, 75],          # rsi_ob
        [(2.0, 3.5), (2.5, 4.0), (3.0, 5.0), (3.5, 5.5), (4.0, 6.0)],  # sl, tp
    ))

    for spike, shadow, rsi_os, rsi_ob, (sl, tp) in vs_grid:
        total_profit = 0
        total_trades = 0

        for pair, ind in data.items():
            n = TRAIN_BARS
            vr = ind["vol_ratio"][:n]
            rsi = ind["rsi"][:n]
            ba = ind["body_abs"][:n]
            ls = ind["lower_shadow"][:n]
            us = ind["upper_shadow"][:n]

            valid = (vr >= spike) & (ba > 0)
            long_sig = valid & (ls > shadow * ba) & (rsi < rsi_os)
            short_sig = valid & (us > shadow * ba) & (rsi > rsi_ob)

            signals = np.zeros(n)
            signals[long_sig] = 1
            signals[short_sig] = -1
            signals[:20] = 0

            p, t, w, d, pf = backtest_vec(
                signals, ind["high"][:n], ind["low"][:n],
                ind["close"][:n], ind["atr"][:n], sl, tp, 96)
            total_profit += p
            total_trades += t

        count += 1
        if total_trades >= 15 and total_profit > best_vs["profit"]:
            best_vs = {"profit": total_profit, "trades": total_trades,
                       "spike": spike, "shadow": shadow,
                       "rsi_os": rsi_os, "rsi_ob": rsi_ob, "sl": sl, "tp": tp}

    print(f"  Tested {count} combos in {time.time()-t1:.0f}s")
    print(f"  BEST TRAIN: profit={best_vs.get('profit',0):+.2%} | trades={best_vs.get('trades',0)}")
    print(f"    spike={best_vs.get('spike')}, shadow={best_vs.get('shadow')}, rsi={best_vs.get('rsi_os')}/{best_vs.get('rsi_ob')}")
    print(f"    sl/tp={best_vs.get('sl')}/{best_vs.get('tp')}")

    # Validate
    print(f"\n  VALIDATION (last 60 days):")
    vs_val = 0
    for pair, ind in data.items():
        n = ind["n"]
        vr = ind["vol_ratio"][TRAIN_BARS:n]
        rsi = ind["rsi"][TRAIN_BARS:n]
        ba = ind["body_abs"][TRAIN_BARS:n]
        ls = ind["lower_shadow"][TRAIN_BARS:n]
        us = ind["upper_shadow"][TRAIN_BARS:n]
        valid = (vr >= best_vs["spike"]) & (ba > 0)
        signals = np.zeros(len(vr))
        signals[valid & (ls > best_vs["shadow"] * ba) & (rsi < best_vs["rsi_os"])] = 1
        signals[valid & (us > best_vs["shadow"] * ba) & (rsi > best_vs["rsi_ob"])] = -1
        p, t, w, d, pf = backtest_vec(
            signals, ind["high"][TRAIN_BARS:n], ind["low"][TRAIN_BARS:n],
            ind["close"][TRAIN_BARS:n], ind["atr"][TRAIN_BARS:n],
            best_vs["sl"], best_vs["tp"], 96)
        vs_val += p
        if t > 0:
            print(f"    {pair}: {p:+.2%} ({t} trades, WR {w*100:.1f}%, PF {pf:.2f})")
    print(f"    TOTAL VAL: {vs_val:+.2%}")

    # ═══════════════════════════════════════════════════════════════════════
    # CB ADX BREAKOUT — vectorized
    # ═══════════════════════════════════════════════════════════════════════
    t2 = time.time()
    print(f"\n{'='*80}")
    print("  CB ADX BREAKOUT")
    print(f"{'='*80}")

    best_cb = {"profit": -999}
    count = 0

    cb_grid = list(product(
        [0.5, 0.6, 0.7, 0.8, 0.9],   # comp
        [0.6, 0.8, 1.0, 1.2],         # vol_min
        [18, 20, 22, 25, 28],          # adx_max
        [72, 75, 78],                  # rsi_max
        [22, 25, 28],                  # rsi_min
        [(2.5, 4.0), (3.0, 5.0), (3.5, 5.5), (4.0, 6.0), (4.5, 7.0)],  # sl, tp
    ))

    for comp, vol_min, adx_max, rsi_max, rsi_min, (sl, tp) in cb_grid:
        total_profit = 0
        total_trades = 0

        for pair, ind in data.items():
            n = TRAIN_BARS
            adx = ind["adx"][:n]
            vr = ind["vol_ratio"][:n]
            rsi = ind["rsi"][:n]
            atr = ind["atr"][:n]
            c = ind["close"][:n]
            r3 = ind["range_3bar"][:n]
            prev_h = ind["prev_h2"][:n]
            prev_l = ind["prev_l2"][:n]

            # ADX low in any of last 3 bars
            adx_low_0 = adx < adx_max
            adx_low_1 = np.roll(adx, 1) < adx_max
            adx_low_2 = np.roll(adx, 2) < adx_max
            adx_recent_low = adx_low_0 | adx_low_1 | adx_low_2

            # Compression
            compressed = (atr > 0) & (r3 / (atr + 1e-10) < comp)

            # Filters
            valid = (vr >= vol_min) & (rsi >= rsi_min) & (rsi <= rsi_max) & adx_recent_low & compressed

            long_sig = valid & (c > prev_h)
            short_sig = valid & (c < prev_l)

            signals = np.zeros(n)
            signals[long_sig] = 1
            signals[short_sig] = -1
            signals[:20] = 0

            p, t, w, d, pf = backtest_vec(
                signals, ind["high"][:n], ind["low"][:n],
                ind["close"][:n], ind["atr"][:n], sl, tp, 96)
            total_profit += p
            total_trades += t

        count += 1
        if total_trades >= 15 and total_profit > best_cb["profit"]:
            best_cb = {"profit": total_profit, "trades": total_trades,
                       "comp": comp, "vol_min": vol_min, "adx_max": adx_max,
                       "rsi_max": rsi_max, "rsi_min": rsi_min, "sl": sl, "tp": tp}

    print(f"  Tested {count} combos in {time.time()-t2:.0f}s")
    print(f"  BEST TRAIN: profit={best_cb.get('profit',0):+.2%} | trades={best_cb.get('trades',0)}")
    print(f"    comp={best_cb.get('comp')}, vol={best_cb.get('vol_min')}, adx_max={best_cb.get('adx_max')}")
    print(f"    rsi={best_cb.get('rsi_min')}/{best_cb.get('rsi_max')}, sl/tp={best_cb.get('sl')}/{best_cb.get('tp')}")

    # Validate
    print(f"\n  VALIDATION (last 60 days):")
    cb_val = 0
    for pair, ind in data.items():
        n = ind["n"]
        adx = ind["adx"][TRAIN_BARS:n]
        vr = ind["vol_ratio"][TRAIN_BARS:n]
        rsi = ind["rsi"][TRAIN_BARS:n]
        atr = ind["atr"][TRAIN_BARS:n]
        c = ind["close"][TRAIN_BARS:n]
        r3 = ind["range_3bar"][TRAIN_BARS:n]
        prev_h = ind["prev_h2"][TRAIN_BARS:n]
        prev_l = ind["prev_l2"][TRAIN_BARS:n]

        adx_low = (adx < best_cb["adx_max"]) | (np.roll(adx, 1) < best_cb["adx_max"]) | (np.roll(adx, 2) < best_cb["adx_max"])
        compressed = (atr > 0) & (r3 / (atr + 1e-10) < best_cb["comp"])
        valid = (vr >= best_cb["vol_min"]) & (rsi >= best_cb["rsi_min"]) & (rsi <= best_cb["rsi_max"]) & adx_low & compressed
        signals = np.zeros(len(adx))
        signals[valid & (c > prev_h)] = 1
        signals[valid & (c < prev_l)] = -1
        p, t, w, d, pf = backtest_vec(
            signals, ind["high"][TRAIN_BARS:n], ind["low"][TRAIN_BARS:n],
            ind["close"][TRAIN_BARS:n], ind["atr"][TRAIN_BARS:n],
            best_cb["sl"], best_cb["tp"], 96)
        cb_val += p
        if t > 0:
            print(f"    {pair}: {p:+.2%} ({t} trades, WR {w*100:.1f}%, PF {pf:.2f})")
    print(f"    TOTAL VAL: {cb_val:+.2%}")

    # ═══════════════════════════════════════════════════════════════════════
    # FULL PERIOD with best params
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  FULL PERIOD (180 days) — Per Pair Breakdown")
    print(f"{'='*80}")

    print("\n  [regime_adaptive]")
    ra_full = 0
    ra_full_trades = 0
    for pair, ind in data.items():
        n = ind["n"]
        adx = ind["adx"][:n]; di_p = ind["di_plus"][:n]; di_m = ind["di_minus"][:n]
        rsi = ind["rsi"][:n]; vr = ind["vol_ratio"][:n]
        cross_up = ind[f"cross_up_{best_ra['ema_f']}_{best_ra['ema_s']}"][:n]
        cross_down = ind[f"cross_down_{best_ra['ema_f']}_{best_ra['ema_s']}"][:n]
        recent_up = maximum_filter1d(cross_up.astype(float), size=best_ra["cross_lb"], origin=best_ra["cross_lb"]//2-1)
        recent_down = maximum_filter1d(cross_down.astype(float), size=best_ra["cross_lb"], origin=best_ra["cross_lb"]//2-1)
        trend_l = (adx >= best_ra["adx_thr"]) & (vr >= best_ra["vol_min"]) & (recent_up > 0) & (di_p > di_m)
        trend_s = (adx >= best_ra["adx_thr"]) & (vr >= best_ra["vol_min"]) & (recent_down > 0) & (di_m > di_p)
        range_l = (~(adx >= best_ra["adx_thr"])) & (vr >= best_ra["vol_min"]) & (rsi < best_ra["rsi_os"])
        range_s = (~(adx >= best_ra["adx_thr"])) & (vr >= best_ra["vol_min"]) & (rsi > best_ra["rsi_ob"])
        signals = np.zeros(n)
        signals[trend_l | range_l] = 1
        signals[trend_s | range_s] = -1
        signals[:STARTUP] = 0
        p, t, w, d, pf = backtest_vec(signals, ind["high"][:n], ind["low"][:n], ind["close"][:n], ind["atr"][:n], best_ra["sl"], best_ra["tp"], 192)
        ra_full += p; ra_full_trades += t
        if t > 0:
            print(f"    {pair}: {p:+.2%} ({t} trades, WR {w*100:.1f}%, PF {pf:.2f}, DD {d:.2%})")
    print(f"    TOTAL: {ra_full:+.2%} ({ra_full_trades} trades)")

    print("\n  [volume_spike_rev]")
    vs_full = 0; vs_full_trades = 0
    for pair, ind in data.items():
        n = ind["n"]
        vr = ind["vol_ratio"][:n]; rsi = ind["rsi"][:n]
        ba = ind["body_abs"][:n]; ls = ind["lower_shadow"][:n]; us = ind["upper_shadow"][:n]
        valid = (vr >= best_vs["spike"]) & (ba > 0)
        signals = np.zeros(n)
        signals[valid & (ls > best_vs["shadow"] * ba) & (rsi < best_vs["rsi_os"])] = 1
        signals[valid & (us > best_vs["shadow"] * ba) & (rsi > best_vs["rsi_ob"])] = -1
        signals[:20] = 0
        p, t, w, d, pf = backtest_vec(signals, ind["high"][:n], ind["low"][:n], ind["close"][:n], ind["atr"][:n], best_vs["sl"], best_vs["tp"], 96)
        vs_full += p; vs_full_trades += t
        if t > 0:
            print(f"    {pair}: {p:+.2%} ({t} trades, WR {w*100:.1f}%, PF {pf:.2f}, DD {d:.2%})")
    print(f"    TOTAL: {vs_full:+.2%} ({vs_full_trades} trades)")

    print("\n  [cb_adx_breakout]")
    cb_full = 0; cb_full_trades = 0
    for pair, ind in data.items():
        n = ind["n"]
        adx = ind["adx"][:n]; vr = ind["vol_ratio"][:n]; rsi = ind["rsi"][:n]
        atr = ind["atr"][:n]; c = ind["close"][:n]
        r3 = ind["range_3bar"][:n]; prev_h = ind["prev_h2"][:n]; prev_l = ind["prev_l2"][:n]
        adx_low = (adx < best_cb["adx_max"]) | (np.roll(adx, 1) < best_cb["adx_max"]) | (np.roll(adx, 2) < best_cb["adx_max"])
        compressed = (atr > 0) & (r3 / (atr + 1e-10) < best_cb["comp"])
        valid = (vr >= best_cb["vol_min"]) & (rsi >= best_cb["rsi_min"]) & (rsi <= best_cb["rsi_max"]) & adx_low & compressed
        signals = np.zeros(n)
        signals[valid & (c > prev_h)] = 1
        signals[valid & (c < prev_l)] = -1
        signals[:20] = 0
        p, t, w, d, pf = backtest_vec(signals, ind["high"][:n], ind["low"][:n], ind["close"][:n], ind["atr"][:n], best_cb["sl"], best_cb["tp"], 96)
        cb_full += p; cb_full_trades += t
        if t > 0:
            print(f"    {pair}: {p:+.2%} ({t} trades, WR {w*100:.1f}%, PF {pf:.2f}, DD {d:.2%})")
    print(f"    TOTAL: {cb_full:+.2%} ({cb_full_trades} trades)")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  Total time: {time.time()-t0:.0f}s\n")

    print(f"  {'Strategy':<22} {'Train':>8} {'Val':>8} {'Full':>8} {'Trades':>7}")
    print(f"  {'-'*60}")
    print(f"  {'regime_adaptive':<22} {best_ra.get('profit',0):+.2%}  {ra_val:+.2%}  {ra_full:+.2%}  {ra_full_trades:>5}")
    print(f"  {'volume_spike_rev':<22} {best_vs.get('profit',0):+.2%}  {vs_val:+.2%}  {vs_full:+.2%}  {vs_full_trades:>5}")
    print(f"  {'cb_adx_breakout':<22} {best_cb.get('profit',0):+.2%}  {cb_val:+.2%}  {cb_full:+.2%}  {cb_full_trades:>5}")
    combined = ra_full + vs_full + cb_full
    print(f"\n  COMBINED (all strategies): {combined:+.2%}")

    print(f"\n  Optimized Parameters:")
    print(f"  regime_adaptive:")
    print(f"    adx_thr: {best_ra.get('adx_thr')}, rsi_os: {best_ra.get('rsi_os')}, rsi_ob: {best_ra.get('rsi_ob')}")
    print(f"    vol_min: {best_ra.get('vol_min')}, ema: {best_ra.get('ema_f')}/{best_ra.get('ema_s')}, cross_lb: {best_ra.get('cross_lb')}")
    print(f"    sl_atr_mult: {best_ra.get('sl')}, tp_atr_mult: {best_ra.get('tp')}")
    print(f"  volume_spike_rev:")
    print(f"    spike_mult: {best_vs.get('spike')}, shadow_mult: {best_vs.get('shadow')}")
    print(f"    rsi_os_thr: {best_vs.get('rsi_os')}, rsi_ob_thr: {best_vs.get('rsi_ob')}")
    print(f"    sl_atr_mult: {best_vs.get('sl')}, tp_atr_mult: {best_vs.get('tp')}")
    print(f"  cb_adx_breakout:")
    print(f"    compression_thr: {best_cb.get('comp')}, vol_min: {best_cb.get('vol_min')}, adx_max: {best_cb.get('adx_max')}")
    print(f"    rsi_min: {best_cb.get('rsi_min')}, rsi_max: {best_cb.get('rsi_max')}")
    print(f"    sl_atr_mult: {best_cb.get('sl')}, tp_atr_mult: {best_cb.get('tp')}")


if __name__ == "__main__":
    from scipy.ndimage import maximum_filter1d
    main()

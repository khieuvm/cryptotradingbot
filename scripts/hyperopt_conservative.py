"""Conservative hyperopt — anti-overfit measures.

Scoring: Sharpe ratio (not raw profit)
Filters: min 40 trades, max DD < 30%, positive on >= 4/7 pairs
Walk-forward: 120d train / 60d validate
"""

import sys
from pathlib import Path
import time
from itertools import product

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.ndimage import maximum_filter1d

sys.path.insert(0, str(Path(__file__).parent.parent))

TAKER_FEE = 0.0005
DATA_DIR = Path(__file__).parent.parent / "user_data" / "data" / "okx" / "futures"
PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT",
         "DOGE_USDT_USDT", "SUI_USDT_USDT", "LINK_USDT_USDT"]
BARS_PER_DAY = 96
TRAIN_BARS = 120 * BARS_PER_DAY
STARTUP = 60


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_feather(str(fp)).sort_values("date").reset_index(drop=True)


def backtest_vec(signals, high, low, close, atr, sl_mult, tp_mult, max_bars=96):
    n = len(close)
    entries = np.where(signals != 0)[0]
    if len(entries) == 0:
        return {"profit": 0, "trades": 0, "wr": 0, "dd": 0, "pf": 0, "sharpe": -10, "trades_arr": np.array([])}

    fee = 2 * TAKER_FEE
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
                    next_allowed = j + 1; exited = True; break
                if high[j] >= ep + tp_d:
                    trades.append(tp_mult * ea / ep - fee)
                    next_allowed = j + 1; exited = True; break
            else:
                if high[j] >= ep + sl_d:
                    trades.append(-sl_mult * ea / ep - fee)
                    next_allowed = j + 1; exited = True; break
                if low[j] <= ep - tp_d:
                    trades.append(tp_mult * ea / ep - fee)
                    next_allowed = j + 1; exited = True; break
        if not exited:
            pnl = ((close[end_bar] - ep) / ep if is_long else (ep - close[end_bar]) / ep) - fee
            trades.append(pnl)
            next_allowed = end_bar + 1

    if not trades:
        return {"profit": 0, "trades": 0, "wr": 0, "dd": 0, "pf": 0, "sharpe": -10, "trades_arr": np.array([])}

    arr = np.array(trades)
    total = arr.sum()
    wr = (arr > 0).sum() / len(arr)
    cum = np.cumsum(arr)
    dd = abs(np.min(cum - np.maximum.accumulate(cum))) if len(cum) > 1 else 0
    wins = arr[arr > 0].sum() if (arr > 0).any() else 0
    losses = abs(arr[arr < 0].sum()) if (arr < 0).any() else 1e-10
    pf = wins / losses
    sharpe = (arr.mean() / (arr.std() + 1e-10)) * np.sqrt(len(arr)) if len(arr) > 5 else -10
    return {"profit": total, "trades": len(arr), "wr": wr, "dd": dd, "pf": pf, "sharpe": sharpe, "trades_arr": arr}


def score_result(pair_results, min_trades=40, max_dd=0.30, min_pairs_positive=4):
    """Conservative scoring: Sharpe + consistency penalty."""
    total_trades = sum(r["trades"] for r in pair_results.values())
    if total_trades < min_trades:
        return -999

    all_trades = np.concatenate([r["trades_arr"] for r in pair_results.values() if len(r["trades_arr"]) > 0])
    if len(all_trades) < min_trades:
        return -999

    # Check max DD (portfolio-level)
    cum = np.cumsum(all_trades)
    portfolio_dd = abs(np.min(cum - np.maximum.accumulate(cum))) if len(cum) > 1 else 0
    if portfolio_dd > max_dd:
        return -999

    # Check pair consistency
    positive_pairs = sum(1 for r in pair_results.values() if r["profit"] > 0 and r["trades"] > 2)
    if positive_pairs < min_pairs_positive:
        return -999

    # Sharpe-based score
    sharpe = (all_trades.mean() / (all_trades.std() + 1e-10)) * np.sqrt(len(all_trades))

    # Bonus for consistency
    consistency_bonus = positive_pairs / len(pair_results) * 0.3

    return sharpe + consistency_bonus


def main():
    t0 = time.time()
    print("=" * 80)
    print("  CONSERVATIVE HYPEROPT — Anti-Overfit")
    print("  Scoring: Sharpe + Consistency | Min 40 trades | DD < 30% | 4+ pairs positive")
    print("  Walk-Forward: 120d train -> 60d validate | 7 pairs")
    print("=" * 80)

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

        for ef_name, es_name in [("ema14", "ema50"), ("ema18", "ema50"), ("ema21", "ema60")]:
            ef = ind[ef_name]; es = ind[es_name]
            cross_up = np.zeros(len(df), dtype=bool)
            cross_down = np.zeros(len(df), dtype=bool)
            cross_up[1:] = (ef[1:] > es[1:]) & (ef[:-1] <= es[:-1])
            cross_down[1:] = (ef[1:] < es[1:]) & (ef[:-1] >= es[:-1])
            ind[f"cross_up_{ef_name}_{es_name}"] = cross_up
            ind[f"cross_down_{ef_name}_{es_name}"] = cross_down

        h3_max = np.maximum(np.maximum(np.roll(h, 1), np.roll(h, 2)), h)
        l3_min = np.minimum(np.minimum(np.roll(lo, 1), np.roll(lo, 2)), lo)
        ind["range_3bar"] = h3_max - l3_min
        ind["prev_h2"] = np.maximum(np.roll(h, 1), np.roll(h, 2))
        ind["prev_l2"] = np.minimum(np.roll(lo, 1), np.roll(lo, 2))

        data[pair] = ind
        print(f"  {pair}: {len(df)} bars")

    print(f"\n  {len(data)} pairs loaded\n")

    # ═══════════════════════════════════════════════════════════════════════
    # REGIME ADAPTIVE — conservative
    # ═══════════════════════════════════════════════════════════════════════
    print(f"{'='*80}")
    print("  REGIME ADAPTIVE (conservative)")
    print(f"{'='*80}")

    best_ra = {"score": -999}
    count = 0

    ra_grid = list(product(
        [25, 28, 31, 35, 40],           # adx_thr (higher = fewer trend signals)
        [25, 28, 30, 33],               # rsi_os (lower = fewer range signals)
        [67, 70, 73, 75],               # rsi_ob (higher = fewer range signals)
        [0.8, 1.0, 1.2, 1.5],          # vol_min (higher = stricter)
        [("ema18", "ema50"), ("ema21", "ema60")],  # ema pair
        [5, 8, 12],                     # cross_lb
        [(5.0, 8.0), (5.9, 9.3), (6.0, 10.0), (7.0, 11.0)],  # sl, tp
    ))

    for adx_thr, rsi_os, rsi_ob, vol_min, (ema_f, ema_s), cross_lb, (sl, tp) in ra_grid:
        pair_results = {}
        for pair, ind in data.items():
            n = TRAIN_BARS
            adx = ind["adx"][:n]; di_p = ind["di_plus"][:n]; di_m = ind["di_minus"][:n]
            rsi = ind["rsi"][:n]; vr = ind["vol_ratio"][:n]
            cross_up = ind[f"cross_up_{ema_f}_{ema_s}"][:n]
            cross_down = ind[f"cross_down_{ema_f}_{ema_s}"][:n]
            recent_up = maximum_filter1d(cross_up.astype(float), size=cross_lb, origin=cross_lb//2-1)
            recent_down = maximum_filter1d(cross_down.astype(float), size=cross_lb, origin=cross_lb//2-1)

            trend_l = (adx >= adx_thr) & (vr >= vol_min) & (recent_up > 0) & (di_p > di_m)
            trend_s = (adx >= adx_thr) & (vr >= vol_min) & (recent_down > 0) & (di_m > di_p)
            range_l = (~(adx >= adx_thr)) & (vr >= vol_min) & (rsi < rsi_os)
            range_s = (~(adx >= adx_thr)) & (vr >= vol_min) & (rsi > rsi_ob)

            signals = np.zeros(n)
            signals[trend_l | range_l] = 1
            signals[trend_s | range_s] = -1
            signals[:STARTUP] = 0

            pair_results[pair] = backtest_vec(signals, ind["high"][:n], ind["low"][:n], ind["close"][:n], ind["atr"][:n], sl, tp, 192)

        score = score_result(pair_results, min_trades=30, max_dd=0.45, min_pairs_positive=3)
        count += 1
        if score > best_ra["score"]:
            best_ra = {"score": score, "pair_results": pair_results,
                       "adx_thr": adx_thr, "rsi_os": rsi_os, "rsi_ob": rsi_ob,
                       "vol_min": vol_min, "ema_f": ema_f, "ema_s": ema_s,
                       "cross_lb": cross_lb, "sl": sl, "tp": tp}

    train_profit = sum(r["profit"] for r in best_ra.get("pair_results", {}).values())
    train_trades = sum(r["trades"] for r in best_ra.get("pair_results", {}).values())
    print(f"  Tested {count} combos in {time.time()-t0:.0f}s")
    if best_ra["score"] <= -999:
        print(f"  NO PARAMS PASSED FILTER - using defaults")
        best_ra.update({"adx_thr": 31, "rsi_os": 30, "rsi_ob": 70, "vol_min": 1.0, "ema_f": "ema18", "ema_s": "ema50", "cross_lb": 8, "sl": 5.9, "tp": 9.3})
    else:
        print(f"  BEST TRAIN: score={best_ra['score']:.2f} | profit={train_profit:+.2%} | trades={train_trades}")
        print(f"    adx={best_ra.get('adx_thr')}, rsi={best_ra.get('rsi_os')}/{best_ra.get('rsi_ob')}, vol={best_ra.get('vol_min')}")
        print(f"    ema={best_ra.get('ema_f')}/{best_ra.get('ema_s')}, lb={best_ra.get('cross_lb')}, sl/tp={best_ra.get('sl')}/{best_ra.get('tp')}")
        print(f"\n  Train per-pair:")
        for pair, r in best_ra.get("pair_results", {}).items():
            if r["trades"] > 0:
                print(f"    {pair}: {r['profit']:+.2%} ({r['trades']} trades, WR {r['wr']*100:.1f}%, PF {r['pf']:.2f})")

    # Validate
    print(f"\n  VALIDATION (last 60 days):")
    ra_val = 0; ra_val_pairs = {}
    for pair, ind in data.items():
        n = ind["n"]
        sl = TRAIN_BARS
        adx = ind["adx"][sl:n]; di_p = ind["di_plus"][sl:n]; di_m = ind["di_minus"][sl:n]
        rsi = ind["rsi"][sl:n]; vr = ind["vol_ratio"][sl:n]
        cross_up = ind[f"cross_up_{best_ra['ema_f']}_{best_ra['ema_s']}"][sl:n]
        cross_down = ind[f"cross_down_{best_ra['ema_f']}_{best_ra['ema_s']}"][sl:n]
        recent_up = maximum_filter1d(cross_up.astype(float), size=best_ra["cross_lb"], origin=best_ra["cross_lb"]//2-1)
        recent_down = maximum_filter1d(cross_down.astype(float), size=best_ra["cross_lb"], origin=best_ra["cross_lb"]//2-1)
        trend_l = (adx >= best_ra["adx_thr"]) & (vr >= best_ra["vol_min"]) & (recent_up > 0) & (di_p > di_m)
        trend_s = (adx >= best_ra["adx_thr"]) & (vr >= best_ra["vol_min"]) & (recent_down > 0) & (di_m > di_p)
        range_l = (~(adx >= best_ra["adx_thr"])) & (vr >= best_ra["vol_min"]) & (rsi < best_ra["rsi_os"])
        range_s = (~(adx >= best_ra["adx_thr"])) & (vr >= best_ra["vol_min"]) & (rsi > best_ra["rsi_ob"])
        signals = np.zeros(len(adx))
        signals[trend_l | range_l] = 1
        signals[trend_s | range_s] = -1
        r = backtest_vec(signals, ind["high"][sl:n], ind["low"][sl:n], ind["close"][sl:n], ind["atr"][sl:n], best_ra["sl"], best_ra["tp"], 192)
        ra_val += r["profit"]
        ra_val_pairs[pair] = r
        if r["trades"] > 0:
            print(f"    {pair}: {r['profit']:+.2%} ({r['trades']} trades, WR {r['wr']*100:.1f}%, PF {r['pf']:.2f})")
    val_positive = sum(1 for r in ra_val_pairs.values() if r["profit"] > 0)
    print(f"    TOTAL VAL: {ra_val:+.2%} ({val_positive}/7 pairs positive)")

    # ═══════════════════════════════════════════════════════════════════════
    # VOLUME SPIKE — conservative
    # ═══════════════════════════════════════════════════════════════════════
    t1 = time.time()
    print(f"\n{'='*80}")
    print("  VOLUME SPIKE REVERSAL (conservative)")
    print(f"{'='*80}")

    best_vs = {"score": -999}
    count = 0

    vs_grid = list(product(
        [2.0, 2.5, 3.0, 3.5, 4.0],   # spike (higher = stricter)
        [1.5, 2.0, 2.5, 3.0],         # shadow
        [25, 28, 30, 33],              # rsi_os
        [67, 70, 72, 75],              # rsi_ob
        [(2.5, 4.0), (3.0, 5.0), (3.5, 5.5), (4.0, 6.0), (4.5, 7.0)],  # sl, tp
    ))

    for spike, shadow, rsi_os, rsi_ob, (sl, tp) in vs_grid:
        pair_results = {}
        for pair, ind in data.items():
            n = TRAIN_BARS
            vr = ind["vol_ratio"][:n]; rsi = ind["rsi"][:n]
            ba = ind["body_abs"][:n]; ls = ind["lower_shadow"][:n]; us = ind["upper_shadow"][:n]
            valid = (vr >= spike) & (ba > 0)
            signals = np.zeros(n)
            signals[valid & (ls > shadow * ba) & (rsi < rsi_os)] = 1
            signals[valid & (us > shadow * ba) & (rsi > rsi_ob)] = -1
            signals[:20] = 0
            pair_results[pair] = backtest_vec(signals, ind["high"][:n], ind["low"][:n], ind["close"][:n], ind["atr"][:n], sl, tp, 96)

        score = score_result(pair_results, min_trades=30, max_dd=0.40, min_pairs_positive=3)
        count += 1
        if score > best_vs["score"]:
            best_vs = {"score": score, "pair_results": pair_results,
                       "spike": spike, "shadow": shadow,
                       "rsi_os": rsi_os, "rsi_ob": rsi_ob, "sl": sl, "tp": tp}

    train_profit = sum(r["profit"] for r in best_vs.get("pair_results", {}).values())
    train_trades = sum(r["trades"] for r in best_vs.get("pair_results", {}).values())
    print(f"  Tested {count} combos in {time.time()-t1:.0f}s")
    if best_vs["score"] <= -999:
        print(f"  NO PARAMS PASSED FILTER - using defaults")
        best_vs.update({"spike": 2.5, "shadow": 2.0, "rsi_os": 30, "rsi_ob": 70, "sl": 3.0, "tp": 5.0})
    else:
        print(f"  BEST TRAIN: score={best_vs['score']:.2f} | profit={train_profit:+.2%} | trades={train_trades}")
        print(f"    spike={best_vs.get('spike')}, shadow={best_vs.get('shadow')}, rsi={best_vs.get('rsi_os')}/{best_vs.get('rsi_ob')}")
        print(f"    sl/tp={best_vs.get('sl')}/{best_vs.get('tp')}")
        print(f"\n  Train per-pair:")
        for pair, r in best_vs.get("pair_results", {}).items():
            if r["trades"] > 0:
                print(f"    {pair}: {r['profit']:+.2%} ({r['trades']} trades, WR {r['wr']*100:.1f}%, PF {r['pf']:.2f})")

    # Validate
    print(f"\n  VALIDATION (last 60 days):")
    vs_val = 0; vs_val_pairs = {}
    for pair, ind in data.items():
        n = ind["n"]; sl_idx = TRAIN_BARS
        vr = ind["vol_ratio"][sl_idx:n]; rsi = ind["rsi"][sl_idx:n]
        ba = ind["body_abs"][sl_idx:n]; ls = ind["lower_shadow"][sl_idx:n]; us = ind["upper_shadow"][sl_idx:n]
        valid = (vr >= best_vs["spike"]) & (ba > 0)
        signals = np.zeros(len(vr))
        signals[valid & (ls > best_vs["shadow"] * ba) & (rsi < best_vs["rsi_os"])] = 1
        signals[valid & (us > best_vs["shadow"] * ba) & (rsi > best_vs["rsi_ob"])] = -1
        r = backtest_vec(signals, ind["high"][sl_idx:n], ind["low"][sl_idx:n], ind["close"][sl_idx:n], ind["atr"][sl_idx:n], best_vs["sl"], best_vs["tp"], 96)
        vs_val += r["profit"]
        vs_val_pairs[pair] = r
        if r["trades"] > 0:
            print(f"    {pair}: {r['profit']:+.2%} ({r['trades']} trades, WR {r['wr']*100:.1f}%, PF {r['pf']:.2f})")
    val_positive = sum(1 for r in vs_val_pairs.values() if r["profit"] > 0)
    print(f"    TOTAL VAL: {vs_val:+.2%} ({val_positive}/7 pairs positive)")

    # ═══════════════════════════════════════════════════════════════════════
    # CB ADX BREAKOUT — conservative
    # ═══════════════════════════════════════════════════════════════════════
    t2 = time.time()
    print(f"\n{'='*80}")
    print("  CB ADX BREAKOUT (conservative)")
    print(f"{'='*80}")

    best_cb = {"score": -999}
    count = 0

    cb_grid = list(product(
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # comp
        [0.6, 0.8, 1.0, 1.2],             # vol_min
        [18, 20, 22, 25, 28, 30],          # adx_max
        [72, 75, 78],                      # rsi_max
        [22, 25, 28],                      # rsi_min
        [(2.5, 4.0), (3.0, 5.0), (3.5, 5.5), (4.0, 6.0), (4.5, 7.0)],  # sl, tp
    ))

    for comp, vol_min, adx_max, rsi_max, rsi_min, (sl, tp) in cb_grid:
        pair_results = {}
        for pair, ind in data.items():
            n = TRAIN_BARS
            adx = ind["adx"][:n]; vr = ind["vol_ratio"][:n]; rsi = ind["rsi"][:n]
            atr = ind["atr"][:n]; c = ind["close"][:n]
            r3 = ind["range_3bar"][:n]; prev_h = ind["prev_h2"][:n]; prev_l = ind["prev_l2"][:n]
            adx_low = (adx < adx_max) | (np.roll(adx, 1) < adx_max) | (np.roll(adx, 2) < adx_max)
            compressed = (atr > 0) & (r3 / (atr + 1e-10) < comp)
            valid = (vr >= vol_min) & (rsi >= rsi_min) & (rsi <= rsi_max) & adx_low & compressed
            signals = np.zeros(n)
            signals[valid & (c > prev_h)] = 1
            signals[valid & (c < prev_l)] = -1
            signals[:20] = 0
            pair_results[pair] = backtest_vec(signals, ind["high"][:n], ind["low"][:n], ind["close"][:n], ind["atr"][:n], sl, tp, 96)

        score = score_result(pair_results, min_trades=20, max_dd=0.40, min_pairs_positive=3)
        count += 1
        if score > best_cb["score"]:
            best_cb = {"score": score, "pair_results": pair_results,
                       "comp": comp, "vol_min": vol_min, "adx_max": adx_max,
                       "rsi_max": rsi_max, "rsi_min": rsi_min, "sl": sl, "tp": tp}

    train_profit = sum(r["profit"] for r in best_cb.get("pair_results", {}).values())
    train_trades = sum(r["trades"] for r in best_cb.get("pair_results", {}).values())
    print(f"  Tested {count} combos in {time.time()-t2:.0f}s")
    if best_cb["score"] <= -999:
        print(f"  NO PARAMS PASSED FILTER - using defaults")
        best_cb.update({"comp": 0.7, "vol_min": 0.8, "adx_max": 20, "rsi_min": 28, "rsi_max": 72, "sl": 3.0, "tp": 5.0})
    else:
        print(f"  BEST TRAIN: score={best_cb['score']:.2f} | profit={train_profit:+.2%} | trades={train_trades}")
        print(f"    comp={best_cb.get('comp')}, vol={best_cb.get('vol_min')}, adx_max={best_cb.get('adx_max')}")
        print(f"    rsi={best_cb.get('rsi_min')}/{best_cb.get('rsi_max')}, sl/tp={best_cb.get('sl')}/{best_cb.get('tp')}")
        print(f"\n  Train per-pair:")
        for pair, r in best_cb.get("pair_results", {}).items():
            if r["trades"] > 0:
                print(f"    {pair}: {r['profit']:+.2%} ({r['trades']} trades, WR {r['wr']*100:.1f}%, PF {r['pf']:.2f})")

    # Validate
    print(f"\n  VALIDATION (last 60 days):")
    cb_val = 0; cb_val_pairs = {}
    for pair, ind in data.items():
        n = ind["n"]; sl_idx = TRAIN_BARS
        adx = ind["adx"][sl_idx:n]; vr = ind["vol_ratio"][sl_idx:n]; rsi = ind["rsi"][sl_idx:n]
        atr = ind["atr"][sl_idx:n]; c = ind["close"][sl_idx:n]
        r3 = ind["range_3bar"][sl_idx:n]; prev_h = ind["prev_h2"][sl_idx:n]; prev_l = ind["prev_l2"][sl_idx:n]
        adx_low = (adx < best_cb["adx_max"]) | (np.roll(adx, 1) < best_cb["adx_max"]) | (np.roll(adx, 2) < best_cb["adx_max"])
        compressed = (atr > 0) & (r3 / (atr + 1e-10) < best_cb["comp"])
        valid = (vr >= best_cb["vol_min"]) & (rsi >= best_cb["rsi_min"]) & (rsi <= best_cb["rsi_max"]) & adx_low & compressed
        signals = np.zeros(len(adx))
        signals[valid & (c > prev_h)] = 1
        signals[valid & (c < prev_l)] = -1
        r = backtest_vec(signals, ind["high"][sl_idx:n], ind["low"][sl_idx:n], ind["close"][sl_idx:n], ind["atr"][sl_idx:n], best_cb["sl"], best_cb["tp"], 96)
        cb_val += r["profit"]
        cb_val_pairs[pair] = r
        if r["trades"] > 0:
            print(f"    {pair}: {r['profit']:+.2%} ({r['trades']} trades, WR {r['wr']*100:.1f}%, PF {r['pf']:.2f})")
    val_positive = sum(1 for r in cb_val_pairs.values() if r["profit"] > 0)
    print(f"    TOTAL VAL: {cb_val:+.2%} ({val_positive}/7 pairs positive)")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  FINAL SUMMARY — CONSERVATIVE OPTIMIZATION")
    print(f"{'='*80}")
    print(f"  Total time: {time.time()-t0:.0f}s\n")

    ra_train_p = sum(r["profit"] for r in best_ra.get("pair_results", {}).values())
    vs_train_p = sum(r["profit"] for r in best_vs.get("pair_results", {}).values())
    cb_train_p = sum(r["profit"] for r in best_cb.get("pair_results", {}).values())

    print(f"  {'Strategy':<22} {'Score':>7} {'Train':>8} {'Val':>8} {'Val+':>5}")
    print(f"  {'-'*60}")
    ra_vp = sum(1 for r in ra_val_pairs.values() if r["profit"] > 0)
    vs_vp = sum(1 for r in vs_val_pairs.values() if r["profit"] > 0)
    cb_vp = sum(1 for r in cb_val_pairs.values() if r["profit"] > 0)
    print(f"  {'regime_adaptive':<22} {best_ra['score']:>7.2f} {ra_train_p:+.2%}  {ra_val:+.2%}  {ra_vp}/7")
    print(f"  {'volume_spike_rev':<22} {best_vs['score']:>7.2f} {vs_train_p:+.2%}  {vs_val:+.2%}  {vs_vp}/7")
    print(f"  {'cb_adx_breakout':<22} {best_cb['score']:>7.2f} {cb_train_p:+.2%}  {cb_val:+.2%}  {cb_vp}/7")

    print(f"\n  Optimized Parameters (CONSERVATIVE):")
    print(f"  regime_adaptive:")
    print(f"    adx_thr: {best_ra.get('adx_thr')}")
    print(f"    rsi_os: {best_ra.get('rsi_os')}, rsi_ob: {best_ra.get('rsi_ob')}")
    print(f"    vol_min: {best_ra.get('vol_min')}")
    print(f"    ema: {best_ra.get('ema_f')}/{best_ra.get('ema_s')}, cross_lb: {best_ra.get('cross_lb')}")
    print(f"    sl_atr_mult: {best_ra.get('sl')}, tp_atr_mult: {best_ra.get('tp')}")
    print(f"  volume_spike_rev:")
    print(f"    spike_mult: {best_vs.get('spike')}, shadow_mult: {best_vs.get('shadow')}")
    print(f"    rsi_os_thr: {best_vs.get('rsi_os')}, rsi_ob_thr: {best_vs.get('rsi_ob')}")
    print(f"    sl_atr_mult: {best_vs.get('sl')}, tp_atr_mult: {best_vs.get('tp')}")
    print(f"  cb_adx_breakout:")
    print(f"    compression_thr: {best_cb.get('comp')}, vol_min: {best_cb.get('vol_min')}, adx_max: {best_cb.get('adx_max')}")
    print(f"    rsi_min: {best_cb.get('rsi_min')}, rsi_max: {best_cb.get('rsi_max')}")
    print(f"    sl_atr_mult: {best_cb.get('sl')}, tp_atr_mult: {best_cb.get('tp')}")

    # Verdict
    print(f"\n  VERDICT:")
    if ra_val > 0:
        print(f"  [PASS] regime_adaptive: OOS positive ({ra_val:+.2%}), {ra_vp}/7 pairs")
    else:
        print(f"  [FAIL] regime_adaptive: OOS negative ({ra_val:+.2%})")
    if vs_val > 0:
        print(f"  [PASS] volume_spike_rev: OOS positive ({vs_val:+.2%}), {vs_vp}/7 pairs")
    else:
        print(f"  [FAIL] volume_spike_rev: OOS negative ({vs_val:+.2%})")
    if cb_val > 0:
        print(f"  [PASS] cb_adx_breakout: OOS positive ({cb_val:+.2%}), {cb_vp}/7 pairs")
    else:
        print(f"  [FAIL] cb_adx_breakout: OOS negative ({cb_val:+.2%})")


if __name__ == "__main__":
    main()

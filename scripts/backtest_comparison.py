"""Compare BEFORE vs AFTER optimization on real OKX data.

BEFORE: old params (adx=25, rsi=28/67, vol=1.0, ema21/60, sl=10, tp=11)
AFTER: new params (adx=31, rsi=25/70, vol=1.5, ema18/50, sl=7, tp=11) + per-pair improvements

Includes limit entry simulation and time-cut improvements.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.ndimage import maximum_filter1d

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "user_data" / "data" / "okx" / "futures"
PAIRS = ["ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT", "DOGE_USDT_USDT", "LINK_USDT_USDT"]
TAKER_FEE = 0.0005
STARTUP = 60


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    return pd.read_feather(str(fp)).sort_values("date").reset_index(drop=True)


def backtest_advanced(signals, high, low, close, atr, sl_mult, tp_mult, max_bars=192,
                      entry_offset_atr=0.0, time_cuts=None, trailing=None):
    """Advanced backtest with limit entry, time cuts, and trailing SL."""
    n = len(close)
    entries = np.where(signals != 0)[0]
    if len(entries) == 0:
        return {"profit": 0, "trades": 0, "wr": 0, "dd": 0, "pf": 0, "sl_count": 0, "tp_count": 0, "tc_count": 0}

    fee = 2 * TAKER_FEE
    trades = []
    sl_count = 0; tp_count = 0; tc_count = 0
    next_allowed = 0

    for idx in entries:
        if idx < next_allowed:
            continue
        ea = atr[idx]
        if ea <= 0 or close[idx] <= 0 or np.isnan(ea):
            continue

        is_long = signals[idx] > 0

        # Limit entry: get better price by offset
        if entry_offset_atr > 0:
            offset = entry_offset_atr * ea
            limit_price = close[idx] - offset if is_long else close[idx] + offset
            # Check if limit was filled in next 3 bars
            filled = False
            for fb in range(idx + 1, min(idx + 4, n)):
                if is_long and low[fb] <= limit_price:
                    ep = limit_price
                    start_bar = fb
                    filled = True
                    break
                elif not is_long and high[fb] >= limit_price:
                    ep = limit_price
                    start_bar = fb
                    filled = True
                    break
            if not filled:
                continue
        else:
            ep = close[idx]
            start_bar = idx + 1

        sl_d = sl_mult * ea
        tp_d = tp_mult * ea
        end_bar = min(idx + max_bars, n - 1)

        # Trailing SL state
        trail_active = False
        trail_sl = None
        if trailing:
            be_trigger_pct = trailing.get("be_trigger", 0.4) * tp_d / ep
            trail_trigger_pct = trailing.get("trail_trigger", 0.6) * tp_d / ep
            trail_dist = trailing.get("trail_dist", 0.4) * ea

        exited = False
        best_profit = 0

        for j in range(start_bar, end_bar + 1):
            if is_long:
                unrealized_pct = (close[j] - ep) / ep
                # SL check
                if trail_active and trail_sl is not None:
                    if low[j] <= trail_sl:
                        pnl = (trail_sl - ep) / ep - fee
                        trades.append(pnl)
                        tp_count += 1
                        next_allowed = j + 1; exited = True; break
                elif low[j] <= ep - sl_d:
                    trades.append(-sl_mult * ea / ep - fee)
                    sl_count += 1
                    next_allowed = j + 1; exited = True; break
                # TP check
                if high[j] >= ep + tp_d:
                    trades.append(tp_mult * ea / ep - fee)
                    tp_count += 1
                    next_allowed = j + 1; exited = True; break
                # Trailing update
                if trailing:
                    curr_profit = (high[j] - ep) / ep
                    best_profit = max(best_profit, curr_profit)
                    if best_profit >= trail_trigger_pct:
                        trail_active = True
                        new_sl = high[j] - trail_dist
                        trail_sl = max(trail_sl or 0, new_sl)
                    elif best_profit >= be_trigger_pct and not trail_active:
                        trail_sl = ep + 0.001 * ep  # break-even + tiny buffer
            else:
                # Short
                if trail_active and trail_sl is not None:
                    if high[j] >= trail_sl:
                        pnl = (ep - trail_sl) / ep - fee
                        trades.append(pnl)
                        tp_count += 1
                        next_allowed = j + 1; exited = True; break
                elif high[j] >= ep + sl_d:
                    trades.append(-sl_mult * ea / ep - fee)
                    sl_count += 1
                    next_allowed = j + 1; exited = True; break
                if low[j] <= ep - tp_d:
                    trades.append(tp_mult * ea / ep - fee)
                    tp_count += 1
                    next_allowed = j + 1; exited = True; break
                if trailing:
                    curr_profit = (ep - low[j]) / ep
                    best_profit = max(best_profit, curr_profit)
                    if best_profit >= trail_trigger_pct:
                        trail_active = True
                        new_sl = low[j] + trail_dist
                        trail_sl = min(trail_sl or 1e18, new_sl)
                    elif best_profit >= be_trigger_pct and not trail_active:
                        trail_sl = ep - 0.001 * ep

            # Time cuts
            if time_cuts and not exited:
                bars_held = j - idx
                hours_held = bars_held * 15 / 60
                curr_pnl = ((close[j] - ep) / ep if is_long else (ep - close[j]) / ep)
                for tc_hours, tc_threshold in time_cuts:
                    if hours_held >= tc_hours and curr_pnl < tc_threshold:
                        trades.append(curr_pnl - fee)
                        tc_count += 1
                        next_allowed = j + 1; exited = True; break
                if exited:
                    break

        if not exited:
            pnl = ((close[end_bar] - ep) / ep if is_long else (ep - close[end_bar]) / ep) - fee
            trades.append(pnl)
            tc_count += 1
            next_allowed = end_bar + 1

    if not trades:
        return {"profit": 0, "trades": 0, "wr": 0, "dd": 0, "pf": 0, "sl_count": 0, "tp_count": 0, "tc_count": 0}

    arr = np.array(trades)
    total = arr.sum()
    wr = (arr > 0).sum() / len(arr)
    cum = np.cumsum(arr)
    dd = abs(np.min(cum - np.maximum.accumulate(cum))) if len(cum) > 1 else 0
    wins = arr[arr > 0].sum() if (arr > 0).any() else 0
    losses = abs(arr[arr < 0].sum()) if (arr < 0).any() else 1e-10
    pf = wins / losses
    return {"profit": total, "trades": len(arr), "wr": wr, "dd": dd, "pf": pf,
            "sl_count": sl_count, "tp_count": tp_count, "tc_count": tc_count}


def gen_regime_signals(ind, adx_thr, rsi_os, rsi_ob, vol_min, ema_f_key, ema_s_key, cross_lb):
    n = len(ind["close"])
    adx = ind["adx"]; di_p = ind["di_plus"]; di_m = ind["di_minus"]
    rsi = ind["rsi"]; vr = ind["vol_ratio"]
    cross_up = ind[f"cross_up_{ema_f_key}_{ema_s_key}"]
    cross_down = ind[f"cross_down_{ema_f_key}_{ema_s_key}"]
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
    return signals


def gen_cb_signals(ind, comp, vol_min, adx_max, rsi_max, rsi_min):
    n = len(ind["close"])
    adx = ind["adx"]; vr = ind["vol_ratio"]; rsi = ind["rsi"]
    atr = ind["atr"]; c = ind["close"]; h = ind["high"]; lo = ind["low"]

    h3_max = np.maximum(np.maximum(np.roll(h, 1), np.roll(h, 2)), h)
    l3_min = np.minimum(np.minimum(np.roll(lo, 1), np.roll(lo, 2)), lo)
    range_3bar = h3_max - l3_min
    prev_h2 = np.maximum(np.roll(h, 1), np.roll(h, 2))
    prev_l2 = np.minimum(np.roll(lo, 1), np.roll(lo, 2))

    adx_low = (adx < adx_max) | (np.roll(adx, 1) < adx_max) | (np.roll(adx, 2) < adx_max)
    compressed = (atr > 0) & (range_3bar / (atr + 1e-10) < comp)
    valid = (vr >= vol_min) & (rsi >= rsi_min) & (rsi <= rsi_max) & adx_low & compressed

    signals = np.zeros(n)
    signals[valid & (c > prev_h2)] = 1
    signals[valid & (c < prev_l2)] = -1
    signals[:20] = 0
    return signals


def main():
    print("=" * 90)
    print("  BEFORE vs AFTER — Real OKX Data Comparison")
    print("  5 pairs: ETH, SOL, SPX, DOGE, LINK | 180 days")
    print("=" * 90)

    # Load data
    data = {}
    for pair in PAIRS:
        df = load_15m(pair)
        c = df["close"].values.astype(float)
        h = df["high"].values.astype(float)
        lo = df["low"].values.astype(float)
        o = df["open"].values.astype(float)
        v = df["volume"].values.astype(float)

        ind = {"close": c, "high": h, "low": lo, "open": o, "n": len(df)}
        adx_r = ta.adx(df["high"], df["low"], df["close"], length=14)
        ind["adx"] = adx_r.iloc[:, 0].values
        ind["di_plus"] = adx_r.iloc[:, 1].values
        ind["di_minus"] = adx_r.iloc[:, 2].values
        ind["rsi"] = ta.rsi(df["close"], length=14).values
        ind["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14).values
        ve = ta.ema(df["volume"].astype(float), length=20).values
        ind["vol_ratio"] = v / (ve + 1e-10)

        for p in [18, 21, 50, 60]:
            ind[f"ema{p}"] = ta.ema(df["close"], length=p).values

        for ef_name, es_name in [("ema18", "ema50"), ("ema21", "ema60")]:
            ef = ind[ef_name]; es = ind[es_name]
            cup = np.zeros(len(df), dtype=bool)
            cdn = np.zeros(len(df), dtype=bool)
            cup[1:] = (ef[1:] > es[1:]) & (ef[:-1] <= es[:-1])
            cdn[1:] = (ef[1:] < es[1:]) & (ef[:-1] >= es[:-1])
            ind[f"cross_up_{ef_name}_{es_name}"] = cup
            ind[f"cross_down_{ef_name}_{es_name}"] = cdn

        body = c - o
        ind["body_abs"] = np.abs(body)
        ind["upper_shadow"] = h - np.maximum(c, o)
        ind["lower_shadow"] = np.minimum(c, o) - lo

        data[pair] = ind

    # Per-pair config for AFTER
    after_ra_config = {
        "ETH_USDT_USDT": {"entry_offset": 0.0, "time_cuts": [(8, -0.01), (12, -0.005), (24, 0.0)], "trailing": None},
        "SOL_USDT_USDT": {"entry_offset": 0.0, "time_cuts": [(12, -0.005), (24, 0.0), (48, 0.005)], "trailing": None},
        "SPX_USDT_USDT": {"entry_offset": 0.1, "time_cuts": [(12, -0.005), (24, 0.0)], "trailing": {"be_trigger": 0.4, "trail_trigger": 0.6, "trail_dist": 0.4}},
        "DOGE_USDT_USDT": {"entry_offset": 0.0, "time_cuts": [(8, -0.005), (12, 0.0), (24, 0.005)], "trailing": None},
        "LINK_USDT_USDT": {"entry_offset": 0.15, "time_cuts": [(12, -0.005), (24, 0.0)], "trailing": None},
    }

    after_cb_config = {
        "ETH_USDT_USDT": {"entry_offset": 0.0, "sl": 3.0},
        "SOL_USDT_USDT": {"entry_offset": 0.1, "sl": 3.5},
        "SPX_USDT_USDT": {"entry_offset": 0.0, "sl": 3.0},
        "DOGE_USDT_USDT": {"entry_offset": 0.1, "sl": 3.5},
        "LINK_USDT_USDT": {"entry_offset": 0.2, "sl": 4.0},
    }

    # ═══════════════════════════════════════════════════════════════════════
    # REGIME ADAPTIVE
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  REGIME ADAPTIVE")
    print(f"  BEFORE: adx=25, rsi=28/67, vol=1.0, ema21/60, lb=5, sl=10.0, tp=11.0")
    print(f"  AFTER:  adx=31, rsi=25/70, vol=1.5, ema18/50, lb=5, sl=7.0, tp=11.0 + per-pair fixes")
    print(f"{'='*90}")

    before_total = 0; after_total = 0
    print(f"\n  {'Pair':<18} {'BEFORE':>22} {'AFTER':>22} {'Delta':>10}")
    print(f"  {'-'*75}")

    for pair, ind in data.items():
        # BEFORE: old params
        sig_before = gen_regime_signals(ind, 25, 28, 67, 1.0, "ema21", "ema60", 5)
        r_before = backtest_advanced(sig_before, ind["high"], ind["low"], ind["close"], ind["atr"],
                                     sl_mult=10.0, tp_mult=11.0, max_bars=192)

        # AFTER: new params + per-pair improvements
        sig_after = gen_regime_signals(ind, 31, 25, 70, 1.5, "ema18", "ema50", 5)
        cfg = after_ra_config[pair]
        tp_mult = 13.0 if pair == "SPX_USDT_USDT" else 11.0
        r_after = backtest_advanced(sig_after, ind["high"], ind["low"], ind["close"], ind["atr"],
                                    sl_mult=7.0, tp_mult=tp_mult, max_bars=192,
                                    entry_offset_atr=cfg["entry_offset"],
                                    time_cuts=cfg["time_cuts"],
                                    trailing=cfg["trailing"])

        before_total += r_before["profit"]
        after_total += r_after["profit"]

        b_str = f"{r_before['profit']:+.1%} ({r_before['trades']}t)"
        a_str = f"{r_after['profit']:+.1%} ({r_after['trades']}t)"
        delta = r_after["profit"] - r_before["profit"]
        d_str = f"{delta:+.1%}"
        print(f"  {pair:<18} {b_str:>22} {a_str:>22} {d_str:>10}")

    print(f"  {'-'*75}")
    delta_total = after_total - before_total
    print(f"  {'TOTAL':<18} {before_total:+.1%}              {after_total:+.1%}              {delta_total:+.1%}")

    # Detailed AFTER breakdown
    print(f"\n  AFTER — Detailed breakdown:")
    for pair, ind in data.items():
        sig_after = gen_regime_signals(ind, 31, 25, 70, 1.5, "ema18", "ema50", 5)
        cfg = after_ra_config[pair]
        tp_mult = 13.0 if pair == "SPX_USDT_USDT" else 11.0
        r = backtest_advanced(sig_after, ind["high"], ind["low"], ind["close"], ind["atr"],
                              sl_mult=7.0, tp_mult=tp_mult, max_bars=192,
                              entry_offset_atr=cfg["entry_offset"],
                              time_cuts=cfg["time_cuts"], trailing=cfg["trailing"])
        if r["trades"] > 0:
            print(f"    {pair}: WR={r['wr']*100:.1f}%, PF={r['pf']:.2f}, DD={r['dd']:.1%} | "
                  f"TP:{r['tp_count']} SL:{r['sl_count']} TC:{r['tc_count']}")

    # ═══════════════════════════════════════════════════════════════════════
    # CB ADX BREAKOUT
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  CB ADX BREAKOUT")
    print(f"  BEFORE: comp=0.8, vol=0.8, adx_max=22, sl=5.0, tp=5.0")
    print(f"  AFTER:  comp=0.9, vol=0.6, adx_max=30, sl=3.0-4.0, tp=5.0 + limit entry")
    print(f"{'='*90}")

    before_total_cb = 0; after_total_cb = 0
    print(f"\n  {'Pair':<18} {'BEFORE':>22} {'AFTER':>22} {'Delta':>10}")
    print(f"  {'-'*75}")

    for pair, ind in data.items():
        # BEFORE
        sig_before = gen_cb_signals(ind, 0.8, 0.8, 22, 72, 22)
        r_before = backtest_advanced(sig_before, ind["high"], ind["low"], ind["close"], ind["atr"],
                                     sl_mult=5.0, tp_mult=5.0, max_bars=96)

        # AFTER
        sig_after = gen_cb_signals(ind, 0.9, 0.6, 30, 72, 22)
        cfg = after_cb_config[pair]
        r_after = backtest_advanced(sig_after, ind["high"], ind["low"], ind["close"], ind["atr"],
                                    sl_mult=cfg["sl"], tp_mult=5.0, max_bars=96,
                                    entry_offset_atr=cfg["entry_offset"])

        before_total_cb += r_before["profit"]
        after_total_cb += r_after["profit"]

        b_str = f"{r_before['profit']:+.1%} ({r_before['trades']}t)"
        a_str = f"{r_after['profit']:+.1%} ({r_after['trades']}t)"
        delta = r_after["profit"] - r_before["profit"]
        d_str = f"{delta:+.1%}"
        print(f"  {pair:<18} {b_str:>22} {a_str:>22} {d_str:>10}")

    print(f"  {'-'*75}")
    delta_cb = after_total_cb - before_total_cb
    print(f"  {'TOTAL':<18} {before_total_cb:+.1%}              {after_total_cb:+.1%}              {delta_cb:+.1%}")

    # Detailed AFTER
    print(f"\n  AFTER — Detailed breakdown:")
    for pair, ind in data.items():
        sig_after = gen_cb_signals(ind, 0.9, 0.6, 30, 72, 22)
        cfg = after_cb_config[pair]
        r = backtest_advanced(sig_after, ind["high"], ind["low"], ind["close"], ind["atr"],
                              sl_mult=cfg["sl"], tp_mult=5.0, max_bars=96,
                              entry_offset_atr=cfg["entry_offset"])
        if r["trades"] > 0:
            print(f"    {pair}: WR={r['wr']*100:.1f}%, PF={r['pf']:.2f}, DD={r['dd']:.1%} | "
                  f"TP:{r['tp_count']} SL:{r['sl_count']} TC:{r['tc_count']}")

    # ═══════════════════════════════════════════════════════════════════════
    # COMBINED SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  COMBINED SUMMARY")
    print(f"{'='*90}")
    combined_before = before_total + before_total_cb
    combined_after = after_total + after_total_cb
    print(f"\n  BEFORE (old params):     {combined_before:+.2%}")
    print(f"  AFTER  (optimized):      {combined_after:+.2%}")
    print(f"  IMPROVEMENT:             {combined_after - combined_before:+.2%}")
    print(f"\n  Key changes applied:")
    print(f"    - regime_adaptive: vol 1.0->1.5, adx 25->31, rsi 28/67->25/70, sl 10->7 ATR")
    print(f"    - SPX: limit entry 0.1 ATR + trailing SL (activate 60% TP) + tp 11->13 ATR")
    print(f"    - ETH/DOGE: aggressive time cut 8h")
    print(f"    - LINK: limit entry 0.15 ATR")
    print(f"    - cb_adx: comp 0.8->0.9, vol 0.8->0.6, adx 22->30, sl 5->3-4 ATR")
    print(f"    - CB LINK: limit entry 0.2 ATR + widen SL to 4.0")
    print(f"    - CB SOL/DOGE: limit entry 0.1 ATR + SL 3.5")


if __name__ == "__main__":
    main()

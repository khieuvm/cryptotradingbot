"""Fast 15m hyperopt — vectorized signal + backtest.

Strategy: Test a focused grid around current best params.
Pre-compute everything with numpy, no per-bar Python loops for signal gen.
"""

import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, str(Path(__file__).parent.parent))

TAKER_FEE = 0.0005
DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT", "NVDA_USDT_USDT"]
BARS_PER_DAY = 96


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp).sort_values("date").reset_index(drop=True)
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


def backtest_signals(signals, high, low, close, atr, sl_mult, tp_mult, max_bars=96):
    """Fast backtest with SL/TP."""
    fee = 2 * TAKER_FEE
    n = len(close)
    trades = []
    i = 0
    while i < n:
        if signals[i] == 0:
            i += 1
            continue
        ep = close[i]
        ea = atr[i]
        if ea <= 0 or ep <= 0 or np.isnan(ea):
            i += 1
            continue
        is_long = signals[i] > 0
        sl_d = sl_mult * ea
        tp_d = tp_mult * ea
        exited = False
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if is_long:
                if low[j] <= ep - sl_d:
                    trades.append(-sl_mult * ea / ep - fee)
                    i = j + 1
                    exited = True
                    break
                if high[j] >= ep + tp_d:
                    trades.append(tp_mult * ea / ep - fee)
                    i = j + 1
                    exited = True
                    break
            else:
                if high[j] >= ep + sl_d:
                    trades.append(-sl_mult * ea / ep - fee)
                    i = j + 1
                    exited = True
                    break
                if low[j] <= ep - tp_d:
                    trades.append(tp_mult * ea / ep - fee)
                    i = j + 1
                    exited = True
                    break
        if not exited:
            ei = min(i + max_bars, n - 1)
            pnl = ((close[ei] - ep) / ep if is_long else (ep - close[ei]) / ep) - fee
            trades.append(pnl)
            i = ei + 1
    if not trades:
        return 0, 0, 0, 0
    arr = np.array(trades)
    wr = (arr > 0).sum() / len(arr)
    cum = np.cumsum(arr)
    dd = abs(np.min(cum - np.maximum.accumulate(cum)))
    return arr.sum(), len(arr), wr, dd


def main():
    t0 = time.time()
    print("=" * 90, flush=True)
    print("15m FAST HYPEROPT", flush=True)
    print("=" * 90, flush=True)

    # Load data
    data = {}
    for pair in PAIRS:
        df = load_15m(pair)
        if df.empty:
            continue
        c = df["close"]
        h = df["high"]
        lo = df["low"]
        v = df["volume"].astype(float)

        ind = {}
        ind["close"] = c.values
        ind["high"] = h.values
        ind["low"] = lo.values
        ind["open"] = df["open"].values

        adx_r = ta.adx(h, lo, c, length=14)
        ind["adx"] = adx_r.iloc[:, 0].values if adx_r is not None else np.full(len(df), 25.0)
        ind["di_plus"] = adx_r.iloc[:, 1].values if adx_r is not None else np.full(len(df), 12.5)
        ind["di_minus"] = adx_r.iloc[:, 2].values if adx_r is not None else np.full(len(df), 12.5)
        ind["rsi"] = ta.rsi(c, length=14).values
        ind["atr"] = ta.atr(h, lo, c, length=14).values

        ve = ta.ema(v, length=20)
        ind["vol_ratio"] = (v / (ve + 1e-10)).values

        for p in [14, 18, 21, 50, 60]:
            ind[f"ema{p}"] = ta.ema(c, length=p).values

        body = c.values - df["open"].values
        ind["body_abs"] = np.abs(body)
        ind["upper_shadow"] = h.values - np.maximum(c.values, df["open"].values)
        ind["lower_shadow"] = np.minimum(c.values, df["open"].values) - lo.values

        data[pair] = ind
        print(f"  {pair}: {len(c)} bars", flush=True)

    n_bars = len(next(iter(data.values()))["close"])
    print(f"\n  Total: {len(data)} pairs, {n_bars} bars each", flush=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # REGIME ADAPTIVE — focused grid
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}", flush=True)
    print("  REGIME ADAPTIVE", flush=True)
    print(f"{'='*90}", flush=True)

    best_ra = {"profit": -999}
    count = 0

    # Reduced but meaningful grid
    for adx_thr in [25, 28, 31, 35]:
        for rsi_os in [28, 30, 33]:
            for rsi_ob in [67, 70, 73]:
                for vol_min in [0.8, 1.0]:
                    for ema_f, ema_s in [("ema14", "ema50"), ("ema18", "ema50"), ("ema21", "ema60")]:
                        for cross_lb in [5, 8, 12]:
                            for sl, tp in [(5.0, 8.0), (5.9, 9.3), (6.0, 10.0), (7.0, 11.0), (5.0, 9.0)]:
                                total_profit = 0
                                total_trades = 0

                                for pair, ind in data.items():
                                    n = len(ind["close"])
                                    adx = ind["adx"]
                                    di_p = ind["di_plus"]
                                    di_m = ind["di_minus"]
                                    rsi = ind["rsi"]
                                    ef = ind[ema_f]
                                    es = ind[ema_s]
                                    vr = ind["vol_ratio"]

                                    signals = np.zeros(n)
                                    for i in range(60, n):
                                        if vr[i] < vol_min:
                                            continue
                                        if adx[i] >= adx_thr:
                                            for k in range(1, cross_lb + 1):
                                                pi = i - k
                                                if pi < 1:
                                                    break
                                                if ef[pi-1] <= es[pi-1] and ef[pi] > es[pi]:
                                                    if di_p[i] > di_m[i]:
                                                        signals[i] = 1
                                                    break
                                                if ef[pi-1] >= es[pi-1] and ef[pi] < es[pi]:
                                                    if di_m[i] > di_p[i]:
                                                        signals[i] = -1
                                                    break
                                        else:
                                            if rsi[i] < rsi_os:
                                                signals[i] = 1
                                            elif rsi[i] > rsi_ob:
                                                signals[i] = -1

                                    p, t, w, d = backtest_signals(
                                        signals, ind["high"], ind["low"],
                                        ind["close"], ind["atr"], sl, tp, 192)
                                    total_profit += p
                                    total_trades += t

                                count += 1
                                if total_trades >= 30 and total_profit > best_ra["profit"]:
                                    best_ra = {"profit": total_profit, "trades": total_trades,
                                               "adx_thr": adx_thr, "rsi_os": rsi_os, "rsi_ob": rsi_ob,
                                               "vol_min": vol_min, "ema_f": ema_f, "ema_s": ema_s,
                                               "cross_lb": cross_lb, "sl": sl, "tp": tp}

    print(f"  Tested {count} combos in {time.time()-t0:.0f}s", flush=True)
    if best_ra.get("trades"):
        print(f"  BEST: profit={best_ra['profit']:+.2%} | trades={best_ra['trades']}", flush=True)
        print(f"    adx_thr={best_ra['adx_thr']}, rsi_os={best_ra['rsi_os']}, rsi_ob={best_ra['rsi_ob']}", flush=True)
        print(f"    vol_min={best_ra['vol_min']}, ema={best_ra['ema_f']}/{best_ra['ema_s']}, cross_lb={best_ra['cross_lb']}", flush=True)
        print(f"    sl={best_ra['sl']}, tp={best_ra['tp']}", flush=True)

        print(f"\n  Per-pair:", flush=True)
        for pair, ind in data.items():
            n = len(ind["close"])
            signals = np.zeros(n)
            adx = ind["adx"]; di_p = ind["di_plus"]; di_m = ind["di_minus"]
            rsi = ind["rsi"]; ef = ind[best_ra["ema_f"]]; es = ind[best_ra["ema_s"]]
            vr = ind["vol_ratio"]
            for i in range(60, n):
                if vr[i] < best_ra["vol_min"]:
                    continue
                if adx[i] >= best_ra["adx_thr"]:
                    for k in range(1, best_ra["cross_lb"] + 1):
                        pi = i - k
                        if pi < 1: break
                        if ef[pi-1] <= es[pi-1] and ef[pi] > es[pi]:
                            if di_p[i] > di_m[i]: signals[i] = 1
                            break
                        if ef[pi-1] >= es[pi-1] and ef[pi] < es[pi]:
                            if di_m[i] > di_p[i]: signals[i] = -1
                            break
                else:
                    if rsi[i] < best_ra["rsi_os"]: signals[i] = 1
                    elif rsi[i] > best_ra["rsi_ob"]: signals[i] = -1

            p, t, w, d = backtest_signals(signals, ind["high"], ind["low"],
                                          ind["close"], ind["atr"], best_ra["sl"], best_ra["tp"], 192)
            if t > 0:
                print(f"    {pair}: {p:+.2%} ({t} trades, WR {w*100:.1f}%, DD {d:.2%})", flush=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # VOLUME SPIKE REVERSAL
    # ═══════════════════════════════════════════════════════════════════════════
    t1 = time.time()
    print(f"\n{'='*90}", flush=True)
    print("  VOLUME SPIKE REVERSAL", flush=True)
    print(f"{'='*90}", flush=True)

    best_vs = {"profit": -999}
    count = 0

    for spike in [2.0, 2.5, 3.0, 3.5]:
        for shadow in [1.5, 2.0, 2.5, 3.0]:
            for rsi_os in [28, 30, 33]:
                for rsi_ob in [67, 68, 70, 72]:
                    for vol_min in [1.0, 1.5, 2.0]:
                        for sl, tp in [(2.0, 3.5), (2.5, 4.0), (3.0, 5.0), (3.5, 5.5)]:
                            total_profit = 0
                            total_trades = 0

                            for pair, ind in data.items():
                                n = len(ind["close"])
                                vr = ind["vol_ratio"]
                                rsi = ind["rsi"]
                                ba = ind["body_abs"]
                                ls = ind["lower_shadow"]
                                us = ind["upper_shadow"]

                                signals = np.zeros(n)
                                for i in range(20, n):
                                    if vr[i] < spike or vr[i] < vol_min or ba[i] <= 0:
                                        continue
                                    if ls[i] > shadow * ba[i] and rsi[i] < rsi_os:
                                        signals[i] = 1
                                    elif us[i] > shadow * ba[i] and rsi[i] > rsi_ob:
                                        signals[i] = -1

                                p, t, w, d = backtest_signals(
                                    signals, ind["high"], ind["low"],
                                    ind["close"], ind["atr"], sl, tp, 96)
                                total_profit += p
                                total_trades += t

                            count += 1
                            if total_trades >= 20 and total_profit > best_vs["profit"]:
                                best_vs = {"profit": total_profit, "trades": total_trades,
                                           "spike": spike, "shadow": shadow,
                                           "rsi_os": rsi_os, "rsi_ob": rsi_ob,
                                           "vol_min": vol_min, "sl": sl, "tp": tp}

    print(f"  Tested {count} combos in {time.time()-t1:.0f}s", flush=True)
    if best_vs.get("trades"):
        print(f"  BEST: profit={best_vs['profit']:+.2%} | trades={best_vs['trades']}", flush=True)
        print(f"    spike={best_vs['spike']}, shadow={best_vs['shadow']}, rsi={best_vs['rsi_os']}/{best_vs['rsi_ob']}", flush=True)
        print(f"    vol_min={best_vs['vol_min']}, sl={best_vs['sl']}, tp={best_vs['tp']}", flush=True)

        print(f"\n  Per-pair:", flush=True)
        for pair, ind in data.items():
            n = len(ind["close"]); vr = ind["vol_ratio"]; rsi = ind["rsi"]
            ba = ind["body_abs"]; ls = ind["lower_shadow"]; us = ind["upper_shadow"]
            signals = np.zeros(n)
            for i in range(20, n):
                if vr[i] < best_vs["spike"] or vr[i] < best_vs["vol_min"] or ba[i] <= 0:
                    continue
                if ls[i] > best_vs["shadow"] * ba[i] and rsi[i] < best_vs["rsi_os"]:
                    signals[i] = 1
                elif us[i] > best_vs["shadow"] * ba[i] and rsi[i] > best_vs["rsi_ob"]:
                    signals[i] = -1
            p, t, w, d = backtest_signals(signals, ind["high"], ind["low"],
                                          ind["close"], ind["atr"], best_vs["sl"], best_vs["tp"], 96)
            if t > 0:
                print(f"    {pair}: {p:+.2%} ({t} trades, WR {w*100:.1f}%, DD {d:.2%})", flush=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # CB ADX BREAKOUT
    # ═══════════════════════════════════════════════════════════════════════════
    t2 = time.time()
    print(f"\n{'='*90}", flush=True)
    print("  CB ADX BREAKOUT", flush=True)
    print(f"{'='*90}", flush=True)

    best_cb = {"profit": -999}
    count = 0

    for comp in [0.5, 0.6, 0.7, 0.8]:
        for vol_min in [0.6, 0.8, 1.0]:
            for adx_max in [18, 20, 22, 25]:
                for adx_lb in [2, 3, 4]:
                    for rsi_max in [72, 75, 78]:
                        for rsi_min in [22, 25, 28]:
                            for sl, tp in [(2.5, 4.0), (3.0, 5.0), (3.5, 5.5), (4.0, 6.0)]:
                                total_profit = 0
                                total_trades = 0

                                for pair, ind in data.items():
                                    n = len(ind["close"])
                                    adx = ind["adx"]; vr = ind["vol_ratio"]
                                    rsi = ind["rsi"]; atr = ind["atr"]
                                    h = ind["high"]; lo = ind["low"]; c = ind["close"]

                                    signals = np.zeros(n)
                                    for i in range(max(20, adx_lb + 3), n):
                                        if vr[i] < vol_min: continue
                                        if rsi[i] > rsi_max or rsi[i] < rsi_min: continue
                                        if adx[i] > adx_max: continue
                                        if adx[i] >= adx[i - adx_lb]: continue
                                        if atr[i] <= 0 or np.isnan(atr[i]): continue

                                        r3 = np.mean([h[i-k] - lo[i-k] for k in range(3)])
                                        if r3 / atr[i] > comp: continue

                                        h3 = max(h[i-2], h[i-1])
                                        l3 = min(lo[i-2], lo[i-1])
                                        if c[i] > h3: signals[i] = 1
                                        elif c[i] < l3: signals[i] = -1

                                    p, t, w, d = backtest_signals(
                                        signals, ind["high"], ind["low"],
                                        ind["close"], ind["atr"], sl, tp, 96)
                                    total_profit += p
                                    total_trades += t

                                count += 1
                                if total_trades >= 20 and total_profit > best_cb["profit"]:
                                    best_cb = {"profit": total_profit, "trades": total_trades,
                                               "comp": comp, "vol_min": vol_min,
                                               "adx_max": adx_max, "adx_lb": adx_lb,
                                               "rsi_max": rsi_max, "rsi_min": rsi_min,
                                               "sl": sl, "tp": tp}

    print(f"  Tested {count} combos in {time.time()-t2:.0f}s", flush=True)
    if best_cb.get("trades"):
        print(f"  BEST: profit={best_cb['profit']:+.2%} | trades={best_cb['trades']}", flush=True)
        print(f"    comp={best_cb['comp']}, vol_min={best_cb['vol_min']}, adx_max={best_cb['adx_max']}, adx_lb={best_cb['adx_lb']}", flush=True)
        print(f"    rsi_max={best_cb['rsi_max']}, rsi_min={best_cb['rsi_min']}, sl={best_cb['sl']}, tp={best_cb['tp']}", flush=True)

        print(f"\n  Per-pair:", flush=True)
        for pair, ind in data.items():
            n = len(ind["close"]); adx = ind["adx"]; vr = ind["vol_ratio"]
            rsi = ind["rsi"]; atr = ind["atr"]; h = ind["high"]; lo = ind["low"]; c = ind["close"]
            signals = np.zeros(n)
            for i in range(max(20, best_cb["adx_lb"] + 3), n):
                if vr[i] < best_cb["vol_min"]: continue
                if rsi[i] > best_cb["rsi_max"] or rsi[i] < best_cb["rsi_min"]: continue
                if adx[i] > best_cb["adx_max"]: continue
                if adx[i] >= adx[i - best_cb["adx_lb"]]: continue
                if atr[i] <= 0 or np.isnan(atr[i]): continue
                r3 = np.mean([h[i-k] - lo[i-k] for k in range(3)])
                if r3 / atr[i] > best_cb["comp"]: continue
                h3 = max(h[i-2], h[i-1]); l3 = min(lo[i-2], lo[i-1])
                if c[i] > h3: signals[i] = 1
                elif c[i] < l3: signals[i] = -1
            p, t, w, d = backtest_signals(signals, ind["high"], ind["low"],
                                          ind["close"], ind["atr"], best_cb["sl"], best_cb["tp"], 96)
            if t > 0:
                print(f"    {pair}: {p:+.2%} ({t} trades, WR {w*100:.1f}%, DD {d:.2%})", flush=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPARE CURRENT vs OPTIMIZED
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}", flush=True)
    print("  CURRENT vs OPTIMIZED", flush=True)
    print(f"{'='*90}", flush=True)

    # Current regime_adaptive: adx=31, rsi_os=30, rsi_ob=73, vol=1.0, ema18/50, lb=8, sl=5.9, tp=9.3
    print("\n  regime_adaptive (current config):", flush=True)
    total_curr = 0
    for pair, ind in data.items():
        n = len(ind["close"]); adx = ind["adx"]; di_p = ind["di_plus"]; di_m = ind["di_minus"]
        rsi = ind["rsi"]; ef = ind["ema18"]; es = ind["ema50"]; vr = ind["vol_ratio"]
        signals = np.zeros(n)
        for i in range(60, n):
            if vr[i] < 1.0: continue
            if adx[i] >= 31:
                for k in range(1, 9):
                    pi = i - k
                    if pi < 1: break
                    if ef[pi-1] <= es[pi-1] and ef[pi] > es[pi]:
                        if di_p[i] > di_m[i]: signals[i] = 1
                        break
                    if ef[pi-1] >= es[pi-1] and ef[pi] < es[pi]:
                        if di_m[i] > di_p[i]: signals[i] = -1
                        break
            else:
                if rsi[i] < 30: signals[i] = 1
                elif rsi[i] > 73: signals[i] = -1
        p, t, w, d = backtest_signals(signals, ind["high"], ind["low"],
                                      ind["close"], ind["atr"], 5.9, 9.3, 192)
        total_curr += p
        if t > 0:
            print(f"    {pair}: {p:+.2%} ({t} trades, WR {w*100:.1f}%)", flush=True)
    print(f"    TOTAL: {total_curr:+.2%}", flush=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}", flush=True)
    print("  FINAL SUMMARY", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"  Total time: {time.time()-t0:.0f}s", flush=True)
    print(f"\n  {'Strategy':<22} {'Profit':<12} {'Trades':<8} {'Params'}", flush=True)
    print(f"  {'-'*75}", flush=True)
    print(f"  {'regime_adaptive':<22} {best_ra.get('profit',0):+.2%}   {best_ra.get('trades',0):<8} "
          f"adx={best_ra.get('adx_thr','?')} rsi={best_ra.get('rsi_os','?')}/{best_ra.get('rsi_ob','?')} "
          f"sl/tp={best_ra.get('sl','?')}/{best_ra.get('tp','?')}", flush=True)
    print(f"  {'volume_spike_rev':<22} {best_vs.get('profit',0):+.2%}   {best_vs.get('trades',0):<8} "
          f"spike={best_vs.get('spike','?')} shadow={best_vs.get('shadow','?')} "
          f"sl/tp={best_vs.get('sl','?')}/{best_vs.get('tp','?')}", flush=True)
    print(f"  {'cb_adx_breakout':<22} {best_cb.get('profit',0):+.2%}   {best_cb.get('trades',0):<8} "
          f"comp={best_cb.get('comp','?')} adx_max={best_cb.get('adx_max','?')} "
          f"sl/tp={best_cb.get('sl','?')}/{best_cb.get('tp','?')}", flush=True)
    print(f"\n  Current regime_adaptive total: {total_curr:+.2%}", flush=True)


if __name__ == "__main__":
    main()

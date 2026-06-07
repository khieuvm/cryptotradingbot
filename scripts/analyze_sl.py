"""Stoploss behavior analysis: What happens after SL hit?

For each strategy x pair:
- How fast does SL get hit?
- After SL, does price come back (sweep) or continue (trend)?
- Long vs Short SL patterns
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
PAIRS = ["ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT"]
TAKER_FEE = 0.0005


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    df = pd.read_feather(fp).sort_values("date").reset_index(drop=True)
    return df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)


def analyze_sl(signals, high, low, close, atr, sl_mult, tp_mult, max_bars):
    n = len(close)
    sl_trades = []
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
        exited = False
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if is_long:
                if low[j] <= ep - sl_mult * ea:
                    sl_price = ep - sl_mult * ea
                    post = {}
                    for look in [3, 6, 12, 24, 48]:
                        if j + look < n:
                            post[f"ret_{look}"] = (close[j + look] - sl_price) / sl_price
                    # Did it go FURTHER down after SL?
                    min_after = min(low[j + 1:j + 25]) if j + 1 < n and j + 25 <= n else low[j]
                    max_after = max(high[j + 1:j + 25]) if j + 1 < n and j + 25 <= n else high[j]
                    post["max_adverse"] = (sl_price - min_after) / ep  # how much further it dropped
                    post["max_favorable"] = (max_after - sl_price) / ep  # how much it recovered
                    sl_trades.append({
                        "dir": "LONG", "bars_to_sl": j - i, **post
                    })
                    i = j + 1
                    exited = True
                    break
                if high[j] >= ep + tp_mult * ea:
                    i = j + 1
                    exited = True
                    break
            else:
                if high[j] >= ep + sl_mult * ea:
                    sl_price = ep + sl_mult * ea
                    post = {}
                    for look in [3, 6, 12, 24, 48]:
                        if j + look < n:
                            post[f"ret_{look}"] = (sl_price - close[j + look]) / sl_price
                    min_after = min(low[j + 1:j + 25]) if j + 1 < n and j + 25 <= n else low[j]
                    max_after = max(high[j + 1:j + 25]) if j + 1 < n and j + 25 <= n else high[j]
                    post["max_adverse"] = (max_after - sl_price) / ep  # how much further it went up
                    post["max_favorable"] = (sl_price - min_after) / ep  # how much it came back
                    sl_trades.append({
                        "dir": "SHORT", "bars_to_sl": j - i, **post
                    })
                    i = j + 1
                    exited = True
                    break
                if low[j] <= ep - tp_mult * ea:
                    i = j + 1
                    exited = True
                    break
        if not exited:
            i = min(i + max_bars, n - 1) + 1
    return sl_trades


def print_sl_analysis(name, sl_trades):
    if not sl_trades:
        print(f"  {name}: 0 SL hits")
        return

    print(f"\n  {name}")
    print(f"  {'─' * 70}")
    print(f"  Total SL hits: {len(sl_trades)}")

    longs = [t for t in sl_trades if t["dir"] == "LONG"]
    shorts = [t for t in sl_trades if t["dir"] == "SHORT"]
    print(f"  Long SL: {len(longs)} | Short SL: {len(shorts)}")

    # Speed to SL
    bars = [t["bars_to_sl"] for t in sl_trades]
    fast = sum(1 for b in bars if b <= 2)
    mid = sum(1 for b in bars if 2 < b <= 8)
    slow = sum(1 for b in bars if 8 < b <= 24)
    vslow = sum(1 for b in bars if b > 24)
    print(f"  Speed: Instant(1-2 bars)={fast} | Fast(3-8)={mid} | Mid(9-24)={slow} | Slow(>24)={vslow}")
    print(f"  Avg bars to SL: {np.mean(bars):.1f} | Median: {np.median(bars):.0f}")

    # Post-SL behavior
    print(f"\n  {'After SL':<12} {'CameBack':<18} {'Continued':<18} {'AvgRet':<10} {'Verdict'}")
    print(f"  {'─' * 70}")
    for look, label in [(3, "45min"), (6, "1.5h"), (12, "3h"), (24, "6h"), (48, "12h")]:
        key = f"ret_{look}"
        rets = np.array([t[key] for t in sl_trades if key in t])
        if len(rets) == 0:
            continue
        came = (rets > 0).sum()
        cont = (rets <= 0).sum()
        avg = rets.mean()
        verdict = "SWEEP (SL too tight)" if came / len(rets) > 0.55 else "TREND (SL correct)" if cont / len(rets) > 0.55 else "MIXED"
        print(f"  {label:<12} {came}/{len(rets)} ({came/len(rets)*100:.0f}%){'':>5} "
              f"{cont}/{len(rets)} ({cont/len(rets)*100:.0f}%){'':>5} "
              f"{avg*100:+.3f}%   {verdict}")

    # Max adverse/favorable after SL
    adv = [t.get("max_adverse", 0) for t in sl_trades]
    fav = [t.get("max_favorable", 0) for t in sl_trades]
    print(f"\n  After SL (next 6h window):")
    print(f"    Max further against: avg {np.mean(adv)*100:.2f}%, max {np.max(adv)*100:.2f}%")
    print(f"    Max recovery:        avg {np.mean(fav)*100:.2f}%, max {np.max(fav)*100:.2f}%")

    would_survive = sum(1 for f in fav if f > 0.01)
    print(f"    Would have recovered >1%: {would_survive}/{len(fav)} ({would_survive/len(fav)*100:.0f}%)")

    # Long vs Short breakdown
    for direction, trades in [("LONG", longs), ("SHORT", shorts)]:
        if not trades:
            continue
        bars_d = [t["bars_to_sl"] for t in trades]
        rets_12 = [t.get("ret_12", 0) for t in trades if "ret_12" in t]
        came = sum(1 for r in rets_12 if r > 0)
        adv_d = [t.get("max_adverse", 0) for t in trades]
        fav_d = [t.get("max_favorable", 0) for t in trades]
        print(f"\n  {direction} SL Detail:")
        print(f"    Count: {len(trades)} | Avg bars: {np.mean(bars_d):.0f} | Fast(<=2): {sum(1 for b in bars_d if b<=2)}")
        print(f"    ComeBack@3h: {came}/{len(rets_12)} ({came/max(len(rets_12),1)*100:.0f}%)")
        print(f"    AvgAdverse: {np.mean(adv_d)*100:.2f}% | AvgRecovery: {np.mean(fav_d)*100:.2f}%")


def main():
    data = {}
    for pair in PAIRS:
        df = load_15m(pair)
        c = df["close"]; h = df["high"]; lo = df["low"]; v = df["volume"].astype(float)
        ind = {}
        ind["c"] = c.values; ind["h"] = h.values; ind["l"] = lo.values
        adx_r = ta.adx(h, lo, c, length=14)
        ind["adx"] = adx_r.iloc[:, 0].values
        ind["dip"] = adx_r.iloc[:, 1].values
        ind["dim"] = adx_r.iloc[:, 2].values
        ind["rsi"] = ta.rsi(c, length=14).values
        ind["atr"] = ta.atr(h, lo, c, length=14).values
        ind["vr"] = (v / (ta.ema(v, length=20) + 1e-10)).values
        ind["e21"] = ta.ema(c, length=21).values
        ind["e60"] = ta.ema(c, length=60).values
        ind["ba"] = np.abs(c.values - df["open"].values)
        ind["us"] = h.values - np.maximum(c.values, df["open"].values)
        ind["ls"] = np.minimum(c.values, df["open"].values) - lo.values
        data[pair] = ind

    print("=" * 90)
    print("STOPLOSS BEHAVIOR ANALYSIS")
    print("Question: After SL hit, does price COME BACK (= SL too tight/sweep)")
    print("          or CONTINUE against us (= SL correct, bad entry)?")
    print("=" * 90)

    for pair in PAIRS:
        ind = data[pair]
        n = len(ind["c"])
        print(f"\n{'=' * 90}")
        print(f"  {pair}")
        print(f"{'=' * 90}")

        # 1. Regime Adaptive
        sig1 = np.zeros(n)
        for i in range(60, n):
            if ind["vr"][i] < 1.0:
                continue
            if ind["adx"][i] >= 25:
                for k in range(1, 6):
                    pi = i - k
                    if pi < 1:
                        break
                    if ind["e21"][pi - 1] <= ind["e60"][pi - 1] and ind["e21"][pi] > ind["e60"][pi]:
                        if ind["dip"][i] > ind["dim"][i]:
                            sig1[i] = 1
                        break
                    if ind["e21"][pi - 1] >= ind["e60"][pi - 1] and ind["e21"][pi] < ind["e60"][pi]:
                        if ind["dim"][i] > ind["dip"][i]:
                            sig1[i] = -1
                        break
            else:
                if ind["rsi"][i] < 28:
                    sig1[i] = 1
                elif ind["rsi"][i] > 67:
                    sig1[i] = -1

        sl1 = analyze_sl(sig1, ind["h"], ind["l"], ind["c"], ind["atr"], 7.0, 11.0, 192)
        print_sl_analysis("REGIME ADAPTIVE (SL=7.0x ATR, TP=11.0x)", sl1)

        # 2. Volume Spike
        sig2 = np.zeros(n)
        for i in range(20, n):
            if ind["vr"][i] < 2.0 or ind["ba"][i] <= 0:
                continue
            if ind["ls"][i] > 3.0 * ind["ba"][i] and ind["rsi"][i] < 33:
                sig2[i] = 1
            elif ind["us"][i] > 3.0 * ind["ba"][i] and ind["rsi"][i] > 72:
                sig2[i] = -1

        sl2 = analyze_sl(sig2, ind["h"], ind["l"], ind["c"], ind["atr"], 3.5, 5.5, 96)
        print_sl_analysis("VOLUME SPIKE REV (SL=3.5x ATR, TP=5.5x)", sl2)

        # 3. CB ADX
        sig3 = np.zeros(n)
        for i in range(22, n):
            if ind["vr"][i] < 0.8:
                continue
            if ind["rsi"][i] > 72 or ind["rsi"][i] < 22:
                continue
            if ind["adx"][i] > 22 or ind["adx"][i] >= ind["adx"][i - 2]:
                continue
            if ind["atr"][i] <= 0 or np.isnan(ind["atr"][i]):
                continue
            r3m = np.mean([ind["h"][i - k] - ind["l"][i - k] for k in range(3)])
            if r3m / ind["atr"][i] > 0.8:
                continue
            h3 = max(ind["h"][i - 2], ind["h"][i - 1])
            l3 = min(ind["l"][i - 2], ind["l"][i - 1])
            if ind["c"][i] > h3:
                sig3[i] = 1
            elif ind["c"][i] < l3:
                sig3[i] = -1

        sl3 = analyze_sl(sig3, ind["h"], ind["l"], ind["c"], ind["atr"], 3.0, 5.0, 96)
        print_sl_analysis("CB ADX BREAKOUT (SL=3.0x ATR, TP=5.0x)", sl3)

    # ══════════════════════════════════════════════════════════════════════════
    # OPTIMAL SL TEST: try different SL levels
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 90}")
    print("  SL SENSITIVITY TEST — What if we change SL?")
    print(f"{'=' * 90}")

    def quick_backtest(signals, high, low, close, atr, sl_mult, tp_mult, max_bars):
        n = len(close); fee = 2 * TAKER_FEE
        trades = []; i = 0
        while i < n:
            if signals[i] == 0: i += 1; continue
            ep = close[i]; ea = atr[i]
            if ea <= 0 or ep <= 0 or np.isnan(ea): i += 1; continue
            is_long = signals[i] > 0; exited = False
            for j in range(i + 1, min(i + max_bars + 1, n)):
                if is_long:
                    if low[j] <= ep - sl_mult * ea: trades.append(-sl_mult * ea / ep - fee); i = j + 1; exited = True; break
                    if high[j] >= ep + tp_mult * ea: trades.append(tp_mult * ea / ep - fee); i = j + 1; exited = True; break
                else:
                    if high[j] >= ep + sl_mult * ea: trades.append(-sl_mult * ea / ep - fee); i = j + 1; exited = True; break
                    if low[j] <= ep - tp_mult * ea: trades.append(tp_mult * ea / ep - fee); i = j + 1; exited = True; break
            if not exited:
                ei = min(i + max_bars, n - 1)
                pnl = ((close[ei] - ep) / ep if is_long else (ep - close[ei]) / ep) - fee
                trades.append(pnl); i = ei + 1
        if not trades: return 0, 0, 0
        arr = np.array(trades)
        return arr.sum(), len(arr), (arr > 0).sum() / len(arr)

    for pair in PAIRS:
        ind = data[pair]; n = len(ind["c"])
        print(f"\n  {pair}:")

        # Regime adaptive signals
        sig1 = np.zeros(n)
        for i in range(60, n):
            if ind["vr"][i] < 1.0: continue
            if ind["adx"][i] >= 25:
                for k in range(1, 6):
                    pi = i - k
                    if pi < 1: break
                    if ind["e21"][pi-1] <= ind["e60"][pi-1] and ind["e21"][pi] > ind["e60"][pi]:
                        if ind["dip"][i] > ind["dim"][i]: sig1[i] = 1
                        break
                    if ind["e21"][pi-1] >= ind["e60"][pi-1] and ind["e21"][pi] < ind["e60"][pi]:
                        if ind["dim"][i] > ind["dip"][i]: sig1[i] = -1
                        break
            else:
                if ind["rsi"][i] < 28: sig1[i] = 1
                elif ind["rsi"][i] > 67: sig1[i] = -1

        print(f"    REGIME ADAPTIVE — SL sweep (TP fixed at ratio):")
        print(f"    {'SL':<6} {'TP':<6} {'Profit':<10} {'Trades':<8} {'WR':<8} {'R:R'}")
        print(f"    {'-' * 55}")
        for sl in [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            tp = sl * 1.57  # Keep same R:R ratio
            p, t, w = quick_backtest(sig1, ind["h"], ind["l"], ind["c"], ind["atr"], sl, tp, 192)
            marker = " <<<" if p > 0.20 else ""
            print(f"    {sl:<6.1f} {tp:<6.1f} {p:+.2%}{'':>2} {t:<8} {w*100:.1f}%   1:{tp/sl:.2f}{marker}")

        # Also test fixed TP with varying SL
        print(f"\n    REGIME ADAPTIVE — Fixed TP=11, vary SL:")
        print(f"    {'SL':<6} {'Profit':<10} {'Trades':<8} {'WR':<8}")
        print(f"    {'-' * 40}")
        for sl in [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]:
            p, t, w = quick_backtest(sig1, ind["h"], ind["l"], ind["c"], ind["atr"], sl, 11.0, 192)
            marker = " <<<" if p > 0.20 else ""
            print(f"    {sl:<6.1f} {p:+.2%}{'':>2} {t:<8} {w*100:.1f}%{marker}")


if __name__ == "__main__":
    main()

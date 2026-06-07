"""Parameter sweep for XAU and NVDA with regime_adaptive signals."""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
TAKER = 0.0005


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    df = pd.read_feather(fp)
    return df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)


def compute_indicators(df):
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"].astype(float)
    df["ema21"] = ta.ema(c, length=21)
    df["ema60"] = ta.ema(c, length=60)
    df["ema200"] = ta.ema(c, length=200)
    df["atr"] = ta.atr(h, lo, c, length=14)
    df["atr_ma"] = ta.ema(df["atr"], length=50)
    df["atr_ratio"] = df["atr"] / (df["atr_ma"] + 1e-10)
    adx = ta.adx(h, lo, c, length=14)
    if adx is not None:
        df["adx"] = adx.iloc[:, 0]
        df["plus_di"] = adx.iloc[:, 1]
        df["minus_di"] = adx.iloc[:, 2]
    else:
        df["adx"] = df["plus_di"] = df["minus_di"] = 0
    df["rsi"] = ta.rsi(c, length=14)
    df["vol_ema"] = ta.ema(v, length=20)
    df["vol_ratio"] = v / (df["vol_ema"] + 1e-10)
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    df["macd_hist"] = macd.iloc[:, 1] if macd is not None else 0
    df["obv"] = ta.obv(c, v)
    df["obv_ema"] = ta.ema(df["obv"], length=20)
    df["obv_rising"] = (df["obv"] > df["obv_ema"]).astype(int)
    try:
        st = ta.supertrend(h, lo, c, length=7, multiplier=3.0)
        st_dir_col = next((col for col in st.columns if "SUPERTd" in col), None)
        df["st_dir"] = st[st_dir_col].fillna(0) if st_dir_col else 0
    except Exception:
        df["st_dir"] = 0
    cross_up = (df["ema21"] > df["ema60"]) & (df["ema21"].shift(1) <= df["ema60"].shift(1))
    cross_down = (df["ema21"] < df["ema60"]) & (df["ema21"].shift(1) >= df["ema60"].shift(1))
    df["cross_up_recent"] = cross_up.rolling(5).max().fillna(0).astype(int)
    df["cross_down_recent"] = cross_down.rolling(5).max().fillna(0).astype(int)
    df["is_bull"] = (c > df["ema200"]).astype(int)
    df["is_bear"] = (c < df["ema200"]).astype(int)
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        df["bb_upper"] = bb.iloc[:, 0]
        df["bb_lower"] = bb.iloc[:, 2]
    return df


def detect_signals(df):
    signals = []
    for i in range(220, len(df)):
        row = df.iloc[i]
        if float(row.get("atr_ratio", 0)) >= 2.2:
            continue
        if float(row.get("vol_ratio", 0)) < 1.0:
            continue
        if float(row.get("volume", 0)) <= 0:
            continue
        is_trending = float(row.get("adx", 0)) > 25
        is_bull = int(row.get("is_bull", 0)) == 1
        is_bear = int(row.get("is_bear", 0)) == 1
        if is_trending:
            if (is_bull
                and int(row.get("cross_up_recent", 0)) == 1
                and float(row.get("ema21", 0)) > float(row.get("ema60", 0))
                and float(row.get("macd_hist", 0)) > 0
                and float(row.get("plus_di", 0)) > float(row.get("minus_di", 0))
                and int(row.get("st_dir", 0)) == 1):
                signals.append((i, "long"))
            if (is_bear
                and int(row.get("cross_down_recent", 0)) == 1
                and float(row.get("ema21", 0)) < float(row.get("ema60", 0))
                and float(row.get("macd_hist", 0)) < 0
                and float(row.get("minus_di", 0)) > float(row.get("plus_di", 0))
                and int(row.get("st_dir", 0)) == -1):
                signals.append((i, "short"))
        else:
            rsi = float(row.get("rsi", 50))
            prev_rsi = float(df.iloc[i - 1].get("rsi", 50))
            if (prev_rsi < 28 and rsi > prev_rsi
                and float(row["close"]) < float(row.get("bb_lower", 0)) * 1.01
                and float(row["close"]) > float(row["open"])
                and int(row.get("obv_rising", 0)) == 1):
                signals.append((i, "long"))
            if (prev_rsi > 67 and rsi < prev_rsi
                and float(row["close"]) > float(row.get("bb_upper", 0)) * 0.99
                and float(row["close"]) < float(row["open"])
                and int(row.get("obv_rising", 0)) == 0):
                signals.append((i, "short"))
    return signals


def simulate(df, signals, sl_mult, tp_mult, max_bars, cooldown=5):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values
    trades = []
    last_bar = -cooldown

    for bar_idx, direction in signals:
        if bar_idx - last_bar < cooldown:
            continue
        if bar_idx + max_bars >= len(close):
            continue
        entry = close[bar_idx]
        a = atr[bar_idx]
        if np.isnan(a) or a <= 0:
            continue

        tp_p = entry + tp_mult * a if direction == "long" else entry - tp_mult * a
        sl_p = entry - sl_mult * a if direction == "long" else entry + sl_mult * a

        exit_p = close[min(bar_idx + max_bars, len(close) - 1)]
        for j in range(1, max_bars + 1):
            idx = bar_idx + j
            if idx >= len(close):
                break
            if direction == "long":
                if low[idx] <= sl_p:
                    exit_p = sl_p
                    break
                if high[idx] >= tp_p:
                    exit_p = tp_p
                    break
            else:
                if high[idx] >= sl_p:
                    exit_p = sl_p
                    break
                if low[idx] <= tp_p:
                    exit_p = tp_p
                    break

        pnl = ((exit_p - entry) / entry if direction == "long" else (entry - exit_p) / entry) - 2 * TAKER
        trades.append(pnl)
        last_bar = bar_idx

    return trades


def sweep(pair_name):
    print(f"\n{'=' * 80}")
    print(f"  {pair_name} -- regime_adaptive Parameter Sweep")
    print(f"{'=' * 80}")

    df = load_15m(pair_name)
    df = compute_indicators(df)
    signals = detect_signals(df)
    print(f"  Total signals: {len(signals)}")
    print()

    header = f"  {'SL':>4} {'TP':>4} {'MaxB':>5} {'Trades':>7} {'WR':>7} {'PnL%':>9} {'PF':>6}"
    print(header)
    print(f"  {'-' * 50}")

    best_pf = 0
    best_params = None
    profitable_configs = []

    for sl in [2, 3, 4, 5, 7, 10]:
        for tp in [3, 4, 5, 7, 9, 11]:
            if tp <= sl:
                continue
            for max_bars in [48, 72, 96]:
                trades = simulate(df, signals, sl, tp, max_bars)
                if len(trades) < 10:
                    continue
                pnls = np.array(trades)
                wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                total = pnls.sum() * 100
                wins = pnls[pnls > 0].sum()
                losses = abs(pnls[pnls < 0].sum())
                pf = wins / losses if losses > 0 else 0

                if pf > best_pf and total > 0:
                    best_pf = pf
                    best_params = (sl, tp, max_bars, len(trades), wr, total, pf)

                if total > 0 and pf > 1.0:
                    profitable_configs.append((sl, tp, max_bars, len(trades), wr, total, pf))

    # Sort by PnL and show top 15
    profitable_configs.sort(key=lambda x: -x[5])
    for cfg in profitable_configs[:15]:
        sl, tp, mb, n, wr, pnl, pf = cfg
        print(f"  {sl:>4} {tp:>4} {mb:>5} {n:>7} {wr:>6.1f}% {pnl:>+8.2f}% {pf:>5.2f}")

    print()
    if best_params:
        sl, tp, mb, n, wr, pnl, pf = best_params
        print(f"  BEST: SL={sl}x TP={tp}x MaxBars={mb} | {n} trades, WR={wr:.1f}%, PnL={pnl:+.2f}%, PF={pf:.2f}")
    else:
        print(f"  No profitable configuration found")

    return best_params


def main():
    print("PARAMETER SWEEP: Finding optimal SL/TP for new pairs")
    print("Strategy: regime_adaptive | Sweep: SL=[2-10], TP=[3-11], MaxBars=[48-96]")

    xau_best = sweep("XAU_USDT_USDT")
    nvda_best = sweep("NVDA_USDT_USDT")

    # Also test volume_spike_rev on XAU with different params
    print(f"\n{'=' * 80}")
    print(f"  XAU_USDT_USDT -- volume_spike_rev Parameter Sweep")
    print(f"{'=' * 80}")

    df = load_15m("XAU_USDT_USDT")
    df = compute_indicators(df)
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"].astype(float)

    # Detect volume spike reversal signals
    vsr_signals = []
    for i in range(20, len(df)):
        row = df.iloc[i]
        vol_ratio = float(row.get("vol_ratio", 0))
        if vol_ratio < 2.0:
            continue
        close_v = float(row["close"])
        open_v = float(row["open"])
        high_v = float(row["high"])
        low_v = float(row["low"])
        rsi = float(row.get("rsi", 50))
        body = abs(close_v - open_v)
        full_range = high_v - low_v
        if full_range <= 0 or body <= 0:
            continue
        shadow_ratio = (full_range - body) / body
        if shadow_ratio >= 2.5:
            lower_shadow = min(close_v, open_v) - low_v
            upper_shadow = high_v - max(close_v, open_v)
            if lower_shadow > full_range * 0.4 and rsi < 35:
                vsr_signals.append((i, "long"))
            if upper_shadow > full_range * 0.4 and rsi > 65:
                vsr_signals.append((i, "short"))

    print(f"  Total VSR signals: {len(vsr_signals)}")
    print()

    header = f"  {'SL':>4} {'TP':>4} {'MaxB':>5} {'Trades':>7} {'WR':>7} {'PnL%':>9} {'PF':>6}"
    print(header)
    print(f"  {'-' * 50}")

    profitable = []
    for sl in [2, 3, 4, 5]:
        for tp in [3, 4, 5, 7, 9]:
            if tp <= sl:
                continue
            for max_bars in [24, 36, 48]:
                trades = simulate(df, vsr_signals, sl, tp, max_bars, cooldown=5)
                if len(trades) < 8:
                    continue
                pnls = np.array(trades)
                wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                total = pnls.sum() * 100
                wins = pnls[pnls > 0].sum()
                losses = abs(pnls[pnls < 0].sum())
                pf = wins / losses if losses > 0 else 0
                if total > 0 and pf > 1.0:
                    profitable.append((sl, tp, max_bars, len(trades), wr, total, pf))

    profitable.sort(key=lambda x: -x[5])
    for cfg in profitable[:10]:
        sl, tp, mb, n, wr, pnl, pf = cfg
        print(f"  {sl:>4} {tp:>4} {mb:>5} {n:>7} {wr:>6.1f}% {pnl:>+8.2f}% {pf:>5.2f}")

    if not profitable:
        print("  No profitable config found for VSR on XAU")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    if xau_best:
        print(f"  XAU regime_adaptive: VIABLE with optimized params")
    else:
        print(f"  XAU regime_adaptive: NOT VIABLE (no profitable config)")
    if nvda_best:
        print(f"  NVDA regime_adaptive: VIABLE with optimized params")
    else:
        print(f"  NVDA regime_adaptive: NOT VIABLE")


if __name__ == "__main__":
    main()

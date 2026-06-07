"""Entry Optimization Analysis: Find optimal entry points using ATR, S/R, EMA, BB.

Instead of entering at market on signal, test different entry strategies:
1. ATR offset (limit order below/above close)
2. Support/Resistance proximity
3. EMA pullback (enter when price touches EMA)
4. Bollinger Band touch (enter at band edge)
5. Combined: Signal + wait for pullback to optimal zone

Compare: raw signal entry vs optimized entry for each strategy x pair.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
PAIRS = ["ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT"]
TAKER_FEE = 0.0005
MAKER_FEE = 0.0002


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    df = pd.read_feather(fp).sort_values("date").reset_index(drop=True)
    return df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)


def backtest_with_entry(signals, high, low, close, open_p, atr, sl_mult, tp_mult,
                        entry_prices, max_bars=96, fee=2*TAKER_FEE):
    """Backtest with custom entry prices (not just close)."""
    n = len(close)
    trades = []
    i = 0
    while i < n:
        if signals[i] == 0:
            i += 1
            continue
        ep = entry_prices[i]
        ea = atr[i]
        if ea <= 0 or ep <= 0 or np.isnan(ea) or np.isnan(ep):
            i += 1
            continue
        is_long = signals[i] > 0
        exited = False

        for j in range(i + 1, min(i + max_bars + 1, n)):
            if is_long:
                if low[j] <= ep - sl_mult * ea:
                    trades.append((-sl_mult * ea / ep - fee, "SL"))
                    i = j + 1
                    exited = True
                    break
                if high[j] >= ep + tp_mult * ea:
                    trades.append((tp_mult * ea / ep - fee, "TP"))
                    i = j + 1
                    exited = True
                    break
            else:
                if high[j] >= ep + sl_mult * ea:
                    trades.append((-sl_mult * ea / ep - fee, "SL"))
                    i = j + 1
                    exited = True
                    break
                if low[j] <= ep - tp_mult * ea:
                    trades.append((tp_mult * ea / ep - fee, "TP"))
                    i = j + 1
                    exited = True
                    break
        if not exited:
            ei = min(i + max_bars, n - 1)
            pnl = ((close[ei] - ep) / ep if is_long else (ep - close[ei]) / ep) - fee
            trades.append((pnl, "TC"))
            i = ei + 1
    return trades


def backtest_limit_entry(signals, high, low, close, atr, sl_mult, tp_mult,
                         limit_offset_atr, max_wait_bars=6, max_hold_bars=96,
                         fee=2*MAKER_FEE):
    """Simulate limit order entry: place order at offset, check if filled within wait bars."""
    n = len(close)
    trades = []
    fills = 0
    misses = 0
    i = 0
    while i < n:
        if signals[i] == 0:
            i += 1
            continue
        ea = atr[i]
        if ea <= 0 or np.isnan(ea):
            i += 1
            continue
        is_long = signals[i] > 0

        # Limit order price
        if is_long:
            limit_price = close[i] - limit_offset_atr * ea
        else:
            limit_price = close[i] + limit_offset_atr * ea

        # Wait for fill
        filled = False
        fill_bar = -1
        for w in range(1, max_wait_bars + 1):
            if i + w >= n:
                break
            if is_long and low[i + w] <= limit_price:
                filled = True
                fill_bar = i + w
                break
            elif not is_long and high[i + w] >= limit_price:
                filled = True
                fill_bar = i + w
                break

        if not filled:
            misses += 1
            i += 1
            continue

        fills += 1
        ep = limit_price
        exited = False

        for j in range(fill_bar + 1, min(fill_bar + max_hold_bars + 1, n)):
            if is_long:
                if low[j] <= ep - sl_mult * ea:
                    trades.append((-sl_mult * ea / ep - fee, "SL"))
                    i = j + 1
                    exited = True
                    break
                if high[j] >= ep + tp_mult * ea:
                    trades.append((tp_mult * ea / ep - fee, "TP"))
                    i = j + 1
                    exited = True
                    break
            else:
                if high[j] >= ep + sl_mult * ea:
                    trades.append((-sl_mult * ea / ep - fee, "SL"))
                    i = j + 1
                    exited = True
                    break
                if low[j] <= ep - tp_mult * ea:
                    trades.append((tp_mult * ea / ep - fee, "TP"))
                    i = j + 1
                    exited = True
                    break
        if not exited:
            ei = min(fill_bar + max_hold_bars, n - 1)
            pnl = ((close[ei] - ep) / ep if is_long else (ep - close[ei]) / ep) - fee
            trades.append((pnl, "TC"))
            i = ei + 1

    return trades, fills, misses


def backtest_pullback_entry(signals, high, low, close, atr, sl_mult, tp_mult,
                            pullback_target, max_wait_bars=12, max_hold_bars=96,
                            fee=2*MAKER_FEE):
    """Wait for pullback to target price before entering."""
    n = len(close)
    trades = []
    fills = 0
    misses = 0
    i = 0
    while i < n:
        if signals[i] == 0:
            i += 1
            continue
        ea = atr[i]
        target = pullback_target[i]
        if ea <= 0 or np.isnan(ea) or np.isnan(target) or target <= 0:
            i += 1
            continue
        is_long = signals[i] > 0

        # Wait for price to reach target
        filled = False
        fill_bar = -1
        for w in range(1, max_wait_bars + 1):
            if i + w >= n:
                break
            if is_long and low[i + w] <= target:
                filled = True
                fill_bar = i + w
                break
            elif not is_long and high[i + w] >= target:
                filled = True
                fill_bar = i + w
                break

        if not filled:
            misses += 1
            i += 1
            continue

        fills += 1
        ep = target
        exited = False

        for j in range(fill_bar + 1, min(fill_bar + max_hold_bars + 1, n)):
            if is_long:
                if low[j] <= ep - sl_mult * ea:
                    trades.append((-sl_mult * ea / ep - fee, "SL"))
                    i = j + 1
                    exited = True
                    break
                if high[j] >= ep + tp_mult * ea:
                    trades.append((tp_mult * ea / ep - fee, "TP"))
                    i = j + 1
                    exited = True
                    break
            else:
                if high[j] >= ep + sl_mult * ea:
                    trades.append((-sl_mult * ea / ep - fee, "SL"))
                    i = j + 1
                    exited = True
                    break
                if low[j] <= ep - tp_mult * ea:
                    trades.append((tp_mult * ea / ep - fee, "TP"))
                    i = j + 1
                    exited = True
                    break
        if not exited:
            ei = min(fill_bar + max_hold_bars, n - 1)
            pnl = ((close[ei] - ep) / ep if is_long else (ep - close[ei]) / ep) - fee
            trades.append((pnl, "TC"))
            i = ei + 1

    return trades, fills, misses


def summarize(trades, label=""):
    if not trades:
        return f"  {label}: 0 trades"
    pnls = np.array([t[0] for t in trades])
    exits = [t[1] for t in trades]
    wr = (pnls > 0).sum() / len(pnls)
    pf = sum(p for p in pnls if p > 0) / max(abs(sum(p for p in pnls if p < 0)), 1e-10)
    tp_rate = exits.count("TP") / len(exits) * 100
    return (f"  {label:<45} PnL={pnls.sum():+.2%} | {len(pnls)} trades | "
            f"WR={wr*100:.1f}% | PF={pf:.2f} | TP%={tp_rate:.0f}%")


def main():
    print("=" * 95)
    print("ENTRY OPTIMIZATION: Better entries using ATR, S/R, EMA, BB")
    print("Hypothesis: Optimized entry = better fill = tighter SL = higher TP rate")
    print("=" * 95)

    for pair in PAIRS:
        df = load_15m(pair)
        n = len(df)
        c = df["close"].values
        h = df["high"].values
        lo = df["low"].values
        o = df["open"].values
        v = df["volume"].values.astype(float)

        # Pre-compute indicators
        atr = ta.atr(df["high"], df["low"], df["close"], length=14).values
        adx_r = ta.adx(df["high"], df["low"], df["close"], length=14)
        adx = adx_r.iloc[:, 0].values
        dip = adx_r.iloc[:, 1].values
        dim = adx_r.iloc[:, 2].values
        rsi = ta.rsi(df["close"], length=14).values
        vr = (v / (ta.ema(df["volume"].astype(float), length=20).values + 1e-10))

        ema8 = ta.ema(df["close"], length=8).values
        ema21 = ta.ema(df["close"], length=21).values
        ema50 = ta.ema(df["close"], length=50).values
        ema60 = ta.ema(df["close"], length=60).values

        bb = ta.bbands(df["close"], length=20, std=2.0)
        bb_upper = bb.iloc[:, 0].values if bb is not None else c
        bb_mid = bb.iloc[:, 1].values if bb is not None else c
        bb_lower = bb.iloc[:, 2].values if bb is not None else c

        # Support/Resistance: rolling high/low
        sup_20 = pd.Series(lo).rolling(20).min().values
        res_20 = pd.Series(h).rolling(20).max().values
        sup_10 = pd.Series(lo).rolling(10).min().values
        res_10 = pd.Series(h).rolling(10).max().values

        print(f"\n{'=' * 95}")
        print(f"  {pair}")
        print(f"{'=' * 95}")

        # ══════════════════════════════════════════════════════════════════
        # Generate signals: REGIME ADAPTIVE
        # ══════════════════════════════════════════════════════════════════
        sig_ra = np.zeros(n)
        for i in range(60, n):
            if vr[i] < 1.0:
                continue
            if adx[i] >= 25:
                for k in range(1, 6):
                    pi = i - k
                    if pi < 1:
                        break
                    if ema21[pi - 1] <= ema60[pi - 1] and ema21[pi] > ema60[pi]:
                        if dip[i] > dim[i]:
                            sig_ra[i] = 1
                        break
                    if ema21[pi - 1] >= ema60[pi - 1] and ema21[pi] < ema60[pi]:
                        if dim[i] > dip[i]:
                            sig_ra[i] = -1
                        break
            else:
                if rsi[i] < 28:
                    sig_ra[i] = 1
                elif rsi[i] > 67:
                    sig_ra[i] = -1

        print("\n  REGIME ADAPTIVE:")

        # A. Baseline: enter at close
        t_base = backtest_with_entry(sig_ra, h, lo, c, o, atr, 7.0, 11.0, c, 192)
        print(summarize(t_base, "A. Baseline (market @ close)"))

        # B. ATR offset limit entries
        for offset in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            t_lim, fills, misses = backtest_limit_entry(
                sig_ra, h, lo, c, atr, 7.0, 11.0, offset, max_wait_bars=6, max_hold_bars=192)
            fill_rate = fills / max(fills + misses, 1) * 100
            print(summarize(t_lim, f"B. Limit {offset:.1f}x ATR (fill={fill_rate:.0f}%)"))

        # C. EMA pullback: wait for price to touch EMA21 (for longs: pullback down to ema21)
        ema_target = np.where(sig_ra > 0, ema21, np.where(sig_ra < 0, ema21, c))
        t_ema, fills, misses = backtest_pullback_entry(
            sig_ra, h, lo, c, atr, 7.0, 11.0, ema_target, max_wait_bars=12, max_hold_bars=192)
        fill_rate = fills / max(fills + misses, 1) * 100
        print(summarize(t_ema, f"C. EMA21 pullback (fill={fill_rate:.0f}%)"))

        # D. EMA8 pullback (faster, more fills)
        ema8_target = np.where(sig_ra > 0, ema8, np.where(sig_ra < 0, ema8, c))
        t_ema8, fills, misses = backtest_pullback_entry(
            sig_ra, h, lo, c, atr, 7.0, 11.0, ema8_target, max_wait_bars=8, max_hold_bars=192)
        fill_rate = fills / max(fills + misses, 1) * 100
        print(summarize(t_ema8, f"D. EMA8 pullback (fill={fill_rate:.0f}%)"))

        # E. BB band entry: long at BB lower, short at BB upper
        bb_target = np.where(sig_ra > 0, bb_lower, np.where(sig_ra < 0, bb_upper, c))
        t_bb, fills, misses = backtest_pullback_entry(
            sig_ra, h, lo, c, atr, 7.0, 11.0, bb_target, max_wait_bars=12, max_hold_bars=192)
        fill_rate = fills / max(fills + misses, 1) * 100
        print(summarize(t_bb, f"E. BB band entry (fill={fill_rate:.0f}%)"))

        # F. Support/Resistance: long near support, short near resistance
        sr_target = np.where(sig_ra > 0, sup_10, np.where(sig_ra < 0, res_10, c))
        t_sr, fills, misses = backtest_pullback_entry(
            sig_ra, h, lo, c, atr, 7.0, 11.0, sr_target, max_wait_bars=12, max_hold_bars=192)
        fill_rate = fills / max(fills + misses, 1) * 100
        print(summarize(t_sr, f"F. S/R (sup10/res10) entry (fill={fill_rate:.0f}%)"))

        # G. Combined: ATR offset + only enter if near EMA or support
        # Smart entry: limit at 0.3x ATR below close for longs, but only if price > ema21
        smart_entry = np.full(n, np.nan)
        for i in range(60, n):
            if sig_ra[i] == 1:
                # Long: entry = max(close - 0.3*ATR, ema21) — don't go below ema21
                smart_entry[i] = max(c[i] - 0.3 * atr[i], ema21[i])
            elif sig_ra[i] == -1:
                # Short: entry = min(close + 0.3*ATR, ema21) — don't go above ema21
                smart_entry[i] = min(c[i] + 0.3 * atr[i], ema21[i])

        t_smart, fills, misses = backtest_pullback_entry(
            sig_ra, h, lo, c, atr, 7.0, 11.0, smart_entry, max_wait_bars=8, max_hold_bars=192)
        fill_rate = fills / max(fills + misses, 1) * 100
        print(summarize(t_smart, f"G. Smart (0.3ATR+EMA21 floor) (fill={fill_rate:.0f}%)"))

        # H. Aggressive smart: 0.5 ATR offset, EMA8 floor, tighter SL
        smart2 = np.full(n, np.nan)
        for i in range(60, n):
            if sig_ra[i] == 1:
                smart2[i] = max(c[i] - 0.5 * atr[i], ema8[i])
            elif sig_ra[i] == -1:
                smart2[i] = min(c[i] + 0.5 * atr[i], ema8[i])

        t_smart2, fills, misses = backtest_pullback_entry(
            sig_ra, h, lo, c, atr, 6.0, 11.0, smart2, max_wait_bars=8, max_hold_bars=192)
        fill_rate = fills / max(fills + misses, 1) * 100
        print(summarize(t_smart2, f"H. Aggr (0.5ATR+EMA8, SL=6) (fill={fill_rate:.0f}%)"))

        # ══════════════════════════════════════════════════════════════════
        # VOLUME SPIKE REV
        # ══════════════════════════════════════════════════════════════════
        sig_vs = np.zeros(n)
        ba = np.abs(c - o)
        ls_v = np.minimum(c, o) - lo
        us_v = h - np.maximum(c, o)
        for i in range(20, n):
            if vr[i] < 2.0 or ba[i] <= 0:
                continue
            if ls_v[i] > 3.0 * ba[i] and rsi[i] < 33:
                sig_vs[i] = 1
            elif us_v[i] > 3.0 * ba[i] and rsi[i] > 72:
                sig_vs[i] = -1

        if sig_vs.any():
            print("\n  VOLUME SPIKE REV:")
            t_base_vs = backtest_with_entry(sig_vs, h, lo, c, o, atr, 3.5, 5.5, c, 96)
            print(summarize(t_base_vs, "A. Baseline (market @ close)"))

            for offset in [0.2, 0.3, 0.5, 0.7, 1.0]:
                t_lim, fills, misses = backtest_limit_entry(
                    sig_vs, h, lo, c, atr, 3.5, 5.5, offset, max_wait_bars=4, max_hold_bars=96)
                fill_rate = fills / max(fills + misses, 1) * 100
                print(summarize(t_lim, f"B. Limit {offset:.1f}x ATR (fill={fill_rate:.0f}%)"))

            # For vol spike: enter at candle body midpoint (reversal confirmation)
            body_mid = np.where(sig_vs > 0, (lo + np.minimum(c, o)) / 2,
                       np.where(sig_vs < 0, (h + np.maximum(c, o)) / 2, c))
            t_mid, fills, misses = backtest_pullback_entry(
                sig_vs, h, lo, c, atr, 3.5, 5.5, body_mid, max_wait_bars=4, max_hold_bars=96)
            fill_rate = fills / max(fills + misses, 1) * 100
            print(summarize(t_mid, f"C. Body midpoint entry (fill={fill_rate:.0f}%)"))

            # BB band entry for vol spike
            bb_vs = np.where(sig_vs > 0, bb_lower, np.where(sig_vs < 0, bb_upper, c))
            t_bb_vs, fills, misses = backtest_pullback_entry(
                sig_vs, h, lo, c, atr, 3.5, 5.5, bb_vs, max_wait_bars=6, max_hold_bars=96)
            fill_rate = fills / max(fills + misses, 1) * 100
            print(summarize(t_bb_vs, f"D. BB band entry (fill={fill_rate:.0f}%)"))

        # ══════════════════════════════════════════════════════════════════
        # CB ADX BREAKOUT
        # ══════════════════════════════════════════════════════════════════
        sig_cb = np.zeros(n)
        for i in range(22, n):
            if vr[i] < 0.8:
                continue
            if rsi[i] > 72 or rsi[i] < 22:
                continue
            if adx[i] > 22 or adx[i] >= adx[i - 2]:
                continue
            if atr[i] <= 0 or np.isnan(atr[i]):
                continue
            r3m = np.mean([h[i - k] - lo[i - k] for k in range(3)])
            if r3m / atr[i] > 0.8:
                continue
            h3 = max(h[i - 2], h[i - 1])
            l3 = min(lo[i - 2], lo[i - 1])
            if c[i] > h3:
                sig_cb[i] = 1
            elif c[i] < l3:
                sig_cb[i] = -1

        if sig_cb.any():
            print("\n  CB ADX BREAKOUT:")
            t_base_cb = backtest_with_entry(sig_cb, h, lo, c, o, atr, 3.0, 5.0, c, 96)
            print(summarize(t_base_cb, "A. Baseline (market @ close)"))

            # Breakout retest: after breakout, wait for pullback to breakout level
            retest_target = np.full(n, np.nan)
            for i in range(22, n):
                if sig_cb[i] == 1:
                    retest_target[i] = max(h[i - 2], h[i - 1])  # Pullback to broken resistance
                elif sig_cb[i] == -1:
                    retest_target[i] = min(lo[i - 2], lo[i - 1])  # Pullback to broken support

            t_retest, fills, misses = backtest_pullback_entry(
                sig_cb, h, lo, c, atr, 3.0, 5.0, retest_target, max_wait_bars=8, max_hold_bars=96)
            fill_rate = fills / max(fills + misses, 1) * 100
            print(summarize(t_retest, f"B. Breakout retest (fill={fill_rate:.0f}%)"))

            for offset in [0.1, 0.2, 0.3, 0.5]:
                t_lim, fills, misses = backtest_limit_entry(
                    sig_cb, h, lo, c, atr, 3.0, 5.0, offset, max_wait_bars=4, max_hold_bars=96)
                fill_rate = fills / max(fills + misses, 1) * 100
                print(summarize(t_lim, f"C. Limit {offset:.1f}x ATR (fill={fill_rate:.0f}%)"))

            # EMA8 retest for breakout
            ema8_cb = np.where(sig_cb > 0, ema8, np.where(sig_cb < 0, ema8, c))
            t_ema8_cb, fills, misses = backtest_pullback_entry(
                sig_cb, h, lo, c, atr, 3.0, 5.0, ema8_cb, max_wait_bars=6, max_hold_bars=96)
            fill_rate = fills / max(fills + misses, 1) * 100
            print(summarize(t_ema8_cb, f"D. EMA8 retest (fill={fill_rate:.0f}%)"))

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print("  KEY INSIGHTS")
    print(f"{'=' * 95}")
    print("""
  Entry optimization strategies tested:
  A. Baseline:     Market order at signal candle close (taker fee 0.10% RT)
  B. ATR offset:   Limit order at close +/- X*ATR (maker fee 0.04% RT)
  C-F. Pullback:   Wait for price to reach EMA/BB/S-R level (maker fee)
  G-H. Smart:      Combined ATR offset with EMA floor

  Better entry = lower effective cost + closer to optimal zone
  Trade-off: fill rate drops with more aggressive limit orders
    """)


if __name__ == "__main__":
    main()

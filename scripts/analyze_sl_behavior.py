"""Analyze stoploss behavior on real OKX data.

For each SL hit:
- Track price after SL: does it rotate back to entry? (= SL too tight)
- Track max adverse excursion (MAE) vs max favorable excursion (MFE)
- Per-pair analysis to recommend: wider SL, trailing, limit entry, etc.
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


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    return pd.read_feather(str(fp)).sort_values("date").reset_index(drop=True)


def analyze_sl_trades(pair, ind, signals, sl_mult, tp_mult, max_bars=192):
    """Detailed trade-by-trade analysis with MAE/MFE and post-SL behavior."""
    n = len(ind["close"])
    c = ind["close"]; h = ind["high"]; lo = ind["low"]; atr = ind["atr"]
    entries = np.where(signals != 0)[0]

    trades = []
    next_allowed = 0

    for idx in entries:
        if idx < next_allowed:
            continue
        ep = c[idx]
        ea = atr[idx]
        if ea <= 0 or ep <= 0 or np.isnan(ea):
            continue

        is_long = signals[idx] > 0
        sl_d = sl_mult * ea
        tp_d = tp_mult * ea
        end_bar = min(idx + max_bars, n - 1)

        # Track MAE and MFE
        mae = 0  # max adverse excursion (worst drawdown during trade)
        mfe = 0  # max favorable excursion (best unrealized profit)
        exit_bar = end_bar
        exit_reason = "timeout"
        exit_pnl = 0

        for j in range(idx + 1, end_bar + 1):
            if is_long:
                unrealized = (c[j] - ep) / ep
                drawdown = (ep - lo[j]) / ep
                runup = (h[j] - ep) / ep
            else:
                unrealized = (ep - c[j]) / ep
                drawdown = (h[j] - ep) / ep
                runup = (ep - lo[j]) / ep

            mae = max(mae, drawdown)
            mfe = max(mfe, runup)

            # Check SL
            if is_long and lo[j] <= ep - sl_d:
                exit_bar = j
                exit_reason = "SL"
                exit_pnl = -sl_mult * ea / ep
                break
            if not is_long and h[j] >= ep + sl_d:
                exit_bar = j
                exit_reason = "SL"
                exit_pnl = -sl_mult * ea / ep
                break
            # Check TP
            if is_long and h[j] >= ep + tp_d:
                exit_bar = j
                exit_reason = "TP"
                exit_pnl = tp_mult * ea / ep
                break
            if not is_long and lo[j] <= ep - tp_d:
                exit_bar = j
                exit_reason = "TP"
                exit_pnl = tp_mult * ea / ep
                break
        else:
            exit_pnl = ((c[end_bar] - ep) / ep if is_long else (ep - c[end_bar]) / ep)

        # Post-exit analysis: what happens in next 48 bars after SL?
        post_sl_rotated = False
        post_sl_max_favorable = 0
        post_sl_bars_to_entry = -1

        if exit_reason == "SL" and exit_bar + 48 < n:
            for k in range(1, 49):
                pk = exit_bar + k
                if pk >= n:
                    break
                if is_long:
                    recovery = (c[pk] - ep) / ep
                    post_fav = (h[pk] - (ep - sl_d)) / ep
                else:
                    recovery = (ep - c[pk]) / ep
                    post_fav = ((ep + sl_d) - lo[pk]) / ep

                post_sl_max_favorable = max(post_sl_max_favorable, post_fav)
                if recovery > 0 and post_sl_bars_to_entry < 0:
                    post_sl_bars_to_entry = k
                    post_sl_rotated = True

        trades.append({
            "entry_bar": idx,
            "exit_bar": exit_bar,
            "is_long": is_long,
            "entry_price": ep,
            "atr": ea,
            "sl_distance_pct": sl_d / ep,
            "tp_distance_pct": tp_d / ep,
            "mae_pct": mae,
            "mfe_pct": mfe,
            "exit_reason": exit_reason,
            "exit_pnl": exit_pnl,
            "post_sl_rotated": post_sl_rotated,
            "post_sl_max_favorable": post_sl_max_favorable,
            "post_sl_bars_to_entry": post_sl_bars_to_entry,
            "hold_bars": exit_bar - idx,
        })
        next_allowed = exit_bar + 1

    return trades


def main():
    print("=" * 90)
    print("  STOPLOSS ANALYSIS — Real OKX Data (5 pairs, exclude BTC/SUI)")
    print("  Strategy: regime_adaptive | Params: adx=31, vol=1.5, sl=7.0, tp=11.0 ATR")
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
        ind["ema18"] = ta.ema(df["close"], length=18).values
        ind["ema50"] = ta.ema(df["close"], length=50).values

        cross_up = np.zeros(len(df), dtype=bool)
        cross_down = np.zeros(len(df), dtype=bool)
        cross_up[1:] = (ind["ema18"][1:] > ind["ema50"][1:]) & (ind["ema18"][:-1] <= ind["ema50"][:-1])
        cross_down[1:] = (ind["ema18"][1:] < ind["ema50"][1:]) & (ind["ema18"][:-1] >= ind["ema50"][:-1])
        ind["cross_up"] = cross_up
        ind["cross_down"] = cross_down

        data[pair] = ind

    # Generate signals and analyze
    # regime_adaptive params: adx=31, rsi_os=25, rsi_ob=70, vol=1.5, ema18/50, lb=5
    SL_MULT = 7.0
    TP_MULT = 11.0

    for pair, ind in data.items():
        n = ind["n"]
        adx = ind["adx"]; di_p = ind["di_plus"]; di_m = ind["di_minus"]
        rsi = ind["rsi"]; vr = ind["vol_ratio"]
        recent_up = maximum_filter1d(ind["cross_up"].astype(float), size=5, origin=1)
        recent_down = maximum_filter1d(ind["cross_down"].astype(float), size=5, origin=1)

        trend_l = (adx >= 31) & (vr >= 1.5) & (recent_up > 0) & (di_p > di_m)
        trend_s = (adx >= 31) & (vr >= 1.5) & (recent_down > 0) & (di_m > di_p)
        range_l = (~(adx >= 31)) & (vr >= 1.5) & (rsi < 25)
        range_s = (~(adx >= 31)) & (vr >= 1.5) & (rsi > 70)

        signals = np.zeros(n)
        signals[trend_l | range_l] = 1
        signals[trend_s | range_s] = -1
        signals[:60] = 0

        trades = analyze_sl_trades(pair, ind, signals, SL_MULT, TP_MULT, max_bars=192)

        if not trades:
            continue

        # Analyze
        sl_trades = [t for t in trades if t["exit_reason"] == "SL"]
        tp_trades = [t for t in trades if t["exit_reason"] == "TP"]
        timeout_trades = [t for t in trades if t["exit_reason"] == "timeout"]

        total = len(trades)
        n_sl = len(sl_trades)
        n_tp = len(tp_trades)
        n_to = len(timeout_trades)

        print(f"\n{'='*90}")
        print(f"  {pair}")
        print(f"{'='*90}")
        print(f"  Total trades: {total} | TP: {n_tp} ({n_tp/total*100:.0f}%) | SL: {n_sl} ({n_sl/total*100:.0f}%) | Timeout: {n_to} ({n_to/total*100:.0f}%)")

        if sl_trades:
            rotated = sum(1 for t in sl_trades if t["post_sl_rotated"])
            pct_rotated = rotated / n_sl * 100

            avg_mae = np.mean([t["mae_pct"] for t in sl_trades]) * 100
            avg_mfe_before_sl = np.mean([t["mfe_pct"] for t in sl_trades]) * 100
            avg_post_fav = np.mean([t["post_sl_max_favorable"] for t in sl_trades]) * 100
            avg_bars_to_entry = np.mean([t["post_sl_bars_to_entry"] for t in sl_trades if t["post_sl_bars_to_entry"] > 0])
            avg_hold = np.mean([t["hold_bars"] for t in sl_trades])

            print(f"\n  SL ANALYSIS ({n_sl} trades):")
            print(f"    Price rotated back to entry after SL: {rotated}/{n_sl} ({pct_rotated:.0f}%)")
            print(f"    Avg hold before SL: {avg_hold:.0f} bars ({avg_hold*15/60:.1f}h)")
            print(f"    Avg MAE (worst drawdown): {avg_mae:.2f}%")
            print(f"    Avg MFE before SL hit: {avg_mfe_before_sl:.2f}%")
            print(f"    Avg post-SL max favorable move: {avg_post_fav:.2f}%")
            if avg_bars_to_entry > 0:
                print(f"    Avg bars to rotate back to entry: {avg_bars_to_entry:.0f} ({avg_bars_to_entry*15/60:.1f}h)")

            # Split by long/short
            sl_longs = [t for t in sl_trades if t["is_long"]]
            sl_shorts = [t for t in sl_trades if not t["is_long"]]
            if sl_longs:
                rot_l = sum(1 for t in sl_longs if t["post_sl_rotated"])
                print(f"    Long SL: {len(sl_longs)} trades, {rot_l} rotated ({rot_l/len(sl_longs)*100:.0f}%)")
            if sl_shorts:
                rot_s = sum(1 for t in sl_shorts if t["post_sl_rotated"])
                print(f"    Short SL: {len(sl_shorts)} trades, {rot_s} rotated ({rot_s/len(sl_shorts)*100:.0f}%)")

            # MFE analysis: could trailing have saved some?
            # If MFE > 50% of TP before SL hit, trailing would have locked profit
            could_trail = sum(1 for t in sl_trades if t["mfe_pct"] > 0.5 * TP_MULT * t["atr"] / t["entry_price"])
            print(f"\n    Trades with MFE > 50% TP before SL: {could_trail}/{n_sl} ({could_trail/n_sl*100:.0f}%) -- trailing SL would help")

            # Entry timing: was entry at bad price?
            # Check if price moved favorably in first 2 bars (good entry) vs immediately adverse
            fast_sl = sum(1 for t in sl_trades if t["hold_bars"] <= 8)  # SL within 2 hours
            slow_sl = sum(1 for t in sl_trades if t["hold_bars"] > 48)  # SL after 12+ hours
            print(f"    Fast SL (<=2h): {fast_sl}/{n_sl} ({fast_sl/n_sl*100:.0f}%) -- entry timing issue")
            print(f"    Slow SL (>12h): {slow_sl}/{n_sl} ({slow_sl/n_sl*100:.0f}%) -- trend reversal, trailing would help")

        if tp_trades:
            avg_mfe_tp = np.mean([t["mfe_pct"] for t in tp_trades]) * 100
            avg_hold_tp = np.mean([t["hold_bars"] for t in tp_trades])
            print(f"\n  TP ANALYSIS ({n_tp} trades):")
            print(f"    Avg hold: {avg_hold_tp:.0f} bars ({avg_hold_tp*15/60:.1f}h)")
            print(f"    Avg MFE: {avg_mfe_tp:.2f}%")

            # Could TP have been higher? (Did price continue much past TP?)
            continued_past_tp = []
            for t in tp_trades:
                if t["exit_bar"] + 24 < n:
                    post_bars = slice(t["exit_bar"], t["exit_bar"] + 24)
                    if t["is_long"]:
                        max_after = np.max(ind["high"][post_bars])
                        extra = (max_after - (t["entry_price"] + TP_MULT * t["atr"])) / t["entry_price"]
                    else:
                        min_after = np.min(ind["low"][post_bars])
                        extra = ((t["entry_price"] - TP_MULT * t["atr"]) - min_after) / t["entry_price"]
                    continued_past_tp.append(extra)
            if continued_past_tp:
                avg_extra = np.mean(continued_past_tp) * 100
                pct_continue = sum(1 for x in continued_past_tp if x > 0.01) / len(continued_past_tp) * 100
                print(f"    Price continued past TP: {pct_continue:.0f}% of the time (avg extra: {avg_extra:+.2f}%)")

        # RECOMMENDATION
        print(f"\n  RECOMMENDATION for {pair}:")
        if sl_trades:
            if pct_rotated > 60:
                print(f"    [!] SL TOO TIGHT: {pct_rotated:.0f}% rotate back. Widen SL or use limit entry.")
                if fast_sl / n_sl > 0.4:
                    print(f"    --> Use LIMIT ENTRY (offset 0.1-0.2 ATR) to get better price")
                else:
                    print(f"    --> Widen SL from {SL_MULT} to {SL_MULT + 1.5:.1f} ATR")
            elif pct_rotated > 40:
                print(f"    [~] SL marginal: {pct_rotated:.0f}% rotate. Consider trailing SL or limit entry.")
                if could_trail / n_sl > 0.3:
                    print(f"    --> Add TRAILING SL (activate at 50% TP, trail at 3 ATR)")
            else:
                print(f"    [OK] SL appropriate: only {pct_rotated:.0f}% rotate back.")

            if slow_sl / n_sl > 0.3:
                print(f"    --> {slow_sl/n_sl*100:.0f}% slow SL: add TIME CUT at 12-24h if profit < 0")

    # ═══════════════════════════════════════════════════════════════════════
    # CB ADX BREAKOUT analysis
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*90}")
    print("  CB ADX BREAKOUT — SL Analysis")
    print("  Params: comp=0.9, adx_max=30, sl=3.0, tp=5.0 ATR")
    print(f"{'='*90}")

    CB_SL = 3.0
    CB_TP = 5.0

    for pair, ind in data.items():
        n = ind["n"]
        adx = ind["adx"]; vr = ind["vol_ratio"]; rsi = ind["rsi"]
        atr_v = ind["atr"]; c = ind["close"]; h = ind["high"]; lo = ind["low"]

        # Compute range_3bar and prev_h2/prev_l2
        h3_max = np.maximum(np.maximum(np.roll(h, 1), np.roll(h, 2)), h)
        l3_min = np.minimum(np.minimum(np.roll(lo, 1), np.roll(lo, 2)), lo)
        range_3bar = h3_max - l3_min
        prev_h2 = np.maximum(np.roll(h, 1), np.roll(h, 2))
        prev_l2 = np.minimum(np.roll(lo, 1), np.roll(lo, 2))

        adx_low = (adx < 30) | (np.roll(adx, 1) < 30) | (np.roll(adx, 2) < 30)
        compressed = (atr_v > 0) & (range_3bar / (atr_v + 1e-10) < 0.9)
        valid = (vr >= 0.6) & (rsi >= 22) & (rsi <= 72) & adx_low & compressed

        signals = np.zeros(n)
        signals[valid & (c > prev_h2)] = 1
        signals[valid & (c < prev_l2)] = -1
        signals[:20] = 0

        trades = analyze_sl_trades(pair, ind, signals, CB_SL, CB_TP, max_bars=96)

        if not trades:
            continue

        sl_trades = [t for t in trades if t["exit_reason"] == "SL"]
        tp_trades = [t for t in trades if t["exit_reason"] == "TP"]
        total = len(trades)
        n_sl = len(sl_trades)
        n_tp = len(tp_trades)

        print(f"\n  {pair}: {total} trades | TP: {n_tp} ({n_tp/total*100:.0f}%) | SL: {n_sl} ({n_sl/total*100:.0f}%)")

        if sl_trades:
            rotated = sum(1 for t in sl_trades if t["post_sl_rotated"])
            pct_rotated = rotated / n_sl * 100
            fast_sl = sum(1 for t in sl_trades if t["hold_bars"] <= 4)
            could_trail = sum(1 for t in sl_trades if t["mfe_pct"] > 0.5 * CB_TP * t["atr"] / t["entry_price"])
            avg_hold = np.mean([t["hold_bars"] for t in sl_trades])

            print(f"    SL rotated back: {rotated}/{n_sl} ({pct_rotated:.0f}%)")
            print(f"    Fast SL (<=1h): {fast_sl}/{n_sl} | Avg hold: {avg_hold:.0f} bars ({avg_hold*15/60:.1f}h)")
            print(f"    Trailing would help: {could_trail}/{n_sl}")

            if pct_rotated > 50:
                print(f"    --> LIMIT ENTRY or widen SL to {CB_SL + 1.0:.1f}")
            elif could_trail / max(n_sl, 1) > 0.3:
                print(f"    --> Add TRAILING (activate 40% TP)")


if __name__ == "__main__":
    main()

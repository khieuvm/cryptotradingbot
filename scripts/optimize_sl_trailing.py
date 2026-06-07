"""SL Optimization + Trailing Analysis.

For each strategy x pair:
1. SL sensitivity sweep (find optimal SL per pair)
2. MFE analysis: how far trades go before coming back
3. Trades that were profitable but ended as SL/timeout
4. Trailing SL backtest: BE lock + trail at various levels
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
PAIRS = ["SOL_USDT_USDT", "SPX_USDT_USDT"]
TAKER_FEE = 0.0005
MAKER_FEE = 0.0002


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    df = pd.read_feather(fp).sort_values("date").reset_index(drop=True)
    return df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)


def compute_indicators(df):
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    o = df["open"].values
    v = df["volume"].astype(float).values
    n = len(c)

    adx_r = ta.adx(df["high"], df["low"], df["close"], length=14)
    adx = adx_r.iloc[:, 0].values
    dip = adx_r.iloc[:, 1].values
    dim = adx_r.iloc[:, 2].values
    rsi = ta.rsi(df["close"], length=14).values
    atr = ta.atr(df["high"], df["low"], df["close"], length=14).values
    e8 = ta.ema(df["close"], length=8).values
    e21 = ta.ema(df["close"], length=21).values
    e60 = ta.ema(df["close"], length=60).values
    vol_ema = ta.ema(df["volume"].astype(float), length=20).values
    vr = v / (vol_ema + 1e-10)

    body = np.abs(c - o)
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(c, o) - lo

    return {
        "c": c, "h": h, "l": lo, "o": o, "v": v, "n": n,
        "adx": adx, "dip": dip, "dim": dim, "rsi": rsi, "atr": atr,
        "e8": e8, "e21": e21, "e60": e60, "vr": vr,
        "body": body, "upper_shadow": upper_shadow, "lower_shadow": lower_shadow,
    }


def generate_signals_regime(ind):
    n = ind["n"]
    signals = []
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
                        signals.append((i, 1))
                    break
                if ind["e21"][pi - 1] >= ind["e60"][pi - 1] and ind["e21"][pi] < ind["e60"][pi]:
                    if ind["dim"][i] > ind["dip"][i]:
                        signals.append((i, -1))
                    break
        else:
            if ind["rsi"][i] < 28:
                signals.append((i, 1))
            elif ind["rsi"][i] > 67:
                signals.append((i, -1))
    return signals


def generate_signals_volspike(ind):
    n = ind["n"]
    signals = []
    for i in range(20, n):
        if ind["vr"][i] < 2.0 or ind["body"][i] <= 0:
            continue
        if ind["lower_shadow"][i] > 3.0 * ind["body"][i] and ind["rsi"][i] < 33:
            signals.append((i, 1))
        elif ind["upper_shadow"][i] > 3.0 * ind["body"][i] and ind["rsi"][i] > 72:
            signals.append((i, -1))
    return signals


def generate_signals_cbadx(ind):
    n = ind["n"]
    signals = []
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
            signals.append((i, 1))
        elif ind["c"][i] < l3:
            signals.append((i, -1))
    return signals


def backtest_with_mfe(signals, ind, sl_mult, tp_mult, max_bars, fee=2*TAKER_FEE):
    """Backtest tracking Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE)."""
    c, h, lo, atr = ind["c"], ind["h"], ind["l"], ind["atr"]
    n = ind["n"]
    trades = []

    for idx, direction in signals:
        ep = c[idx]
        ea = atr[idx]
        if ea <= 0 or ep <= 0 or np.isnan(ea):
            continue
        is_long = direction > 0
        sl_dist = sl_mult * ea
        tp_dist = tp_mult * ea

        exit_reason = "timeout"
        exit_price = ep
        bars_held = 0
        mfe = 0.0  # max favorable excursion (best unrealized profit)
        mae = 0.0  # max adverse excursion (worst unrealized loss)
        mfe_bar = 0
        mae_bar = 0

        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            bars_held = j - idx
            if is_long:
                unrealized_best = (h[j] - ep) / ep
                unrealized_worst = (lo[j] - ep) / ep
            else:
                unrealized_best = (ep - lo[j]) / ep
                unrealized_worst = (ep - h[j]) / ep

            if unrealized_best > mfe:
                mfe = unrealized_best
                mfe_bar = bars_held
            if unrealized_worst < mae:
                mae = unrealized_worst
                mae_bar = bars_held

            if is_long:
                if lo[j] <= ep - sl_dist:
                    exit_price = ep - sl_dist
                    exit_reason = "SL"
                    break
                if h[j] >= ep + tp_dist:
                    exit_price = ep + tp_dist
                    exit_reason = "TP"
                    break
            else:
                if h[j] >= ep + sl_dist:
                    exit_price = ep + sl_dist
                    exit_reason = "SL"
                    break
                if lo[j] <= ep - tp_dist:
                    exit_price = ep - tp_dist
                    exit_reason = "TP"
                    break
        else:
            ei = min(idx + max_bars, n - 1)
            exit_price = c[ei]
            bars_held = ei - idx

        if is_long:
            pnl = (exit_price - ep) / ep - fee
        else:
            pnl = (ep - exit_price) / ep - fee

        sl_pct = sl_dist / ep
        tp_pct = tp_dist / ep

        trades.append({
            "idx": idx, "dir": "LONG" if is_long else "SHORT",
            "entry": ep, "exit": exit_price, "exit_reason": exit_reason,
            "pnl": pnl, "sl_pct": sl_pct, "tp_pct": tp_pct,
            "mfe": mfe, "mae": mae, "mfe_bar": mfe_bar, "mae_bar": mae_bar,
            "bars_held": bars_held,
        })
    return trades


def backtest_trailing(signals, ind, sl_mult, tp_mult, max_bars, trail_config, fee=2*MAKER_FEE):
    """Backtest with trailing SL logic.

    trail_config: dict with:
      - be_trigger: fraction of TP to lock breakeven (e.g. 0.3 = 30% of TP distance)
      - be_offset: how much above entry to lock BE (e.g. 0.1 = lock at +0.1x ATR from entry)
      - trail_trigger: fraction of TP to start trailing (e.g. 0.5)
      - trail_step: trail locks at (current_high - trail_step * ATR)
    """
    c, h, lo, atr = ind["c"], ind["h"], ind["l"], ind["atr"]
    n = ind["n"]
    trades = []

    be_trigger = trail_config.get("be_trigger", 0.4)
    be_offset = trail_config.get("be_offset", 0.1)
    trail_trigger = trail_config.get("trail_trigger", 0.6)
    trail_dist = trail_config.get("trail_dist", 0.4)

    for idx, direction in signals:
        ep = c[idx]
        ea = atr[idx]
        if ea <= 0 or ep <= 0 or np.isnan(ea):
            continue
        is_long = direction > 0
        sl_dist = sl_mult * ea
        tp_dist = tp_mult * ea

        # Trailing state
        current_sl = sl_dist
        be_locked = False
        trailing = False
        highest_profit = 0.0
        mfe = 0.0

        exit_reason = "timeout"
        exit_price = ep
        bars_held = 0

        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            bars_held = j - idx

            if is_long:
                current_profit_pct = (h[j] - ep) / ep
                bar_low_profit = (lo[j] - ep) / ep
            else:
                current_profit_pct = (ep - lo[j]) / ep
                bar_low_profit = (ep - h[j]) / ep

            if current_profit_pct > highest_profit:
                highest_profit = current_profit_pct
            if current_profit_pct > mfe:
                mfe = current_profit_pct

            # Phase 1: Check if BE should be locked
            be_level_pct = be_trigger * tp_dist / ep
            if not be_locked and highest_profit >= be_level_pct:
                be_locked = True
                current_sl = -(be_offset * ea / ep)  # negative = profit side

            # Phase 2: Check if trailing should start
            trail_level_pct = trail_trigger * tp_dist / ep
            if not trailing and highest_profit >= trail_level_pct:
                trailing = True

            # Phase 3: Trail the SL
            if trailing:
                trail_sl_pct = highest_profit - (trail_dist * tp_dist / ep)
                if trail_sl_pct > current_sl:
                    current_sl = -trail_sl_pct  # negative = profit side

            # Check SL hit
            if is_long:
                if be_locked or trailing:
                    sl_price = ep * (1 - current_sl) if current_sl < 0 else ep - current_sl * ep
                    # Recompute: current_sl is stored as negative when in profit
                    actual_sl_price = ep + (-current_sl) * ep if current_sl < 0 else ep - sl_dist
                else:
                    actual_sl_price = ep - sl_dist

                if lo[j] <= actual_sl_price:
                    exit_price = actual_sl_price
                    if trailing:
                        exit_reason = "trail_SL"
                    elif be_locked:
                        exit_reason = "BE_SL"
                    else:
                        exit_reason = "SL"
                    break
                if h[j] >= ep + tp_dist:
                    exit_price = ep + tp_dist
                    exit_reason = "TP"
                    break
            else:
                if be_locked or trailing:
                    actual_sl_price = ep - (-current_sl) * ep if current_sl < 0 else ep + sl_dist
                else:
                    actual_sl_price = ep + sl_dist

                if h[j] >= actual_sl_price:
                    exit_price = actual_sl_price
                    if trailing:
                        exit_reason = "trail_SL"
                    elif be_locked:
                        exit_reason = "BE_SL"
                    else:
                        exit_reason = "SL"
                    break
                if lo[j] <= ep - tp_dist:
                    exit_price = ep - tp_dist
                    exit_reason = "TP"
                    break
        else:
            ei = min(idx + max_bars, n - 1)
            exit_price = c[ei]
            bars_held = ei - idx

        if is_long:
            pnl = (exit_price - ep) / ep - fee
        else:
            pnl = (ep - exit_price) / ep - fee

        trades.append({
            "idx": idx, "dir": "LONG" if is_long else "SHORT",
            "entry": ep, "exit": exit_price, "exit_reason": exit_reason,
            "pnl": pnl, "mfe": mfe, "bars_held": bars_held,
            "sl_pct": sl_dist / ep, "tp_pct": tp_dist / ep,
        })
    return trades


def sl_sweep(signals, ind, tp_mult, max_bars):
    """Test multiple SL levels."""
    results = []
    for sl in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        trades = backtest_with_mfe(signals, ind, sl, tp_mult, max_bars)
        if not trades:
            continue
        pnls = np.array([t["pnl"] for t in trades])
        n_trades = len(pnls)
        wr = (pnls > 0).sum() / n_trades
        pf = abs(pnls[pnls > 0].sum() / pnls[pnls <= 0].sum()) if pnls[pnls <= 0].sum() != 0 else 99
        tp_count = sum(1 for t in trades if t["exit_reason"] == "TP")
        sl_count = sum(1 for t in trades if t["exit_reason"] == "SL")
        results.append({
            "sl": sl, "trades": n_trades, "wr": wr, "pf": pf,
            "pnl": pnls.sum(), "tp_rate": tp_count / n_trades,
            "sl_rate": sl_count / n_trades,
        })
    return results


def analyze_mfe_mae(trades):
    """Analyze MFE/MAE patterns: how many trades were profitable but ended as loss."""
    if not trades:
        return {}

    sl_trades = [t for t in trades if t["exit_reason"] == "SL"]
    tp_trades = [t for t in trades if t["exit_reason"] == "TP"]
    timeout_trades = [t for t in trades if t["exit_reason"] == "timeout"]

    # Trades that hit SL but had positive MFE (came back first)
    sl_with_profit = [t for t in sl_trades if t["mfe"] > 0.002]  # > 0.2% profit at some point
    sl_mfe_half_tp = [t for t in sl_trades if t["mfe"] >= t["tp_pct"] * 0.3]  # reached 30% of TP
    sl_mfe_most_tp = [t for t in sl_trades if t["mfe"] >= t["tp_pct"] * 0.6]  # reached 60% of TP

    # Timeout trades that were profitable at some point
    timeout_positive_mfe = [t for t in timeout_trades if t["mfe"] > 0.003]
    timeout_half_tp = [t for t in timeout_trades if t["mfe"] >= t["tp_pct"] * 0.5]

    all_mfe = [t["mfe"] for t in trades]
    all_mae = [t["mae"] for t in trades]
    sl_mfe = [t["mfe"] for t in sl_trades] if sl_trades else [0]

    return {
        "total": len(trades),
        "tp_count": len(tp_trades),
        "sl_count": len(sl_trades),
        "timeout_count": len(timeout_trades),
        "sl_had_profit": len(sl_with_profit),
        "sl_reached_30pct_tp": len(sl_mfe_half_tp),
        "sl_reached_60pct_tp": len(sl_mfe_most_tp),
        "timeout_had_profit": len(timeout_positive_mfe),
        "timeout_reached_50pct_tp": len(timeout_half_tp),
        "avg_mfe_all": np.mean(all_mfe),
        "avg_mae_all": np.mean(all_mae),
        "avg_mfe_sl_trades": np.mean(sl_mfe),
        "avg_mfe_bar_sl": np.mean([t["mfe_bar"] for t in sl_trades]) if sl_trades else 0,
        "pct_sl_comeback": len(sl_with_profit) / max(len(sl_trades), 1),
        "pct_sl_reached_30tp": len(sl_mfe_half_tp) / max(len(sl_trades), 1),
        "pct_sl_reached_60tp": len(sl_mfe_most_tp) / max(len(sl_trades), 1),
    }


def main():
    print("=" * 95)
    print("STOPLOSS OPTIMIZATION + TRAILING ANALYSIS")
    print("Period: 2026-01-01 to 2026-05-21 (142 days)")
    print("=" * 95)

    strategy_config = {
        "regime_adaptive": {"tp": 11.0, "max_bars": 192, "current_sl": 7.0},
        "volume_spike_rev": {"tp": 5.5, "max_bars": 96, "current_sl": 3.5},
        "cb_adx_breakout": {"tp": 5.0, "max_bars": 96, "current_sl": 3.0},
    }

    trail_configs = [
        {"name": "Conservative", "be_trigger": 0.4, "be_offset": 0.2, "trail_trigger": 0.7, "trail_dist": 0.5},
        {"name": "Moderate", "be_trigger": 0.3, "be_offset": 0.1, "trail_trigger": 0.5, "trail_dist": 0.4},
        {"name": "Aggressive", "be_trigger": 0.25, "be_offset": 0.05, "trail_trigger": 0.4, "trail_dist": 0.3},
        {"name": "Tight", "be_trigger": 0.2, "be_offset": 0.0, "trail_trigger": 0.35, "trail_dist": 0.25},
    ]

    for pair in PAIRS:
        df = load_15m(pair)
        ind = compute_indicators(df)
        print(f"\n\n{'#' * 95}")
        print(f"# {pair}")
        print(f"{'#' * 95}")

        sigs = {
            "regime_adaptive": generate_signals_regime(ind),
            "volume_spike_rev": generate_signals_volspike(ind),
            "cb_adx_breakout": generate_signals_cbadx(ind),
        }

        for strat_name, sig_list in sigs.items():
            cfg = strategy_config[strat_name]
            tp = cfg["tp"]
            max_bars = cfg["max_bars"]
            current_sl = cfg["current_sl"]

            print(f"\n  {'=' * 85}")
            print(f"  {strat_name.upper()} | Current SL={current_sl}x ATR | TP={tp}x ATR")
            print(f"  {'=' * 85}")

            # ── 1. SL SWEEP ──
            print(f"\n  [1] SL SENSITIVITY SWEEP (TP fixed at {tp}x ATR)")
            print(f"  {'SL':<6} {'Trades':<8} {'WR':<8} {'PF':<7} {'PnL':<12} {'TP%':<8} {'SL%':<8} {'Note'}")
            print(f"  {'-' * 80}")
            sweep = sl_sweep(sig_list, ind, tp, max_bars)
            best_pnl = -999
            best_sl = current_sl
            for r in sweep:
                marker = ""
                if r["sl"] == current_sl:
                    marker = " <-- CURRENT"
                if r["pnl"] > best_pnl:
                    best_pnl = r["pnl"]
                    best_sl = r["sl"]
                print(f"  {r['sl']:<6.1f} {r['trades']:<8} {r['wr']*100:<7.1f}% {r['pf']:<7.2f} "
                      f"{r['pnl']*100:<+11.2f}% {r['tp_rate']*100:<7.0f}% {r['sl_rate']*100:<7.0f}%{marker}")

            best_r = next((r for r in sweep if r["sl"] == best_sl), None)
            curr_r = next((r for r in sweep if r["sl"] == current_sl), None)
            if best_r and curr_r:
                print(f"\n  BEST SL: {best_sl}x ATR -> PnL {best_r['pnl']*100:+.2f}% "
                      f"(vs current {current_sl}x: {curr_r['pnl']*100:+.2f}%, diff: {(best_r['pnl']-curr_r['pnl'])*100:+.2f}%)")

            # ── 2. MFE/MAE ANALYSIS ──
            trades_mfe = backtest_with_mfe(sig_list, ind, current_sl, tp, max_bars)
            mfe_stats = analyze_mfe_mae(trades_mfe)

            print(f"\n  [2] MFE/MAE ANALYSIS (SL={current_sl}x)")
            print(f"  {'-' * 80}")
            print(f"  Total trades: {mfe_stats['total']} | TP: {mfe_stats['tp_count']} | "
                  f"SL: {mfe_stats['sl_count']} | Timeout: {mfe_stats['timeout_count']}")
            print(f"  ")
            print(f"  Avg MFE (all trades): {mfe_stats['avg_mfe_all']*100:.3f}%")
            print(f"  Avg MAE (all trades): {mfe_stats['avg_mae_all']*100:.3f}%")
            print(f"  Avg MFE of SL'd trades: {mfe_stats['avg_mfe_sl_trades']*100:.3f}%")
            print(f"  Avg bar at MFE (SL trades): {mfe_stats['avg_mfe_bar_sl']:.0f}")
            print(f"  ")
            print(f"  SL trades that HAD profit (>0.2%): {mfe_stats['sl_had_profit']}/{mfe_stats['sl_count']} "
                  f"({mfe_stats['pct_sl_comeback']*100:.0f}%)")
            print(f"  SL trades that reached 30% of TP: {mfe_stats['sl_reached_30pct_tp']}/{mfe_stats['sl_count']} "
                  f"({mfe_stats['pct_sl_reached_30tp']*100:.0f}%)")
            print(f"  SL trades that reached 60% of TP: {mfe_stats['sl_reached_60pct_tp']}/{mfe_stats['sl_count']} "
                  f"({mfe_stats['pct_sl_reached_60tp']*100:.0f}%)")
            print(f"  ")
            print(f"  Timeout trades that had profit: {mfe_stats['timeout_had_profit']}/{mfe_stats['timeout_count']}")
            print(f"  Timeout trades that reached 50% TP: {mfe_stats['timeout_reached_50pct_tp']}/{mfe_stats['timeout_count']}")

            salvageable = mfe_stats['sl_reached_30pct_tp'] + mfe_stats['timeout_reached_50pct_tp']
            print(f"\n  SALVAGEABLE TRADES (trailing could save): {salvageable} "
                  f"({salvageable/max(mfe_stats['total'],1)*100:.0f}% of all trades)")

            # ── 3. TRAILING SL TEST ──
            print(f"\n  [3] TRAILING SL BACKTEST")
            print(f"  {'Config':<14} {'Trades':<7} {'WR':<7} {'PF':<7} {'PnL':<11} "
                  f"{'TP':<5} {'SL':<5} {'BE':<5} {'Trail':<6} {'Timeout':<7}")
            print(f"  {'-' * 85}")

            # Baseline (no trailing)
            baseline_trades = backtest_with_mfe(sig_list, ind, current_sl, tp, max_bars)
            if baseline_trades:
                bp = np.array([t["pnl"] for t in baseline_trades])
                bwr = (bp > 0).sum() / len(bp)
                bpf = abs(bp[bp > 0].sum() / bp[bp <= 0].sum()) if bp[bp <= 0].sum() != 0 else 99
                btp = sum(1 for t in baseline_trades if t["exit_reason"] == "TP")
                bsl = sum(1 for t in baseline_trades if t["exit_reason"] == "SL")
                bto = sum(1 for t in baseline_trades if t["exit_reason"] == "timeout")
                print(f"  {'No Trail':<14} {len(bp):<7} {bwr*100:<6.1f}% {bpf:<7.2f} "
                      f"{bp.sum()*100:<+10.2f}% {btp:<5} {bsl:<5} {'--':<5} {'--':<6} {bto:<7}")

            for tcfg in trail_configs:
                trail_trades = backtest_trailing(sig_list, ind, current_sl, tp, max_bars, tcfg)
                if not trail_trades:
                    continue
                tp_arr = np.array([t["pnl"] for t in trail_trades])
                wr = (tp_arr > 0).sum() / len(tp_arr)
                pf = abs(tp_arr[tp_arr > 0].sum() / tp_arr[tp_arr <= 0].sum()) if tp_arr[tp_arr <= 0].sum() != 0 else 99
                tp_c = sum(1 for t in trail_trades if t["exit_reason"] == "TP")
                sl_c = sum(1 for t in trail_trades if t["exit_reason"] == "SL")
                be_c = sum(1 for t in trail_trades if t["exit_reason"] == "BE_SL")
                tr_c = sum(1 for t in trail_trades if t["exit_reason"] == "trail_SL")
                to_c = sum(1 for t in trail_trades if t["exit_reason"] == "timeout")
                print(f"  {tcfg['name']:<14} {len(tp_arr):<7} {wr*100:<6.1f}% {pf:<7.2f} "
                      f"{tp_arr.sum()*100:<+10.2f}% {tp_c:<5} {sl_c:<5} {be_c:<5} {tr_c:<6} {to_c:<7}")

            # ── 4. COMBINED: BEST SL + BEST TRAILING ──
            print(f"\n  [4] COMBINED OPTIMIZATION: Best SL + Best Trail")
            print(f"  {'-' * 85}")

            # Test best SL with each trail config
            print(f"  Testing SL={best_sl}x + trailing configs:")
            print(f"  {'Config':<14} {'Trades':<7} {'WR':<7} {'PF':<7} {'PnL':<11} "
                  f"{'TP':<5} {'SL':<5} {'BE':<5} {'Trail':<6}")
            print(f"  {'-' * 70}")

            best_combined_pnl = -999
            best_combined_name = ""
            for tcfg in trail_configs:
                trail_trades = backtest_trailing(sig_list, ind, best_sl, tp, max_bars, tcfg)
                if not trail_trades:
                    continue
                tp_arr = np.array([t["pnl"] for t in trail_trades])
                wr = (tp_arr > 0).sum() / len(tp_arr)
                pf = abs(tp_arr[tp_arr > 0].sum() / tp_arr[tp_arr <= 0].sum()) if tp_arr[tp_arr <= 0].sum() != 0 else 99
                tp_c = sum(1 for t in trail_trades if t["exit_reason"] == "TP")
                sl_c = sum(1 for t in trail_trades if t["exit_reason"] == "SL")
                be_c = sum(1 for t in trail_trades if t["exit_reason"] == "BE_SL")
                tr_c = sum(1 for t in trail_trades if t["exit_reason"] == "trail_SL")
                total_pnl = tp_arr.sum()
                if total_pnl > best_combined_pnl:
                    best_combined_pnl = total_pnl
                    best_combined_name = tcfg["name"]
                print(f"  {tcfg['name']:<14} {len(tp_arr):<7} {wr*100:<6.1f}% {pf:<7.2f} "
                      f"{total_pnl*100:<+10.2f}% {tp_c:<5} {sl_c:<5} {be_c:<5} {tr_c:<6}")

            # Also test no-trail with best SL
            no_trail_best = backtest_with_mfe(sig_list, ind, best_sl, tp, max_bars)
            if no_trail_best:
                nt_pnls = np.array([t["pnl"] for t in no_trail_best])
                nt_pnl = nt_pnls.sum()
                nt_wr = (nt_pnls > 0).sum() / len(nt_pnls)
                nt_pf = abs(nt_pnls[nt_pnls > 0].sum() / nt_pnls[nt_pnls <= 0].sum()) if nt_pnls[nt_pnls <= 0].sum() != 0 else 99
                print(f"  {'No Trail':<14} {len(nt_pnls):<7} {nt_wr*100:<6.1f}% {nt_pf:<7.2f} "
                      f"{nt_pnl*100:<+10.2f}%")

            print(f"\n  >>> RECOMMENDATION: SL={best_sl}x ATR + {best_combined_name} trailing")
            if curr_r:
                print(f"  >>> Improvement: {curr_r['pnl']*100:+.2f}% -> {best_combined_pnl*100:+.2f}% "
                      f"({(best_combined_pnl - curr_r['pnl'])*100:+.2f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 95}")
    print("FINAL RECOMMENDATIONS")
    print(f"{'=' * 95}")
    print(f"\n  {'Strategy':<20} {'Pair':<18} {'Current SL':<12} {'Best SL':<10} "
          f"{'Trail':<14} {'PnL Gain'}")
    print(f"  {'-' * 90}")

    for pair in PAIRS:
        df = load_15m(pair)
        ind = compute_indicators(df)
        sig_map = {
            "regime_adaptive": generate_signals_regime(ind),
            "volume_spike_rev": generate_signals_volspike(ind),
            "cb_adx_breakout": generate_signals_cbadx(ind),
        }
        for strat_name, sig_list in sig_map.items():
            cfg = strategy_config[strat_name]
            tp = cfg["tp"]
            max_bars = cfg["max_bars"]
            current_sl = cfg["current_sl"]

            # Find best SL
            sweep = sl_sweep(sig_list, ind, tp, max_bars)
            best_sl_val = current_sl
            best_pnl_val = -999
            for r in sweep:
                if r["pnl"] > best_pnl_val:
                    best_pnl_val = r["pnl"]
                    best_sl_val = r["sl"]

            curr_pnl = next((r["pnl"] for r in sweep if r["sl"] == current_sl), 0)

            # Find best trail at best SL
            best_trail_name = "None"
            best_trail_pnl = best_pnl_val
            for tcfg in trail_configs:
                tt = backtest_trailing(sig_list, ind, best_sl_val, tp, max_bars, tcfg)
                if tt:
                    tp_arr = np.array([t["pnl"] for t in tt])
                    if tp_arr.sum() > best_trail_pnl:
                        best_trail_pnl = tp_arr.sum()
                        best_trail_name = tcfg["name"]

            gain = best_trail_pnl - curr_pnl
            print(f"  {strat_name:<20} {pair:<18} {current_sl:<12.1f} {best_sl_val:<10.1f} "
                  f"{best_trail_name:<14} {gain*100:+.2f}%")


if __name__ == "__main__":
    main()

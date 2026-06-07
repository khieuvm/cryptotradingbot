"""Backtest with optimized entry strategies per pair per strategy.

Reports: WR, PnL, SL%, TP%, entry method, real entry price vs signal price,
avg improvement, per strategy per pair breakdown.

Entry methods:
- market: enter at close of signal candle (taker fee)
- limit_atr: limit order at close +/- X*ATR offset (maker fee, fill window)
- ema8_retest: wait for price to retest EMA8 after breakout (maker fee)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from datetime import datetime

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

    bb = ta.bbands(df["close"], length=20, std=2.0)
    bb_upper = bb.iloc[:, 0].values if bb is not None else c.copy()
    bb_lower = bb.iloc[:, 2].values if bb is not None else c.copy()

    body = np.abs(c - o)
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(c, o) - lo

    return {
        "c": c, "h": h, "l": lo, "o": o, "v": v, "n": n,
        "adx": adx, "dip": dip, "dim": dim, "rsi": rsi, "atr": atr,
        "e8": e8, "e21": e21, "e60": e60, "vr": vr,
        "bb_upper": bb_upper, "bb_lower": bb_lower,
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


def backtest_market(signals, ind, sl_mult, tp_mult, max_bars):
    """Baseline: market entry at close, taker fees."""
    c, h, lo, atr = ind["c"], ind["h"], ind["l"], ind["atr"]
    n = ind["n"]
    fee = 2 * TAKER_FEE
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

        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            bars_held = j - idx
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
            "signal_idx": idx, "direction": "LONG" if is_long else "SHORT",
            "signal_price": ep, "entry_price": ep,
            "entry_improvement": 0.0,
            "exit_price": exit_price, "exit_reason": exit_reason,
            "pnl": pnl, "sl_pct": sl_pct, "tp_pct": tp_pct,
            "bars_held": bars_held, "fee_type": "taker",
            "entry_method": "market",
        })
    return trades


def backtest_limit_atr(signals, ind, sl_mult, tp_mult, max_bars, atr_offset, fill_window=3):
    """Limit entry at close +/- atr_offset*ATR, maker fees."""
    c, h, lo, atr = ind["c"], ind["h"], ind["l"], ind["atr"]
    n = ind["n"]
    fee = 2 * MAKER_FEE
    trades = []

    for idx, direction in signals:
        sp = c[idx]
        ea = atr[idx]
        if ea <= 0 or sp <= 0 or np.isnan(ea):
            continue
        is_long = direction > 0

        if atr_offset == 0:
            ep = sp
            filled = True
        else:
            if is_long:
                limit_price = sp - atr_offset * ea
            else:
                limit_price = sp + atr_offset * ea

            filled = False
            fill_bar = idx
            for j in range(idx + 1, min(idx + fill_window + 1, n)):
                if is_long and lo[j] <= limit_price:
                    filled = True
                    fill_bar = j
                    ep = limit_price
                    break
                elif not is_long and h[j] >= limit_price:
                    filled = True
                    fill_bar = j
                    ep = limit_price
                    break

            if not filled:
                continue

        sl_dist = sl_mult * ea
        tp_dist = tp_mult * ea
        start_bar = fill_bar + 1 if atr_offset > 0 else idx + 1

        exit_reason = "timeout"
        exit_price = ep
        bars_held = 0

        for j in range(start_bar, min(idx + max_bars + 1, n)):
            bars_held = j - (fill_bar if atr_offset > 0 else idx)
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
            bars_held = ei - (fill_bar if atr_offset > 0 else idx)

        if is_long:
            pnl = (exit_price - ep) / ep - fee
        else:
            pnl = (ep - exit_price) / ep - fee

        improvement = abs(sp - ep) / sp if sp != ep else 0.0
        sl_pct = sl_dist / ep
        tp_pct = tp_dist / ep

        trades.append({
            "signal_idx": idx, "direction": "LONG" if is_long else "SHORT",
            "signal_price": sp, "entry_price": ep,
            "entry_improvement": improvement,
            "exit_price": exit_price, "exit_reason": exit_reason,
            "pnl": pnl, "sl_pct": sl_pct, "tp_pct": tp_pct,
            "bars_held": bars_held, "fee_type": "maker",
            "entry_method": f"limit_{atr_offset}xATR",
        })
    return trades


def backtest_ema8_retest(signals, ind, sl_mult, tp_mult, max_bars, fill_window=4):
    """EMA8 retest entry: after breakout signal, wait for pullback to EMA8."""
    c, h, lo, atr, e8 = ind["c"], ind["h"], ind["l"], ind["atr"], ind["e8"]
    n = ind["n"]
    fee = 2 * MAKER_FEE
    trades = []

    for idx, direction in signals:
        sp = c[idx]
        ea = atr[idx]
        if ea <= 0 or sp <= 0 or np.isnan(ea):
            continue
        is_long = direction > 0

        filled = False
        fill_bar = idx
        ep = sp

        for j in range(idx + 1, min(idx + fill_window + 1, n)):
            ema8_val = e8[j]
            if is_long and lo[j] <= ema8_val:
                filled = True
                fill_bar = j
                ep = ema8_val
                break
            elif not is_long and h[j] >= ema8_val:
                filled = True
                fill_bar = j
                ep = ema8_val
                break

        if not filled:
            continue

        sl_dist = sl_mult * ea
        tp_dist = tp_mult * ea

        exit_reason = "timeout"
        exit_price = ep
        bars_held = 0

        for j in range(fill_bar + 1, min(idx + max_bars + 1, n)):
            bars_held = j - fill_bar
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
            bars_held = ei - fill_bar

        if is_long:
            pnl = (exit_price - ep) / ep - fee
        else:
            pnl = (ep - exit_price) / ep - fee

        improvement = abs(sp - ep) / sp
        sl_pct = sl_dist / ep
        tp_pct = tp_dist / ep

        trades.append({
            "signal_idx": idx, "direction": "LONG" if is_long else "SHORT",
            "signal_price": sp, "entry_price": ep,
            "entry_improvement": improvement,
            "exit_price": exit_price, "exit_reason": exit_reason,
            "pnl": pnl, "sl_pct": sl_pct, "tp_pct": tp_pct,
            "bars_held": bars_held, "fee_type": "maker",
            "entry_method": "ema8_retest",
        })
    return trades


def compute_stats(trades):
    if not trades:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    wins = (pnls > 0).sum()
    wr = wins / n
    total_pnl = pnls.sum()
    avg_pnl = pnls.mean()
    avg_win = pnls[pnls > 0].mean() if wins > 0 else 0
    avg_loss = pnls[pnls <= 0].mean() if (n - wins) > 0 else 0
    pf = abs(pnls[pnls > 0].sum() / pnls[pnls <= 0].sum()) if pnls[pnls <= 0].sum() != 0 else 99.0

    tp_count = sum(1 for t in trades if t["exit_reason"] == "TP")
    sl_count = sum(1 for t in trades if t["exit_reason"] == "SL")
    timeout_count = sum(1 for t in trades if t["exit_reason"] == "timeout")

    sl_pcts = [t["sl_pct"] for t in trades]
    tp_pcts = [t["tp_pct"] for t in trades]
    improvements = [t["entry_improvement"] for t in trades]
    bars_held = [t["bars_held"] for t in trades]

    longs = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]
    long_pnl = sum(t["pnl"] for t in longs)
    short_pnl = sum(t["pnl"] for t in shorts)
    long_wr = sum(1 for t in longs if t["pnl"] > 0) / max(len(longs), 1)
    short_wr = sum(1 for t in shorts if t["pnl"] > 0) / max(len(shorts), 1)

    # Max drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = dd.max() if len(dd) > 0 else 0

    return {
        "trades": n, "wins": wins, "wr": wr,
        "total_pnl": total_pnl, "avg_pnl": avg_pnl,
        "avg_win": avg_win, "avg_loss": avg_loss, "pf": pf,
        "tp_count": tp_count, "sl_count": sl_count, "timeout_count": timeout_count,
        "tp_rate": tp_count / n, "sl_rate": sl_count / n,
        "avg_sl_pct": np.mean(sl_pcts), "avg_tp_pct": np.mean(tp_pcts),
        "avg_improvement": np.mean(improvements),
        "avg_bars_held": np.mean(bars_held),
        "long_count": len(longs), "short_count": len(shorts),
        "long_pnl": long_pnl, "short_pnl": short_pnl,
        "long_wr": long_wr, "short_wr": short_wr,
        "max_dd": max_dd,
        "fee_type": trades[0]["fee_type"] if trades else "taker",
        "entry_method": trades[0]["entry_method"] if trades else "market",
    }


def print_report(pair, strategy_name, baseline_stats, optimized_stats, trades):
    if baseline_stats is None and optimized_stats is None:
        return

    print(f"\n  {'=' * 80}")
    print(f"  {strategy_name.upper()} | {pair}")
    print(f"  {'=' * 80}")

    for label, stats in [("BASELINE (market)", baseline_stats), ("OPTIMIZED", optimized_stats)]:
        if stats is None:
            continue
        marker = " <<<" if label == "OPTIMIZED" else ""
        print(f"\n  [{label}] Entry: {stats['entry_method']} | Fee: {stats['fee_type']}{marker}")
        print(f"  {'-' * 75}")
        print(f"  Trades: {stats['trades']} | WR: {stats['wr']*100:.1f}% | PF: {stats['pf']:.2f}")
        print(f"  PnL Total: {stats['total_pnl']*100:+.2f}% | Avg PnL/trade: {stats['avg_pnl']*100:+.3f}%")
        print(f"  Avg Win: {stats['avg_win']*100:+.3f}% | Avg Loss: {stats['avg_loss']*100:.3f}%")
        print(f"  Max DD: {stats['max_dd']*100:.2f}%")
        print(f"  ")
        print(f"  Exit Breakdown: TP={stats['tp_count']} ({stats['tp_rate']*100:.0f}%) | "
              f"SL={stats['sl_count']} ({stats['sl_rate']*100:.0f}%) | "
              f"Timeout={stats['timeout_count']}")
        print(f"  SL Distance: {stats['avg_sl_pct']*100:.2f}% | TP Distance: {stats['avg_tp_pct']*100:.2f}%")
        print(f"  Avg Entry Improvement: {stats['avg_improvement']*100:.3f}%")
        print(f"  Avg Bars Held: {stats['avg_bars_held']:.1f}")
        print(f"  ")
        print(f"  LONG:  {stats['long_count']} trades | WR={stats['long_wr']*100:.1f}% | PnL={stats['long_pnl']*100:+.2f}%")
        print(f"  SHORT: {stats['short_count']} trades | WR={stats['short_wr']*100:.1f}% | PnL={stats['short_pnl']*100:+.2f}%")

    if baseline_stats and optimized_stats:
        print(f"\n  {'~' * 75}")
        print(f"  IMPROVEMENT SUMMARY:")
        pnl_diff = optimized_stats['total_pnl'] - baseline_stats['total_pnl']
        wr_diff = optimized_stats['wr'] - baseline_stats['wr']
        pf_diff = optimized_stats['pf'] - baseline_stats['pf']
        fill_rate = optimized_stats['trades'] / max(baseline_stats['trades'], 1)
        print(f"  PnL:     {baseline_stats['total_pnl']*100:+.2f}% -> {optimized_stats['total_pnl']*100:+.2f}%  ({pnl_diff*100:+.2f}%)")
        print(f"  WR:      {baseline_stats['wr']*100:.1f}% -> {optimized_stats['wr']*100:.1f}%  ({wr_diff*100:+.1f}pp)")
        print(f"  PF:      {baseline_stats['pf']:.2f} -> {optimized_stats['pf']:.2f}  ({pf_diff:+.2f})")
        print(f"  Trades:  {baseline_stats['trades']} -> {optimized_stats['trades']}  (fill rate: {fill_rate*100:.0f}%)")
        print(f"  Fee:     {baseline_stats['fee_type']}(0.10% RT) -> {optimized_stats['fee_type']}(0.04% RT)")


def print_sample_trades(trades, n_samples=5):
    if not trades:
        return
    print(f"\n  Sample Trades (first {min(n_samples, len(trades))}):")
    print(f"  {'Dir':<6} {'Signal$':<10} {'Entry$':<10} {'Improv':<8} {'Exit$':<10} {'Reason':<8} {'PnL':<10} {'Bars'}")
    print(f"  {'-' * 75}")
    for t in trades[:n_samples]:
        print(f"  {t['direction']:<6} {t['signal_price']:<10.2f} {t['entry_price']:<10.2f} "
              f"{t['entry_improvement']*100:+.3f}% {t['exit_price']:<10.2f} "
              f"{t['exit_reason']:<8} {t['pnl']*100:+.3f}%  {t['bars_held']}")


def main():
    print("=" * 90)
    print("OPTIMIZED ENTRY BACKTEST REPORT")
    print(f"Period: 2026-01-01 to 2026-05-21 (142 days)")
    print(f"Pairs: {', '.join(PAIRS)}")
    print(f"Strategies: regime_adaptive, volume_spike_rev, cb_adx_breakout")
    print("=" * 90)

    # Config: per pair per strategy optimized entries
    entry_config = {
        "regime_adaptive": {
            "SOL_USDT_USDT": {"method": "market", "atr_offset": 0.0},
            "SPX_USDT_USDT": {"method": "limit_atr", "atr_offset": 0.1, "fill_window": 3},
        },
        "volume_spike_rev": {
            "SOL_USDT_USDT": {"method": "market", "atr_offset": 0.0},
            "SPX_USDT_USDT": {"method": "market", "atr_offset": 0.0},
        },
        "cb_adx_breakout": {
            "SOL_USDT_USDT": {"method": "ema8_retest", "fill_window": 4},
            "SPX_USDT_USDT": {"method": "market", "atr_offset": 0.0},
        },
    }

    sl_tp_config = {
        "regime_adaptive": {"sl": 7.0, "tp": 11.0, "max_bars": 192},
        "volume_spike_rev": {"sl": 3.5, "tp": 5.5, "max_bars": 96},
        "cb_adx_breakout": {"sl": 3.0, "tp": 5.0, "max_bars": 96},
    }

    all_results = []

    for pair in PAIRS:
        df = load_15m(pair)
        ind = compute_indicators(df)
        print(f"\n\n{'#' * 90}")
        print(f"# PAIR: {pair} | Bars: {ind['n']} | Period: 142 days")
        print(f"{'#' * 90}")

        # Generate all signals
        sigs_regime = generate_signals_regime(ind)
        sigs_volspike = generate_signals_volspike(ind)
        sigs_cbadx = generate_signals_cbadx(ind)

        strategy_signals = {
            "regime_adaptive": sigs_regime,
            "volume_spike_rev": sigs_volspike,
            "cb_adx_breakout": sigs_cbadx,
        }

        for strat_name, sigs in strategy_signals.items():
            cfg = sl_tp_config[strat_name]
            ecfg = entry_config[strat_name][pair]
            sl, tp, max_bars = cfg["sl"], cfg["tp"], cfg["max_bars"]

            # Baseline: market entry
            baseline_trades = backtest_market(sigs, ind, sl, tp, max_bars)
            baseline_stats = compute_stats(baseline_trades)

            # Optimized entry
            method = ecfg["method"]
            if method == "market":
                opt_trades = baseline_trades
                opt_stats = baseline_stats
            elif method == "limit_atr":
                offset = ecfg.get("atr_offset", 0.1)
                fw = ecfg.get("fill_window", 3)
                opt_trades = backtest_limit_atr(sigs, ind, sl, tp, max_bars, offset, fw)
                opt_stats = compute_stats(opt_trades)
            elif method == "ema8_retest":
                fw = ecfg.get("fill_window", 4)
                opt_trades = backtest_ema8_retest(sigs, ind, sl, tp, max_bars, fw)
                opt_stats = compute_stats(opt_trades)

            print_report(pair, strat_name, baseline_stats, opt_stats, opt_trades)
            print_sample_trades(opt_trades)

            if opt_stats:
                all_results.append({
                    "pair": pair, "strategy": strat_name,
                    "method": method, **opt_stats,
                })

    # Final summary table
    print(f"\n\n{'=' * 90}")
    print("FINAL SUMMARY — ALL STRATEGIES x ALL PAIRS (OPTIMIZED ENTRIES)")
    print(f"{'=' * 90}")
    print(f"\n{'Pair':<18} {'Strategy':<20} {'Method':<14} {'Trades':<7} {'WR':<7} {'PF':<6} "
          f"{'PnL':<10} {'SL%':<7} {'TP%':<7} {'Improv':<8} {'MaxDD':<8}")
    print(f"{'-' * 110}")

    total_pnl = 0
    total_trades = 0
    for r in all_results:
        print(f"{r['pair']:<18} {r['strategy']:<20} {r['entry_method']:<14} "
              f"{r['trades']:<7} {r['wr']*100:<6.1f}% {r['pf']:<6.2f} "
              f"{r['total_pnl']*100:<+9.2f}% {r['avg_sl_pct']*100:<6.2f}% {r['avg_tp_pct']*100:<6.2f}% "
              f"{r['avg_improvement']*100:<7.3f}% {r['max_dd']*100:<7.2f}%")
        total_pnl += r['total_pnl']
        total_trades += r['trades']

    print(f"{'-' * 110}")
    print(f"{'TOTAL':<18} {'ALL':<20} {'':<14} {total_trades:<7} {'':<7} {'':<6} "
          f"{total_pnl*100:<+9.2f}%")

    # Portfolio-level metrics
    print(f"\n\n{'=' * 90}")
    print("PORTFOLIO SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Total PnL (all strategies combined): {total_pnl*100:+.2f}%")
    print(f"  Total Trades: {total_trades}")
    print(f"  Avg PnL/Trade: {total_pnl/max(total_trades,1)*100:+.3f}%")
    print(f"  Active Pairs: {len(PAIRS)}")
    print(f"  Leverage: 3x -> Leveraged PnL: {total_pnl*3*100:+.2f}%")
    print(f"  Period: 142 days (~4.7 months)")
    print(f"  Monthly avg (unleveraged): {total_pnl/4.7*100:+.2f}%/month")
    print(f"  Monthly avg (3x leveraged): {total_pnl*3/4.7*100:+.2f}%/month")


if __name__ == "__main__":
    main()

"""Combined optimization: Best Entry + Best SL + Trailing (where applicable).

Compare CURRENT config vs FULLY OPTIMIZED config per pair per strategy.
Shows the cumulative effect of all improvements together.
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


def gen_regime(ind):
    n = ind["n"]; signals = []
    for i in range(60, n):
        if ind["vr"][i] < 1.0: continue
        if ind["adx"][i] >= 25:
            for k in range(1, 6):
                pi = i - k
                if pi < 1: break
                if ind["e21"][pi-1] <= ind["e60"][pi-1] and ind["e21"][pi] > ind["e60"][pi]:
                    if ind["dip"][i] > ind["dim"][i]: signals.append((i, 1))
                    break
                if ind["e21"][pi-1] >= ind["e60"][pi-1] and ind["e21"][pi] < ind["e60"][pi]:
                    if ind["dim"][i] > ind["dip"][i]: signals.append((i, -1))
                    break
        else:
            if ind["rsi"][i] < 28: signals.append((i, 1))
            elif ind["rsi"][i] > 67: signals.append((i, -1))
    return signals


def gen_volspike(ind):
    n = ind["n"]; signals = []
    for i in range(20, n):
        if ind["vr"][i] < 2.0 or ind["body"][i] <= 0: continue
        if ind["lower_shadow"][i] > 3.0 * ind["body"][i] and ind["rsi"][i] < 33:
            signals.append((i, 1))
        elif ind["upper_shadow"][i] > 3.0 * ind["body"][i] and ind["rsi"][i] > 72:
            signals.append((i, -1))
    return signals


def gen_cbadx(ind):
    n = ind["n"]; signals = []
    for i in range(22, n):
        if ind["vr"][i] < 0.8: continue
        if ind["rsi"][i] > 72 or ind["rsi"][i] < 22: continue
        if ind["adx"][i] > 22 or ind["adx"][i] >= ind["adx"][i-2]: continue
        if ind["atr"][i] <= 0 or np.isnan(ind["atr"][i]): continue
        r3m = np.mean([ind["h"][i-k] - ind["l"][i-k] for k in range(3)])
        if r3m / ind["atr"][i] > 0.8: continue
        h3 = max(ind["h"][i-2], ind["h"][i-1])
        l3 = min(ind["l"][i-2], ind["l"][i-1])
        if ind["c"][i] > h3: signals.append((i, 1))
        elif ind["c"][i] < l3: signals.append((i, -1))
    return signals


def backtest_full(signals, ind, sl_mult, tp_mult, max_bars, entry_cfg, trail_cfg=None):
    """Full backtest with entry optimization + optional trailing SL."""
    c, h, lo, atr, e8 = ind["c"], ind["h"], ind["l"], ind["atr"], ind["e8"]
    n = ind["n"]

    method = entry_cfg.get("method", "market")
    atr_offset = entry_cfg.get("atr_offset", 0.0)
    fill_window = entry_cfg.get("fill_window", 3)

    if method == "market":
        fee = 2 * TAKER_FEE
    else:
        fee = 2 * MAKER_FEE

    use_trail = trail_cfg is not None
    if use_trail:
        be_trigger = trail_cfg["be_trigger"]
        be_offset = trail_cfg["be_offset"]
        trail_trigger = trail_cfg["trail_trigger"]
        trail_dist = trail_cfg["trail_dist"]

    trades = []
    for idx, direction in signals:
        sp = c[idx]
        ea = atr[idx]
        if ea <= 0 or sp <= 0 or np.isnan(ea):
            continue
        is_long = direction > 0

        # ── ENTRY ──
        if method == "market":
            ep = sp
            fill_bar = idx
        elif method == "limit_atr":
            filled = False
            if atr_offset == 0:
                ep = sp; fill_bar = idx; filled = True
            else:
                limit_price = sp - atr_offset * ea if is_long else sp + atr_offset * ea
                for j in range(idx + 1, min(idx + fill_window + 1, n)):
                    if is_long and lo[j] <= limit_price:
                        ep = limit_price; fill_bar = j; filled = True; break
                    elif not is_long and h[j] >= limit_price:
                        ep = limit_price; fill_bar = j; filled = True; break
            if not filled:
                continue
        elif method == "ema8_retest":
            filled = False
            for j in range(idx + 1, min(idx + fill_window + 1, n)):
                ema8_val = e8[j]
                if is_long and lo[j] <= ema8_val:
                    ep = ema8_val; fill_bar = j; filled = True; break
                elif not is_long and h[j] >= ema8_val:
                    ep = ema8_val; fill_bar = j; filled = True; break
            if not filled:
                continue
        else:
            ep = sp; fill_bar = idx

        sl_dist = sl_mult * ea
        tp_dist = tp_mult * ea

        # ── TRADE SIMULATION ──
        exit_reason = "timeout"
        exit_price = ep
        bars_held = 0
        highest_profit = 0.0
        mfe = 0.0
        current_sl_price = ep - sl_dist if is_long else ep + sl_dist
        be_locked = False
        trailing = False

        for j in range(fill_bar + 1, min(idx + max_bars + 1, n)):
            bars_held = j - fill_bar

            if is_long:
                bar_best = (h[j] - ep) / ep
                bar_worst_price = lo[j]
            else:
                bar_best = (ep - lo[j]) / ep
                bar_worst_price = h[j]

            if bar_best > highest_profit:
                highest_profit = bar_best
            if bar_best > mfe:
                mfe = bar_best

            # Trailing logic (only if enabled)
            if use_trail:
                be_level = be_trigger * tp_dist / ep
                if not be_locked and highest_profit >= be_level:
                    be_locked = True
                    if is_long:
                        current_sl_price = ep + be_offset * ea
                    else:
                        current_sl_price = ep - be_offset * ea

                trail_level = trail_trigger * tp_dist / ep
                if not trailing and highest_profit >= trail_level:
                    trailing = True

                if trailing:
                    if is_long:
                        new_trail = h[j] - trail_dist * tp_dist
                        if new_trail > current_sl_price:
                            current_sl_price = new_trail
                    else:
                        new_trail = lo[j] + trail_dist * tp_dist
                        if new_trail < current_sl_price:
                            current_sl_price = new_trail

            # Check SL
            if is_long:
                if lo[j] <= current_sl_price:
                    exit_price = current_sl_price
                    if trailing: exit_reason = "trail_SL"
                    elif be_locked: exit_reason = "BE_SL"
                    else: exit_reason = "SL"
                    break
                if h[j] >= ep + tp_dist:
                    exit_price = ep + tp_dist
                    exit_reason = "TP"
                    break
            else:
                if h[j] >= current_sl_price:
                    exit_price = current_sl_price
                    if trailing: exit_reason = "trail_SL"
                    elif be_locked: exit_reason = "BE_SL"
                    else: exit_reason = "SL"
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

        trades.append({
            "idx": idx, "dir": "LONG" if is_long else "SHORT",
            "signal_price": sp, "entry_price": ep,
            "exit_price": exit_price, "exit_reason": exit_reason,
            "pnl": pnl, "mfe": mfe, "bars_held": bars_held,
            "sl_pct": sl_dist / ep, "tp_pct": tp_dist / ep,
            "entry_improvement": abs(sp - ep) / sp if sp != ep else 0.0,
        })
    return trades


def stats(trades):
    if not trades:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    wins = (pnls > 0).sum()
    wr = wins / n
    pf = abs(pnls[pnls > 0].sum() / pnls[pnls <= 0].sum()) if pnls[pnls <= 0].sum() != 0 else 99
    tp_c = sum(1 for t in trades if t["exit_reason"] == "TP")
    sl_c = sum(1 for t in trades if t["exit_reason"] == "SL")
    be_c = sum(1 for t in trades if t["exit_reason"] == "BE_SL")
    tr_c = sum(1 for t in trades if t["exit_reason"] == "trail_SL")
    to_c = sum(1 for t in trades if t["exit_reason"] == "timeout")
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = (peak - equity).max()
    avg_win = pnls[pnls > 0].mean() if wins > 0 else 0
    avg_loss = pnls[pnls <= 0].mean() if (n - wins) > 0 else 0
    longs = [t for t in trades if t["dir"] == "LONG"]
    shorts = [t for t in trades if t["dir"] == "SHORT"]
    long_pnl = sum(t["pnl"] for t in longs)
    short_pnl = sum(t["pnl"] for t in shorts)
    long_wr = sum(1 for t in longs if t["pnl"] > 0) / max(len(longs), 1)
    short_wr = sum(1 for t in shorts if t["pnl"] > 0) / max(len(shorts), 1)
    return {
        "trades": n, "wr": wr, "pf": pf, "pnl": pnls.sum(),
        "avg_win": avg_win, "avg_loss": avg_loss, "max_dd": max_dd,
        "tp": tp_c, "sl": sl_c, "be": be_c, "trail": tr_c, "timeout": to_c,
        "long_count": len(longs), "short_count": len(shorts),
        "long_pnl": long_pnl, "short_pnl": short_pnl,
        "long_wr": long_wr, "short_wr": short_wr,
        "avg_sl_pct": np.mean([t["sl_pct"] for t in trades]),
        "avg_tp_pct": np.mean([t["tp_pct"] for t in trades]),
        "avg_entry_impr": np.mean([t["entry_improvement"] for t in trades]),
        "avg_bars": np.mean([t["bars_held"] for t in trades]),
    }


def main():
    # ══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION: CURRENT vs OPTIMIZED
    # ══════════════════════════════════════════════════════════════════════════
    configs = {
        "regime_adaptive": {
            "SOL_USDT_USDT": {
                "current": {"sl": 7.0, "tp": 11.0, "max_bars": 192, "entry": {"method": "market"}, "trail": None},
                "optimized": {"sl": 10.0, "tp": 11.0, "max_bars": 192, "entry": {"method": "market"}, "trail": None},
            },
            "SPX_USDT_USDT": {
                "current": {"sl": 7.0, "tp": 11.0, "max_bars": 192, "entry": {"method": "market"}, "trail": None},
                "optimized": {"sl": 9.0, "tp": 11.0, "max_bars": 192,
                              "entry": {"method": "limit_atr", "atr_offset": 0.1, "fill_window": 3},
                              "trail": {"be_trigger": 0.4, "be_offset": 0.2, "trail_trigger": 0.7, "trail_dist": 0.5}},
            },
        },
        "volume_spike_rev": {
            "SOL_USDT_USDT": {
                "current": {"sl": 3.5, "tp": 5.5, "max_bars": 96, "entry": {"method": "market"}, "trail": None},
                "optimized": {"sl": 3.5, "tp": 5.5, "max_bars": 96, "entry": {"method": "market"}, "trail": None},
            },
            "SPX_USDT_USDT": {
                "current": {"sl": 3.5, "tp": 5.5, "max_bars": 96, "entry": {"method": "market"}, "trail": None},
                "optimized": {"sl": 2.5, "tp": 5.5, "max_bars": 96, "entry": {"method": "market"}, "trail": None},
            },
        },
        "cb_adx_breakout": {
            "SOL_USDT_USDT": {
                "current": {"sl": 3.0, "tp": 5.0, "max_bars": 96, "entry": {"method": "market"}, "trail": None},
                "optimized": {"sl": 7.0, "tp": 5.0, "max_bars": 96, "entry": {"method": "ema8_retest", "fill_window": 4}, "trail": None},
            },
            "SPX_USDT_USDT": {
                "current": {"sl": 3.0, "tp": 5.0, "max_bars": 96, "entry": {"method": "market"}, "trail": None},
                "optimized": {"sl": 2.5, "tp": 5.0, "max_bars": 96, "entry": {"method": "market"}, "trail": None},
            },
        },
    }

    gen_funcs = {
        "regime_adaptive": gen_regime,
        "volume_spike_rev": gen_volspike,
        "cb_adx_breakout": gen_cbadx,
    }

    print("=" * 100)
    print("COMBINED OPTIMIZATION ANALYSIS: Entry + SL + Trailing")
    print("CURRENT vs FULLY OPTIMIZED — Per Strategy Per Pair")
    print(f"Period: 2026-01-01 to 2026-05-21 (142 days)")
    print("=" * 100)

    all_current_pnl = 0
    all_optimized_pnl = 0
    all_current_trades = 0
    all_optimized_trades = 0
    summary_rows = []

    for pair in PAIRS:
        df = load_15m(pair)
        ind = compute_indicators(df)
        print(f"\n{'#' * 100}")
        print(f"# {pair}")
        print(f"{'#' * 100}")

        for strat_name in ["regime_adaptive", "volume_spike_rev", "cb_adx_breakout"]:
            sig_list = gen_funcs[strat_name](ind)
            cfg = configs[strat_name][pair]
            cur = cfg["current"]
            opt = cfg["optimized"]

            # Run CURRENT
            cur_trades = backtest_full(sig_list, ind, cur["sl"], cur["tp"], cur["max_bars"],
                                       cur["entry"], cur["trail"])
            cur_stats = stats(cur_trades)

            # Run OPTIMIZED
            opt_trades = backtest_full(sig_list, ind, opt["sl"], opt["tp"], opt["max_bars"],
                                       opt["entry"], opt["trail"])
            opt_stats = stats(opt_trades)

            if cur_stats is None or opt_stats is None:
                continue

            all_current_pnl += cur_stats["pnl"]
            all_optimized_pnl += opt_stats["pnl"]
            all_current_trades += cur_stats["trades"]
            all_optimized_trades += opt_stats["trades"]

            # Describe changes
            changes = []
            if cur["sl"] != opt["sl"]:
                changes.append(f"SL: {cur['sl']}x -> {opt['sl']}x")
            if cur["entry"]["method"] != opt["entry"]["method"]:
                changes.append(f"Entry: {cur['entry']['method']} -> {opt['entry']['method']}")
            elif opt["entry"].get("atr_offset", 0) != cur["entry"].get("atr_offset", 0):
                changes.append(f"Entry: +{opt['entry']['atr_offset']}x ATR offset")
            if opt["trail"] is not None and cur["trail"] is None:
                changes.append("Trail: +Conservative")
            change_str = " | ".join(changes) if changes else "No change"

            print(f"\n  {'=' * 90}")
            print(f"  {strat_name.upper()} | {pair}")
            print(f"  Changes: {change_str}")
            print(f"  {'=' * 90}")

            # Side-by-side comparison
            print(f"\n  {'Metric':<22} {'CURRENT':<20} {'OPTIMIZED':<20} {'CHANGE'}")
            print(f"  {'-' * 80}")
            print(f"  {'Trades':<22} {cur_stats['trades']:<20} {opt_stats['trades']:<20} {opt_stats['trades']-cur_stats['trades']:+d}")
            print(f"  {'Win Rate':<22} {cur_stats['wr']*100:<19.1f}% {opt_stats['wr']*100:<19.1f}% {(opt_stats['wr']-cur_stats['wr'])*100:+.1f}pp")
            print(f"  {'Profit Factor':<22} {cur_stats['pf']:<20.2f} {opt_stats['pf']:<20.2f} {opt_stats['pf']-cur_stats['pf']:+.2f}")
            print(f"  {'Total PnL':<22} {cur_stats['pnl']*100:<+19.2f}% {opt_stats['pnl']*100:<+19.2f}% {(opt_stats['pnl']-cur_stats['pnl'])*100:+.2f}%")
            print(f"  {'Avg PnL/Trade':<22} {cur_stats['pnl']/cur_stats['trades']*100:<+19.3f}% {opt_stats['pnl']/opt_stats['trades']*100:<+19.3f}% {(opt_stats['pnl']/opt_stats['trades']-cur_stats['pnl']/cur_stats['trades'])*100:+.3f}%")
            print(f"  {'Avg Win':<22} {cur_stats['avg_win']*100:<+19.3f}% {opt_stats['avg_win']*100:<+19.3f}%")
            print(f"  {'Avg Loss':<22} {cur_stats['avg_loss']*100:<19.3f}% {opt_stats['avg_loss']*100:<19.3f}%")
            print(f"  {'Max Drawdown':<22} {cur_stats['max_dd']*100:<19.2f}% {opt_stats['max_dd']*100:<19.2f}% {(opt_stats['max_dd']-cur_stats['max_dd'])*100:+.2f}%")
            print(f"  {'SL Distance':<22} {cur_stats['avg_sl_pct']*100:<19.2f}% {opt_stats['avg_sl_pct']*100:<19.2f}%")
            print(f"  {'TP Distance':<22} {cur_stats['avg_tp_pct']*100:<19.2f}% {opt_stats['avg_tp_pct']*100:<19.2f}%")
            print(f"  {'Entry Improvement':<22} {0:<19.3f}% {opt_stats['avg_entry_impr']*100:<19.3f}%")
            print(f"  {'Avg Hold (bars)':<22} {cur_stats['avg_bars']:<19.0f} {opt_stats['avg_bars']:<19.0f}")
            print(f"  ")
            print(f"  {'Exit: TP':<22} {cur_stats['tp']:<20} {opt_stats['tp']:<20}")
            print(f"  {'Exit: SL':<22} {cur_stats['sl']:<20} {opt_stats['sl']:<20}")
            print(f"  {'Exit: BE':<22} {'--':<20} {opt_stats['be']:<20}")
            print(f"  {'Exit: Trail':<22} {'--':<20} {opt_stats['trail']:<20}")
            print(f"  {'Exit: Timeout':<22} {cur_stats['timeout']:<20} {opt_stats['timeout']:<20}")
            print(f"  ")
            print(f"  {'LONG PnL':<22} {cur_stats['long_pnl']*100:<+19.2f}% {opt_stats['long_pnl']*100:<+19.2f}%  ({cur_stats['long_count']}t WR={cur_stats['long_wr']*100:.0f}% -> {opt_stats['long_count']}t WR={opt_stats['long_wr']*100:.0f}%)")
            print(f"  {'SHORT PnL':<22} {cur_stats['short_pnl']*100:<+19.2f}% {opt_stats['short_pnl']*100:<+19.2f}%  ({cur_stats['short_count']}t WR={cur_stats['short_wr']*100:.0f}% -> {opt_stats['short_count']}t WR={opt_stats['short_wr']*100:.0f}%)")

            summary_rows.append({
                "pair": pair, "strategy": strat_name,
                "change": change_str,
                "cur_pnl": cur_stats["pnl"], "opt_pnl": opt_stats["pnl"],
                "cur_wr": cur_stats["wr"], "opt_wr": opt_stats["wr"],
                "cur_pf": cur_stats["pf"], "opt_pf": opt_stats["pf"],
                "cur_trades": cur_stats["trades"], "opt_trades": opt_stats["trades"],
                "cur_dd": cur_stats["max_dd"], "opt_dd": opt_stats["max_dd"],
            })

    # ══════════════════════════════════════════════════════════════════════════
    # PORTFOLIO SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}")
    print("PORTFOLIO SUMMARY: CURRENT vs OPTIMIZED")
    print(f"{'=' * 100}")

    print(f"\n  {'Strategy':<20} {'Pair':<18} {'Changes':<35} {'CUR PnL':<12} {'OPT PnL':<12} {'DIFF'}")
    print(f"  {'-' * 100}")
    for r in summary_rows:
        diff = (r["opt_pnl"] - r["cur_pnl"]) * 100
        print(f"  {r['strategy']:<20} {r['pair']:<18} {r['change']:<35} "
              f"{r['cur_pnl']*100:+.2f}%{'':>3} {r['opt_pnl']*100:+.2f}%{'':>3} {diff:+.2f}%")

    print(f"  {'-' * 100}")
    diff_total = (all_optimized_pnl - all_current_pnl) * 100
    print(f"  {'TOTAL':<20} {'ALL':<18} {'':<35} "
          f"{all_current_pnl*100:+.2f}%{'':>3} {all_optimized_pnl*100:+.2f}%{'':>3} {diff_total:+.2f}%")

    print(f"\n\n  {'=' * 90}")
    print(f"  BOTTOM LINE")
    print(f"  {'=' * 90}")
    print(f"  CURRENT SYSTEM:    {all_current_pnl*100:+.2f}% over 142d ({all_current_trades} trades)")
    print(f"  OPTIMIZED SYSTEM:  {all_optimized_pnl*100:+.2f}% over 142d ({all_optimized_trades} trades)")
    print(f"  IMPROVEMENT:       {diff_total:+.2f}% absolute")
    print(f"  RELATIVE GAIN:     {(all_optimized_pnl/max(all_current_pnl,0.001)-1)*100:+.1f}%")
    print(f"  ")
    print(f"  3x Leverage:")
    print(f"    Current:   {all_current_pnl*3*100:+.2f}%")
    print(f"    Optimized: {all_optimized_pnl*3*100:+.2f}%")
    print(f"  ")
    print(f"  Monthly (4.7 months):")
    print(f"    Current:   {all_current_pnl/4.7*100:+.2f}%/mo (unlev) | {all_current_pnl*3/4.7*100:+.2f}%/mo (3x)")
    print(f"    Optimized: {all_optimized_pnl/4.7*100:+.2f}%/mo (unlev) | {all_optimized_pnl*3/4.7*100:+.2f}%/mo (3x)")

    print(f"\n\n  {'=' * 90}")
    print(f"  CHANGES SUMMARY (what to update in config/base.yaml)")
    print(f"  {'=' * 90}")
    print(f"""
  regime_adaptive:
    SOL: SL 7.0x -> 10.0x ATR | Entry: market (no change) | No trailing
    SPX: SL 7.0x -> 9.0x ATR  | Entry: limit 0.1x ATR offset | Trail: Conservative
         Trail params: BE at 40% TP, lock +0.2x ATR | Trail at 70% TP, dist 0.5x TP

  volume_spike_rev:
    SOL: No change (SL 3.5x, market entry)
    SPX: SL 3.5x -> 2.5x ATR | Entry: market (no change) | No trailing

  cb_adx_breakout:
    SOL: SL 3.0x -> 7.0x ATR | Entry: EMA8 retest (4-bar window) | No trailing
    SPX: SL 3.0x -> 2.5x ATR | Entry: market (no change) | No trailing
""")


if __name__ == "__main__":
    main()

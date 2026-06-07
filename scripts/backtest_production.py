"""Final production backtest with all optimizations + stake weighting.

Tests the FULL optimized config:
- Optimized entry (limit/market/ema8 per pair)
- Optimized SL (per pair)
- Trailing SL (SPX regime_adaptive only)
- Capital allocation (stake weights by WR/PF performance)

Reports PnL weighted by stake allocation.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
PAIRS = ["SOL_USDT_USDT", "SPX_USDT_USDT"]
TAKER_FEE = 0.0005
MAKER_FEE = 0.0002
BASE_STAKE = 50  # USDT base position
LEVERAGE = 3


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    df = pd.read_feather(fp).sort_values("date").reset_index(drop=True)
    return df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)


def compute_indicators(df):
    c = df["close"].values; h = df["high"].values; lo = df["low"].values
    o = df["open"].values; v = df["volume"].astype(float).values; n = len(c)
    adx_r = ta.adx(df["high"], df["low"], df["close"], length=14)
    adx = adx_r.iloc[:, 0].values; dip = adx_r.iloc[:, 1].values; dim = adx_r.iloc[:, 2].values
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
    return {"c": c, "h": h, "l": lo, "o": o, "v": v, "n": n,
            "adx": adx, "dip": dip, "dim": dim, "rsi": rsi, "atr": atr,
            "e8": e8, "e21": e21, "e60": e60, "vr": vr,
            "body": body, "upper_shadow": upper_shadow, "lower_shadow": lower_shadow}


def gen_regime(ind):
    n = ind["n"]; signals = []
    for i in range(60, n):
        if ind["vr"][i] < 1.0: continue
        if ind["adx"][i] >= 25:
            for k in range(1, 6):
                pi = i - k
                if pi < 1: break
                if ind["e21"][pi-1] <= ind["e60"][pi-1] and ind["e21"][pi] > ind["e60"][pi]:
                    if ind["dip"][i] > ind["dim"][i]: signals.append((i, 1)); break
                if ind["e21"][pi-1] >= ind["e60"][pi-1] and ind["e21"][pi] < ind["e60"][pi]:
                    if ind["dim"][i] > ind["dip"][i]: signals.append((i, -1)); break
        else:
            if ind["rsi"][i] < 28: signals.append((i, 1))
            elif ind["rsi"][i] > 67: signals.append((i, -1))
    return signals


def gen_volspike(ind):
    n = ind["n"]; signals = []
    for i in range(20, n):
        if ind["vr"][i] < 2.0 or ind["body"][i] <= 0: continue
        if ind["lower_shadow"][i] > 3.0 * ind["body"][i] and ind["rsi"][i] < 33: signals.append((i, 1))
        elif ind["upper_shadow"][i] > 3.0 * ind["body"][i] and ind["rsi"][i] > 72: signals.append((i, -1))
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


def backtest_production(signals, ind, cfg):
    """Production backtest with entry opt + SL opt + trailing + stake weighting."""
    c, h, lo, atr, e8 = ind["c"], ind["h"], ind["l"], ind["atr"], ind["e8"]
    n = ind["n"]
    sl_mult = cfg["sl"]; tp_mult = cfg["tp"]; max_bars = cfg["max_bars"]
    entry_cfg = cfg["entry"]; trail_cfg = cfg.get("trail")
    stake_mult = cfg.get("stake_mult", 1.0)

    method = entry_cfg.get("method", "market")
    atr_offset = entry_cfg.get("atr_offset", 0.0)
    fill_window = entry_cfg.get("fill_window", 3)
    fee = 2 * MAKER_FEE if method != "market" else 2 * TAKER_FEE

    use_trail = trail_cfg is not None and trail_cfg.get("enabled", False)
    if use_trail:
        be_trigger = trail_cfg["be_trigger"]
        be_offset_mult = trail_cfg["be_offset"]
        trail_trigger = trail_cfg["trail_trigger"]
        trail_dist_mult = trail_cfg["trail_dist"]

    trades = []
    for idx, direction in signals:
        sp = c[idx]; ea = atr[idx]
        if ea <= 0 or sp <= 0 or np.isnan(ea): continue
        is_long = direction > 0

        # Entry
        if method == "market":
            ep = sp; fill_bar = idx
        elif method == "limit_atr":
            filled = False
            limit_price = sp - atr_offset * ea if is_long else sp + atr_offset * ea
            for j in range(idx + 1, min(idx + fill_window + 1, n)):
                if is_long and lo[j] <= limit_price:
                    ep = limit_price; fill_bar = j; filled = True; break
                elif not is_long and h[j] >= limit_price:
                    ep = limit_price; fill_bar = j; filled = True; break
            if not filled: continue
        elif method == "ema8_retest":
            filled = False
            for j in range(idx + 1, min(idx + fill_window + 1, n)):
                if is_long and lo[j] <= e8[j]:
                    ep = e8[j]; fill_bar = j; filled = True; break
                elif not is_long and h[j] >= e8[j]:
                    ep = e8[j]; fill_bar = j; filled = True; break
            if not filled: continue
        else:
            ep = sp; fill_bar = idx

        sl_dist = sl_mult * ea; tp_dist = tp_mult * ea
        current_sl_price = ep - sl_dist if is_long else ep + sl_dist
        be_locked = False; trailing = False; highest_profit = 0.0
        exit_reason = "timeout"; exit_price = ep; bars_held = 0

        for j in range(fill_bar + 1, min(idx + max_bars + 1, n)):
            bars_held = j - fill_bar
            bar_best = (h[j] - ep) / ep if is_long else (ep - lo[j]) / ep
            if bar_best > highest_profit: highest_profit = bar_best

            if use_trail:
                be_level = be_trigger * tp_dist / ep
                if not be_locked and highest_profit >= be_level:
                    be_locked = True
                    current_sl_price = ep + be_offset_mult * ea if is_long else ep - be_offset_mult * ea
                trail_level = trail_trigger * tp_dist / ep
                if not trailing and highest_profit >= trail_level:
                    trailing = True
                if trailing:
                    if is_long:
                        new_trail = h[j] - trail_dist_mult * tp_dist
                        if new_trail > current_sl_price: current_sl_price = new_trail
                    else:
                        new_trail = lo[j] + trail_dist_mult * tp_dist
                        if new_trail < current_sl_price: current_sl_price = new_trail

            if is_long:
                if lo[j] <= current_sl_price:
                    exit_price = current_sl_price
                    exit_reason = "trail_SL" if trailing else ("BE_SL" if be_locked else "SL")
                    break
                if h[j] >= ep + tp_dist:
                    exit_price = ep + tp_dist; exit_reason = "TP"; break
            else:
                if h[j] >= current_sl_price:
                    exit_price = current_sl_price
                    exit_reason = "trail_SL" if trailing else ("BE_SL" if be_locked else "SL")
                    break
                if lo[j] <= ep - tp_dist:
                    exit_price = ep - tp_dist; exit_reason = "TP"; break
        else:
            ei = min(idx + max_bars, n - 1)
            exit_price = c[ei]; bars_held = ei - fill_bar

        pnl_pct = ((exit_price - ep) / ep if is_long else (ep - exit_price) / ep) - fee
        stake_usdt = BASE_STAKE * stake_mult
        pnl_usdt = pnl_pct * stake_usdt * LEVERAGE

        trades.append({
            "idx": idx, "dir": "LONG" if is_long else "SHORT",
            "entry": ep, "exit": exit_price, "exit_reason": exit_reason,
            "pnl_pct": pnl_pct, "pnl_usdt": pnl_usdt,
            "stake_usdt": stake_usdt, "bars_held": bars_held,
            "sl_pct": sl_dist / ep, "tp_pct": tp_dist / ep,
        })
    return trades


def main():
    # Production config
    prod_config = {
        "regime_adaptive": {
            "SOL_USDT_USDT": {"sl": 10.0, "tp": 11.0, "max_bars": 192, "stake_mult": 1.0,
                              "entry": {"method": "market"}, "trail": None},
            "SPX_USDT_USDT": {"sl": 9.0, "tp": 11.0, "max_bars": 192, "stake_mult": 2.0,
                              "entry": {"method": "limit_atr", "atr_offset": 0.1, "fill_window": 3},
                              "trail": {"enabled": True, "be_trigger": 0.4, "be_offset": 0.2,
                                        "trail_trigger": 0.7, "trail_dist": 0.5}},
        },
        "volume_spike_rev": {
            "SOL_USDT_USDT": {"sl": 3.5, "tp": 5.5, "max_bars": 96, "stake_mult": 1.2,
                              "entry": {"method": "market"}, "trail": None},
            "SPX_USDT_USDT": {"sl": 2.5, "tp": 5.5, "max_bars": 96, "stake_mult": 2.0,
                              "entry": {"method": "market"}, "trail": None},
        },
        "cb_adx_breakout": {
            "SOL_USDT_USDT": {"sl": 7.0, "tp": 5.0, "max_bars": 96, "stake_mult": 1.5,
                              "entry": {"method": "ema8_retest", "fill_window": 4}, "trail": None},
            "SPX_USDT_USDT": {"sl": 2.5, "tp": 5.0, "max_bars": 96, "stake_mult": 0.8,
                              "entry": {"method": "market"}, "trail": None},
        },
    }

    gen_funcs = {"regime_adaptive": gen_regime, "volume_spike_rev": gen_volspike, "cb_adx_breakout": gen_cbadx}

    print("=" * 100)
    print("PRODUCTION BACKTEST — FULLY OPTIMIZED + STAKE WEIGHTED")
    print(f"Period: 2026-01-01 to 2026-05-21 (142 days)")
    print(f"Base stake: {BASE_STAKE} USDT | Leverage: {LEVERAGE}x")
    print(f"Capital allocation: SPX regime 2.0x | SPX volspike 2.0x | SOL cbadx 1.5x | SOL volspike 1.2x | SOL regime 1.0x | SPX cbadx 0.8x")
    print("=" * 100)

    all_trades = []
    results = []

    for pair in PAIRS:
        df = load_15m(pair)
        ind = compute_indicators(df)

        for strat_name, gen_fn in gen_funcs.items():
            sig_list = gen_fn(ind)
            cfg = prod_config[strat_name][pair]
            trades = backtest_production(sig_list, ind, cfg)
            all_trades.extend(trades)

            if not trades:
                continue

            pnls = np.array([t["pnl_pct"] for t in trades])
            usdt_pnls = np.array([t["pnl_usdt"] for t in trades])
            n_trades = len(pnls)
            wins = (pnls > 0).sum()
            wr = wins / n_trades
            pf = abs(pnls[pnls > 0].sum() / pnls[pnls <= 0].sum()) if pnls[pnls <= 0].sum() != 0 else 99
            tp_c = sum(1 for t in trades if t["exit_reason"] == "TP")
            sl_c = sum(1 for t in trades if t["exit_reason"] == "SL")
            be_c = sum(1 for t in trades if t["exit_reason"] == "BE_SL")
            tr_c = sum(1 for t in trades if t["exit_reason"] == "trail_SL")
            to_c = sum(1 for t in trades if t["exit_reason"] == "timeout")
            longs = [t for t in trades if t["dir"] == "LONG"]
            shorts = [t for t in trades if t["dir"] == "SHORT"]
            long_pnl = sum(t["pnl_usdt"] for t in longs)
            short_pnl = sum(t["pnl_usdt"] for t in shorts)
            equity = np.cumsum(usdt_pnls)
            peak = np.maximum.accumulate(equity)
            max_dd_usdt = (peak - equity).max()

            results.append({
                "pair": pair, "strategy": strat_name,
                "stake_mult": cfg["stake_mult"],
                "trades": n_trades, "wr": wr, "pf": pf,
                "pnl_pct": pnls.sum(), "pnl_usdt": usdt_pnls.sum(),
                "tp": tp_c, "sl": sl_c, "be": be_c, "trail": tr_c, "timeout": to_c,
                "long_pnl_usdt": long_pnl, "short_pnl_usdt": short_pnl,
                "long_count": len(longs), "short_count": len(shorts),
                "long_wr": sum(1 for t in longs if t["pnl_pct"] > 0) / max(len(longs), 1),
                "short_wr": sum(1 for t in shorts if t["pnl_pct"] > 0) / max(len(shorts), 1),
                "max_dd_usdt": max_dd_usdt,
                "avg_sl_pct": np.mean([t["sl_pct"] for t in trades]),
                "avg_tp_pct": np.mean([t["tp_pct"] for t in trades]),
                "entry_method": cfg["entry"]["method"],
                "sl_mult": cfg["sl"],
                "has_trail": cfg.get("trail") is not None and cfg["trail"].get("enabled", False),
            })

    # Print detailed results
    for r in results:
        trail_str = " + TRAIL" if r["has_trail"] else ""
        print(f"\n  {'=' * 90}")
        print(f"  {r['strategy'].upper()} | {r['pair']} | Stake: {r['stake_mult']}x ({BASE_STAKE*r['stake_mult']:.0f} USDT)")
        print(f"  Entry: {r['entry_method']} | SL: {r['sl_mult']}x ATR{trail_str}")
        print(f"  {'=' * 90}")
        print(f"  Trades: {r['trades']} | WR: {r['wr']*100:.1f}% | PF: {r['pf']:.2f}")
        print(f"  PnL: {r['pnl_pct']*100:+.2f}% = ${r['pnl_usdt']:+.2f} USDT ({LEVERAGE}x lev)")
        print(f"  Max DD: ${r['max_dd_usdt']:.2f}")
        print(f"  SL dist: {r['avg_sl_pct']*100:.2f}% | TP dist: {r['avg_tp_pct']*100:.2f}%")
        print(f"  Exit: TP={r['tp']} | SL={r['sl']} | BE={r['be']} | Trail={r['trail']} | Timeout={r['timeout']}")
        print(f"  LONG:  {r['long_count']} trades, WR={r['long_wr']*100:.0f}%, ${r['long_pnl_usdt']:+.2f}")
        print(f"  SHORT: {r['short_count']} trades, WR={r['short_wr']*100:.0f}%, ${r['short_pnl_usdt']:+.2f}")

    # Summary table
    print(f"\n\n{'=' * 100}")
    print("SUMMARY TABLE — STAKE WEIGHTED PRODUCTION RESULTS")
    print(f"{'=' * 100}")
    print(f"\n  {'Strategy':<20} {'Pair':<18} {'Stake':<8} {'Trades':<7} {'WR':<7} {'PF':<6} "
          f"{'PnL%':<10} {'PnL$':<12} {'SL%':<7} {'Entry':<12} {'Trail'}")
    print(f"  {'-' * 110}")

    total_usdt = 0
    total_trades = 0
    for r in results:
        trail_mark = "YES" if r["has_trail"] else "—"
        print(f"  {r['strategy']:<20} {r['pair']:<18} {r['stake_mult']:<8.1f} {r['trades']:<7} "
              f"{r['wr']*100:<6.1f}% {r['pf']:<6.2f} {r['pnl_pct']*100:<+9.2f}% "
              f"${r['pnl_usdt']:<+11.2f} {r['avg_sl_pct']*100:<6.2f}% {r['entry_method']:<12} {trail_mark}")
        total_usdt += r["pnl_usdt"]
        total_trades += r["trades"]

    print(f"  {'-' * 110}")
    print(f"  {'TOTAL':<20} {'ALL':<18} {'':<8} {total_trades:<7} {'':<7} {'':<6} "
          f"{'':<10} ${total_usdt:<+11.2f}")

    # Portfolio metrics
    all_usdt_pnls = np.array([t["pnl_usdt"] for t in all_trades])
    all_usdt_pnls_sorted = all_usdt_pnls[np.argsort([t["idx"] for t in all_trades])]
    equity_curve = np.cumsum(all_usdt_pnls_sorted)
    portfolio_peak = np.maximum.accumulate(equity_curve)
    portfolio_dd = (portfolio_peak - equity_curve).max()

    # Win/loss streaks
    win_streak = 0; max_win_streak = 0; loss_streak = 0; max_loss_streak = 0
    for p in all_usdt_pnls_sorted:
        if p > 0:
            win_streak += 1; loss_streak = 0
            max_win_streak = max(max_win_streak, win_streak)
        else:
            loss_streak += 1; win_streak = 0
            max_loss_streak = max(max_loss_streak, loss_streak)

    overall_wr = (all_usdt_pnls > 0).sum() / len(all_usdt_pnls)

    print(f"\n\n{'=' * 100}")
    print("PORTFOLIO PERFORMANCE (ALL STRATEGIES COMBINED)")
    print(f"{'=' * 100}")
    print(f"""
  Period:              142 days (2026-01-01 to 2026-05-21)
  Base Stake:          {BASE_STAKE} USDT per trade
  Leverage:            {LEVERAGE}x

  Total Trades:        {total_trades}
  Overall Win Rate:    {overall_wr*100:.1f}%
  Total PnL:           ${total_usdt:+,.2f} USDT

  Portfolio Max DD:    ${portfolio_dd:,.2f}
  Max Win Streak:      {max_win_streak}
  Max Loss Streak:     {max_loss_streak}

  Monthly PnL:         ${total_usdt/4.7:+,.2f}/month
  Daily PnL:           ${total_usdt/142:+,.2f}/day
  Per-Trade Avg:       ${total_usdt/total_trades:+,.2f}/trade

  Annualized:          ${total_usdt/142*365:+,.2f}/year
""")

    # Capital allocation summary
    print(f"  {'=' * 90}")
    print(f"  CAPITAL ALLOCATION (Stake Weights by WR/PF)")
    print(f"  {'=' * 90}")
    print(f"""
  HIGH CONVICTION (2.0x = {BASE_STAKE*2} USDT):
    - SPX regime_adaptive:  WR=71.4%, PF=1.38 -> ${next((r['pnl_usdt'] for r in results if r['pair']=='SPX_USDT_USDT' and r['strategy']=='regime_adaptive'), 0):+.2f}
    - SPX volume_spike_rev: WR=69.0%, PF=5.90 -> ${next((r['pnl_usdt'] for r in results if r['pair']=='SPX_USDT_USDT' and r['strategy']=='volume_spike_rev'), 0):+.2f}

  MEDIUM (1.2-1.5x = {BASE_STAKE*1.2:.0f}-{BASE_STAKE*1.5:.0f} USDT):
    - SOL cb_adx_breakout:  WR=62.5%, PF=1.62 -> ${next((r['pnl_usdt'] for r in results if r['pair']=='SOL_USDT_USDT' and r['strategy']=='cb_adx_breakout'), 0):+.2f}
    - SOL volume_spike_rev: WR=50.0%, PF=2.21 -> ${next((r['pnl_usdt'] for r in results if r['pair']=='SOL_USDT_USDT' and r['strategy']=='volume_spike_rev'), 0):+.2f}

  BASE (1.0x = {BASE_STAKE} USDT):
    - SOL regime_adaptive:  WR=52.2%, PF=1.27 -> ${next((r['pnl_usdt'] for r in results if r['pair']=='SOL_USDT_USDT' and r['strategy']=='regime_adaptive'), 0):+.2f}

  REDUCED (0.8x = {BASE_STAKE*0.8:.0f} USDT):
    - SPX cb_adx_breakout:  WR=43.6%, PF=1.28 -> ${next((r['pnl_usdt'] for r in results if r['pair']=='SPX_USDT_USDT' and r['strategy']=='cb_adx_breakout'), 0):+.2f}
""")


if __name__ == "__main__":
    main()

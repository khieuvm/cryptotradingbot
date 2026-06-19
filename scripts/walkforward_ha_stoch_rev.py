"""Walk-forward OOS validation for ha_stoch_rev (ETH shorts-only).

Protocol: 60d IS / 30d OOS, 30d step (3 windows).
Grade A requirement: OOS PF > 1.5, OOS WR > 52%, MaxDD < 10%.
Monte Carlo: shuffle signal timestamps 200x per window, measure p-value.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.config import AppConfig
from engine.events import Direction
from strategies import get_strategy_class

TAKER_FEE = 0.0005
FUNDING_PER_8H = 0.0001

WALK_FORWARD_SPLITS = [
    # (IS_start, IS_end, OOS_start, OOS_end)
    ("2026-01-01", "2026-03-01", "2026-03-01", "2026-04-01"),
    ("2026-02-01", "2026-04-01", "2026-04-01", "2026-05-01"),
    ("2026-03-01", "2026-05-01", "2026-05-01", "2026-06-13"),
]


@dataclass
class Trade:
    pair: str
    direction: Direction
    entry_time: datetime
    entry_rate: float
    strategy: str
    tag: str
    exit_time: datetime | None = None
    exit_rate: float | None = None
    exit_reason: str | None = None
    profit_pct: float = 0.0
    hold_hours: float = 0.0


def load_eth_data() -> pd.DataFrame:
    data_dir = ROOT / "data" / "okx" / "futures"
    fp = data_dir / "ETH_USDT_USDT-15m-futures.feather"
    if not fp.exists():
        print(f"ERROR: Data file not found at {fp}")
        sys.exit(1)
    df = pd.read_feather(fp).sort_values("date").reset_index(drop=True)
    return df


def compute_funding(entry: datetime, exit_: datetime) -> float:
    hours = (exit_ - entry).total_seconds() / 3600
    if hours <= 4:
        return 0.0
    return (hours / 8) * FUNDING_PER_8H


def run_backtest(strategy, df_full: pd.DataFrame, start: str, end: str) -> list[Trade]:
    """Run ha_stoch_rev on df_full, filtering to [start, end] for trade collection.
    Uses full df_full for indicator warm-up then filters for actual OOS/IS window."""
    df = df_full[(df_full["date"] >= start) & (df_full["date"] <= end)].reset_index(drop=True)

    # Need warmup — prepend 200 bars of data before start
    start_dt = pd.Timestamp(start, tz="UTC")
    pre_df = df_full[df_full["date"] < start].iloc[-200:]
    df_warm = pd.concat([pre_df, df]).reset_index(drop=True)

    df_warm = strategy.compute_indicators(df_warm, {"pair": "ETH/USDT:USDT"})

    atr_col = next((c for c in df_warm.columns if c.endswith("_atr") and "_atr_" not in c), None)
    sl_mult = float(strategy.config.exit.get("sl_atr_mult", 2.5))
    startup = strategy.startup_candle_count

    # Only trade in the target window
    window_start_idx = len(pre_df)
    open_trade: Trade | None = None
    trades: list[Trade] = []

    for i in range(max(startup, window_start_idx), len(df_warm)):
        row = df_warm.iloc[i]
        t = pd.Timestamp(row["date"])
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row[atr_col]) if atr_col else 0

        if open_trade is not None:
            entry_rate = open_trade.entry_rate
            if open_trade.direction == Direction.SHORT:
                raw = (entry_rate - close) / entry_rate
            else:
                raw = (close - entry_rate) / entry_rate
            profit = raw - 2 * TAKER_FEE

            trade_info = {
                "current_profit": profit,
                "current_time": t.to_pydatetime(),
                "entry_time": open_trade.entry_time,
                "entry_rate": entry_rate,
                "is_short": open_trade.direction == Direction.SHORT,
            }
            exit_req = strategy.detect_exits(df_warm.iloc[: i + 1], "ETH/USDT:USDT", trade_info)

            sl_pct = sl_mult * atr / entry_rate if atr > 0 and entry_rate > 0 else 0.05
            should_exit = False
            exit_reason = ""
            exit_price = close

            if open_trade.direction == Direction.SHORT:
                sl_price = entry_rate * (1 + sl_pct)
                if high >= sl_price:
                    should_exit, exit_reason, exit_price = True, "SL_HIT", sl_price
            else:
                sl_price = entry_rate * (1 - sl_pct)
                if low <= sl_price:
                    should_exit, exit_reason, exit_price = True, "SL_HIT", sl_price

            if not should_exit and exit_req is not None:
                should_exit, exit_reason, exit_price = True, exit_req.reason, close

            if should_exit:
                if open_trade.direction == Direction.SHORT:
                    p_raw = (entry_rate - exit_price) / entry_rate
                else:
                    p_raw = (exit_price - entry_rate) / entry_rate
                hold = (t.to_pydatetime() - open_trade.entry_time).total_seconds() / 3600
                funding = compute_funding(open_trade.entry_time, t.to_pydatetime())
                open_trade.exit_time = t.to_pydatetime()
                open_trade.exit_rate = exit_price
                open_trade.exit_reason = exit_reason
                open_trade.profit_pct = p_raw - 2 * TAKER_FEE - funding
                open_trade.hold_hours = hold
                trades.append(open_trade)
                open_trade = None

        if open_trade is None and t >= start_dt:
            sigs = strategy.detect_entries(df_warm.iloc[: i + 1], "ETH/USDT:USDT")
            if sigs:
                s = sigs[0]
                open_trade = Trade(
                    pair="ETH/USDT:USDT", direction=s.direction,
                    entry_time=t.to_pydatetime(), entry_rate=close,
                    strategy="ha_stoch_rev", tag=s.tag,
                )

    # Force-close any open trade
    if open_trade is not None:
        last = df_warm.iloc[-1]
        c = float(last["close"])
        et = pd.Timestamp(last["date"]).to_pydatetime()
        if open_trade.direction == Direction.SHORT:
            p_raw = (open_trade.entry_rate - c) / open_trade.entry_rate
        else:
            p_raw = (c - open_trade.entry_rate) / open_trade.entry_rate
        hold = (et - open_trade.entry_time).total_seconds() / 3600
        funding = compute_funding(open_trade.entry_time, et)
        open_trade.profit_pct = p_raw - 2 * TAKER_FEE - funding
        open_trade.hold_hours = hold
        open_trade.exit_reason = "end_of_data"
        open_trade.exit_time = et
        trades.append(open_trade)

    return trades


def summary(trades: list[Trade]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0, "pf": 0, "max_dd": 0}
    wins = sum(1 for t in trades if t.profit_pct > 0)
    gross_w = sum(t.profit_pct for t in trades if t.profit_pct > 0)
    gross_l = abs(sum(t.profit_pct for t in trades if t.profit_pct <= 0))
    pf = gross_w / max(0.0001, gross_l)
    peak, dd, cum = 0.0, 0.0, 0.0
    for t in trades:
        cum += t.profit_pct
        if cum > peak:
            peak = cum
        if (cum - peak) < dd:
            dd = cum - peak
    return {"n": n, "wr": wins / n, "pf": pf, "max_dd": abs(dd) * 100}


def monte_carlo_pvalue(trades: list[Trade], actual_pf: float, n_perms: int = 200) -> float:
    """Estimate p-value by randomly flipping trade directions (sign-flip permutation).

    Each permutation randomly assigns +/- to each trade P&L magnitude, then computes PF.
    p-value = fraction of permutations that achieve PF >= actual PF.
    """
    if len(trades) < 5:
        return 1.0
    magnitudes = [abs(t.profit_pct) for t in trades]
    beat = 0
    for _ in range(n_perms):
        signs = [random.choice([-1, 1]) for _ in magnitudes]
        pnls = [m * s for m, s in zip(magnitudes, signs)]
        gw = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p <= 0))
        perm_pf = gw / max(0.0001, gl)
        if perm_pf >= actual_pf:
            beat += 1
    return beat / n_perms


def main():
    print("=" * 78)
    print("WALK-FORWARD OOS VALIDATION: ha_stoch_rev")
    print("Protocol: 60d IS / 30d OOS, 30d step | ETH shorts-only")
    print("=" * 78)

    cfg = AppConfig("backtest")
    strat_cfg = cfg.get_strategy_config("ha_stoch_rev")
    StrategyClass = get_strategy_class("ha_stoch_rev")
    strategy = StrategyClass(strat_cfg)

    print(f"\nConfig: allow_longs={strat_cfg.entry.get('allow_longs', True)}, "
          f"ha_run_min={strat_cfg.entry.get('ha_run_min', 2)}, "
          f"stoch_k={strat_cfg.entry.get('stoch_k', 14)}, "
          f"sl_atr_mult={strat_cfg.exit.get('sl_atr_mult', 2.5)}, "
          f"tp_atr_mult={strat_cfg.exit.get('tp_atr_mult', 2.5)}")

    df_full = load_eth_data()
    print(f"\nData: {len(df_full)} bars "
          f"({df_full['date'].iloc[0].date()} to {df_full['date'].iloc[-1].date()})")

    oos_results = []
    print(f"\n{'Split':<6} {'IS Window':<25} {'OOS Window':<25} "
          f"{'Trades':>7} {'WR':>7} {'PF':>7} {'MaxDD%':>8} {'p-val':>7}")
    print("-" * 90)

    for i, (is_s, is_e, oos_s, oos_e) in enumerate(WALK_FORWARD_SPLITS):
        oos_trades = run_backtest(strategy, df_full, oos_s, oos_e)
        s = summary(oos_trades)
        p_val = monte_carlo_pvalue(oos_trades, s["pf"]) if s["n"] >= 5 else 1.0
        oos_results.append((s, p_val, oos_trades))

        flag = "(*)" if s["pf"] >= 1.5 and s["wr"] >= 0.52 else "   "
        print(f"  {i+1:<4} IS:{is_s}->{is_e}  OOS:{oos_s}->{oos_e}  "
              f"{s['n']:>7} {s['wr']:>6.1%} {s['pf']:>7.2f} {s['max_dd']:>7.2f}% "
              f"{p_val:>7.3f} {flag}")

        if oos_trades:
            reasons = Counter(t.exit_reason for t in oos_trades)
            for reason, cnt in reasons.most_common(3):
                avg_p = np.mean([t.profit_pct for t in oos_trades if t.exit_reason == reason]) * 100
                print(f"       {reason:<20} {cnt:>4}  avg P&L: {avg_p:+.3f}%")
        print()

    # ── Aggregate OOS assessment ───────────────────────────────────────────────
    all_oos = [t for _, _, ts in oos_results for t in ts]
    agg = summary(all_oos)
    avg_pf = np.mean([s["pf"] for s, _, _ in oos_results if s["n"] > 0])
    avg_wr = np.mean([s["wr"] for s, _, _ in oos_results if s["n"] > 0])
    avg_p = np.mean([p for _, p, _ in oos_results if oos_results[0][0]["n"] > 0])

    print("=" * 78)
    print("AGGREGATE OOS SUMMARY")
    print("=" * 78)
    print(f"  Total OOS trades:   {agg['n']}")
    print(f"  Overall OOS WR:     {agg['wr']:.1%}")
    print(f"  Overall OOS PF:     {agg['pf']:.2f}")
    print(f"  Max OOS Drawdown:   {agg['max_dd']:.2f}%")
    print(f"  Avg split PF:       {avg_pf:.2f}")
    print(f"  Avg split WR:       {avg_wr:.1%}")
    print(f"  Avg MC p-value:     {avg_p:.3f}")

    # Grade check
    grade_a = (agg["pf"] >= 1.5 and agg["wr"] >= 0.52 and agg["max_dd"] < 10.0 and avg_p < 0.05)
    grade_b = (agg["pf"] >= 1.3 and agg["wr"] >= 0.48 and agg["max_dd"] < 15.0 and avg_p < 0.10)

    print()
    if grade_a:
        print("  OOS GRADE: A  [PASS] Promote to dry-run")
    elif grade_b:
        print("  OOS GRADE: B  [PASS] Promote to dry-run (reduced stake)")
    else:
        print("  OOS GRADE: FAIL  [x] In-sample only - do not promote")

    print(f"  Grade A requires: PF>=1.5, WR>=52%, MaxDD<10%, p<0.05")
    print(f"  Got:              PF={agg['pf']:.2f}, WR={agg['wr']:.1%}, MaxDD={agg['max_dd']:.1f}%, p={avg_p:.3f}")

    # ── Save results ──────────────────────────────────────────────────────────
    out = ROOT / "backtest_results" / "walkforward_ha_stoch_rev.txt"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("ha_stoch_rev Walk-Forward OOS Validation\n")
        f.write("ETH/USDT:USDT, shorts-only\n")
        f.write(f"Protocol: 60d IS / 30d OOS, 3 splits\n\n")
        for i, (s, p_val, _) in enumerate(oos_results):
            _, _, oos_s, oos_e = WALK_FORWARD_SPLITS[i]
            f.write(f"Split {i+1}: OOS {oos_s} → {oos_e}\n")
            f.write(f"  Trades={s['n']}, WR={s['wr']:.1%}, PF={s['pf']:.2f}, "
                    f"MaxDD={s['max_dd']:.1f}%, MC_p={p_val:.3f}\n\n")
        f.write(f"\nAggregate OOS: Trades={agg['n']}, WR={agg['wr']:.1%}, "
                f"PF={agg['pf']:.2f}, MaxDD={agg['max_dd']:.1f}%\n")
        f.write(f"Grade A: {'PASS' if grade_a else 'FAIL'}, Grade B: {'PASS' if grade_b else 'FAIL'}\n")

    print(f"\n  Results saved to: {out}")


if __name__ == "__main__":
    main()

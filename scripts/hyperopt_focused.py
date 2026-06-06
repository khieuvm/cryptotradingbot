"""Focused hyperopt on promising strategies only.

Quick approach: test key parameter combinations for strategies that
show potential but need tuning.
"""

from __future__ import annotations

import sys
import itertools
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.config import AppConfig, StrategyConfig
from engine.events import Direction
from strategies import get_strategy_class

TAKER_FEE = 0.0005


def fast_backtest(strategy_name, config, pair, timerange):
    """Minimal backtest — returns (trades, win_rate, profit_factor, total_profit, max_dd)."""
    data_dir = Path(__file__).parent.parent / "data" / "okx" / "futures"
    filename = pair.replace("/", "_").replace(":", "_") + "-15m-futures.feather"
    filepath = data_dir / filename
    if not filepath.exists():
        return (0, 0, 0, 0, 0)

    df = pd.read_feather(filepath).sort_values("date").reset_index(drop=True)
    start, end = timerange
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    if len(df) < config.startup_candle_count + 30:
        return (0, 0, 0, 0, 0)

    StrategyClass = get_strategy_class(strategy_name)
    strategy = StrategyClass(config)
    df = strategy.compute_indicators(df, {"pair": pair})

    sl_mult = config.exit.get("sl_atr_mult", 2.0)
    open_trade = None
    profits = []
    start_idx = config.startup_candle_count

    for i in range(start_idx, len(df)):
        bar = df.iloc[i]
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        current_time = pd.Timestamp(bar["date"])

        if open_trade is not None:
            entry_rate = open_trade["entry_rate"]
            direction = open_trade["direction"]
            if direction == Direction.LONG:
                current_profit = (close - entry_rate) / entry_rate - 2 * TAKER_FEE
            else:
                current_profit = (entry_rate - close) / entry_rate - 2 * TAKER_FEE

            trade_info = {
                "current_profit": current_profit,
                "current_time": current_time.to_pydatetime(),
                "entry_time": open_trade["entry_time"],
                "entry_rate": entry_rate,
                "is_short": direction == Direction.SHORT,
            }
            sub_df = df.iloc[: i + 1]
            exit_req = strategy.detect_exits(sub_df, pair, trade_info)

            atr_col = None
            for col in df.columns:
                if col.endswith("_atr") and not col.endswith("_atr_ma"):
                    atr_col = col
                    break
            atr = float(bar.get(atr_col, 0)) if atr_col else 0
            sl_pct = sl_mult * atr / entry_rate if atr > 0 and entry_rate > 0 else 0.05

            should_exit = False
            exit_price = close
            if direction == Direction.LONG:
                if low <= entry_rate * (1 - sl_pct):
                    should_exit = True
                    exit_price = entry_rate * (1 - sl_pct)
            else:
                if high >= entry_rate * (1 + sl_pct):
                    should_exit = True
                    exit_price = entry_rate * (1 + sl_pct)

            if not should_exit and exit_req is not None:
                should_exit = True
                exit_price = close

            if should_exit:
                if direction == Direction.LONG:
                    profit = (exit_price - entry_rate) / entry_rate - 2 * TAKER_FEE
                else:
                    profit = (entry_rate - exit_price) / entry_rate - 2 * TAKER_FEE
                profits.append(profit)
                open_trade = None

        if open_trade is None:
            sub_df = df.iloc[: i + 1]
            signals = strategy.detect_entries(sub_df, pair)
            if signals:
                sig = signals[0]
                open_trade = {
                    "direction": sig.direction,
                    "entry_rate": close,
                    "entry_time": current_time.to_pydatetime(),
                }

    n = len(profits)
    if n == 0:
        return (0, 0, 0, 0, 0)

    wins = sum(1 for p in profits if p > 0)
    wr = wins / n
    gw = sum(p for p in profits if p > 0)
    gl = abs(sum(p for p in profits if p <= 0))
    pf = gw / max(0.0001, gl)
    total = sum(profits)
    peak = dd = cum = 0.0
    for p in profits:
        cum += p
        peak = max(peak, cum)
        dd = min(dd, cum - peak)

    return (n, wr, pf, total, abs(dd))


def make_config(base_cfg, strategy_name, entry_override, exit_override=None):
    """Create modified StrategyConfig."""
    strat_cfg = base_cfg.get_strategy_config(strategy_name)
    entry = {**strat_cfg.entry, **entry_override}
    exit_params = {**strat_cfg.exit, **(exit_override or {})}
    return StrategyConfig(
        name=strat_cfg.name, grade=strat_cfg.grade, pairs=strat_cfg.pairs,
        timeframe=strat_cfg.timeframe, startup_candle_count=strat_cfg.startup_candle_count,
        entry=entry, exit=exit_params, stake=strat_cfg.stake,
        protections=strat_cfg.protections, enabled=True,
    )


def grid_search(strategy_name, param_grid, exit_grid, pairs, base_cfg, timerange):
    """Grid search over entry and exit params."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combos = list(itertools.product(*values))

    print(f"\n  Testing {len(all_combos)} entry combos x {len(exit_grid)} exit combos = {len(all_combos)*len(exit_grid)} total")

    best_per_pair = {}

    for pair in pairs:
        best_score = -999
        best_result = None
        best_params = {}

        for combo in all_combos:
            entry_params = dict(zip(keys, combo))
            for exit_params in exit_grid:
                cfg = make_config(base_cfg, strategy_name, entry_params, exit_params)
                n, wr, pf, total, dd = fast_backtest(strategy_name, cfg, pair, timerange)

                if n < 8:
                    continue

                wr_bonus = 1.0 if wr >= 0.35 else 0.6
                trade_norm = min(1.0, n / 40)
                score = pf * wr_bonus * trade_norm - dd * 3

                if score > best_score:
                    best_score = score
                    best_result = (n, wr, pf, total, dd)
                    best_params = {**entry_params, **exit_params}

        best_per_pair[pair] = (best_result, best_params, best_score)

    return best_per_pair


def main():
    print("=" * 80)
    print("FOCUSED HYPEROPT — Promising Strategies Only")
    print("=" * 80)

    cfg = AppConfig("backtest")
    pairs = cfg.get_pairs()
    timerange = ("2026-01-01", "2026-05-22")

    exit_grid = [
        {"sl_atr_mult": 1.5, "tp_atr_mult": 2.5},
        {"sl_atr_mult": 2.0, "tp_atr_mult": 3.0},
        {"sl_atr_mult": 2.0, "tp_atr_mult": 4.0},
        {"sl_atr_mult": 2.5, "tp_atr_mult": 4.0},
        {"sl_atr_mult": 3.0, "tp_atr_mult": 5.0},
    ]

    # Strategy-specific param grids (focused on fixing the over-trading issue)
    strategies_to_opt = {
        "volume_spike_rev": {
            "spike_mult": [2.0, 2.5, 3.0],
            "shadow_body_ratio": [1.5, 2.0, 2.5],
            "rsi_os_thr": [30, 35],
            "rsi_ob_thr": [65, 70],
            "vol_min": [1.5, 2.0],
        },
        "donchian_breakout": {
            "dc_length": [12, 16, 20, 24],
            "adx_min": [20, 25, 30],
            "vol_min": [1.2, 1.5, 2.0],
            "dedup_bars": [8, 12, 16],
        },
        "nr7_breakout": {
            "range_atr_mult": [0.4, 0.5, 0.6],
            "vol_min": [1.5, 2.0, 2.5],
            "dedup_bars": [8, 12, 16],
        },
        "nr4_breakout": {
            "range_atr_mult": [0.3, 0.4, 0.5],
            "vol_min": [1.5, 2.0, 2.5],
            "dedup_bars": [8, 12, 16],
        },
        "vwap_meanrev": {
            "vwap_std_mult": [2.5, 3.0, 3.5],
            "rsi_os_thr": [25, 28, 30],
            "rsi_ob_thr": [70, 72, 75],
            "vol_min": [1.5, 2.0],
        },
        "micro_pullback": {
            "adx_min": [28, 32, 35],
            "vol_min": [1.5, 2.0, 2.5],
            "ema_fast": [5, 8],
        },
        "cb_adx_breakout": {
            "compression_thr": [0.6, 0.7, 0.8],
            "adx_max": [20, 25, 30],
            "vol_min": [0.6, 0.8, 1.0],
        },
    }

    all_results = {}

    for strat_name, param_grid in strategies_to_opt.items():
        print(f"\n{'=' * 80}")
        print(f"STRATEGY: {strat_name}")
        print(f"{'=' * 80}")

        results = grid_search(strat_name, param_grid, exit_grid, pairs, cfg, timerange)
        all_results[strat_name] = results

        for pair, (result, params, score) in results.items():
            if result and result[0] > 0:
                n, wr, pf, total, dd = result
                print(f"  {pair:20s} | T:{n:3d} WR:{wr:.1%} PF:{pf:.2f} P:{total:+.2%} DD:{dd:.2%} | {params}")
            else:
                print(f"  {pair:20s} | No viable result")

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL OPTIMIZED SUMMARY")
    print(f"{'=' * 80}")

    for strat_name, results in all_results.items():
        total_trades = 0
        total_profit = 0.0
        max_dd = 0.0
        viable_pairs = 0

        for pair, (result, params, score) in results.items():
            if result and result[0] >= 8:
                n, wr, pf, total, dd = result
                total_trades += n
                total_profit += total
                max_dd = max(max_dd, dd)
                viable_pairs += 1

        if total_trades > 0:
            print(f"  {strat_name:25s} | Viable pairs: {viable_pairs}/5 | T:{total_trades:3d} | "
                  f"Profit:{total_profit:+.2%} | MaxDD:{max_dd:.2%}")
        else:
            print(f"  {strat_name:25s} | REJECTED (no viable results)")


if __name__ == "__main__":
    main()

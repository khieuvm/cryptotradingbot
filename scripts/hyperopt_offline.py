"""Offline Hyperopt: optimize strategy parameters per pair.

Uses grid search + random search on key parameters for each strategy/pair combination.
Objective: maximize Profit Factor while maintaining WR > 30% and trades > 15.
"""

from __future__ import annotations

import sys
import itertools
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.config import AppConfig, StrategyConfig
from engine.events import Direction
from strategies import get_strategy_class

TAKER_FEE = 0.0005


@dataclass
class OptResult:
    strategy: str
    pair: str
    params: dict
    trades: int
    win_rate: float
    profit_factor: float
    total_profit: float
    max_drawdown: float
    avg_duration_h: float


def run_single_backtest(
    strategy_name: str,
    config: StrategyConfig,
    pair: str,
    timerange: tuple[str, str],
) -> OptResult:
    """Run backtest with given config, return metrics."""
    data_dir = Path(__file__).parent.parent / "data" / "okx" / "futures"
    filename = pair.replace("/", "_").replace(":", "_") + "-15m-futures.feather"
    filepath = data_dir / filename
    if not filepath.exists():
        return OptResult(strategy_name, pair, {}, 0, 0, 0, 0, 0, 0)

    df = pd.read_feather(filepath).sort_values("date").reset_index(drop=True)
    start, end = timerange
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    if len(df) < config.startup_candle_count + 30:
        return OptResult(strategy_name, pair, {}, 0, 0, 0, 0, 0, 0)

    StrategyClass = get_strategy_class(strategy_name)
    strategy = StrategyClass(config)
    df = strategy.compute_indicators(df, {"pair": pair})

    sl_mult = config.exit.get("sl_atr_mult", 2.0)
    open_trade = None
    trades = []
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

            # ATR SL check
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
                sl_price = entry_rate * (1 - sl_pct)
                if low <= sl_price:
                    should_exit = True
                    exit_price = sl_price
            else:
                sl_price = entry_rate * (1 + sl_pct)
                if high >= sl_price:
                    should_exit = True
                    exit_price = sl_price

            if not should_exit and exit_req is not None:
                should_exit = True
                exit_price = close

            if should_exit:
                if direction == Direction.LONG:
                    profit = (exit_price - entry_rate) / entry_rate - 2 * TAKER_FEE
                else:
                    profit = (entry_rate - exit_price) / entry_rate - 2 * TAKER_FEE
                trades.append({
                    "profit": profit,
                    "duration_h": (current_time - open_trade["entry_time"]).total_seconds() / 3600,
                })
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

    # Close remaining
    if open_trade is not None:
        last = df.iloc[-1]
        close = float(last["close"])
        d = open_trade["direction"]
        profit = ((close - open_trade["entry_rate"]) / open_trade["entry_rate"] - 2 * TAKER_FEE) \
            if d == Direction.LONG else \
            ((open_trade["entry_rate"] - close) / open_trade["entry_rate"] - 2 * TAKER_FEE)
        trades.append({"profit": profit, "duration_h": 24})

    # Metrics
    n = len(trades)
    if n == 0:
        return OptResult(strategy_name, pair, dict(config.entry), 0, 0, 0, 0, 0, 0)

    wins = sum(1 for t in trades if t["profit"] > 0)
    wr = wins / n
    gross_wins = sum(t["profit"] for t in trades if t["profit"] > 0)
    gross_losses = abs(sum(t["profit"] for t in trades if t["profit"] <= 0))
    pf = gross_wins / max(0.0001, gross_losses)
    total_profit = sum(t["profit"] for t in trades)

    peak = dd = cum = 0.0
    for t in trades:
        cum += t["profit"]
        peak = max(peak, cum)
        dd = min(dd, cum - peak)

    avg_dur = sum(t["duration_h"] for t in trades) / n

    return OptResult(
        strategy=strategy_name, pair=pair,
        params=dict(config.entry),
        trades=n, win_rate=wr, profit_factor=pf,
        total_profit=total_profit, max_drawdown=abs(dd),
        avg_duration_h=avg_dur,
    )


# Parameter search spaces per strategy
PARAM_SPACES = {
    "nr7_breakout": {
        "range_atr_mult": [0.6, 0.7, 0.8, 0.9],
        "vol_min": [0.8, 1.0, 1.2, 1.5],
        "dedup_bars": [3, 5, 7],
    },
    "nr4_breakout": {
        "range_atr_mult": [0.5, 0.6, 0.7, 0.8],
        "vol_min": [0.8, 1.0, 1.2, 1.5],
        "dedup_bars": [3, 4, 6],
    },
    "donchian_breakout": {
        "dc_length": [8, 12, 16, 20],
        "adx_min": [15, 20, 25],
        "vol_min": [0.8, 1.0, 1.2],
    },
    "cb_adx_breakout": {
        "compression_thr": [0.5, 0.6, 0.7],
        "adx_max": [15, 20, 25],
        "vol_min": [0.8, 1.0, 1.2],
    },
    "cb_nr7_breakout": {
        "compression_thr": [0.6, 0.7, 0.8],
        "nr_lookback": [5, 7, 9],
        "vol_min": [0.6, 0.8, 1.0],
    },
    "vwap_meanrev": {
        "vwap_std_mult": [1.5, 2.0, 2.5, 3.0],
        "rsi_os_thr": [25, 30, 35],
        "rsi_ob_thr": [65, 70, 75],
    },
    "micro_pullback": {
        "ema_fast": [5, 8, 10],
        "adx_min": [20, 25, 30],
        "vol_min": [0.6, 0.8, 1.0],
    },
    "volume_spike_rev": {
        "spike_mult": [1.5, 2.0, 2.5, 3.0],
        "shadow_body_ratio": [1.5, 2.0, 2.5],
        "rsi_os_thr": [30, 35, 40],
    },
    # Existing strategies
    "regime_adaptive": {
        "cross_lookback": [5, 8, 12],
        "vol_min": [0.8, 1.0, 1.5],
        "rsi_os": [28, 30, 34],
    },
    "meanrev_confluence": {
        "rsi_buy": [25, 28, 30, 33],
        "vol_mult": [1.0, 1.2, 1.5],
        "bb_std": [1.8, 2.0, 2.2],
    },
    "trend_composite": {
        "adx_min": [22, 25, 28],
        "vol_mult": [1.0, 1.2, 1.5],
        "cross_lookback": [2, 3, 5],
    },
    "compression_breakout": {
        "compression_thr": [0.5, 0.6, 0.7],
        "vol_min": [0.8, 1.0, 1.2],
        "dedup_bars": [4, 6, 8],
    },
    "volatility_compression": {
        "kc_mult": [1.5, 1.8, 2.0],
        "min_squeeze_bars": [6, 8, 10],
        "vol_min": [1.2, 1.5, 1.8],
    },
}

# Exit parameter spaces (searched separately)
EXIT_SPACES = {
    "sl_tp_ratio": [
        {"sl_atr_mult": 1.5, "tp_atr_mult": 2.5},
        {"sl_atr_mult": 2.0, "tp_atr_mult": 3.0},
        {"sl_atr_mult": 2.0, "tp_atr_mult": 3.5},
        {"sl_atr_mult": 2.5, "tp_atr_mult": 4.0},
        {"sl_atr_mult": 3.0, "tp_atr_mult": 5.0},
    ],
}


def hyperopt_strategy(
    strategy_name: str,
    pair: str,
    base_config: AppConfig,
    timerange: tuple[str, str],
    max_evals: int = 50,
) -> OptResult:
    """Run hyperopt for a single strategy/pair combination."""
    strat_cfg = base_config.get_strategy_config(strategy_name)
    space = PARAM_SPACES.get(strategy_name, {})

    if not space:
        return run_single_backtest(strategy_name, strat_cfg, pair, timerange)

    # Generate all combinations
    keys = list(space.keys())
    values = list(space.values())
    all_combos = list(itertools.product(*values))

    # Limit evaluations
    if len(all_combos) > max_evals:
        combos_to_test = random.sample(all_combos, max_evals)
    else:
        combos_to_test = all_combos

    best_result = None
    best_score = -999

    for combo in combos_to_test:
        params = dict(zip(keys, combo))
        test_entry = {**strat_cfg.entry, **params}
        test_cfg = StrategyConfig(
            name=strat_cfg.name,
            grade=strat_cfg.grade,
            pairs=strat_cfg.pairs,
            timeframe=strat_cfg.timeframe,
            startup_candle_count=strat_cfg.startup_candle_count,
            entry=test_entry,
            exit=strat_cfg.exit,
            stake=strat_cfg.stake,
            protections=strat_cfg.protections,
            enabled=True,
        )

        result = run_single_backtest(strategy_name, test_cfg, pair, timerange)

        # Score: PF * sqrt(trades) * wr_bonus, penalize < 15 trades
        if result.trades < 10:
            score = -10
        else:
            wr_bonus = 1.0 if result.win_rate >= 0.30 else 0.5
            trade_bonus = min(1.0, result.trades / 30)
            score = result.profit_factor * trade_bonus * wr_bonus - result.max_drawdown * 2

        if score > best_score:
            best_score = score
            best_result = result

    # Also try different SL/TP ratios with best entry params
    if best_result and best_result.trades > 0:
        for exit_combo in EXIT_SPACES["sl_tp_ratio"]:
            test_exit = {**strat_cfg.exit, **exit_combo}
            test_cfg = StrategyConfig(
                name=strat_cfg.name,
                grade=strat_cfg.grade,
                pairs=strat_cfg.pairs,
                timeframe=strat_cfg.timeframe,
                startup_candle_count=strat_cfg.startup_candle_count,
                entry={**strat_cfg.entry, **best_result.params} if best_result.params else strat_cfg.entry,
                exit=test_exit,
                stake=strat_cfg.stake,
                protections=strat_cfg.protections,
                enabled=True,
            )
            result = run_single_backtest(strategy_name, test_cfg, pair, timerange)
            if result.trades >= 10:
                wr_bonus = 1.0 if result.win_rate >= 0.30 else 0.5
                trade_bonus = min(1.0, result.trades / 30)
                score = result.profit_factor * trade_bonus * wr_bonus - result.max_drawdown * 2
                if score > best_score:
                    best_score = score
                    best_result = result

    return best_result or OptResult(strategy_name, pair, {}, 0, 0, 0, 0, 0, 0)


def main():
    print("=" * 80)
    print("HYPEROPT — Per-Strategy Per-Pair Parameter Optimization")
    print("=" * 80)

    cfg = AppConfig("backtest")
    pairs = cfg.get_pairs()
    timerange = ("2026-01-01", "2026-05-22")

    # All strategies to optimize (existing + new)
    all_strategies = [
        "regime_adaptive", "meanrev_confluence", "trend_composite",
        "compression_breakout", "volatility_compression",
        "nr7_breakout", "nr4_breakout", "donchian_breakout",
        "cb_adx_breakout", "cb_nr7_breakout",
        "vwap_meanrev", "micro_pullback", "volume_spike_rev",
    ]

    # Check which strategies are actually importable
    available = []
    for name in all_strategies:
        try:
            get_strategy_class(name)
            available.append(name)
        except (KeyError, Exception) as e:
            print(f"  SKIP {name}: {e}")

    print(f"\nStrategies: {len(available)}")
    print(f"Pairs: {pairs}")
    print(f"Timerange: {timerange[0]} to {timerange[1]}")
    print(f"Max evals per combo: 50")
    print()

    all_results: list[OptResult] = []

    for strat_name in available:
        strat_cfg = cfg.get_strategy_config(strat_name)
        strat_pairs = [p for p in strat_cfg.pairs if p in pairs]

        print(f"\n{'-' * 80}")
        print(f"Optimizing: {strat_name}")
        print(f"{'-' * 80}")

        for pair in strat_pairs:
            result = hyperopt_strategy(strat_name, pair, cfg, timerange)
            all_results.append(result)

            if result.trades > 0:
                print(
                    f"  {pair:20s} | T:{result.trades:3d} | "
                    f"WR:{result.win_rate:.1%} | PF:{result.profit_factor:.2f} | "
                    f"Profit:{result.total_profit:+.2%} | "
                    f"DD:{result.max_drawdown:.2%} | "
                    f"Best params: {result.params}"
                )
            else:
                print(f"  {pair:20s} | No viable params found")

    # Final report
    print(f"\n{'=' * 80}")
    print("FINAL REPORT — Best Parameters Per Strategy")
    print(f"{'=' * 80}")

    for strat_name in available:
        strat_results = [r for r in all_results if r.strategy == strat_name]
        total_trades = sum(r.trades for r in strat_results)
        total_profit = sum(r.total_profit for r in strat_results)
        total_wins = sum(int(r.trades * r.win_rate) for r in strat_results)
        max_dd = max((r.max_drawdown for r in strat_results), default=0)
        all_pf_trades = [(r.profit_factor, r.trades) for r in strat_results if r.trades > 0]
        avg_pf = sum(pf * t for pf, t in all_pf_trades) / max(1, sum(t for _, t in all_pf_trades))

        wr = total_wins / max(1, total_trades)

        # Grade assignment
        if avg_pf > 1.5 and wr > 0.52 and max_dd < 0.10:
            grade = "A"
        elif avg_pf > 1.3 and wr > 0.48 and max_dd < 0.15:
            grade = "B"
        elif avg_pf > 1.1 and wr > 0.45 and max_dd < 0.20:
            grade = "C"
        else:
            grade = "F"

        print(
            f"  {strat_name:25s} | Grade: {grade} | "
            f"T:{total_trades:3d} | WR:{wr:.1%} | PF:{avg_pf:.2f} | "
            f"Profit:{total_profit:+.2%} | MaxDD:{max_dd:.2%}"
        )

    grand_total = sum(r.total_trades for r in all_results)
    grand_profit = sum(r.total_profit for r in all_results)
    print(f"\n  {'PORTFOLIO TOTAL':25s} | T:{grand_total:3d} | Profit:{grand_profit:+.2%}")


if __name__ == "__main__":
    main()

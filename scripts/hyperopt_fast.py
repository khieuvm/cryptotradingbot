"""Fast Hyperopt: pre-compute indicators once, then iterate thresholds.

Key optimization: indicators are computed ONCE per strategy/pair, then we test
different entry/exit thresholds by scanning the pre-computed columns directly.
This is 50-100x faster than re-running full bar-by-bar simulation per combo.
"""
from __future__ import annotations

import sys
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.config import AppConfig, StrategyConfig
from engine.events import Direction
from strategies import get_strategy_class

TAKER_FEE = 0.0005


def load_data(pair: str, timerange: tuple[str, str]) -> pd.DataFrame:
    data_dir = Path(__file__).parent.parent / "data" / "okx" / "futures"
    filename = pair.replace("/", "_").replace(":", "_") + "-15m-futures.feather"
    filepath = data_dir / filename
    if not filepath.exists():
        return pd.DataFrame()
    df = pd.read_feather(filepath).sort_values("date").reset_index(drop=True)
    start, end = timerange
    return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def vectorized_backtest(df: pd.DataFrame, entries: np.ndarray, directions: np.ndarray,
                        sl_mult: float, tp_mult: float, atr_col: str,
                        startup: int, dedup_bars: int = 6) -> dict:
    """Fast vectorized-ish backtest given entry signals array."""
    n = len(df)
    if n < startup + 30:
        return {"trades": 0, "wr": 0, "pf": 0, "profit": 0, "dd": 0}

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    atr = df[atr_col].values.astype(float) if atr_col in df.columns else np.full(n, 0.01)

    profits = []
    last_trade_bar = -dedup_bars - 1
    in_trade = False
    entry_rate = 0.0
    entry_dir = 1
    entry_bar = 0

    for i in range(startup, n):
        if in_trade:
            sl_pct = sl_mult * atr[i] / entry_rate if atr[i] > 0 and entry_rate > 0 else 0.05
            tp_pct = tp_mult * atr[i] / entry_rate if atr[i] > 0 and entry_rate > 0 else 0.10

            if entry_dir == 1:  # LONG
                if low[i] <= entry_rate * (1 - sl_pct):
                    profit = -sl_pct - 2 * TAKER_FEE
                    profits.append(profit)
                    in_trade = False
                elif high[i] >= entry_rate * (1 + tp_pct):
                    profit = tp_pct - 2 * TAKER_FEE
                    profits.append(profit)
                    in_trade = False
                elif i - entry_bar > 96:  # 24h time cut
                    profit = (close[i] - entry_rate) / entry_rate - 2 * TAKER_FEE
                    profits.append(profit)
                    in_trade = False
            else:  # SHORT
                if high[i] >= entry_rate * (1 + sl_pct):
                    profit = -sl_pct - 2 * TAKER_FEE
                    profits.append(profit)
                    in_trade = False
                elif low[i] <= entry_rate * (1 - tp_pct):
                    profit = tp_pct - 2 * TAKER_FEE
                    profits.append(profit)
                    in_trade = False
                elif i - entry_bar > 96:
                    profit = (entry_rate - close[i]) / entry_rate - 2 * TAKER_FEE
                    profits.append(profit)
                    in_trade = False

        if not in_trade and entries[i] and (i - last_trade_bar >= dedup_bars):
            entry_rate = close[i]
            entry_dir = directions[i]
            entry_bar = i
            last_trade_bar = i
            in_trade = True

    # Close remaining
    if in_trade:
        if entry_dir == 1:
            profit = (close[-1] - entry_rate) / entry_rate - 2 * TAKER_FEE
        else:
            profit = (entry_rate - close[-1]) / entry_rate - 2 * TAKER_FEE
        profits.append(profit)

    ntrades = len(profits)
    if ntrades == 0:
        return {"trades": 0, "wr": 0, "pf": 0, "profit": 0, "dd": 0}

    wins = sum(1 for p in profits if p > 0)
    wr = wins / ntrades
    gw = sum(p for p in profits if p > 0)
    gl = abs(sum(p for p in profits if p <= 0))
    pf = gw / max(0.0001, gl)
    total = sum(profits)
    peak = dd = cum = 0.0
    for p in profits:
        cum += p
        peak = max(peak, cum)
        dd = min(dd, cum - peak)

    return {"trades": ntrades, "wr": wr, "pf": pf, "profit": total, "dd": abs(dd)}


def find_atr_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.endswith("_atr") and not col.endswith("_atr_ma"):
            return col
    return ""


# === Strategy-specific vectorized signal generators ===

def gen_signals_nr7(df, startup, nr_lookback=7, range_atr_mult=0.8, vol_min=1.0):
    n = len(df)
    entries = np.zeros(n, dtype=bool)
    directions = np.zeros(n, dtype=int)

    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    ranges = h - l
    atr_col = find_atr_col(df)
    atr = df[atr_col].values.astype(float) if atr_col else np.ones(n)
    vol_col = [col for col in df.columns if "volume_ratio" in col or col == "volume"]
    vol = df[vol_col[0]].values.astype(float) if vol_col else np.ones(n)

    for i in range(max(startup, nr_lookback), n):
        bar_range = ranges[i]
        lookback_ranges = ranges[i - nr_lookback + 1:i + 1]
        if bar_range <= 0 or atr[i] <= 0:
            continue
        if bar_range > np.min(lookback_ranges) * 1.001:
            continue
        if bar_range > range_atr_mult * atr[i]:
            continue
        if vol_col and vol[i] < vol_min:
            continue
        # Direction: close vs open
        o = (h[i] + l[i]) / 2  # approximate open
        entries[i] = True
        directions[i] = 1 if c[i] > o else -1

    return entries, directions


def gen_signals_nr4(df, startup, nr_lookback=4, range_atr_mult=0.7, vol_min=1.0):
    return gen_signals_nr7(df, startup, nr_lookback, range_atr_mult, vol_min)


def gen_signals_donchian(df, startup, dc_length=12, adx_min=20, vol_min=1.0, **kwargs):
    n = len(df)
    entries = np.zeros(n, dtype=bool)
    directions = np.zeros(n, dtype=int)

    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    adx_col = [col for col in df.columns if "adx" in col.lower() and "plus" not in col.lower() and "minus" not in col.lower()]
    adx = df[adx_col[0]].values.astype(float) if adx_col else np.full(n, 30.0)
    vol_col = [col for col in df.columns if "volume_ratio" in col]
    vol = df[vol_col[0]].values.astype(float) if vol_col else np.ones(n)

    for i in range(max(startup, dc_length + 1), n):
        if adx[i] < adx_min:
            continue
        if vol_col and vol[i] < vol_min:
            continue
        dc_high = np.max(h[i - dc_length:i])
        dc_low = np.min(l[i - dc_length:i])
        if c[i] > dc_high:
            entries[i] = True
            directions[i] = 1
        elif c[i] < dc_low:
            entries[i] = True
            directions[i] = -1

    return entries, directions


def gen_signals_vwap(df, startup, vwap_std_mult=2.0, rsi_os_thr=30, rsi_ob_thr=70, vol_min=1.0, **kwargs):
    n = len(df)
    entries = np.zeros(n, dtype=bool)
    directions = np.zeros(n, dtype=int)

    c = df["close"].values.astype(float)
    rsi_col = [col for col in df.columns if "rsi" in col.lower()]
    rsi = df[rsi_col[0]].values.astype(float) if rsi_col else np.full(n, 50.0)
    vol_col = [col for col in df.columns if "volume_ratio" in col]
    vol = df[vol_col[0]].values.astype(float) if vol_col else np.ones(n)

    # Compute rolling VWAP-like mean and std
    window = 96
    rolling_mean = pd.Series(c).rolling(window).mean().values
    rolling_std = pd.Series(c).rolling(window).std().values

    for i in range(max(startup, window), n):
        if np.isnan(rolling_mean[i]) or np.isnan(rolling_std[i]) or rolling_std[i] == 0:
            continue
        if vol_col and vol[i] < vol_min:
            continue
        zscore = (c[i] - rolling_mean[i]) / rolling_std[i]

        if zscore < -vwap_std_mult and rsi[i] < rsi_os_thr:
            entries[i] = True
            directions[i] = 1
        elif zscore > vwap_std_mult and rsi[i] > rsi_ob_thr:
            entries[i] = True
            directions[i] = -1

    return entries, directions


def gen_signals_micro_pullback(df, startup, ema_fast=8, adx_min=25, vol_min=0.8, **kwargs):
    n = len(df)
    entries = np.zeros(n, dtype=bool)
    directions = np.zeros(n, dtype=int)

    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)

    ema = pd.Series(c).ewm(span=ema_fast).mean().values
    ema_slow = pd.Series(c).ewm(span=21).mean().values
    adx_col = [col for col in df.columns if "adx" in col.lower() and "plus" not in col.lower() and "minus" not in col.lower()]
    adx = df[adx_col[0]].values.astype(float) if adx_col else np.full(n, 30.0)
    vol_col = [col for col in df.columns if "volume_ratio" in col]
    vol = df[vol_col[0]].values.astype(float) if vol_col else np.ones(n)

    for i in range(max(startup, 22), n):
        if adx[i] < adx_min:
            continue
        if vol_col and vol[i] < vol_min:
            continue

        # Uptrend: EMA fast > EMA slow, pullback touches EMA fast
        if ema[i] > ema_slow[i] and l[i] <= ema[i] <= h[i]:
            entries[i] = True
            directions[i] = 1
        # Downtrend: EMA fast < EMA slow, pullback touches EMA fast
        elif ema[i] < ema_slow[i] and l[i] <= ema[i] <= h[i]:
            entries[i] = True
            directions[i] = -1

    return entries, directions


def gen_signals_volume_spike(df, startup, spike_mult=2.0, shadow_body_ratio=2.0,
                             rsi_os_thr=35, rsi_ob_thr=65, vol_min=1.5, **kwargs):
    n = len(df)
    entries = np.zeros(n, dtype=bool)
    directions = np.zeros(n, dtype=int)

    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)
    rsi_col = [col for col in df.columns if "rsi" in col.lower()]
    rsi = df[rsi_col[0]].values.astype(float) if rsi_col else np.full(n, 50.0)

    vol_ma = pd.Series(v).rolling(20).mean().values

    for i in range(max(startup, 21), n):
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            continue
        if v[i] < spike_mult * vol_ma[i]:
            continue

        body = abs(c[i] - o[i])
        if body < 0.0001:
            body = 0.0001

        upper_shadow = h[i] - max(c[i], o[i])
        lower_shadow = min(c[i], o[i]) - l[i]

        # Hammer (bullish reversal) - long lower shadow
        if lower_shadow / body >= shadow_body_ratio and rsi[i] < rsi_os_thr:
            entries[i] = True
            directions[i] = 1
        # Shooting star (bearish reversal) - long upper shadow
        elif upper_shadow / body >= shadow_body_ratio and rsi[i] > rsi_ob_thr:
            entries[i] = True
            directions[i] = -1

    return entries, directions


def gen_signals_cb_adx(df, startup, compression_thr=0.6, adx_max=20, vol_min=1.0, **kwargs):
    n = len(df)
    entries = np.zeros(n, dtype=bool)
    directions = np.zeros(n, dtype=int)

    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    atr_col = find_atr_col(df)
    atr = df[atr_col].values.astype(float) if atr_col else np.ones(n)
    adx_col = [col for col in df.columns if "adx" in col.lower() and "plus" not in col.lower() and "minus" not in col.lower()]
    adx = df[adx_col[0]].values.astype(float) if adx_col else np.full(n, 30.0)
    vol_col = [col for col in df.columns if "volume_ratio" in col]
    vol = df[vol_col[0]].values.astype(float) if vol_col else np.ones(n)

    for i in range(max(startup, 5), n):
        if atr[i] <= 0:
            continue
        # 3-bar compression
        max_range = max(h[j] - l[j] for j in range(i - 2, i + 1))
        if max_range > compression_thr * atr[i]:
            continue
        # ADX < threshold (any of last 3 bars)
        if not any(adx[max(0, i - 2):i + 1] < adx_max):
            continue
        if vol_col and vol[i] < vol_min:
            continue
        entries[i] = True
        directions[i] = 1 if c[i] > o[i] else -1

    return entries, directions


# Signal generator registry
SIGNAL_GENERATORS = {
    "nr7_breakout": gen_signals_nr7,
    "nr4_breakout": gen_signals_nr4,
    "donchian_breakout": gen_signals_donchian,
    "vwap_meanrev": gen_signals_vwap,
    "micro_pullback": gen_signals_micro_pullback,
    "volume_spike_rev": gen_signals_volume_spike,
    "cb_adx_breakout": gen_signals_cb_adx,
}

# Parameter grids
PARAM_GRIDS = {
    "nr7_breakout": {
        "nr_lookback": [7],
        "range_atr_mult": [0.5, 0.6, 0.7, 0.8],
        "vol_min": [1.0, 1.5, 2.0],
    },
    "nr4_breakout": {
        "nr_lookback": [4],
        "range_atr_mult": [0.4, 0.5, 0.6, 0.7],
        "vol_min": [1.0, 1.5, 2.0],
    },
    "donchian_breakout": {
        "dc_length": [12, 16, 20, 24],
        "adx_min": [20, 25, 30],
        "vol_min": [1.2, 1.5, 2.0],
    },
    "vwap_meanrev": {
        "vwap_std_mult": [2.0, 2.5, 3.0, 3.5],
        "rsi_os_thr": [25, 28, 30],
        "rsi_ob_thr": [70, 72, 75],
        "vol_min": [1.0, 1.5, 2.0],
    },
    "micro_pullback": {
        "ema_fast": [5, 8, 10],
        "adx_min": [25, 30, 35],
        "vol_min": [1.0, 1.5, 2.0],
    },
    "volume_spike_rev": {
        "spike_mult": [2.0, 2.5, 3.0],
        "shadow_body_ratio": [1.5, 2.0, 2.5],
        "rsi_os_thr": [28, 32, 35],
        "rsi_ob_thr": [65, 68, 72],
        "vol_min": [1.5, 2.0],
    },
    "cb_adx_breakout": {
        "compression_thr": [0.5, 0.6, 0.7],
        "adx_max": [18, 22, 25],
        "vol_min": [0.8, 1.0, 1.2],
    },
}

EXIT_GRID = [
    {"sl": 1.5, "tp": 2.5},
    {"sl": 2.0, "tp": 3.0},
    {"sl": 2.0, "tp": 4.0},
    {"sl": 2.5, "tp": 4.0},
    {"sl": 3.0, "tp": 5.0},
]


def hyperopt_single(strategy_name, pair, df, startup):
    """Optimize one strategy on one pair. Returns best result dict."""
    gen_func = SIGNAL_GENERATORS.get(strategy_name)
    if gen_func is None:
        return None

    grid = PARAM_GRIDS.get(strategy_name, {})
    keys = list(grid.keys())
    values = list(grid.values())
    all_combos = list(itertools.product(*values))

    atr_col = find_atr_col(df)
    best_score = -999
    best_result = None
    best_params = {}
    tested = 0

    for combo in all_combos:
        params = dict(zip(keys, combo))
        entries, directions = gen_func(df, startup, **params)

        for exit_combo in EXIT_GRID:
            sl_mult = exit_combo["sl"]
            tp_mult = exit_combo["tp"]
            dedup = params.get("dedup_bars", 6)

            result = vectorized_backtest(
                df, entries, directions, sl_mult, tp_mult, atr_col, startup, dedup
            )
            tested += 1

            n = result["trades"]
            if n < 8:
                continue

            wr = result["wr"]
            pf = result["pf"]
            dd = result["dd"]
            wr_bonus = 1.0 if wr >= 0.35 else 0.6
            trade_norm = min(1.0, n / 40)
            score = pf * wr_bonus * trade_norm - dd * 3

            if score > best_score:
                best_score = score
                best_result = result
                best_params = {**params, "sl_atr_mult": sl_mult, "tp_atr_mult": tp_mult}

    return {"result": best_result, "params": best_params, "score": best_score, "tested": tested}


def main():
    print("=" * 80)
    print("FAST HYPEROPT - Vectorized Parameter Optimization")
    print("=" * 80)

    cfg = AppConfig("backtest")
    pairs = cfg.get_pairs()
    timerange = ("2026-01-01", "2026-05-22")

    strategies_to_opt = list(PARAM_GRIDS.keys())

    # Pre-compute indicators for each strategy/pair
    print(f"\nStrategies: {strategies_to_opt}")
    print(f"Pairs: {pairs}")
    print(f"Timerange: {timerange[0]} to {timerange[1]}")

    all_results = {}

    for strat_name in strategies_to_opt:
        strat_cfg = cfg.get_strategy_config(strat_name)
        StrategyClass = get_strategy_class(strat_name)

        print(f"\n{'=' * 80}")
        print(f"STRATEGY: {strat_name}")
        print(f"{'=' * 80}")

        strat_results = {}

        for pair in strat_cfg.pairs:
            if pair not in pairs:
                continue

            df = load_data(pair, timerange)
            if df.empty or len(df) < strat_cfg.startup_candle_count + 50:
                print(f"  {pair:20s} | No data")
                continue

            # Compute indicators once
            strategy = StrategyClass(strat_cfg)
            df = strategy.compute_indicators(df, {"pair": pair})

            # Run hyperopt
            opt = hyperopt_single(strat_name, pair, df, strat_cfg.startup_candle_count)
            if opt and opt["result"] and opt["result"]["trades"] > 0:
                r = opt["result"]
                print(
                    f"  {pair:20s} | T:{r['trades']:3d} WR:{r['wr']:.1%} PF:{r['pf']:.2f} "
                    f"P:{r['profit']:+.2%} DD:{r['dd']:.2%} | {opt['params']} "
                    f"(tested {opt['tested']})"
                )
                strat_results[pair] = opt
            else:
                print(f"  {pair:20s} | No viable result (tested {opt['tested'] if opt else 0})")
                strat_results[pair] = None

        all_results[strat_name] = strat_results

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL OPTIMIZED SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'Strategy':<25s} | {'Viable':<8s} | {'Trades':<7s} | {'WR':<7s} | {'PF':<6s} | {'Profit':<10s} | {'MaxDD':<8s} | Grade")
    print("-" * 100)

    for strat_name, results in all_results.items():
        total_trades = 0
        total_profit = 0.0
        max_dd = 0.0
        viable_pairs = 0
        total_wr_weighted = 0.0

        for pair, opt in results.items():
            if opt and opt["result"] and opt["result"]["trades"] >= 8:
                r = opt["result"]
                total_trades += r["trades"]
                total_profit += r["profit"]
                max_dd = max(max_dd, r["dd"])
                viable_pairs += 1
                total_wr_weighted += r["wr"] * r["trades"]

        avg_wr = total_wr_weighted / max(1, total_trades)
        avg_pf = 0
        if total_trades > 0:
            # Recalculate aggregate PF
            gw = gl = 0
            for pair, opt in results.items():
                if opt and opt["result"] and opt["result"]["trades"] >= 8:
                    r = opt["result"]
                    # Approximate from profit and wr
                    gw += r["profit"] if r["profit"] > 0 else 0
                    gl += abs(r["profit"]) if r["profit"] < 0 else 0
            avg_pf = (total_profit + gl) / max(0.0001, gl) if gl > 0 else (2.0 if total_profit > 0 else 0)

        # Grade
        if avg_pf > 1.5 and avg_wr > 0.50 and max_dd < 0.10:
            grade = "A"
        elif avg_pf > 1.3 and avg_wr > 0.45 and max_dd < 0.15:
            grade = "B"
        elif avg_pf > 1.1 and avg_wr > 0.40 and max_dd < 0.20:
            grade = "C"
        else:
            grade = "F"

        status = f"{viable_pairs}/{len(results)}"
        print(
            f"  {strat_name:<23s} | {status:<8s} | {total_trades:<7d} | {avg_wr:<6.1%} | "
            f"{avg_pf:<5.2f} | {total_profit:<+9.2%} | {max_dd:<7.2%} | {grade}"
        )

    # Also run baseline strategies (regime_adaptive, meanrev, trend, compression, vol_compression)
    print(f"\n{'=' * 80}")
    print("EXISTING STRATEGIES (baseline, no hyperopt)")
    print(f"{'=' * 80}")

    existing = ["regime_adaptive", "meanrev_confluence", "trend_composite",
                "compression_breakout", "volatility_compression"]

    for strat_name in existing:
        strat_cfg = cfg.get_strategy_config(strat_name)
        StrategyClass = get_strategy_class(strat_name)
        total_trades = 0
        total_profit = 0.0
        max_dd = 0.0
        viable = 0

        for pair in strat_cfg.pairs:
            if pair not in pairs:
                continue
            df = load_data(pair, timerange)
            if df.empty or len(df) < strat_cfg.startup_candle_count + 50:
                continue

            strategy = StrategyClass(strat_cfg)
            df_ind = strategy.compute_indicators(df, {"pair": pair})

            # Quick bar-by-bar for existing strategies (they have complex entry logic)
            open_trade = None
            profits = []
            sl_mult = strat_cfg.exit.get("sl_atr_mult", 2.0)
            tp_mult = strat_cfg.exit.get("tp_atr_mult", 3.0)
            atr_col = find_atr_col(df_ind)

            for i in range(strat_cfg.startup_candle_count, len(df_ind)):
                bar = df_ind.iloc[i]
                close_val = float(bar["close"])
                high_val = float(bar["high"])
                low_val = float(bar["low"])
                current_time = pd.Timestamp(bar["date"])

                if open_trade is not None:
                    entry_rate = open_trade["entry_rate"]
                    direction = open_trade["direction"]
                    atr_val = float(bar.get(atr_col, 0)) if atr_col else 0
                    sl_pct = sl_mult * atr_val / entry_rate if atr_val > 0 and entry_rate > 0 else 0.05
                    tp_pct = tp_mult * atr_val / entry_rate if atr_val > 0 and entry_rate > 0 else 0.10

                    should_exit = False
                    exit_price = close_val

                    if direction == Direction.LONG:
                        if low_val <= entry_rate * (1 - sl_pct):
                            should_exit = True
                            exit_price = entry_rate * (1 - sl_pct)
                        elif high_val >= entry_rate * (1 + tp_pct):
                            should_exit = True
                            exit_price = entry_rate * (1 + tp_pct)
                    else:
                        if high_val >= entry_rate * (1 + sl_pct):
                            should_exit = True
                            exit_price = entry_rate * (1 + sl_pct)
                        elif low_val <= entry_rate * (1 - tp_pct):
                            should_exit = True
                            exit_price = entry_rate * (1 - tp_pct)

                    # Time cut at 24h
                    if not should_exit and i - open_trade["bar"] > 96:
                        should_exit = True
                        exit_price = close_val

                    if should_exit:
                        if direction == Direction.LONG:
                            profit = (exit_price - entry_rate) / entry_rate - 2 * TAKER_FEE
                        else:
                            profit = (entry_rate - exit_price) / entry_rate - 2 * TAKER_FEE
                        profits.append(profit)
                        open_trade = None

                if open_trade is None:
                    sub_df = df_ind.iloc[:i + 1]
                    signals = strategy.detect_entries(sub_df, pair)
                    if signals:
                        sig = signals[0]
                        open_trade = {
                            "direction": sig.direction,
                            "entry_rate": close_val,
                            "bar": i,
                        }

            ntrades = len(profits)
            if ntrades > 0:
                wins = sum(1 for p in profits if p > 0)
                wr = wins / ntrades
                gw = sum(p for p in profits if p > 0)
                gl_val = abs(sum(p for p in profits if p <= 0))
                pf = gw / max(0.0001, gl_val)
                tot = sum(profits)
                pk = mdd = cm = 0.0
                for p in profits:
                    cm += p
                    pk = max(pk, cm)
                    mdd = min(mdd, cm - pk)

                total_trades += ntrades
                total_profit += tot
                max_dd = max(max_dd, abs(mdd))
                if ntrades >= 5:
                    viable += 1

                print(f"  {strat_name:23s} {pair:20s} | T:{ntrades:3d} WR:{wr:.1%} PF:{pf:.2f} P:{tot:+.2%} DD:{abs(mdd):.2%}")

        if total_trades > 0:
            print(f"  {'>>> TOTAL':23s} {'':20s} | T:{total_trades:3d} P:{total_profit:+.2%} MaxDD:{max_dd:.2%}")

    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

"""Vectorized hyperopt for 5m scalping strategies.

Pre-computes all indicators once, then iterates parameter combinations.
Much faster than bar-by-bar backtest for finding optimal parameters.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, str(Path(__file__).parent.parent))

TAKER_FEE = 0.0005
DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT"]


def load_5m_data(pair_file: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{pair_file}-5m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp).sort_values("date").reset_index(drop=True)
    start = "2026-01-01"
    end = "2026-05-21"
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    return df


def vectorized_backtest(entries: np.ndarray, exits_sl: np.ndarray, exits_tp: np.ndarray,
                        close: np.ndarray, atr: np.ndarray, sl_mult: float, tp_mult: float,
                        max_bars: int = 18) -> dict:
    """Vectorized backtest: for each entry, check if SL or TP hits first within max_bars."""
    n = len(close)
    trades = []

    i = 0
    while i < n:
        if not entries[i]:
            i += 1
            continue

        entry_price = close[i]
        entry_atr = atr[i]
        if entry_atr <= 0 or entry_price <= 0:
            i += 1
            continue

        sl_dist = sl_mult * entry_atr
        tp_dist = tp_mult * entry_atr

        # Determine direction from signal (positive = long implied by entry logic)
        # We'll handle direction in the signal generator
        is_long = entries[i] > 0  # positive = long, negative = short

        if is_long:
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        # Scan forward for exit
        exited = False
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if is_long:
                # Check SL (low touches SL)
                if exits_sl[j] <= sl_price:
                    pnl = -sl_mult * entry_atr / entry_price - 2 * TAKER_FEE
                    trades.append(pnl)
                    exited = True
                    i = j + 1
                    break
                # Check TP (high touches TP)
                if exits_tp[j] >= tp_price:
                    pnl = tp_mult * entry_atr / entry_price - 2 * TAKER_FEE
                    trades.append(pnl)
                    exited = True
                    i = j + 1
                    break
            else:
                if exits_tp[j] >= sl_price:
                    pnl = -sl_mult * entry_atr / entry_price - 2 * TAKER_FEE
                    trades.append(pnl)
                    exited = True
                    i = j + 1
                    break
                if exits_sl[j] <= tp_price:
                    pnl = tp_mult * entry_atr / entry_price - 2 * TAKER_FEE
                    trades.append(pnl)
                    exited = True
                    i = j + 1
                    break

        if not exited:
            # Time cut: exit at close of max_bars
            exit_idx = min(i + max_bars, n - 1)
            if is_long:
                pnl = (close[exit_idx] - entry_price) / entry_price - 2 * TAKER_FEE
            else:
                pnl = (entry_price - close[exit_idx]) / entry_price - 2 * TAKER_FEE
            trades.append(pnl)
            i = exit_idx + 1

    if not trades:
        return {"trades": 0, "wr": 0, "pf": 0, "profit": 0, "max_dd": 0}

    trades_arr = np.array(trades)
    wins = trades_arr[trades_arr > 0]
    losses = trades_arr[trades_arr <= 0]
    wr = len(wins) / len(trades_arr)
    pf = wins.sum() / max(abs(losses.sum()), 0.0001) if len(losses) > 0 else 99.0

    cum = np.cumsum(trades_arr)
    peak = np.maximum.accumulate(cum)
    dd = np.min(cum - peak)

    return {
        "trades": len(trades_arr),
        "wr": wr,
        "pf": pf,
        "profit": trades_arr.sum(),
        "max_dd": abs(dd),
    }


# =============================================================================
# SIGNAL GENERATORS
# =============================================================================

def gen_squeeze_signals(df: pd.DataFrame, bb_len: int, bb_std: float,
                        kc_len: int, kc_mult: float, min_squeeze: int,
                        vol_min: float) -> np.ndarray:
    """Generate squeeze breakout signals."""
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    v = df["volume"].values.astype(float)
    n = len(df)

    bb = ta.bbands(df["close"], length=bb_len, std=bb_std)
    if bb is None:
        return np.zeros(n)
    bb_upper = bb.iloc[:, 0].values
    bb_lower = bb.iloc[:, 2].values

    kc_mid = ta.ema(df["close"], length=kc_len).values
    kc_atr = ta.atr(df["high"], df["low"], df["close"], length=kc_len).values
    kc_upper = kc_mid + kc_mult * kc_atr
    kc_lower = kc_mid - kc_mult * kc_atr

    vol_ema = ta.ema(df["volume"].astype(float), length=20).values
    vol_ratio = v / (vol_ema + 1e-10)

    # Squeeze ON: BB inside KC
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    # Count consecutive squeeze bars
    squeeze_count = np.zeros(n)
    for i in range(1, n):
        if squeeze_on[i]:
            squeeze_count[i] = squeeze_count[i - 1] + 1

    # Squeeze fire: was on, now off
    signals = np.zeros(n)
    for i in range(1, n):
        if squeeze_on[i - 1] and not squeeze_on[i]:
            if squeeze_count[i - 1] >= min_squeeze and vol_ratio[i] >= vol_min:
                # Direction: close vs BB mid
                if c[i] > bb_upper[i]:
                    signals[i] = 1  # Long
                elif c[i] < bb_lower[i]:
                    signals[i] = -1  # Short

    return signals


def gen_nr_signals(df: pd.DataFrame, nr_lookback: int, range_atr_max: float,
                   adx_max: float, vol_min: float) -> np.ndarray:
    """Generate NR breakout signals."""
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    v = df["volume"].values.astype(float)
    n = len(df)

    bar_range = h - lo
    atr = ta.atr(df["high"], df["low"], df["close"], length=14).values

    adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
    adx = adx_result.iloc[:, 0].values if adx_result is not None else np.full(n, 25.0)

    vol_ema = ta.ema(df["volume"].astype(float), length=20).values
    vol_ratio = v / (vol_ema + 1e-10)

    signals = np.zeros(n)

    for i in range(nr_lookback + 1, n):
        # Previous bar must be NR (narrowest in lookback)
        prev_range = bar_range[i - 1]
        lookback_min = np.min(bar_range[i - nr_lookback:i])

        if prev_range > lookback_min * 1.01:
            continue
        if prev_range / (atr[i - 1] + 1e-10) > range_atr_max:
            continue
        if adx[i] > adx_max:
            continue
        if vol_ratio[i] < vol_min:
            continue

        prev_high = h[i - 1]
        prev_low = lo[i - 1]

        if c[i] > prev_high:
            signals[i] = 1
        elif c[i] < prev_low:
            signals[i] = -1

    return signals


def gen_volume_climax_signals(df: pd.DataFrame, spike_mult: float,
                              shadow_ratio: float, rsi_os: float,
                              rsi_ob: float) -> np.ndarray:
    """Generate volume climax reversal signals."""
    c = df["close"].values
    o = df["open"].values
    h = df["high"].values
    lo = df["low"].values
    v = df["volume"].values.astype(float)
    n = len(df)

    vol_ema = ta.ema(df["volume"].astype(float), length=20).values
    vol_ratio = v / (vol_ema + 1e-10)

    rsi = ta.rsi(df["close"], length=9).values

    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is None:
        return np.zeros(n)
    bb_upper = bb.iloc[:, 0].values
    bb_lower = bb.iloc[:, 2].values

    body = c - o
    body_abs = np.abs(body)
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(c, o) - lo

    signals = np.zeros(n)

    for i in range(20, n):
        if vol_ratio[i] < spike_mult:
            continue

        # Long: hammer + RSI oversold + lower BB touch
        is_hammer = (lower_shadow[i] > shadow_ratio * body_abs[i]) and body_abs[i] > 0
        if is_hammer and rsi[i] < rsi_os and lo[i] <= bb_lower[i]:
            signals[i] = 1
            continue

        # Short: shooting star + RSI overbought + upper BB touch
        is_star = (upper_shadow[i] > shadow_ratio * body_abs[i]) and body_abs[i] > 0
        if is_star and rsi[i] > rsi_ob and h[i] >= bb_upper[i]:
            signals[i] = -1

    return signals


# =============================================================================
# HYPEROPT
# =============================================================================

def run_hyperopt():
    print("=" * 75)
    print("5m SCALPING HYPEROPT — Finding Optimal Parameters")
    print("=" * 75)

    # Load all data
    all_data = {}
    for pair in PAIRS:
        df = load_5m_data(pair)
        if not df.empty:
            all_data[pair] = df
            print(f"  Loaded {pair}: {len(df)} bars")

    print(f"\n{'=' * 75}")

    # ─── SQUEEZE BREAKOUT HYPEROPT ────────────────────────────────────────
    print("\n## SQUEEZE BREAKOUT 5m — Hyperopt")
    print("-" * 75)

    best_squeeze = {"profit": -999, "params": {}}
    squeeze_params = [
        (bb_len, bb_std, kc_mult, min_sq, vol_min, sl, tp)
        for bb_len in [14, 20]
        for bb_std in [1.8, 2.0, 2.2]
        for kc_mult in [1.0, 1.2, 1.5]
        for min_sq in [2, 3, 5]
        for vol_min in [0.8, 1.0, 1.3]
        for sl in [1.0, 1.5, 2.0]
        for tp in [1.5, 2.0, 2.5, 3.0]
    ]
    print(f"  Testing {len(squeeze_params)} parameter combinations...")

    for bb_len, bb_std, kc_mult, min_sq, vol_min, sl, tp in squeeze_params:
        total_profit = 0
        total_trades = 0
        for pair, df in all_data.items():
            signals = gen_squeeze_signals(df, bb_len, bb_std, bb_len, kc_mult, min_sq, vol_min)
            atr = ta.atr(df["high"], df["low"], df["close"], length=14).values
            entries = signals
            result = vectorized_backtest(
                entries, df["low"].values, df["high"].values,
                df["close"].values, atr, sl, tp, max_bars=18
            )
            total_profit += result["profit"]
            total_trades += result["trades"]

        if total_trades >= 15 and total_profit > best_squeeze["profit"]:
            best_squeeze = {
                "profit": total_profit,
                "trades": total_trades,
                "params": {"bb_len": bb_len, "bb_std": bb_std, "kc_mult": kc_mult,
                           "min_sq": min_sq, "vol_min": vol_min, "sl": sl, "tp": tp}
            }

    if best_squeeze.get("trades"):
        print(f"  BEST: profit={best_squeeze['profit']:+.2%} | trades={best_squeeze['trades']}")
        print(f"  Params: {best_squeeze['params']}")
    else:
        print("  No viable parameter set found (all < 30 trades)")

    # ─── NR BREAKOUT HYPEROPT ─────────────────────────────────────────────
    print("\n## NR BREAKOUT 5m — Hyperopt")
    print("-" * 75)

    best_nr = {"profit": -999, "params": {}}
    nr_params = [
        (lookback, range_max, adx_max, vol_min, sl, tp)
        for lookback in [4, 7, 10]
        for range_max in [0.4, 0.5, 0.6, 0.7]
        for adx_max in [18, 20, 22, 25]
        for vol_min in [1.0, 1.3, 1.5]
        for sl in [1.0, 1.5, 2.0]
        for tp in [2.0, 2.5, 3.0, 4.0]
    ]
    print(f"  Testing {len(nr_params)} parameter combinations...")

    for lookback, range_max, adx_max, vol_min, sl, tp in nr_params:
        total_profit = 0
        total_trades = 0
        for pair, df in all_data.items():
            signals = gen_nr_signals(df, lookback, range_max, adx_max, vol_min)
            atr = ta.atr(df["high"], df["low"], df["close"], length=14).values
            result = vectorized_backtest(
                signals, df["low"].values, df["high"].values,
                df["close"].values, atr, sl, tp, max_bars=18
            )
            total_profit += result["profit"]
            total_trades += result["trades"]

        if total_trades >= 50 and total_profit > best_nr["profit"]:
            best_nr = {
                "profit": total_profit,
                "trades": total_trades,
                "params": {"lookback": lookback, "range_max": range_max, "adx_max": adx_max,
                           "vol_min": vol_min, "sl": sl, "tp": tp}
            }

    print(f"  BEST: profit={best_nr['profit']:+.2%} | trades={best_nr['trades']}")
    print(f"  Params: {best_nr['params']}")

    # ─── VOLUME CLIMAX HYPEROPT ───────────────────────────────────────────
    print("\n## VOLUME CLIMAX 5m — Hyperopt")
    print("-" * 75)

    best_vc = {"profit": -999, "params": {}}
    vc_params = [
        (spike, shadow, rsi_os, rsi_ob, sl, tp)
        for spike in [2.5, 3.0, 4.0, 5.0]
        for shadow in [1.5, 2.0, 2.5, 3.0]
        for rsi_os in [20, 25, 30]
        for rsi_ob in [70, 75, 80]
        for sl in [1.0, 1.5, 2.0]
        for tp in [1.5, 2.0, 2.5, 3.0]
    ]
    print(f"  Testing {len(vc_params)} parameter combinations...")

    for spike, shadow, rsi_os, rsi_ob, sl, tp in vc_params:
        total_profit = 0
        total_trades = 0
        for pair, df in all_data.items():
            signals = gen_volume_climax_signals(df, spike, shadow, rsi_os, rsi_ob)
            atr = ta.atr(df["high"], df["low"], df["close"], length=14).values
            result = vectorized_backtest(
                signals, df["low"].values, df["high"].values,
                df["close"].values, atr, sl, tp, max_bars=12
            )
            total_profit += result["profit"]
            total_trades += result["trades"]

        if total_trades >= 30 and total_profit > best_vc["profit"]:
            best_vc = {
                "profit": total_profit,
                "trades": total_trades,
                "params": {"spike": spike, "shadow": shadow, "rsi_os": rsi_os,
                           "rsi_ob": rsi_ob, "sl": sl, "tp": tp}
            }

    if best_vc.get("trades"):
        print(f"  BEST: profit={best_vc['profit']:+.2%} | trades={best_vc['trades']}")
        print(f"  Params: {best_vc['params']}")
    else:
        print("  No viable parameter set found")

    # ─── SUMMARY ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("HYPEROPT SUMMARY")
    print(f"{'=' * 75}")
    print(f"  squeeze_breakout_5m: {best_squeeze.get('profit', 0):+.2%} ({best_squeeze.get('trades', 0)} trades)")
    print(f"  nr_breakout_5m:      {best_nr.get('profit', 0):+.2%} ({best_nr.get('trades', 0)} trades)")
    print(f"  volume_climax_5m:    {best_vc.get('profit', 0):+.2%} ({best_vc.get('trades', 0)} trades)")


if __name__ == "__main__":
    run_hyperopt()

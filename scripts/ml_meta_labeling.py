"""ML Experiment 2: Meta-Labeling on 15m regime_adaptive signals.

Instead of predicting direction, ML answers a SIMPLER question:
"Given that regime_adaptive just fired a signal, will this trade be profitable?"

Workflow:
1. Reproduce regime_adaptive signals on 15m data
2. For each signal, compute 5m microstructure features at signal time
3. Label: 1 = trade hit TP before SL (winner), 0 = trade hit SL (loser)
4. Train binary classifier to filter signals
5. Only take signals where ML confidence > threshold

Expected benefit: regime_adaptive already has ~52% WR. If ML filters out
the worst 30-40% of signals, WR could rise to 60-65%, dramatically improving PF.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
MAKER = 0.0002
TAKER = 0.0005
PAIRS_15M = ["SOL_USDT_USDT", "SPX_USDT_USDT"]


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp)
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


def load_5m(pair):
    fp = DATA_DIR / f"{pair}-5m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp)
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


def compute_ra_indicators(df):
    """Reproduce regime_adaptive indicators on 15m data."""
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    df["ra_ema_fast"] = ta.ema(c, length=21)
    df["ra_ema_slow"] = ta.ema(c, length=60)
    df["ra_ema200"] = ta.ema(c, length=200)

    df["ra_atr"] = ta.atr(h, lo, c, length=14)
    df["ra_atr_ma"] = ta.ema(df["ra_atr"], length=50)
    df["ra_atr_ratio"] = df["ra_atr"] / (df["ra_atr_ma"] + 1e-10)

    adx = ta.adx(h, lo, c, length=14)
    if adx is not None:
        df["ra_adx"] = adx.iloc[:, 0]
        df["ra_plus_di"] = adx.iloc[:, 1]
        df["ra_minus_di"] = adx.iloc[:, 2]
    else:
        df["ra_adx"] = df["ra_plus_di"] = df["ra_minus_di"] = 0.0

    df["ra_rsi"] = ta.rsi(c, length=14)

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["ra_macd_hist"] = macd.iloc[:, 1]
    else:
        df["ra_macd_hist"] = 0.0

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        df["ra_bb_upper"] = bb.iloc[:, 0]
        df["ra_bb_mid"] = bb.iloc[:, 1]
        df["ra_bb_lower"] = bb.iloc[:, 2]
    else:
        df["ra_bb_upper"] = df["ra_bb_lower"] = df["ra_bb_mid"] = c

    df["ra_vol_ema"] = ta.ema(v, length=20)
    df["ra_vol_ratio"] = v / (df["ra_vol_ema"] + 1e-10)

    df["ra_obv"] = ta.obv(c, v)
    df["ra_obv_ema"] = ta.ema(df["ra_obv"], length=20)
    df["ra_obv_rising"] = (df["ra_obv"] > df["ra_obv_ema"]).astype(int)

    try:
        st = ta.supertrend(h, lo, c, length=7, multiplier=3.0)
        if st is not None:
            st_dir_col = next((col for col in st.columns if "SUPERTd" in col), None)
            df["ra_st_dir"] = st[st_dir_col].fillna(0) if st_dir_col else 0
        else:
            df["ra_st_dir"] = 0
    except Exception:
        df["ra_st_dir"] = 0

    df["ra_is_bull"] = (c > df["ra_ema200"]).astype(int)
    df["ra_is_bear"] = (c < df["ra_ema200"]).astype(int)

    cross_up = (df["ra_ema_fast"] > df["ra_ema_slow"]) & (
        df["ra_ema_fast"].shift(1) <= df["ra_ema_slow"].shift(1)
    )
    cross_down = (df["ra_ema_fast"] < df["ra_ema_slow"]) & (
        df["ra_ema_fast"].shift(1) >= df["ra_ema_slow"].shift(1)
    )
    df["ra_cross_up_recent"] = cross_up.rolling(5).max().fillna(0).astype(int)
    df["ra_cross_down_recent"] = cross_down.rolling(5).max().fillna(0).astype(int)

    return df


def detect_ra_signals(df):
    """Detect regime_adaptive signals. Returns list of (bar_index, direction, regime)."""
    signals = []
    adx_thr = 25
    rsi_os = 28
    rsi_ob = 67
    vol_min = 1.0
    atr_spike = 2.2

    for i in range(220, len(df)):
        row = df.iloc[i]

        if float(row.get("ra_atr_ratio", 0)) >= atr_spike:
            continue
        if float(row.get("ra_vol_ratio", 0)) < vol_min:
            continue
        if float(row.get("volume", 0)) <= 0:
            continue

        is_trending = float(row.get("ra_adx", 0)) > adx_thr
        is_bull = int(row.get("ra_is_bull", 0)) == 1
        is_bear = int(row.get("ra_is_bear", 0)) == 1

        if is_trending:
            if (
                is_bull
                and int(row.get("ra_cross_up_recent", 0)) == 1
                and float(row.get("ra_ema_fast", 0)) > float(row.get("ra_ema_slow", 0))
                and float(row.get("ra_macd_hist", 0)) > 0
                and float(row.get("ra_plus_di", 0)) > float(row.get("ra_minus_di", 0))
                and int(row.get("ra_st_dir", 0)) == 1
            ):
                signals.append((i, "long", "trending"))

            if (
                is_bear
                and int(row.get("ra_cross_down_recent", 0)) == 1
                and float(row.get("ra_ema_fast", 0)) < float(row.get("ra_ema_slow", 0))
                and float(row.get("ra_macd_hist", 0)) < 0
                and float(row.get("ra_minus_di", 0)) > float(row.get("ra_plus_di", 0))
                and int(row.get("ra_st_dir", 0)) == -1
            ):
                signals.append((i, "short", "trending"))
        else:
            rsi = float(row.get("ra_rsi", 50))
            prev_rsi = float(df.iloc[i - 1].get("ra_rsi", 50))

            if (
                prev_rsi < rsi_os
                and rsi > prev_rsi
                and float(row["close"]) < float(row.get("ra_bb_lower", 0)) * 1.01
                and float(row["close"]) > float(row["open"])
                and int(row.get("ra_obv_rising", 0)) == 1
            ):
                signals.append((i, "long", "ranging"))

            if (
                prev_rsi > rsi_ob
                and rsi < prev_rsi
                and float(row["close"]) > float(row.get("ra_bb_upper", 0)) * 0.99
                and float(row["close"]) < float(row["open"])
                and int(row.get("ra_obv_rising", 0)) == 0
            ):
                signals.append((i, "short", "ranging"))

    return signals


def label_signals_triple_barrier(df_15m, signals, sl_mult=10.0, tp_mult=11.0, max_bars=96):
    """Label each signal with triple-barrier on 15m bars.

    Uses regime_adaptive config: SL=10x ATR, TP=11x ATR, max 96 bars (24h).
    """
    close = df_15m["close"].values
    high = df_15m["high"].values
    low = df_15m["low"].values
    atr = df_15m["ra_atr"].values

    labels = []

    for bar_idx, direction, regime in signals:
        if bar_idx + max_bars >= len(close):
            labels.append(0)
            continue

        entry = close[bar_idx]
        a = atr[bar_idx]
        if np.isnan(a) or a <= 0:
            labels.append(0)
            continue

        if direction == "long":
            tp_price = entry + tp_mult * a
            sl_price = entry - sl_mult * a
        else:
            tp_price = entry - tp_mult * a
            sl_price = entry + sl_mult * a

        result = 0
        for j in range(1, max_bars + 1):
            idx = bar_idx + j
            if idx >= len(close):
                break

            if direction == "long":
                if low[idx] <= sl_price:
                    result = -1
                    break
                if high[idx] >= tp_price:
                    result = 1
                    break
            else:
                if high[idx] >= sl_price:
                    result = -1
                    break
                if low[idx] <= tp_price:
                    result = 1
                    break

        # Time-cut: check profit at max_bars
        if result == 0:
            exit_price = close[min(bar_idx + max_bars, len(close) - 1)]
            pnl = (exit_price - entry) / entry if direction == "long" else (entry - exit_price) / entry
            result = 1 if pnl > 0 else -1

        labels.append(1 if result == 1 else 0)

    return np.array(labels)


def compute_meta_features(df_15m, signals):
    """Compute features AT THE TIME of each signal for meta-labeling.

    These features help ML decide if THIS PARTICULAR signal is likely to win.
    Includes signal context, market microstructure, and regime quality metrics.
    """
    features_list = []

    for bar_idx, direction, regime in signals:
        row = df_15m.iloc[bar_idx]
        feat = {}

        # Signal context
        feat["direction_long"] = 1.0 if direction == "long" else 0.0
        feat["regime_trending"] = 1.0 if regime == "trending" else 0.0

        # Trend strength at signal time
        feat["adx"] = float(row.get("ra_adx", 0))
        feat["di_diff"] = float(row.get("ra_plus_di", 0)) - float(row.get("ra_minus_di", 0))
        feat["ema_spread"] = (float(row.get("ra_ema_fast", 0)) - float(row.get("ra_ema_slow", 0))) / (float(row.get("ra_ema_slow", 1)) + 1e-10)
        feat["price_vs_ema200"] = (float(row["close"]) - float(row.get("ra_ema200", row["close"]))) / (float(row.get("ra_ema200", row["close"])) + 1e-10)

        # Momentum
        feat["rsi"] = float(row.get("ra_rsi", 50))
        feat["macd_hist"] = float(row.get("ra_macd_hist", 0))

        # Volatility context
        feat["atr_ratio"] = float(row.get("ra_atr_ratio", 1))
        feat["atr_pct"] = float(row.get("ra_atr", 0)) / (float(row["close"]) + 1e-10)

        # Volume context
        feat["vol_ratio"] = float(row.get("ra_vol_ratio", 1))

        # BB position
        bb_upper = float(row.get("ra_bb_upper", row["close"]))
        bb_lower = float(row.get("ra_bb_lower", row["close"]))
        bb_width = bb_upper - bb_lower
        feat["bb_position"] = (float(row["close"]) - bb_lower) / (bb_width + 1e-10)
        feat["bb_width_pct"] = bb_width / (float(row["close"]) + 1e-10)

        # Recent price action (momentum into signal)
        if bar_idx >= 5:
            c = df_15m["close"].values
            feat["ret_1bar"] = (c[bar_idx] - c[bar_idx - 1]) / (c[bar_idx - 1] + 1e-10)
            feat["ret_3bar"] = (c[bar_idx] - c[bar_idx - 3]) / (c[bar_idx - 3] + 1e-10)
            feat["ret_5bar"] = (c[bar_idx] - c[bar_idx - 5]) / (c[bar_idx - 5] + 1e-10)
            feat["ret_12bar"] = (c[bar_idx] - c[bar_idx - 12]) / (c[bar_idx - 12] + 1e-10) if bar_idx >= 12 else 0
            feat["ret_24bar"] = (c[bar_idx] - c[bar_idx - 24]) / (c[bar_idx - 24] + 1e-10) if bar_idx >= 24 else 0
        else:
            feat["ret_1bar"] = feat["ret_3bar"] = feat["ret_5bar"] = 0
            feat["ret_12bar"] = feat["ret_24bar"] = 0

        # Candle structure at signal
        feat["body_pct"] = (float(row["close"]) - float(row["open"])) / (float(row["close"]) + 1e-10)
        feat["range_pct"] = (float(row["high"]) - float(row["low"])) / (float(row["close"]) + 1e-10)

        # Time of day
        dt = pd.to_datetime(row["date"])
        feat["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)
        feat["is_asia"] = 1.0 if 0 <= dt.hour < 8 else 0.0
        feat["is_europe"] = 1.0 if 8 <= dt.hour < 13 else 0.0
        feat["is_us"] = 1.0 if 13 <= dt.hour < 21 else 0.0

        # Recent volatility trend
        if bar_idx >= 10:
            atr_vals = df_15m["ra_atr"].values
            feat["atr_trend"] = (atr_vals[bar_idx] - atr_vals[bar_idx - 10]) / (atr_vals[bar_idx - 10] + 1e-10)
        else:
            feat["atr_trend"] = 0

        # OBV momentum
        feat["obv_rising"] = float(row.get("ra_obv_rising", 0))

        # SuperTrend alignment
        feat["st_dir"] = float(row.get("ra_st_dir", 0))
        feat["st_aligned"] = 1.0 if (
            (direction == "long" and feat["st_dir"] == 1) or
            (direction == "short" and feat["st_dir"] == -1)
        ) else 0.0

        features_list.append(feat)

    return pd.DataFrame(features_list)


def walk_forward_meta(features, labels, signals, pair_name,
                      train_signals=100, min_test_signals=30):
    """Walk-forward meta-labeling evaluation.

    Since signals are sparse (few per day), use signal count for splitting
    instead of fixed time windows.
    """
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 12,
        "max_depth": 3,
        "min_child_samples": 20,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    n = len(features)
    if n < train_signals + min_test_signals:
        return None, None

    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = {thr: [] for thr in thresholds}

    # Split into folds: 60% train, 40% test (to have enough signals)
    # But walk-forward: multiple train/test splits
    step = max(min_test_signals, n // 5)

    fold_num = 0
    start = 0
    while start + train_signals + min_test_signals <= n:
        fold_num += 1
        train_end = start + train_signals
        test_end = min(train_end + step, n)

        X_train = features.iloc[start:train_end]
        y_train = labels[start:train_end]
        X_test = features.iloc[train_end:test_end]
        y_test = labels[train_end:test_end]

        if len(X_train) < 50 or len(X_test) < 15:
            start += step
            continue

        pw = sum(y_train == 0) / max(sum(y_train == 1), 1)
        model = lgb.train(
            {**params, "scale_pos_weight": pw},
            lgb.Dataset(X_train, label=y_train),
            num_boost_round=200,
        )

        prob = model.predict(X_test)

        for thr in thresholds:
            for idx in range(len(X_test)):
                if prob[idx] >= thr:
                    results[thr].append((fold_num, y_test[idx], prob[idx]))

        start += step

    return results, fold_num


def evaluate_meta_results(results, signals, labels, pair_name, sl_mult, tp_mult):
    """Compare filtered vs unfiltered performance."""
    # Baseline: all signals without ML filter
    base_wr = labels.mean() * 100
    base_n = len(labels)

    # Compute average ATR pct for PnL estimation
    avg_tp_pct = tp_mult * 0.003  # approximate for 15m
    avg_sl_pct = sl_mult * 0.003

    base_pf = (base_wr / 100 * avg_tp_pct) / ((1 - base_wr / 100) * avg_sl_pct + 1e-10)

    print(f"\n  BASELINE (no ML filter): {base_n} signals, WR={base_wr:.1f}%, est. PF={base_pf:.2f}")
    print(f"  (SL={sl_mult}x ATR, TP={tp_mult}x ATR)")

    print(f"\n  {'Thr':<6} {'Signals':<9} {'Taken':<7} {'Filter%':<9} "
          f"{'WR':<8} {'WR Gain':<9} {'Est PF':<8} {'Consistent?'}")
    print(f"  {'-' * 75}")

    for thr in sorted(results.keys()):
        trades = results[thr]
        if not trades:
            continue

        taken = len(trades)
        filter_pct = (1 - taken / base_n) * 100 if base_n > 0 else 0
        wins = sum(1 for _, label, _ in trades if label == 1)
        wr = wins / taken * 100 if taken > 0 else 0
        wr_gain = wr - base_wr

        pf = (wr / 100 * avg_tp_pct) / ((1 - wr / 100) * avg_sl_pct + 1e-10)

        # Per-fold consistency
        folds = {}
        for fold_num, label, _ in trades:
            folds.setdefault(fold_num, []).append(label)

        consistent = True
        fold_details = []
        for f in sorted(folds.keys()):
            f_wr = sum(folds[f]) / len(folds[f]) * 100
            if f_wr < base_wr:
                consistent = False
            fold_details.append(f"F{f}:{f_wr:.0f}%")

        status = "YES" if consistent and len(folds) >= 2 else "no"
        marker = " <<<" if wr_gain > 5 and consistent else ""

        print(f"  {thr:<6.2f} {base_n:<9} {taken:<7} {filter_pct:<8.1f}% "
              f"{wr:<7.1f}% {wr_gain:<+8.1f}% {pf:<7.2f} {status} ({', '.join(fold_details)}){marker}")

    return base_wr, base_pf


def main():
    print("=" * 90)
    print("ML EXPERIMENT 2: Meta-Labeling (Filter 15m regime_adaptive signals)")
    print("Question: 'Which regime_adaptive signals will be winners?'")
    print("Benefit: Even +5pp WR improvement on a 52% base dramatically improves PF")
    print("=" * 90)

    # Test with both the actual production SL/TP config
    sl_tp_configs = [
        (10.0, 11.0, 96, "Production (SL=10, TP=11, 24h)"),
        (7.0, 11.0, 96, "Tighter SL (SL=7, TP=11, 24h)"),
        (5.0, 8.0, 48, "Faster (SL=5, TP=8, 12h)"),
    ]

    for pair in PAIRS_15M:
        print(f"\n{'=' * 90}")
        print(f"  {pair}")
        print(f"{'=' * 90}")

        df_15m = load_15m(pair)
        if df_15m.empty:
            print(f"  No 15m data for {pair}")
            continue

        df_15m = compute_ra_indicators(df_15m)

        # Detect signals
        signals = detect_ra_signals(df_15m)
        print(f"  Total signals detected: {len(signals)}")

        if len(signals) < 50:
            print(f"  Too few signals for meta-labeling")
            continue

        # Signal breakdown
        n_long = sum(1 for _, d, _ in signals if d == "long")
        n_short = sum(1 for _, d, _ in signals if d == "short")
        n_trend = sum(1 for _, _, r in signals if r == "trending")
        n_range = sum(1 for _, _, r in signals if r == "ranging")
        print(f"  Long={n_long}, Short={n_short} | Trending={n_trend}, Ranging={n_range}")

        # Compute meta-features
        meta_features = compute_meta_features(df_15m, signals)
        print(f"  Meta-features: {meta_features.shape[1]} columns")

        for sl_mult, tp_mult, max_bars, config_name in sl_tp_configs:
            print(f"\n  -- {config_name} --")

            # Label signals
            labels = label_signals_triple_barrier(
                df_15m, signals, sl_mult=sl_mult, tp_mult=tp_mult, max_bars=max_bars
            )

            base_wr = labels.mean() * 100
            print(f"  Signal quality: {sum(labels)}/{len(labels)} winners ({base_wr:.1f}% WR)")

            if sum(labels) < 10 or sum(labels == 0) < 10:
                print(f"  Too imbalanced for ML training")
                continue

            # Walk-forward meta-labeling
            results, num_folds = walk_forward_meta(
                meta_features, labels, signals, pair,
                train_signals=max(80, len(signals) // 3),
                min_test_signals=max(20, len(signals) // 6),
            )

            if results is None:
                print(f"  Not enough signals for walk-forward")
                continue

            evaluate_meta_results(results, signals, labels, pair, sl_mult, tp_mult)

    # ─── WHAT THIS MEANS ─────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("INTERPRETATION")
    print(f"{'=' * 90}")
    print("""
  Success criteria for meta-labeling:
  1. WR gain > +5pp (meaningful filter improvement)
  2. Consistent across folds (not overfitting to one regime)
  3. Filter removes < 60% of signals (still enough trading)
  4. PF improvement (from 1.2 -> 1.5+)

  If successful, implementation path:
  1. regime_adaptive fires signal normally
  2. At confirm_trade_entry(), ML model evaluates signal
  3. If P(win) < threshold -> reject trade
  4. If P(win) >= threshold -> allow trade
  5. Retrain model every 30 days on latest data

  Key advantage over standalone ML:
  - ML doesn't need to predict direction (hard problem)
  - ML only needs to predict quality of an EXISTING signal (easier)
  - Base strategy already has proven edge
  - ML just removes the noise signals
    """)


if __name__ == "__main__":
    main()

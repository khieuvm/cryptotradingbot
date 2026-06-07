"""ML Ensemble for 5m Crypto Scalping.

Pipeline:
1. Feature engineering (60+ features)
2. Label generation (future return with SL/TP simulation)
3. Walk-forward train/test split (no look-ahead)
4. LightGBM classifier: predict profitable trade opportunities
5. Feature importance analysis

Target: identify bars where entering long/short has >50% probability of profit
after accounting for fees (0.10% RT taker, or 0.04% maker).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
TAKER = 0.0005  # per side
PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT"]


def load(pair):
    fp = DATA_DIR / f"{pair}-5m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp)
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate 60+ features from OHLCV data."""
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    o = df["open"]
    v = df["volume"].astype(float)

    feat = pd.DataFrame(index=df.index)

    # ── Price Action ─────────────────────────────────────────────────────
    feat["body_pct"] = (c - o) / (c + 1e-10)
    feat["upper_shadow_pct"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (c + 1e-10)
    feat["lower_shadow_pct"] = (pd.concat([c, o], axis=1).min(axis=1) - lo) / (c + 1e-10)
    feat["range_pct"] = (h - lo) / (c + 1e-10)

    # Returns over various lookbacks
    for lb in [1, 2, 3, 5, 8, 13]:
        feat[f"ret_{lb}"] = c.pct_change(lb)

    # High/low relative to recent range
    for w in [5, 10, 20]:
        feat[f"pos_in_range_{w}"] = (c - lo.rolling(w).min()) / (h.rolling(w).max() - lo.rolling(w).min() + 1e-10)

    # Gap (open vs prev close)
    feat["gap_pct"] = (o - c.shift(1)) / (c.shift(1) + 1e-10)

    # ── Momentum ─────────────────────────────────────────────────────────
    feat["rsi_3"] = ta.rsi(c, length=3)
    feat["rsi_9"] = ta.rsi(c, length=9)
    feat["rsi_14"] = ta.rsi(c, length=14)

    # RSI rate of change
    feat["rsi_9_roc"] = feat["rsi_9"] - feat["rsi_9"].shift(3)

    # Stochastic
    stoch = ta.stoch(h, lo, c, k=14, d=3)
    if stoch is not None:
        feat["stoch_k"] = stoch.iloc[:, 0]
        feat["stoch_d"] = stoch.iloc[:, 1]
        feat["stoch_diff"] = feat["stoch_k"] - feat["stoch_d"]

    # MACD
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat["macd_hist"] = macd.iloc[:, 2]  # histogram
        feat["macd_hist_change"] = feat["macd_hist"] - feat["macd_hist"].shift(1)

    # CCI
    feat["cci_14"] = ta.cci(h, lo, c, length=14)

    # Williams %R
    feat["willr_14"] = ta.willr(h, lo, c, length=14)

    # ── Trend ────────────────────────────────────────────────────────────
    ema8 = ta.ema(c, length=8)
    ema21 = ta.ema(c, length=21)
    ema50 = ta.ema(c, length=50)

    feat["ema8_dist"] = (c - ema8) / (c + 1e-10)
    feat["ema21_dist"] = (c - ema21) / (c + 1e-10)
    feat["ema50_dist"] = (c - ema50) / (c + 1e-10)
    feat["ema8_21_diff"] = (ema8 - ema21) / (c + 1e-10)
    feat["ema21_50_diff"] = (ema21 - ema50) / (c + 1e-10)
    feat["ema8_slope"] = (ema8 - ema8.shift(3)) / (c + 1e-10)

    # ADX + DI
    adx_r = ta.adx(h, lo, c, length=14)
    if adx_r is not None:
        feat["adx"] = adx_r.iloc[:, 0]
        feat["plus_di"] = adx_r.iloc[:, 1]
        feat["minus_di"] = adx_r.iloc[:, 2]
        feat["di_diff"] = feat["plus_di"] - feat["minus_di"]
        feat["adx_change"] = feat["adx"] - feat["adx"].shift(3)

    # ── Volatility ───────────────────────────────────────────────────────
    atr14 = ta.atr(h, lo, c, length=14)
    atr5 = ta.atr(h, lo, c, length=5)
    feat["atr14_pct"] = atr14 / (c + 1e-10)
    feat["atr5_pct"] = atr5 / (c + 1e-10)
    feat["atr_ratio"] = atr5 / (atr14 + 1e-10)  # Short ATR vs long ATR (expansion/contraction)
    feat["range_vs_atr"] = (h - lo) / (atr14 + 1e-10)

    # BB
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bb_upper = bb.iloc[:, 0]
        bb_mid = bb.iloc[:, 1]
        bb_lower = bb.iloc[:, 2]
        feat["bb_pos"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)
        feat["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
        feat["bb_width_change"] = feat["bb_width"] - feat["bb_width"].shift(5)

    # Keltner Channel position
    kc_mid = ta.ema(c, length=20)
    kc_atr = ta.atr(h, lo, c, length=20)
    kc_upper = kc_mid + 1.5 * kc_atr
    kc_lower = kc_mid - 1.5 * kc_atr
    feat["kc_pos"] = (c - kc_lower) / (kc_upper - kc_lower + 1e-10)

    # Squeeze indicator
    if bb is not None:
        feat["squeeze_on"] = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(float)

    # ── Volume ───────────────────────────────────────────────────────────
    vol_ema20 = ta.ema(v, length=20)
    feat["vol_ratio"] = v / (vol_ema20 + 1e-10)
    feat["vol_ratio_3bar"] = v.rolling(3).mean() / (vol_ema20 + 1e-10)

    # OBV slope
    obv = ta.obv(c, v)
    if obv is not None:
        feat["obv_slope"] = (obv - obv.shift(5)) / (obv.abs().rolling(20).mean() + 1e-10)

    # Volume-price relationship
    feat["vol_price_corr"] = c.rolling(10).corr(v)

    # Buy/sell volume estimation
    feat["buy_vol_pct"] = ((c - lo) / (h - lo + 1e-10))  # Approximate buyer strength

    # ── Temporal ──────────────────────────────────────────────────────────
    dt = pd.to_datetime(df["date"])
    feat["hour"] = dt.dt.hour
    feat["minute"] = dt.dt.minute
    feat["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    feat["dow"] = dt.dt.dayofweek

    # Session markers
    feat["is_asia"] = ((dt.dt.hour >= 0) & (dt.dt.hour < 8)).astype(float)
    feat["is_europe"] = ((dt.dt.hour >= 8) & (dt.dt.hour < 13)).astype(float)
    feat["is_us"] = ((dt.dt.hour >= 13) & (dt.dt.hour < 21)).astype(float)

    # Pre-funding settlement (within 1h of 00, 08, 16 UTC)
    feat["pre_funding"] = (
        ((dt.dt.hour == 23) | (dt.dt.hour == 7) | (dt.dt.hour == 15))
    ).astype(float)

    # ── Pattern / Structure ──────────────────────────────────────────────
    # Consecutive same-direction bars
    green = (c > o).astype(int)
    feat["consec_green"] = green.groupby((green != green.shift()).cumsum()).cumcount()
    red = (c < o).astype(int)
    feat["consec_red"] = red.groupby((red != red.shift()).cumsum()).cumcount()

    # Distance from recent extremes
    feat["dist_from_5h_high"] = (c - h.rolling(60).max()) / (c + 1e-10)
    feat["dist_from_5h_low"] = (c - lo.rolling(60).min()) / (c + 1e-10)

    # Bar count since last volume spike (>2x)
    vol_spike = (v / (vol_ema20 + 1e-10)) > 2.0
    feat["bars_since_vol_spike"] = (~vol_spike).groupby(vol_spike.cumsum()).cumcount()
    feat["bars_since_vol_spike"] = feat["bars_since_vol_spike"].clip(upper=50)

    return feat


# =============================================================================
# LABEL GENERATION
# =============================================================================

def generate_labels(df: pd.DataFrame, sl_mult=1.5, tp_mult=2.0, max_bars=12):
    """
    For each bar, determine if a long or short trade would be profitable.
    Labels:
      0 = no profitable trade (both directions lose or neutral)
      1 = long profitable
      2 = short profitable
    """
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    atr = ta.atr(df["high"], df["low"], df["close"], length=14).values
    n = len(c)

    labels = np.zeros(n, dtype=int)
    long_profit = np.zeros(n)
    short_profit = np.zeros(n)

    for i in range(n - max_bars - 1):
        a = atr[i]
        if a <= 0 or c[i] <= 0:
            continue

        entry = c[i]

        # Long trade outcome
        sl_long = entry - sl_mult * a
        tp_long = entry + tp_mult * a
        long_pnl = 0.0
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if lo[j] <= sl_long:
                long_pnl = -sl_mult * a / entry - 2 * TAKER
                break
            if h[j] >= tp_long:
                long_pnl = tp_mult * a / entry - 2 * TAKER
                break
        else:
            exit_idx = min(i + max_bars, n - 1)
            long_pnl = (c[exit_idx] - entry) / entry - 2 * TAKER

        # Short trade outcome
        sl_short = entry + sl_mult * a
        tp_short = entry - tp_mult * a
        short_pnl = 0.0
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if h[j] >= sl_short:
                short_pnl = -sl_mult * a / entry - 2 * TAKER
                break
            if lo[j] <= tp_short:
                short_pnl = tp_mult * a / entry - 2 * TAKER
                break
        else:
            exit_idx = min(i + max_bars, n - 1)
            short_pnl = (entry - c[exit_idx]) / entry - 2 * TAKER

        long_profit[i] = long_pnl
        short_profit[i] = short_pnl

        # Label: best direction if profitable
        if long_pnl > 0 and long_pnl >= short_pnl:
            labels[i] = 1
        elif short_pnl > 0 and short_pnl > long_pnl:
            labels[i] = 2
        # else: 0 (no trade)

    return labels, long_profit, short_profit


# =============================================================================
# WALK-FORWARD TRAINING
# =============================================================================

def walk_forward_backtest(features_all, labels_all, profit_long_all, profit_short_all,
                          train_days=60, test_days=30, bars_per_day=288):
    """
    Walk-forward: train on 60 days, test on next 30 days, step forward 30 days.
    """
    train_size = train_days * bars_per_day
    test_size = test_days * bars_per_day
    n = len(features_all)

    all_predictions = []
    all_true_labels = []
    all_profits = []
    fold_results = []

    fold = 0
    start = 0

    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size

        X_train = features_all.iloc[start:train_end]
        y_train = labels_all[start:train_end]
        X_test = features_all.iloc[train_end:test_end]
        y_test = labels_all[train_end:test_end]
        long_prof_test = profit_long_all[train_end:test_end]
        short_prof_test = profit_short_all[train_end:test_end]

        # Drop rows with NaN features
        train_mask = ~X_train.isna().any(axis=1)
        test_mask = ~X_test.isna().any(axis=1)

        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask.values]
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask.values]
        long_prof_clean = long_prof_test[test_mask.values]
        short_prof_clean = short_prof_test[test_mask.values]

        if len(X_train_clean) < 100 or len(X_test_clean) < 50:
            start += test_size
            fold += 1
            continue

        # Train LightGBM
        # Binary: predict "trade" (1) vs "no trade" (0)
        # Then a second model for direction
        y_train_binary = (y_train_clean > 0).astype(int)
        y_test_binary = (y_test_clean > 0).astype(int)

        # Train with class weights (since "no trade" is ~60% of bars)
        train_data = lgb.Dataset(X_train_clean, label=y_train_binary)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 50,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "scale_pos_weight": sum(y_train_binary == 0) / max(sum(y_train_binary == 1), 1),
            "verbose": -1,
        }

        model = lgb.train(params, train_data, num_boost_round=200)

        # Predict probability of "trade opportunity"
        proba = model.predict(X_test_clean)

        # Direction model (only on bars where label > 0)
        dir_mask_train = y_train_clean > 0
        if sum(dir_mask_train) > 50:
            X_dir_train = X_train_clean[dir_mask_train]
            y_dir_train = (y_train_clean[dir_mask_train] == 1).astype(int)  # 1=long, 0=short

            dir_data = lgb.Dataset(X_dir_train, label=y_dir_train)
            dir_params = params.copy()
            dir_params["scale_pos_weight"] = sum(y_dir_train == 0) / max(sum(y_dir_train == 1), 1)

            dir_model = lgb.train(dir_params, dir_data, num_boost_round=150)
            dir_proba = dir_model.predict(X_test_clean)
        else:
            dir_proba = np.full(len(X_test_clean), 0.5)

        # Simulate trading with threshold
        threshold = 0.55  # Only trade when model confidence > 55%
        trades_profit = []
        n_trades = 0

        for idx in range(len(X_test_clean)):
            if proba[idx] > threshold:
                n_trades += 1
                # Choose direction
                if dir_proba[idx] > 0.55:
                    trades_profit.append(long_prof_clean[idx])
                elif dir_proba[idx] < 0.45:
                    trades_profit.append(short_prof_clean[idx])
                # else: skip ambiguous direction

        fold_profit = sum(trades_profit) if trades_profit else 0
        fold_trades = len(trades_profit)
        fold_wr = len([p for p in trades_profit if p > 0]) / max(len(trades_profit), 1)

        fold_results.append({
            "fold": fold,
            "train_bars": len(X_train_clean),
            "test_bars": len(X_test_clean),
            "trades": fold_trades,
            "wr": fold_wr,
            "profit": fold_profit,
        })

        all_profits.extend(trades_profit)
        start += test_size
        fold += 1

    return fold_results, all_profits, model if fold > 0 else None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 85)
    print("ML ENSEMBLE — 5m Crypto Scalping (LightGBM)")
    print("Walk-forward: 60d train / 30d test / 30d step")
    print("Features: 60+ | Target: profitable trade detection")
    print("=" * 85)

    all_fold_results = []
    all_profits = []
    feature_importances = {}

    for pair in PAIRS:
        print(f"\n{'─' * 85}")
        print(f"PAIR: {pair}")
        print(f"{'─' * 85}")

        df = load(pair)
        if df.empty:
            print("  No data")
            continue

        print(f"  Bars: {len(df)} | Engineering features...")
        features = engineer_features(df)
        print(f"  Features: {features.shape[1]} columns")

        print("  Generating labels (SL=1.5x ATR, TP=2.0x ATR, max_bars=12)...")
        labels, long_prof, short_prof = generate_labels(df, sl_mult=1.5, tp_mult=2.0, max_bars=12)

        # Label distribution
        n_long = sum(labels == 1)
        n_short = sum(labels == 2)
        n_none = sum(labels == 0)
        total = len(labels)
        print(f"  Labels: long={n_long} ({n_long/total*100:.1f}%) | "
              f"short={n_short} ({n_short/total*100:.1f}%) | "
              f"none={n_none} ({n_none/total*100:.1f}%)")

        # Walk-forward
        print("  Running walk-forward training...")
        fold_results, profits, model = walk_forward_backtest(
            features, labels, long_prof, short_prof,
            train_days=60, test_days=30, bars_per_day=288
        )

        if not fold_results:
            print("  Not enough data for walk-forward")
            continue

        # Print fold results
        print(f"\n  {'Fold':<6} {'Trades':<8} {'WR%':<8} {'Profit%':<10}")
        print(f"  {'-'*35}")
        for r in fold_results:
            wr_str = f"{r['wr']*100:.1f}%" if r['trades'] > 0 else "--"
            print(f"  {r['fold']:<6} {r['trades']:<8} {wr_str:<8} {r['profit']*100:>+8.2f}%")

        total_trades = sum(r["trades"] for r in fold_results)
        total_profit = sum(r["profit"] for r in fold_results)
        total_wr = len([p for p in profits if p > 0]) / max(len(profits), 1)

        print(f"\n  TOTAL: {total_trades} trades | WR={total_wr*100:.1f}% | "
              f"Profit={total_profit*100:+.2f}%")

        all_fold_results.extend(fold_results)
        all_profits.extend(profits)

        # Feature importance from last model
        if model is not None:
            imp = model.feature_importance(importance_type="gain")
            feat_names = features.columns.tolist()
            for fname, score in zip(feat_names, imp):
                feature_importances[fname] = feature_importances.get(fname, 0) + score

    # ─── AGGREGATE RESULTS ────────────────────────────────────────────────
    print(f"\n{'=' * 85}")
    print("AGGREGATE RESULTS (all pairs combined)")
    print(f"{'=' * 85}")

    if all_profits:
        arr = np.array(all_profits)
        total_trades = len(arr)
        total_wr = len(arr[arr > 0]) / total_trades * 100
        total_pnl = arr.sum() * 100
        avg_pnl = arr.mean() * 100

        wins = arr[arr > 0]
        losses = arr[arr <= 0]
        pf = wins.sum() / max(abs(losses.sum()), 0.0001) if len(losses) > 0 else 99.0

        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        max_dd = abs(np.min(cum - peak)) * 100

        print(f"  Total trades: {total_trades}")
        print(f"  Win rate: {total_wr:.1f}%")
        print(f"  Profit factor: {pf:.2f}")
        print(f"  Total profit: {total_pnl:+.2f}%")
        print(f"  Avg per trade: {avg_pnl:+.4f}%")
        print(f"  Max drawdown: {max_dd:.2f}%")

        # Is it better than random?
        print(f"\n  Breakeven WR (for 1.5:2.0 R:R + fees): ~47%")
        if total_wr > 47:
            print(f"  >>> MODEL SHOWS EDGE: WR {total_wr:.1f}% > 47% breakeven <<<")
        else:
            print(f"  Model WR {total_wr:.1f}% is below breakeven threshold.")

    # Feature importance
    if feature_importances:
        print(f"\n{'=' * 85}")
        print("TOP 20 FEATURES (by gain importance)")
        print(f"{'=' * 85}")
        sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        for i, (fname, score) in enumerate(sorted_feats[:20], 1):
            print(f"  {i:>2}. {fname:<25} {score:>10.1f}")

    # Different threshold analysis
    if all_profits:
        print(f"\n{'=' * 85}")
        print("THRESHOLD SENSITIVITY")
        print(f"{'=' * 85}")
        print("  (Re-running with different confidence thresholds would go here)")
        print("  Current threshold: 0.55 (model must be >55% confident to trade)")

    # Final verdict
    print(f"\n{'=' * 85}")
    print("CONCLUSION")
    print(f"{'=' * 85}")
    if all_profits and np.array(all_profits).sum() > 0:
        print("  ML ENSEMBLE SHOWS PROMISE on 5m data.")
        print("  Next steps:")
        print("    1. Hyperopt threshold and model params")
        print("    2. Add more features (order book, funding rate if available)")
        print("    3. Test with maker fees (should improve further)")
        print("    4. Implement as freqtrade strategy")
    else:
        print("  ML ensemble also struggles on 5m data.")
        print("  Even non-linear feature combinations cannot find consistent edge.")
        print("  Consider:")
        print("    1. Adding external features (funding rate, OI, liquidation data)")
        print("    2. Larger training window")
        print("    3. Different target (e.g., predict next-bar direction only)")
        print("    4. Ensemble of multiple timeframes as features")


if __name__ == "__main__":
    main()

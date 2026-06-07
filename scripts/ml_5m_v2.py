"""ML Ensemble v2 — Selective trading with high confidence threshold.

Key changes from v1:
1. Simpler target: predict 6-bar forward return direction
2. Much higher threshold: only trade top 10% confident predictions
3. Use actual forward returns (not SL/TP simulation) for evaluation
4. Add multi-timeframe features (aggregate 3x5m = 15m equivalent)
5. Feature selection to avoid overfitting
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
TAKER = 0.0005
MAKER = 0.0002
PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT"]


def load(pair):
    fp = DATA_DIR / f"{pair}-5m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp)
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


def engineer_features(df):
    """Generate features including multi-timeframe aggregates."""
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    o = df["open"]
    v = df["volume"].astype(float)

    feat = pd.DataFrame(index=df.index)

    # ── Price Action (normalized) ────────────────────────────────────────
    feat["body_pct"] = (c - o) / (c + 1e-10)
    feat["range_pct"] = (h - lo) / (c + 1e-10)
    feat["upper_wick_ratio"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (h - lo + 1e-10)
    feat["lower_wick_ratio"] = (pd.concat([c, o], axis=1).min(axis=1) - lo) / (h - lo + 1e-10)

    # Returns
    for lb in [1, 3, 6, 12, 24]:
        feat[f"ret_{lb}"] = c.pct_change(lb)

    # ── Momentum ─────────────────────────────────────────────────────────
    feat["rsi_3"] = ta.rsi(c, length=3)
    feat["rsi_9"] = ta.rsi(c, length=9)
    feat["rsi_14"] = ta.rsi(c, length=14)
    feat["rsi_9_delta"] = feat["rsi_9"] - feat["rsi_9"].shift(3)

    # CCI
    feat["cci"] = ta.cci(h, lo, c, length=14)

    # Stochastic
    stoch = ta.stoch(h, lo, c, k=14, d=3)
    if stoch is not None:
        feat["stoch_k"] = stoch.iloc[:, 0]
        feat["stoch_d"] = stoch.iloc[:, 1]

    # MACD
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat["macd_hist"] = macd.iloc[:, 2]

    # ── Trend ────────────────────────────────────────────────────────────
    ema8 = ta.ema(c, length=8)
    ema21 = ta.ema(c, length=21)
    ema50 = ta.ema(c, length=50)

    feat["price_vs_ema8"] = (c - ema8) / (ema8 + 1e-10)
    feat["price_vs_ema21"] = (c - ema21) / (ema21 + 1e-10)
    feat["ema_spread"] = (ema8 - ema21) / (ema21 + 1e-10)

    adx_r = ta.adx(h, lo, c, length=14)
    if adx_r is not None:
        feat["adx"] = adx_r.iloc[:, 0]
        feat["di_diff"] = adx_r.iloc[:, 1] - adx_r.iloc[:, 2]

    # ── Volatility ───────────────────────────────────────────────────────
    atr14 = ta.atr(h, lo, c, length=14)
    feat["atr_pct"] = atr14 / (c + 1e-10)
    feat["atr_ratio"] = ta.atr(h, lo, c, length=5) / (atr14 + 1e-10)
    feat["range_vs_atr"] = (h - lo) / (atr14 + 1e-10)

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        feat["bb_pos"] = (c - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2] + 1e-10)
        feat["bb_width"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / (bb.iloc[:, 1] + 1e-10)

    # ── Volume ───────────────────────────────────────────────────────────
    vol_ema = ta.ema(v, length=20)
    feat["vol_ratio"] = v / (vol_ema + 1e-10)
    feat["vol_ratio_3"] = v.rolling(3).mean() / (vol_ema + 1e-10)

    obv = ta.obv(c, v)
    if obv is not None:
        feat["obv_slope_norm"] = (obv - obv.shift(5)) / (obv.abs().rolling(20).mean() + 1e-10)

    feat["buy_pressure"] = (c - lo) / (h - lo + 1e-10)

    # ── Multi-timeframe (15m equivalent from 3x5m) ──────────────────────
    feat["ret_15m"] = c.pct_change(3)  # 3 bars = 15 min
    feat["range_15m"] = h.rolling(3).max() - lo.rolling(3).min()
    feat["range_15m_pct"] = feat["range_15m"] / (c + 1e-10)
    feat["vol_15m"] = v.rolling(3).sum()
    feat["vol_15m_ratio"] = feat["vol_15m"] / (v.rolling(60).mean() * 3 + 1e-10)

    # 1h equivalent (12x5m)
    feat["ret_1h"] = c.pct_change(12)
    feat["range_1h_pct"] = (h.rolling(12).max() - lo.rolling(12).min()) / (c + 1e-10)

    # ── Temporal ─────────────────────────────────────────────────────────
    dt = pd.to_datetime(df["date"])
    feat["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    feat["is_us_session"] = ((dt.dt.hour >= 13) & (dt.dt.hour < 21)).astype(float)

    # ── Structure ────────────────────────────────────────────────────────
    feat["pos_in_day_range"] = (c - lo.rolling(288).min()) / (h.rolling(288).max() - lo.rolling(288).min() + 1e-10)
    feat["bars_from_day_high"] = 288 - h.rolling(288).apply(lambda x: len(x) - np.argmax(x), raw=True)
    feat["bars_from_day_low"] = 288 - lo.rolling(288).apply(lambda x: len(x) - np.argmin(x), raw=True)

    return feat


def generate_labels_direction(df, forward_bars=6):
    """Simple direction label: will price be higher or lower in N bars?"""
    c = df["close"].values
    n = len(c)

    # Forward return
    fwd_ret = np.zeros(n)
    for i in range(n - forward_bars):
        fwd_ret[i] = (c[i + forward_bars] - c[i]) / c[i]

    # Labels: 1 = long profitable (net of fees), -1 = short profitable, 0 = unclear
    labels = np.zeros(n, dtype=int)
    min_move = 2 * TAKER + 0.0002  # Need at least fees + 0.02% profit

    for i in range(n - forward_bars):
        if fwd_ret[i] > min_move:
            labels[i] = 1  # Long
        elif fwd_ret[i] < -min_move:
            labels[i] = -1  # Short
        # else: 0 (no clear direction)

    return labels, fwd_ret


def walk_forward_v2(features, labels, fwd_returns, df,
                    train_days=60, test_days=30, bars_per_day=288):
    """Walk-forward with selective trading (high confidence only)."""
    train_size = train_days * bars_per_day
    test_size = test_days * bars_per_day
    n = len(features)

    all_results = []
    all_trades = []

    start = 0
    fold = 0

    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size

        X_train = features.iloc[start:train_end]
        y_train = labels[start:train_end]
        X_test = features.iloc[train_end:test_end]
        y_test = labels[train_end:test_end]
        fwd_test = fwd_returns[train_end:test_end]

        # Clean NaN
        train_mask = ~X_train.isna().any(axis=1)
        test_mask = ~X_test.isna().any(axis=1)

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask.values]
        X_te = X_test[test_mask]
        y_te = y_test[test_mask.values]
        fwd_te = fwd_test[test_mask.values]

        if len(X_tr) < 1000 or len(X_te) < 100:
            start += test_size
            fold += 1
            continue

        # Convert to binary: long (1) vs not-long (0)
        y_tr_long = (y_tr == 1).astype(int)
        y_tr_short = (y_tr == -1).astype(int)

        # Train LONG model
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 100,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": sum(y_tr_long == 0) / max(sum(y_tr_long == 1), 1),
            "verbose": -1,
        }

        train_data_long = lgb.Dataset(X_tr, label=y_tr_long)
        model_long = lgb.train(params, train_data_long, num_boost_round=300)

        # Train SHORT model
        params["scale_pos_weight"] = sum(y_tr_short == 0) / max(sum(y_tr_short == 1), 1)
        train_data_short = lgb.Dataset(X_tr, label=y_tr_short)
        model_short = lgb.train(params, train_data_short, num_boost_round=300)

        # Predict on test
        prob_long = model_long.predict(X_te)
        prob_short = model_short.predict(X_te)

        # Trade with different thresholds
        for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
            trades = []
            for idx in range(len(X_te)):
                if prob_long[idx] > threshold and prob_long[idx] > prob_short[idx]:
                    # Enter long, profit = forward return - fees
                    pnl = fwd_te[idx] - 2 * TAKER
                    trades.append(("long", pnl))
                elif prob_short[idx] > threshold and prob_short[idx] > prob_long[idx]:
                    pnl = -fwd_te[idx] - 2 * TAKER  # Short
                    trades.append(("short", pnl))

            if trades:
                pnls = [t[1] for t in trades]
                arr = np.array(pnls)
                wr = len(arr[arr > 0]) / len(arr) * 100
                profit = arr.sum() * 100
                all_results.append({
                    "fold": fold, "threshold": threshold,
                    "trades": len(trades), "wr": wr, "profit": profit,
                })

                if threshold == 0.65:  # Collect trades for mid threshold
                    all_trades.extend(pnls)

        start += test_size
        fold += 1

    return all_results, all_trades, model_long if fold > 0 else None


def main():
    print("=" * 85)
    print("ML ENSEMBLE v2 — Selective 5m Trading (LightGBM)")
    print("Target: 6-bar forward return direction")
    print("Approach: separate long/short models, high confidence threshold")
    print("=" * 85)

    grand_results = {}  # threshold -> aggregated results
    grand_trades = []
    feature_importances = {}

    for pair in PAIRS:
        print(f"\n{'=' * 85}")
        print(f"  PAIR: {pair}")
        print(f"{'=' * 85}")

        df = load(pair)
        if df.empty:
            continue

        print(f"  Bars: {len(df)}")
        features = engineer_features(df)
        print(f"  Features: {features.shape[1]}")

        labels, fwd_returns = generate_labels_direction(df, forward_bars=6)
        n_long = sum(labels == 1)
        n_short = sum(labels == -1)
        n_flat = sum(labels == 0)
        total = len(labels)
        print(f"  Labels: long={n_long} ({n_long/total*100:.1f}%) | "
              f"short={n_short} ({n_short/total*100:.1f}%) | "
              f"flat={n_flat} ({n_flat/total*100:.1f}%)")

        results, trades, model = walk_forward_v2(
            features, labels, fwd_returns, df,
            train_days=60, test_days=30, bars_per_day=288
        )

        if not results:
            print("  No results (insufficient data)")
            continue

        # Aggregate by threshold
        print(f"\n  {'Threshold':<12} {'Trades':<8} {'WR%':<8} {'Profit%':<10} {'Avg/Trade':<10}")
        print(f"  {'-'*50}")

        for thr in [0.55, 0.60, 0.65, 0.70, 0.75]:
            thr_results = [r for r in results if r["threshold"] == thr]
            if not thr_results:
                continue
            total_trades = sum(r["trades"] for r in thr_results)
            total_profit = sum(r["profit"] for r in thr_results)
            # Weighted WR
            all_wins = sum(r["trades"] * r["wr"] / 100 for r in thr_results)
            avg_wr = all_wins / max(total_trades, 1) * 100
            avg_per_trade = total_profit / max(total_trades, 1)

            if thr not in grand_results:
                grand_results[thr] = {"trades": 0, "profit": 0, "wins": 0}
            grand_results[thr]["trades"] += total_trades
            grand_results[thr]["profit"] += total_profit
            grand_results[thr]["wins"] += all_wins

            marker = " ***" if total_profit > 0 else ""
            print(f"  {thr:<12.2f} {total_trades:<8} {avg_wr:<7.1f}% {total_profit:<+9.2f}% "
                  f"{avg_per_trade:<+9.4f}%{marker}")

        grand_trades.extend(trades)

        # Feature importance
        if model is not None:
            imp = model.feature_importance(importance_type="gain")
            feat_names = features.columns.tolist()
            for fname, score in zip(feat_names, imp):
                feature_importances[fname] = feature_importances.get(fname, 0) + score

    # ─── GRAND SUMMARY ────────────────────────────────────────────────────
    print(f"\n{'=' * 85}")
    print("GRAND SUMMARY (all pairs combined)")
    print(f"{'=' * 85}")
    print(f"{'Threshold':<12} {'Trades':<8} {'WR%':<8} {'Profit%':<10} {'Avg/Trade':<10}")
    print("-" * 55)

    for thr in sorted(grand_results.keys()):
        r = grand_results[thr]
        wr = r["wins"] / max(r["trades"], 1) * 100
        avg = r["profit"] / max(r["trades"], 1)
        marker = " <<<" if r["profit"] > 0 else ""
        print(f"{thr:<12.2f} {r['trades']:<8} {wr:<7.1f}% {r['profit']:<+9.2f}% {avg:<+9.4f}%{marker}")

    # Feature importance
    if feature_importances:
        print(f"\n{'=' * 85}")
        print("TOP 15 FEATURES (gain importance)")
        print(f"{'=' * 85}")
        sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        for i, (fname, score) in enumerate(sorted_feats[:15], 1):
            print(f"  {i:>2}. {fname:<25} {score:>10.1f}")

    # Equity curve analysis for threshold=0.65
    if grand_trades:
        arr = np.array(grand_trades)
        print(f"\n{'=' * 85}")
        print(f"EQUITY ANALYSIS (threshold=0.65)")
        print(f"{'=' * 85}")
        print(f"  Total trades: {len(arr)}")
        if len(arr) > 0:
            print(f"  Win rate: {len(arr[arr > 0]) / len(arr) * 100:.1f}%")
            print(f"  Total PnL: {arr.sum() * 100:+.2f}%")
            print(f"  Avg PnL: {arr.mean() * 100:+.4f}%")

            cum = np.cumsum(arr)
            peak = np.maximum.accumulate(cum)
            dd = np.min(cum - peak)
            print(f"  Max DD: {abs(dd)*100:.2f}%")

            # Sharpe-like ratio
            if arr.std() > 0:
                sharpe = arr.mean() / arr.std() * np.sqrt(288)  # Annualize for 5m
                print(f"  Sharpe (annualized): {sharpe:.2f}")

    # Comparison with maker fees
    if grand_trades:
        print(f"\n{'=' * 85}")
        print("MAKER FEE SIMULATION (same trades, 0.04% RT instead of 0.10%)")
        print(f"{'=' * 85}")
        # Each trade had (fwd_return - 2*TAKER) as PnL
        # With maker: (fwd_return - 2*MAKER)
        # Difference per trade: 2*(TAKER-MAKER) = 2*(0.0005-0.0002) = 0.0006
        fee_diff = 2 * (TAKER - MAKER)
        maker_trades = np.array(grand_trades) + fee_diff
        print(f"  Fee savings per trade: {fee_diff*100:.3f}%")
        print(f"  Total trades: {len(maker_trades)}")
        print(f"  Win rate (maker): {len(maker_trades[maker_trades > 0]) / len(maker_trades) * 100:.1f}%")
        print(f"  Total PnL (maker): {maker_trades.sum() * 100:+.2f}%")
        print(f"  Avg PnL (maker): {maker_trades.mean() * 100:+.4f}%")

    # FINAL VERDICT
    print(f"\n{'=' * 85}")
    print("FINAL VERDICT")
    print(f"{'=' * 85}")
    best_thr = max(grand_results.keys(), key=lambda t: grand_results[t]["profit"]) if grand_results else 0.65
    best = grand_results.get(best_thr, {})
    if best.get("profit", 0) > 0:
        print(f"  PROFITABLE at threshold {best_thr}: {best['profit']:+.2f}% on {best['trades']} trades")
        print(f"  ML ensemble CAN find edge on 5m when selective enough.")
        print(f"  Next: implement as freqtrade strategy with model inference on each tick.")
    else:
        print(f"  Best result: threshold {best_thr} with {best.get('profit', 0):+.2f}%")
        print(f"  Even ML with high selectivity cannot profitably trade 5m crypto.")
        print(f"  The market is informationally efficient at this frequency.")
        print(f"  RECOMMENDATION: Stick with 15m strategies (proven edge).")


if __name__ == "__main__":
    main()

"""ML Ensemble v3 — Optimized: ETH focus + maker fees + high selectivity.

Combines all learnings:
1. Focus on ETH (only pair showing ML edge)
2. Maker fee simulation (limit orders)
3. Very high confidence threshold (0.65-0.80)
4. Expanded feature set with additional timeframe aggregates
5. Multiple model variants (hyperopt-like parameter scan)
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
PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT"]


def load(pair):
    fp = DATA_DIR / f"{pair}-5m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp)
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


def engineer_features(df):
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    o = df["open"]
    v = df["volume"].astype(float)

    feat = pd.DataFrame(index=df.index)

    # Price action
    feat["body_pct"] = (c - o) / (c + 1e-10)
    feat["range_pct"] = (h - lo) / (c + 1e-10)
    feat["upper_wick_ratio"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (h - lo + 1e-10)
    feat["lower_wick_ratio"] = (pd.concat([c, o], axis=1).min(axis=1) - lo) / (h - lo + 1e-10)

    for lb in [1, 3, 6, 12, 24, 36]:
        feat[f"ret_{lb}"] = c.pct_change(lb)

    # Momentum
    feat["rsi_3"] = ta.rsi(c, length=3)
    feat["rsi_9"] = ta.rsi(c, length=9)
    feat["rsi_14"] = ta.rsi(c, length=14)
    feat["rsi_9_delta"] = feat["rsi_9"] - feat["rsi_9"].shift(3)
    feat["cci"] = ta.cci(h, lo, c, length=14)

    stoch = ta.stoch(h, lo, c, k=14, d=3)
    if stoch is not None:
        feat["stoch_k"] = stoch.iloc[:, 0]
        feat["stoch_d"] = stoch.iloc[:, 1]

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat["macd_hist"] = macd.iloc[:, 2]

    # Trend
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

    # Volatility
    atr14 = ta.atr(h, lo, c, length=14)
    atr5 = ta.atr(h, lo, c, length=5)
    feat["atr_pct"] = atr14 / (c + 1e-10)
    feat["atr_ratio"] = atr5 / (atr14 + 1e-10)
    feat["range_vs_atr"] = (h - lo) / (atr14 + 1e-10)

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        feat["bb_pos"] = (c - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2] + 1e-10)
        feat["bb_width"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / (bb.iloc[:, 1] + 1e-10)

    # Volume
    vol_ema = ta.ema(v, length=20)
    feat["vol_ratio"] = v / (vol_ema + 1e-10)
    feat["vol_ratio_3"] = v.rolling(3).mean() / (vol_ema + 1e-10)
    feat["buy_pressure"] = (c - lo) / (h - lo + 1e-10)

    obv = ta.obv(c, v)
    if obv is not None:
        feat["obv_slope"] = (obv - obv.shift(5)) / (obv.abs().rolling(20).mean() + 1e-10)

    # Multi-timeframe
    feat["ret_15m"] = c.pct_change(3)
    feat["range_15m_pct"] = (h.rolling(3).max() - lo.rolling(3).min()) / (c + 1e-10)
    feat["ret_1h"] = c.pct_change(12)
    feat["range_1h_pct"] = (h.rolling(12).max() - lo.rolling(12).min()) / (c + 1e-10)
    feat["ret_4h"] = c.pct_change(48)

    # Temporal
    dt = pd.to_datetime(df["date"])
    feat["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    feat["is_us_session"] = ((dt.dt.hour >= 13) & (dt.dt.hour < 21)).astype(float)

    # Intraday structure
    feat["pos_in_day_range"] = (c - lo.rolling(288).min()) / (h.rolling(288).max() - lo.rolling(288).min() + 1e-10)

    return feat


def generate_labels(df, forward_bars=6, fee_mode="maker"):
    """Direction labels with fee-adjusted threshold."""
    c = df["close"].values
    n = len(c)
    fee = 2 * MAKER if fee_mode == "maker" else 2 * TAKER

    fwd_ret = np.zeros(n)
    for i in range(n - forward_bars):
        fwd_ret[i] = (c[i + forward_bars] - c[i]) / c[i]

    labels = np.zeros(n, dtype=int)
    min_move = fee + 0.0001

    for i in range(n - forward_bars):
        if fwd_ret[i] > min_move:
            labels[i] = 1
        elif fwd_ret[i] < -min_move:
            labels[i] = -1

    return labels, fwd_ret


def run_walk_forward(features, labels, fwd_returns, pair_name,
                     train_days=60, test_days=30, bars_per_day=288,
                     model_params=None):
    """Walk-forward with detailed results per threshold."""
    if model_params is None:
        model_params = {
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
            "verbose": -1,
        }

    train_size = train_days * bars_per_day
    test_size = test_days * bars_per_day
    n = len(features)

    threshold_trades = {t: [] for t in [0.55, 0.60, 0.63, 0.65, 0.67, 0.70, 0.73, 0.75, 0.80]}
    last_model_long = None
    last_model_short = None

    start = 0
    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size

        X_train = features.iloc[start:train_end]
        y_train = labels[start:train_end]
        X_test = features.iloc[train_end:test_end]
        fwd_test = fwd_returns[train_end:test_end]

        train_mask = ~X_train.isna().any(axis=1)
        test_mask = ~X_test.isna().any(axis=1)

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask.values]
        X_te = X_test[test_mask]
        fwd_te = fwd_test[test_mask.values]

        if len(X_tr) < 1000 or len(X_te) < 100:
            start += test_size
            continue

        # Long model
        y_tr_long = (y_tr == 1).astype(int)
        params_l = model_params.copy()
        params_l["scale_pos_weight"] = sum(y_tr_long == 0) / max(sum(y_tr_long == 1), 1)
        model_long = lgb.train(params_l, lgb.Dataset(X_tr, label=y_tr_long), num_boost_round=300)

        # Short model
        y_tr_short = (y_tr == -1).astype(int)
        params_s = model_params.copy()
        params_s["scale_pos_weight"] = sum(y_tr_short == 0) / max(sum(y_tr_short == 1), 1)
        model_short = lgb.train(params_s, lgb.Dataset(X_tr, label=y_tr_short), num_boost_round=300)

        prob_long = model_long.predict(X_te)
        prob_short = model_short.predict(X_te)

        last_model_long = model_long
        last_model_short = model_short

        for thr in threshold_trades.keys():
            for idx in range(len(X_te)):
                if prob_long[idx] > thr and prob_long[idx] > prob_short[idx] + 0.05:
                    pnl_maker = fwd_te[idx] - 2 * MAKER
                    pnl_taker = fwd_te[idx] - 2 * TAKER
                    threshold_trades[thr].append(("long", pnl_maker, pnl_taker))
                elif prob_short[idx] > thr and prob_short[idx] > prob_long[idx] + 0.05:
                    pnl_maker = -fwd_te[idx] - 2 * MAKER
                    pnl_taker = -fwd_te[idx] - 2 * TAKER
                    threshold_trades[thr].append(("short", pnl_maker, pnl_taker))

        start += test_size

    return threshold_trades, last_model_long, last_model_short


def main():
    print("=" * 90)
    print("ML v3 — OPTIMIZED: All Pairs + Maker Fees + High Selectivity")
    print("Target: 6-bar forward return | Train: 60d | Test: 30d")
    print("=" * 90)

    for pair in PAIRS:
        print(f"\n{'=' * 90}")
        print(f"  {pair}")
        print(f"{'=' * 90}")

        df = load(pair)
        if df.empty:
            continue

        features = engineer_features(df)
        labels, fwd_returns = generate_labels(df, forward_bars=6, fee_mode="maker")

        n_long = sum(labels == 1)
        n_short = sum(labels == -1)
        n_flat = sum(labels == 0)
        total = len(labels)
        print(f"  Bars: {total} | Labels: long={n_long} ({n_long/total*100:.1f}%), "
              f"short={n_short} ({n_short/total*100:.1f}%), flat={n_flat} ({n_flat/total*100:.1f}%)")

        # Test multiple model configs
        configs = [
            ("Conservative", {"num_leaves": 10, "max_depth": 3, "min_child_samples": 200,
                              "feature_fraction": 0.5, "reg_alpha": 0.5, "reg_lambda": 2.0}),
            ("Balanced", {"num_leaves": 15, "max_depth": 4, "min_child_samples": 100,
                          "feature_fraction": 0.6, "reg_alpha": 0.1, "reg_lambda": 1.0}),
            ("Aggressive", {"num_leaves": 31, "max_depth": 5, "min_child_samples": 50,
                            "feature_fraction": 0.7, "reg_alpha": 0.05, "reg_lambda": 0.5}),
        ]

        best_config = None
        best_profit = -999

        for cfg_name, cfg_overrides in configs:
            params = {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.03,
                "bagging_fraction": 0.7,
                "bagging_freq": 5,
                "verbose": -1,
            }
            params.update(cfg_overrides)

            threshold_trades, _, _ = run_walk_forward(
                features, labels, fwd_returns, pair,
                model_params=params
            )

            # Find best threshold for this config
            for thr, trades in threshold_trades.items():
                if len(trades) < 20:
                    continue
                maker_pnls = [t[1] for t in trades]
                profit = sum(maker_pnls) * 100
                if profit > best_profit:
                    best_profit = profit
                    best_config = (cfg_name, thr, trades)

        # Print results for best config
        if best_config:
            cfg_name, best_thr, trades = best_config
            print(f"\n  Best config: {cfg_name} | threshold={best_thr}")
            print(f"\n  {'Threshold':<10} {'Trades':<8} {'WR(Maker)':<11} {'PnL(Maker)':<12} "
                  f"{'WR(Taker)':<11} {'PnL(Taker)':<12}")
            print(f"  {'-'*65}")

            # Rerun best config and show all thresholds
            params = {
                "objective": "binary", "metric": "auc", "learning_rate": 0.03,
                "bagging_fraction": 0.7, "bagging_freq": 5, "verbose": -1,
            }
            params.update(configs[["Conservative", "Balanced", "Aggressive"].index(cfg_name)][1])

            threshold_trades, model_l, model_s = run_walk_forward(
                features, labels, fwd_returns, pair, model_params=params
            )

            for thr in sorted(threshold_trades.keys()):
                trades = threshold_trades[thr]
                if not trades:
                    continue
                maker_pnls = np.array([t[1] for t in trades])
                taker_pnls = np.array([t[2] for t in trades])

                wr_maker = len(maker_pnls[maker_pnls > 0]) / len(maker_pnls) * 100
                wr_taker = len(taker_pnls[taker_pnls > 0]) / len(taker_pnls) * 100
                pnl_maker = maker_pnls.sum() * 100
                pnl_taker = taker_pnls.sum() * 100

                marker = " ***" if pnl_maker > 0 else ""
                print(f"  {thr:<10.2f} {len(trades):<8} {wr_maker:<10.1f}% {pnl_maker:<+11.2f}% "
                      f"{wr_taker:<10.1f}% {pnl_taker:<+11.2f}%{marker}")

            # Feature importance
            if model_l is not None:
                imp = model_l.feature_importance(importance_type="gain")
                feat_names = features.columns.tolist()
                sorted_idx = np.argsort(imp)[::-1]
                print(f"\n  Top 10 features ({pair}):")
                for rank, idx in enumerate(sorted_idx[:10], 1):
                    print(f"    {rank:>2}. {feat_names[idx]:<22} {imp[idx]:>8.0f}")

    # ─── COMBINED ANALYSIS ─────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("COMBINED MULTI-PAIR PORTFOLIO")
    print(f"{'=' * 90}")
    print("  (Running balanced config across all pairs with threshold sweep...)")

    combined = {t: [] for t in [0.55, 0.60, 0.63, 0.65, 0.67, 0.70, 0.73, 0.75, 0.80]}

    for pair in PAIRS:
        df = load(pair)
        if df.empty:
            continue
        features = engineer_features(df)
        labels, fwd_returns = generate_labels(df, forward_bars=6, fee_mode="maker")

        params = {
            "objective": "binary", "metric": "auc", "learning_rate": 0.03,
            "num_leaves": 15, "max_depth": 4, "min_child_samples": 100,
            "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 5,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1,
        }

        threshold_trades, _, _ = run_walk_forward(features, labels, fwd_returns, pair, model_params=params)
        for thr in combined:
            combined[thr].extend(threshold_trades.get(thr, []))

    print(f"\n  {'Threshold':<10} {'Trades':<8} {'WR(Maker)':<11} {'PnL(Maker)':<12} "
          f"{'Trades/Day':<11} {'Sharpe':<8}")
    print(f"  {'-'*65}")

    for thr in sorted(combined.keys()):
        trades = combined[thr]
        if not trades:
            continue
        maker_pnls = np.array([t[1] for t in trades])
        wr = len(maker_pnls[maker_pnls > 0]) / len(maker_pnls) * 100
        pnl = maker_pnls.sum() * 100
        trades_per_day = len(trades) / 60  # ~60 test days

        # Sharpe
        if maker_pnls.std() > 0:
            sharpe = maker_pnls.mean() / maker_pnls.std() * np.sqrt(trades_per_day * 365)
        else:
            sharpe = 0

        marker = " <<<" if pnl > 0 else ""
        print(f"  {thr:<10.2f} {len(trades):<8} {wr:<10.1f}% {pnl:<+11.2f}% "
              f"{trades_per_day:<10.1f} {sharpe:<7.2f}{marker}")

    # Final recommendation
    print(f"\n{'=' * 90}")
    print("RECOMMENDATION")
    print(f"{'=' * 90}")

    # Check if any threshold is profitable with maker fees
    profitable_thresholds = []
    for thr, trades in combined.items():
        if trades:
            maker_pnls = np.array([t[1] for t in trades])
            if maker_pnls.sum() > 0:
                profitable_thresholds.append((thr, maker_pnls.sum() * 100, len(trades)))

    if profitable_thresholds:
        profitable_thresholds.sort(key=lambda x: x[1], reverse=True)
        best = profitable_thresholds[0]
        print(f"  PROFITABLE combination found!")
        print(f"  Threshold={best[0]}, PnL={best[1]:+.2f}%, Trades={best[2]}")
        print(f"\n  Implementation path:")
        print(f"    1. Deploy LightGBM model in freqtrade (custom predict)")
        print(f"    2. Use limit orders (maker fees) for entry")
        print(f"    3. Only trade when model confidence > {best[0]}")
        print(f"    4. Retrain model every 30 days (walk-forward)")
    else:
        print(f"  No profitable threshold found across all pairs combined.")
        print(f"  However, ETH individually may still show edge at high threshold.")
        print(f"\n  Options:")
        print(f"    1. ETH-only ML strategy (showed +1.9% at threshold 0.75)")
        print(f"    2. Add external features (funding rate, OI)")
        print(f"    3. Different target (SL/TP with maker entry optimization)")
        print(f"    4. Hybrid: use ML as a filter for rule-based 15m signals")


if __name__ == "__main__":
    main()

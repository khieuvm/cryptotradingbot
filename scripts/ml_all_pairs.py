"""ML Walk-Forward on ALL pairs: BTC, ETH, SOL, SPX, XAU (5m).

Uses the proven v3 approach: Conservative model, 60d train / 30d test.
Reports per-pair viability for production deployment.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb
import joblib

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

MAKER = 0.0002
TAKER = 0.0005
ALL_PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT", "XAU_USDT_USDT"]


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

    feat["body_pct"] = (c - o) / (c + 1e-10)
    feat["range_pct"] = (h - lo) / (c + 1e-10)
    max_co = pd.concat([c, o], axis=1).max(axis=1)
    min_co = pd.concat([c, o], axis=1).min(axis=1)
    feat["upper_wick_ratio"] = (h - max_co) / (h - lo + 1e-10)
    feat["lower_wick_ratio"] = (min_co - lo) / (h - lo + 1e-10)

    for lb in [1, 3, 6, 12, 24, 36]:
        feat[f"ret_{lb}"] = c.pct_change(lb)

    feat["rsi_3"] = ta.rsi(c, length=3)
    feat["rsi_9"] = ta.rsi(c, length=9)
    feat["rsi_14"] = ta.rsi(c, length=14)
    feat["rsi_9_delta"] = feat["rsi_9"] - feat["rsi_9"].shift(3)
    feat["cci"] = ta.cci(h, lo, c, length=14)

    stoch = ta.stoch(h, lo, c, k=14, d=3)
    if stoch is not None:
        feat["stoch_k"] = stoch.iloc[:, 0]
        feat["stoch_d"] = stoch.iloc[:, 1]
    else:
        feat["stoch_k"] = 50.0
        feat["stoch_d"] = 50.0

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat["macd_hist"] = macd.iloc[:, 2]
    else:
        feat["macd_hist"] = 0.0

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
    else:
        feat["adx"] = 25.0
        feat["di_diff"] = 0.0

    atr14 = ta.atr(h, lo, c, length=14)
    atr5 = ta.atr(h, lo, c, length=5)
    feat["atr_pct"] = atr14 / (c + 1e-10)
    feat["atr_ratio"] = atr5 / (atr14 + 1e-10)
    feat["range_vs_atr"] = (h - lo) / (atr14 + 1e-10)

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        feat["bb_pos"] = (c - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2] + 1e-10)
        feat["bb_width"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / (bb.iloc[:, 1] + 1e-10)
    else:
        feat["bb_pos"] = 0.5
        feat["bb_width"] = 0.0

    vol_ema = ta.ema(v, length=20)
    feat["vol_ratio"] = v / (vol_ema + 1e-10)
    feat["vol_ratio_3"] = v.rolling(3).mean() / (vol_ema + 1e-10)
    feat["buy_pressure"] = (c - lo) / (h - lo + 1e-10)

    obv = ta.obv(c, v)
    if obv is not None:
        feat["obv_slope"] = (obv - obv.shift(5)) / (obv.abs().rolling(20).mean() + 1e-10)
    else:
        feat["obv_slope"] = 0.0

    feat["ret_15m"] = c.pct_change(3)
    feat["range_15m_pct"] = (h.rolling(3).max() - lo.rolling(3).min()) / (c + 1e-10)
    feat["ret_1h"] = c.pct_change(12)
    feat["range_1h_pct"] = (h.rolling(12).max() - lo.rolling(12).min()) / (c + 1e-10)
    feat["ret_4h"] = c.pct_change(48)

    dt = pd.to_datetime(df["date"])
    feat["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    feat["is_us_session"] = ((dt.dt.hour >= 13) & (dt.dt.hour < 21)).astype(float)
    feat["pos_in_day_range"] = (c - lo.rolling(288).min()) / (h.rolling(288).max() - lo.rolling(288).min() + 1e-10)

    return feat


def walk_forward(features, labels, fwd_returns, train_days=60, test_days=30, bars_per_day=288):
    """Walk-forward with Conservative model config."""
    train_size = train_days * bars_per_day
    test_size = test_days * bars_per_day
    n = len(features)

    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.03,
        "num_leaves": 10, "max_depth": 3, "min_child_samples": 200,
        "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 5,
        "reg_alpha": 0.5, "reg_lambda": 2.0, "verbose": -1,
    }

    threshold_results = {t: [] for t in [0.55, 0.60, 0.63, 0.65, 0.67, 0.70, 0.75]}
    fold = 0
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
            fold += 1
            continue

        # Long model
        y_long = (y_tr == 1).astype(int)
        params_l = params.copy()
        params_l["scale_pos_weight"] = sum(y_long == 0) / max(sum(y_long == 1), 1)
        model_l = lgb.train(params_l, lgb.Dataset(X_tr, label=y_long), num_boost_round=300)

        # Short model
        y_short = (y_tr == -1).astype(int)
        params_s = params.copy()
        params_s["scale_pos_weight"] = sum(y_short == 0) / max(sum(y_short == 1), 1)
        model_s = lgb.train(params_s, lgb.Dataset(X_tr, label=y_short), num_boost_round=300)

        # Predict
        prob_l = model_l.predict(X_te)
        prob_s = model_s.predict(X_te)

        gap = 0.05
        for thr in threshold_results:
            long_mask = (prob_l > thr) & (prob_l > prob_s + gap)
            short_mask = (prob_s > thr) & (prob_s > prob_l + gap)

            long_pnl = fwd_te[long_mask] - 2 * MAKER
            short_pnl = -fwd_te[short_mask] - 2 * MAKER
            all_pnl = np.concatenate([long_pnl, short_pnl])
            threshold_results[thr].extend(all_pnl.tolist())

        start += test_size
        fold += 1

    return threshold_results, fold


def main():
    print("=" * 90)
    print("ML WALK-FORWARD: ALL PAIRS (5m) — Conservative Model")
    print("Train: 60d | Test: 30d | Step: 30d | Maker fees (0.04% RT)")
    print("=" * 90)

    pair_summary = {}

    for pair in ALL_PAIRS:
        print(f"\n{'='*90}")
        print(f"  {pair}")
        print(f"{'='*90}")

        df = load(pair)
        if df.empty or len(df) < 100:
            print(f"  No data")
            continue

        features = engineer_features(df)
        c = df["close"].values
        n = len(c)

        # Labels
        fwd_ret = np.zeros(n)
        for i in range(n - 6):
            fwd_ret[i] = (c[i + 6] - c[i]) / c[i]

        min_move = 2 * MAKER + 0.0001
        labels = np.zeros(n, dtype=int)
        for i in range(n - 6):
            if fwd_ret[i] > min_move:
                labels[i] = 1
            elif fwd_ret[i] < -min_move:
                labels[i] = -1

        n_long = sum(labels == 1)
        n_short = sum(labels == -1)
        print(f"  Bars: {n} | Labels: L={n_long} ({n_long/n*100:.0f}%), S={n_short} ({n_short/n*100:.0f}%)")

        results, n_folds = walk_forward(features, labels, fwd_ret)
        print(f"  Folds: {n_folds}")

        print(f"\n  {'Thr':<6} {'Trades':<8} {'WR%':<8} {'PnL(M)':<10} {'PnL(T)':<10} {'Avg/Trd':<10} {'/Day':<6}")
        print(f"  {'-'*60}")

        best_thr = None
        best_pnl = -999

        for thr in sorted(results.keys()):
            trades = results[thr]
            if not trades:
                continue
            arr = np.array(trades)
            wr = len(arr[arr > 0]) / len(arr) * 100
            pnl_maker = arr.sum() * 100
            # Taker equivalent
            taker_arr = arr - 2 * (TAKER - MAKER)
            pnl_taker = taker_arr.sum() * 100
            avg = arr.mean() * 100
            per_day = len(arr) / (n_folds * 30)

            marker = " ***" if pnl_maker > 0 else ""
            print(f"  {thr:<6.2f} {len(arr):<8} {wr:<7.1f}% {pnl_maker:<+9.2f}% "
                  f"{pnl_taker:<+9.2f}% {avg:<+9.4f}% {per_day:<5.1f}{marker}")

            if pnl_maker > best_pnl:
                best_pnl = pnl_maker
                best_thr = thr

        pair_summary[pair] = {
            "best_thr": best_thr, "best_pnl": best_pnl,
            "results": results, "folds": n_folds,
        }

    # ─── FINAL SUMMARY ────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("FINAL SUMMARY — All Pairs")
    print(f"{'='*90}")
    print(f"{'Pair':<20} {'Best Thr':<10} {'Best PnL(Maker)':<16} {'Trades':<8} {'WR%':<8} {'Viable?':<8}")
    print("-" * 75)

    viable_pairs = []
    for pair, info in pair_summary.items():
        thr = info["best_thr"]
        if thr is None:
            print(f"{pair:<20} {'--':<10} {'--':<16} {'--':<8} {'--':<8} NO")
            continue
        trades = info["results"][thr]
        arr = np.array(trades) if trades else np.array([0])
        wr = len(arr[arr > 0]) / max(len(arr), 1) * 100
        pnl = info["best_pnl"]
        viable = "YES" if pnl > 0 and len(trades) >= 30 else "NO"
        print(f"{pair:<20} {thr:<10.2f} {pnl:<+15.2f}% {len(trades):<8} {wr:<7.1f}% {viable:<8}")
        if viable == "YES":
            viable_pairs.append(pair)

    print(f"\n  Viable pairs for ML 5m: {viable_pairs if viable_pairs else 'NONE'}")

    # Train and save models for viable pairs
    if viable_pairs:
        print(f"\n{'='*90}")
        print("TRAINING PRODUCTION MODELS (last 60d)")
        print(f"{'='*90}")

        for pair in viable_pairs:
            df = load(pair)
            features = engineer_features(df)
            c = df["close"].values
            n = len(c)

            fwd_ret = np.zeros(n)
            for i in range(n - 6):
                fwd_ret[i] = (c[i + 6] - c[i]) / c[i]
            min_move = 2 * MAKER + 0.0001
            labels = np.zeros(n, dtype=int)
            for i in range(n - 6):
                if fwd_ret[i] > min_move: labels[i] = 1
                elif fwd_ret[i] < -min_move: labels[i] = -1

            # Train on last 60 days
            train_start = n - 60 * 288
            X_tr = features.iloc[train_start:]
            y_tr = labels[train_start:]
            mask = ~X_tr.isna().any(axis=1)
            X_tr = X_tr[mask]
            y_tr = y_tr[mask.values]

            params = {
                "objective": "binary", "metric": "auc", "learning_rate": 0.03,
                "num_leaves": 10, "max_depth": 3, "min_child_samples": 200,
                "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 5,
                "reg_alpha": 0.5, "reg_lambda": 2.0, "verbose": -1,
            }

            # Long
            y_long = (y_tr == 1).astype(int)
            params_l = params.copy()
            params_l["scale_pos_weight"] = sum(y_long == 0) / max(sum(y_long == 1), 1)
            model_l = lgb.train(params_l, lgb.Dataset(X_tr, label=y_long), num_boost_round=300)

            # Short
            y_short = (y_tr == -1).astype(int)
            params_s = params.copy()
            params_s["scale_pos_weight"] = sum(y_short == 0) / max(sum(y_short == 1), 1)
            model_s = lgb.train(params_s, lgb.Dataset(X_tr, label=y_short), num_boost_round=300)

            # Save
            pair_clean = pair.replace("/", "_").replace(":", "_")
            model_l.save_model(str(MODEL_DIR / f"ml_5m_{pair_clean}_long.txt"))
            model_s.save_model(str(MODEL_DIR / f"ml_5m_{pair_clean}_short.txt"))
            joblib.dump(features.columns.tolist(), MODEL_DIR / f"ml_5m_{pair_clean}_features.pkl")
            print(f"  Saved: {pair_clean} (long + short models)")


if __name__ == "__main__":
    main()

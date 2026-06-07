"""Train and serialize LightGBM models for 5m ML scalping strategy.

Produces model files that the strategy loads at runtime.
Retrain monthly by running this script with --retrain flag.

Output: models/ml_5m_{pair}_{direction}.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb
import joblib

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "okx" / "futures"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MAKER = 0.0002
PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT"]  # Only pairs with edge


def load(pair):
    fp = DATA_DIR / f"{pair}-5m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp)
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


def engineer_features(df):
    """Feature engineering — must match strategy's compute exactly."""
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    o = df["open"]
    v = df["volume"].astype(float)

    feat = pd.DataFrame(index=df.index)

    feat["body_pct"] = (c - o) / (c + 1e-10)
    feat["range_pct"] = (h - lo) / (c + 1e-10)
    feat["upper_wick_ratio"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (h - lo + 1e-10)
    feat["lower_wick_ratio"] = (pd.concat([c, o], axis=1).min(axis=1) - lo) / (h - lo + 1e-10)

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

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat["macd_hist"] = macd.iloc[:, 2]

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

    atr14 = ta.atr(h, lo, c, length=14)
    atr5 = ta.atr(h, lo, c, length=5)
    feat["atr_pct"] = atr14 / (c + 1e-10)
    feat["atr_ratio"] = atr5 / (atr14 + 1e-10)
    feat["range_vs_atr"] = (h - lo) / (atr14 + 1e-10)

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        feat["bb_pos"] = (c - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2] + 1e-10)
        feat["bb_width"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / (bb.iloc[:, 1] + 1e-10)

    vol_ema = ta.ema(v, length=20)
    feat["vol_ratio"] = v / (vol_ema + 1e-10)
    feat["vol_ratio_3"] = v.rolling(3).mean() / (vol_ema + 1e-10)
    feat["buy_pressure"] = (c - lo) / (h - lo + 1e-10)

    obv = ta.obv(c, v)
    if obv is not None:
        feat["obv_slope"] = (obv - obv.shift(5)) / (obv.abs().rolling(20).mean() + 1e-10)

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


def generate_labels(df, forward_bars=6):
    c = df["close"].values
    n = len(c)
    fee = 2 * MAKER
    min_move = fee + 0.0001

    fwd_ret = np.zeros(n)
    for i in range(n - forward_bars):
        fwd_ret[i] = (c[i + forward_bars] - c[i]) / c[i]

    labels = np.zeros(n, dtype=int)
    for i in range(n - forward_bars):
        if fwd_ret[i] > min_move:
            labels[i] = 1
        elif fwd_ret[i] < -min_move:
            labels[i] = -1

    return labels


def train_models(pair):
    """Train long and short models on all available data."""
    print(f"\nTraining models for {pair}...")
    df = load(pair)
    if df.empty:
        print(f"  No data for {pair}")
        return

    features = engineer_features(df)
    labels = generate_labels(df, forward_bars=6)

    mask = ~features.isna().any(axis=1)
    X = features[mask]
    y = labels[mask.values]

    # Use last 60 days as validation, rest as training
    val_size = 60 * 288
    X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    feature_names = X.columns.tolist()

    # Long model
    y_long_train = (y_train == 1).astype(int)
    y_long_val = (y_val == 1).astype(int)

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
        "verbose": -1,
    }

    params_l = params.copy()
    params_l["scale_pos_weight"] = sum(y_long_train == 0) / max(sum(y_long_train == 1), 1)

    train_data = lgb.Dataset(X_train, label=y_long_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_long_val, feature_name=feature_names, reference=train_data)

    model_long = lgb.train(
        params_l, train_data, num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # Short model
    y_short_train = (y_train == -1).astype(int)
    y_short_val = (y_val == -1).astype(int)

    params_s = params.copy()
    params_s["scale_pos_weight"] = sum(y_short_train == 0) / max(sum(y_short_train == 1), 1)

    train_data_s = lgb.Dataset(X_train, label=y_short_train, feature_name=feature_names)
    val_data_s = lgb.Dataset(X_val, label=y_short_val, feature_name=feature_names, reference=train_data_s)

    model_short = lgb.train(
        params_s, train_data_s, num_boost_round=500,
        valid_sets=[val_data_s],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # Save models
    pair_clean = pair.replace("/", "_").replace(":", "_")
    model_long.save_model(str(MODEL_DIR / f"ml_5m_{pair_clean}_long.txt"))
    model_short.save_model(str(MODEL_DIR / f"ml_5m_{pair_clean}_short.txt"))

    # Save feature names for validation
    joblib.dump(feature_names, MODEL_DIR / f"ml_5m_{pair_clean}_features.pkl")

    # Validation performance
    prob_long = model_long.predict(X_val)
    prob_short = model_short.predict(X_val)

    threshold = 0.65
    correct = 0
    total = 0
    for i in range(len(X_val)):
        if prob_long.flat[i] > threshold and prob_long.flat[i] > prob_short.flat[i] + 0.05:
            total += 1
            if y_val[i] == 1:
                correct += 1
        elif prob_short.flat[i] > threshold and prob_short.flat[i] > prob_long.flat[i] + 0.05:
            total += 1
            if y_val[i] == -1:
                correct += 1

    accuracy = correct / max(total, 1)
    print(f"  Saved: {MODEL_DIR / f'ml_5m_{pair_clean}_*.txt'}")
    print(f"  Validation (thr=0.65): {total} signals, accuracy={accuracy*100:.1f}%")
    print(f"  Long model: {model_long.best_iteration} rounds, short: {model_short.best_iteration} rounds")


def main():
    print("=" * 70)
    print("ML 5m MODEL TRAINING")
    print("=" * 70)

    for pair in PAIRS:
        train_models(pair)

    print(f"\nModels saved to: {MODEL_DIR}")
    print("Feature count:", len(joblib.load(MODEL_DIR / "ml_5m_BTC_USDT_USDT_features.pkl")))


if __name__ == "__main__":
    main()

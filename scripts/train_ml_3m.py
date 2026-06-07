"""Train ML model for SOL 3m scalping strategy.

Trains ExtraTrees on the most recent 30 days of SOL/USDT 3m data.
Saves model to models/ml_scalping_sol_3m_model.pkl
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
import joblib

DATA_DIR = Path("user_data/data/okx/futures")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "body_pct", "range_pct", "upper_wick", "lower_wick",
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_8", "ret_13", "ret_21",
    "atr_5", "atr_14", "atr_ratio", "atr_pct", "bb_pos", "bb_width",
    "rsi_3", "rsi_7", "rsi_14", "rsi_delta", "stoch_k", "stoch_d",
    "ema8_dist", "ema21_dist", "ema50_dist", "ema_spread",
    "vol_ratio", "vol_ratio_3", "obv_slope", "buy_pressure",
    "macd_hist", "adx",
    "hour_sin", "hour_cos", "is_us_session", "is_asia_session",
    "tick_direction", "tick_run", "spread_proxy",
]


def load_and_prepare():
    fpath = DATA_DIR / "SOL_USDT_USDT-3m-futures.feather"
    if not fpath.exists():
        print(f"ERROR: {fpath} not found")
        return None
    df = pd.read_feather(fpath)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_features(df):
    c, h, lo, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    df["body_pct"] = (c - o) / (c + 1e-10)
    df["range_pct"] = (h - lo) / (c + 1e-10)
    df["upper_wick"] = (h - np.maximum(c, o)) / (h - lo + 1e-10)
    df["lower_wick"] = (np.minimum(c, o) - lo) / (h - lo + 1e-10)

    for lb in [1, 2, 3, 5, 8, 13, 21]:
        df[f"ret_{lb}"] = c.pct_change(lb)

    tr = np.maximum(h - lo, np.maximum(abs(h - c.shift(1)), abs(lo - c.shift(1))))
    df["atr_5"] = tr.rolling(5).mean()
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr_5"] / (df["atr_14"] + 1e-10)
    df["atr_pct"] = df["atr_14"] / (c + 1e-10)

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_pos"] = (c - sma20) / (2 * std20 + 1e-10)
    df["bb_width"] = (4 * std20) / (sma20 + 1e-10)

    for p in [3, 7, 14]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        df[f"rsi_{p}"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df["rsi_delta"] = df["rsi_7"] - df["rsi_7"].shift(3)

    low14, high14 = lo.rolling(14).min(), h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    for p in [8, 21, 50]:
        ema = c.ewm(span=p, adjust=False).mean()
        df[f"ema{p}_dist"] = (c - ema) / (c + 1e-10)
    df["ema_spread"] = df["ema8_dist"] - df["ema50_dist"]

    vol_ema = v.rolling(20).mean()
    df["vol_ratio"] = v / (vol_ema + 1e-10)
    df["vol_ratio_3"] = v.rolling(3).mean() / (vol_ema + 1e-10)
    obv = (np.sign(c.diff()) * v).cumsum()
    df["obv_slope"] = obv.diff(5) / (c * 5 + 1e-10)
    df["buy_pressure"] = (c - lo) / (h - lo + 1e-10)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (c + 1e-10)

    plus_dm = (h.diff()).clip(lower=0)
    minus_dm = (-lo.diff()).clip(lower=0)
    atr_adx = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr_adx + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr_adx + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df["adx"] = dx.rolling(14).mean()

    df["hour_sin"] = np.sin(2 * np.pi * df["date"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["date"].dt.hour / 24)
    df["is_us_session"] = ((df["date"].dt.hour >= 13) & (df["date"].dt.hour <= 21)).astype(float)
    df["is_asia_session"] = ((df["date"].dt.hour >= 0) & (df["date"].dt.hour <= 8)).astype(float)

    df["tick_direction"] = np.sign(c.diff())
    ticks = df["tick_direction"].values
    runs = np.zeros(len(df))
    for i in range(1, len(ticks)):
        if ticks[i] == ticks[i - 1] and ticks[i] != 0:
            runs[i] = runs[i - 1] + 1
    df["tick_run"] = runs
    df["spread_proxy"] = (h - lo) / (c + 1e-10)

    return df


def main():
    print("=" * 70)
    print("Training ML Scalping SOL 3m Model")
    print("=" * 70)

    df = load_and_prepare()
    if df is None:
        return

    print(f"Data: {len(df)} bars, {df['date'].min()} to {df['date'].max()}")
    df = compute_features(df)

    # Use last 30 days for training
    train_window = 14400  # 30 days * 480 bars/day
    train_df = df.tail(train_window).copy()
    print(f"Training window: {len(train_df)} bars")

    # Fixed-horizon labels (6-bar forward)
    c = train_df["close"].values
    atr = train_df["atr_14"].values
    horizon = 6
    n = len(c)
    labels = np.zeros(n, dtype=int)

    for i in range(n - horizon):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        fwd_ret = (c[i + horizon] - c[i]) / c[i]
        thresh = max(0.0006, 0.3 * atr[i] / c[i])
        if fwd_ret > thresh:
            labels[i] = 1
        elif fwd_ret < -thresh:
            labels[i] = -1

    X = train_df[FEATURE_COLS].values
    valid_mask = (labels != 0) & np.all(np.isfinite(X), axis=1)
    X_train = X[valid_mask]
    y_train = (labels[valid_mask] == 1).astype(int)

    print(f"Valid samples: {len(X_train)} (long={y_train.sum()}, short={len(y_train)-y_train.sum()})")
    print(f"Label balance: {y_train.mean()*100:.1f}% long / {(1-y_train.mean())*100:.1f}% short")

    model = ExtraTreesClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50,
        max_features="sqrt", class_weight="balanced", n_jobs=-1,
    )
    print("\nTraining ExtraTrees...")
    model.fit(X_train, y_train)

    # In-sample metrics
    proba = model.predict_proba(X_train)[:, 1]
    for thr in [0.55, 0.58, 0.60, 0.65]:
        long_mask = proba > thr
        short_mask = proba < (1 - thr)
        trades = long_mask.sum() + short_mask.sum()
        if trades > 0:
            long_correct = (y_train[long_mask] == 1).sum() if long_mask.sum() > 0 else 0
            short_correct = (y_train[short_mask] == 0).sum() if short_mask.sum() > 0 else 0
            wr = (long_correct + short_correct) / trades * 100
            print(f"  @{thr:.2f}: {trades} trades, WR={wr:.1f}% (in-sample)")

    # Save model
    model_path = MODEL_DIR / "ml_scalping_sol_3m_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")

    # Feature importances
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\nTop 10 features:")
    for i in top_idx:
        print(f"  {FEATURE_COLS[i]:20s} {importances[i]:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()

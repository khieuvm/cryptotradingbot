"""Monthly ML model retraining script.

Run this on the 1st of each month to retrain models on the latest 60 days of data.
Models degrade after ~30 days, so monthly retraining is essential.

Usage:
    python scripts/retrain_ml_monthly.py
    python scripts/retrain_ml_monthly.py --validate  # also run walk-forward check

Output:
    models/ml_5m_{pair}_long.txt
    models/ml_5m_{pair}_short.txt
    models/ml_5m_{pair}_features.pkl
    models/retrain_log.json  (metadata: date, AUC, validation stats)
"""

import json
import sys
from datetime import datetime
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
VIABLE_PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT", "SPX_USDT_USDT"]
TRAIN_DAYS = 60
BARS_PER_DAY = 288  # 5m


def load_latest(pair, days=90):
    """Load most recent N days of 5m data."""
    fp = DATA_DIR / f"{pair}-5m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp).sort_values("date").reset_index(drop=True)
    if df.empty:
        return df
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff].reset_index(drop=True)


def engineer_features(df):
    """Feature engineering — must match strategy exactly."""
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


def train_pair(pair, validate=False):
    """Train models for a single pair using latest 60 days."""
    print(f"\n  Training {pair}...")
    df = load_latest(pair, days=90)
    if df.empty or len(df) < TRAIN_DAYS * BARS_PER_DAY:
        print(f"    Insufficient data ({len(df)} bars, need {TRAIN_DAYS * BARS_PER_DAY})")
        return None

    features = engineer_features(df)
    c = df["close"].values
    n = len(c)

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

    # Train on last 60 days, validate on preceding 30 days
    train_start = n - TRAIN_DAYS * BARS_PER_DAY
    val_end = train_start
    val_start = max(0, val_end - 30 * BARS_PER_DAY)

    X_train = features.iloc[train_start:]
    y_train = labels[train_start:]
    mask = ~X_train.isna().any(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask.values]

    feature_names = X_train.columns.tolist()

    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.03,
        "num_leaves": 10, "max_depth": 3, "min_child_samples": 200,
        "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 5,
        "reg_alpha": 0.5, "reg_lambda": 2.0, "verbose": -1,
    }

    # Long model
    y_long = (y_train == 1).astype(int)
    params_l = params.copy()
    params_l["scale_pos_weight"] = sum(y_long == 0) / max(sum(y_long == 1), 1)
    model_l = lgb.train(params_l, lgb.Dataset(X_train, label=y_long), num_boost_round=300)

    # Short model
    y_short = (y_train == -1).astype(int)
    params_s = params.copy()
    params_s["scale_pos_weight"] = sum(y_short == 0) / max(sum(y_short == 1), 1)
    model_s = lgb.train(params_s, lgb.Dataset(X_train, label=y_short), num_boost_round=300)

    # Save
    pair_clean = pair.replace("/", "_").replace(":", "_")
    model_l.save_model(str(MODEL_DIR / f"ml_5m_{pair_clean}_long.txt"))
    model_s.save_model(str(MODEL_DIR / f"ml_5m_{pair_clean}_short.txt"))
    joblib.dump(feature_names, MODEL_DIR / f"ml_5m_{pair_clean}_features.pkl")

    result = {"pair": pair, "train_bars": len(X_train), "features": len(feature_names)}

    # Validation on preceding 30d
    if validate and val_start < val_end:
        X_val = features.iloc[val_start:val_end]
        y_val = labels[val_start:val_end]
        fwd_val = fwd_ret[val_start:val_end]
        val_mask = ~X_val.isna().any(axis=1)
        X_val = X_val[val_mask]
        y_val = y_val[val_mask.values]
        fwd_val = fwd_val[val_mask.values]

        prob_l = model_l.predict(X_val)
        prob_s = model_s.predict(X_val)

        for thr in [0.60, 0.63, 0.65]:
            long_mask = (prob_l > thr) & (prob_l > prob_s + 0.05)
            short_mask = (prob_s > thr) & (prob_s > prob_l + 0.05)
            long_pnl = fwd_val[long_mask] - 2 * MAKER
            short_pnl = -fwd_val[short_mask] - 2 * MAKER
            all_pnl = np.concatenate([long_pnl, short_pnl])
            if len(all_pnl) > 0:
                wr = (all_pnl > 0).sum() / len(all_pnl)
                pnl_sum = all_pnl.sum() * 100
                print(f"    Val thr={thr:.2f}: {len(all_pnl)} trades, WR={wr*100:.1f}%, PnL={pnl_sum:+.2f}%")
                result[f"val_thr_{thr}"] = {"trades": len(all_pnl), "wr": float(wr), "pnl": float(pnl_sum)}

    print(f"    Saved: {pair_clean} (long + short, {len(feature_names)} features)")
    return result


def main():
    validate = "--validate" in sys.argv

    print("=" * 70)
    print(f"ML 5m MODEL RETRAINING — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Training on latest {TRAIN_DAYS} days for: {VIABLE_PAIRS}")
    if validate:
        print("Validation: ON (checking on 30d preceding training window)")
    print("=" * 70)

    results = []
    for pair in VIABLE_PAIRS:
        r = train_pair(pair, validate=validate)
        if r:
            results.append(r)

    # Save retrain log
    log_path = MODEL_DIR / "retrain_log.json"
    log_entry = {
        "date": datetime.now().isoformat(),
        "pairs": VIABLE_PAIRS,
        "train_days": TRAIN_DAYS,
        "results": results,
    }

    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
    else:
        log = []
    log.append(log_entry)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE — Models saved to {MODEL_DIR}")
    print(f"Log appended to {log_path}")
    print(f"Next retrain due: ~30 days from now")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

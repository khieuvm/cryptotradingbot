"""Train ML Enhanced 3m models offline.

Usage:
    python scripts/train_ml_3m_enhanced.py

Trains all 5 models for ml_scalping_enhanced_3m:
  1. ExtraTrees (base model 1)
  2. LightGBM (base model 2)
  3. LogisticRegression (stacking meta-learner)
  4. LightGBM (meta-labeling filter)
  5. GaussianHMM (regime detector)

Saves to models/ml_enhanced_3m_*.pkl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed. Run: pip install lightgbm")
    sys.exit(1)

try:
    from hmmlearn import hmm
except ImportError:
    print("ERROR: hmmlearn not installed. Run: pip install hmmlearn")
    sys.exit(1)

MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data" / "okx"

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


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 40 features for the ML model."""
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    o = df["open"]
    v = df["volume"].astype(float)

    df["body_pct"] = (c - o) / (c + 1e-10)
    df["range_pct"] = (h - lo) / (c + 1e-10)
    df["upper_wick"] = (h - np.maximum(c, o)) / (h - lo + 1e-10)
    df["lower_wick"] = (np.minimum(c, o) - lo) / (h - lo + 1e-10)

    for lb in [1, 2, 3, 5, 8, 13, 21]:
        df[f"ret_{lb}"] = c.pct_change(lb)

    tr = np.maximum(h - lo, np.maximum(abs(h - c.shift(1)), abs(lo - c.shift(1))))
    atr5 = tr.rolling(5).mean()
    atr14 = tr.rolling(14).mean()
    df["atr_5"] = atr5
    df["atr_14"] = atr14
    df["atr_ratio"] = atr5 / (atr14 + 1e-10)
    df["atr_pct"] = atr14 / (c + 1e-10)

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

    low14 = lo.rolling(14).min()
    high14 = h.rolling(14).max()
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

    dt = pd.to_datetime(df["date"])
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df["is_us_session"] = ((dt.dt.hour >= 13) & (dt.dt.hour <= 21)).astype(float)
    df["is_asia_session"] = ((dt.dt.hour >= 0) & (dt.dt.hour <= 8)).astype(float)

    tick_dir = np.sign(c.diff())
    df["tick_direction"] = tick_dir
    ticks = tick_dir.values
    runs = np.zeros(len(df))
    for i in range(1, len(ticks)):
        if ticks[i] == ticks[i - 1] and ticks[i] != 0:
            runs[i] = runs[i - 1] + 1
    df["tick_run"] = runs
    df["spread_proxy"] = (h - lo) / (c + 1e-10)

    return df


def load_data() -> pd.DataFrame:
    """Load SOL 3m data from freqtrade data directory."""
    patterns = [
        DATA_DIR / "SOL_USDT_USDT-3m-futures.feather",
        DATA_DIR / "SOL_USDT_USDT-3m.feather",
        Path(__file__).parent.parent / "user_data" / "data" / "okx" / "SOL_USDT_USDT-3m-futures.feather",
        Path(__file__).parent.parent / "user_data" / "data" / "okx" / "futures" / "SOL_USDT_USDT-3m-futures.feather",
    ]
    for p in patterns:
        if p.exists():
            df = pd.read_feather(p)
            print(f"Loaded {len(df)} candles from {p}")
            return df

    json_patterns = [
        DATA_DIR / "SOL_USDT_USDT-3m-futures.json",
        Path(__file__).parent.parent / "user_data" / "data" / "okx" / "SOL_USDT_USDT-3m-futures.json",
    ]
    for p in json_patterns:
        if p.exists():
            df = pd.read_json(p)
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            print(f"Loaded {len(df)} candles from {p}")
            return df

    print("ERROR: No SOL 3m data found. Download with:")
    print("  python ft_run.py download-data --exchange okx --pairs SOL/USDT:USDT -t 3m --days 90")
    sys.exit(1)


def main():
    print("=" * 60)
    print("ML Enhanced 3m Training — Stacking + Meta-Label + HMM")
    print("=" * 60)

    df = load_data()
    df = compute_features(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"After feature computation: {len(df)} valid rows")

    # Use last 30 days for training
    train_window = 14400  # 30d at 3m
    window = df.tail(train_window).copy()
    print(f"Training window: {len(window)} candles (last 30d)")

    c = window["close"].values
    atr = window["atr_14"].values

    # Label: fixed-horizon 6 bars
    horizon = 6
    labels = np.zeros(len(window), dtype=int)
    for i in range(len(c) - horizon):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        fwd_ret = (c[i + horizon] - c[i]) / c[i]
        thresh = max(0.0006, 0.3 * atr[i] / c[i])
        if fwd_ret > thresh:
            labels[i] = 1
        elif fwd_ret < -thresh:
            labels[i] = -1

    X_all = window[FEATURE_COLS].values
    valid_mask = (labels != 0) & np.all(np.isfinite(X_all), axis=1)
    X = X_all[valid_mask]
    y = (labels[valid_mask] == 1).astype(int)

    print(f"Labeled samples: {len(X)} (long={y.sum()}, short={len(y)-y.sum()})")

    if len(X) < 2000:
        print("ERROR: Not enough labeled samples. Need 2000+, got", len(X))
        sys.exit(1)

    # === LAYER 1: Stacking Ensemble ===
    print("\n--- Layer 1: Stacking Ensemble ---")
    kf = TimeSeriesSplit(n_splits=5)
    oof_et = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))

    et_params = dict(
        n_estimators=200, max_depth=6, min_samples_leaf=50,
        max_features="sqrt", class_weight="balanced", n_jobs=-1,
    )
    lgb_params = dict(
        n_estimators=200, max_depth=3, num_leaves=8,
        min_child_samples=200, learning_rate=0.03,
        colsample_bytree=0.5, subsample=0.7, subsample_freq=5,
        reg_alpha=0.5, reg_lambda=2.0,
        class_weight="balanced", verbose=-1, n_jobs=-1,
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y[train_idx]

        et = ExtraTreesClassifier(**et_params)
        et.fit(X_tr, y_tr)
        oof_et[val_idx] = et.predict_proba(X_val)[:, 1]

        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]

    # Final base models (trained on full data)
    et_final = ExtraTreesClassifier(**et_params)
    et_final.fit(X, y)

    lgb_final = lgb.LGBMClassifier(**lgb_params)
    lgb_final.fit(X, y)

    # Meta-learner trained on full-model predictions
    # (LogisticRegression with 2 features has no overfitting risk)
    full_et = et_final.predict_proba(X)[:, 1]
    full_lgb = lgb_final.predict_proba(X)[:, 1]
    stack_features = np.column_stack([full_et, full_lgb])
    meta_learner = LogisticRegression(C=1.0, max_iter=1000)
    meta_learner.fit(stack_features, y)

    stacked_probs = meta_learner.predict_proba(stack_features)[:, 1]
    stack_acc = ((stacked_probs > 0.5).astype(int) == y).mean()
    print(f"  ET OOF accuracy: {((oof_et[oof_et > 0] > 0.5).astype(int) == y[oof_et > 0]).mean():.4f}")
    print(f"  LGB OOF accuracy: {((oof_lgb[oof_lgb > 0] > 0.5).astype(int) == y[oof_lgb > 0]).mean():.4f}")
    print(f"  Stacked accuracy (full): {stack_acc:.4f}")

    # === LAYER 2: Meta-Labeling ===
    print("\n--- Layer 2: Meta-Labeling ---")
    # Use full-model predictions (not OOF) for meta-labeling since
    # the strategy at inference will use full-model outputs
    full_et_probs = et_final.predict_proba(X)[:, 1]
    full_lgb_probs = lgb_final.predict_proba(X)[:, 1]
    full_stack = meta_learner.predict_proba(np.column_stack([full_et_probs, full_lgb_probs]))[:, 1]

    threshold = 0.58
    signal_mask = (full_stack > threshold) | (full_stack < (1 - threshold))
    print(f"  Signals above threshold: {signal_mask.sum()}")

    meta_label_model = None
    if signal_mask.sum() > 200:
        signal_indices = np.where(valid_mask)[0][signal_mask]
        meta_labels = np.zeros(signal_mask.sum(), dtype=int)

        for j, idx in enumerate(signal_indices):
            if idx + horizon >= len(c):
                continue
            fwd = (c[idx + horizon] - c[idx]) / c[idx]
            is_long = full_stack[signal_mask][j] > 0.5
            meta_labels[j] = 1 if (is_long and fwd > 0) or (not is_long and fwd < 0) else 0

        meta_X = np.column_stack([
            full_et_probs[signal_mask],
            full_lgb_probs[signal_mask],
            full_stack[signal_mask],
            np.abs(full_et_probs[signal_mask] - full_lgb_probs[signal_mask]),
            X[signal_mask][:, FEATURE_COLS.index("adx")],
            X[signal_mask][:, FEATURE_COLS.index("atr_pct")],
            X[signal_mask][:, FEATURE_COLS.index("bb_width")],
            X[signal_mask][:, FEATURE_COLS.index("vol_ratio")],
            X[signal_mask][:, FEATURE_COLS.index("rsi_14")],
            X[signal_mask][:, FEATURE_COLS.index("hour_sin")],
            X[signal_mask][:, FEATURE_COLS.index("hour_cos")],
            X[signal_mask][:, FEATURE_COLS.index("is_us_session")],
            X[signal_mask][:, FEATURE_COLS.index("ema_spread")],
            X[signal_mask][:, FEATURE_COLS.index("ema8_dist")],
            X[signal_mask][:, FEATURE_COLS.index("atr_ratio")],
            X[signal_mask][:, FEATURE_COLS.index("ret_1")],
            X[signal_mask][:, FEATURE_COLS.index("ret_3")],
            X[signal_mask][:, FEATURE_COLS.index("ret_5")],
            X[signal_mask][:, FEATURE_COLS.index("stoch_k")],
            X[signal_mask][:, FEATURE_COLS.index("buy_pressure")],
        ])

        valid_meta = np.all(np.isfinite(meta_X), axis=1)
        if valid_meta.sum() > 100:
            meta_label_model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=3, num_leaves=8,
                min_child_samples=50, learning_rate=0.05,
                class_weight="balanced", verbose=-1, n_jobs=-1,
            )
            meta_label_model.fit(meta_X[valid_meta], meta_labels[valid_meta])
            ml_acc = (meta_label_model.predict(meta_X[valid_meta]) == meta_labels[valid_meta]).mean()
            print(f"  Meta-label accuracy: {ml_acc:.4f}")
            print(f"  Meta-label WR of signals: {meta_labels[valid_meta].mean():.4f}")
        else:
            print("  Not enough valid meta-features, skipping meta-label")

    # === LAYER 3: Regime HMM ===
    print("\n--- Layer 3: Regime HMM ---")
    hmm_features = np.column_stack([
        window["atr_pct"].values,
        window["adx"].values / 100.0,
        window["vol_ratio"].values / 5.0,
    ])
    hmm_clean = hmm_features[np.all(np.isfinite(hmm_features), axis=1)]
    print(f"  HMM training samples: {len(hmm_clean)}")

    hmm_model = None
    if len(hmm_clean) > 500:
        hmm_model = hmm.GaussianHMM(
            n_components=3, covariance_type="full",
            n_iter=100, random_state=42,
        )
        hmm_model.fit(hmm_clean)

        states = hmm_model.predict(hmm_clean)
        for s in range(3):
            mask = states == s
            if mask.sum() > 0:
                avg_atr = hmm_clean[mask, 0].mean()
                avg_adx = hmm_clean[mask, 1].mean() * 100
                avg_vol = hmm_clean[mask, 2].mean() * 5
                print(f"  State {s}: {mask.sum()} bars, avg_atr={avg_atr:.4f}, adx={avg_adx:.1f}, vol={avg_vol:.2f}")

    # === SAVE MODELS ===
    print("\n--- Saving Models ---")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(et_final, MODEL_DIR / "ml_enhanced_3m_et.pkl")
    joblib.dump(lgb_final, MODEL_DIR / "ml_enhanced_3m_lgb.pkl")
    joblib.dump(meta_learner, MODEL_DIR / "ml_enhanced_3m_stacker.pkl")
    if meta_label_model is not None:
        joblib.dump(meta_label_model, MODEL_DIR / "ml_enhanced_3m_metalabel.pkl")
    if hmm_model is not None:
        joblib.dump(hmm_model, MODEL_DIR / "ml_enhanced_3m_hmm.pkl")

    print(f"\nModels saved to {MODEL_DIR}/")
    print("  - ml_enhanced_3m_et.pkl (ExtraTrees)")
    print("  - ml_enhanced_3m_lgb.pkl (LightGBM)")
    print("  - ml_enhanced_3m_stacker.pkl (LogisticRegression)")
    if meta_label_model:
        print("  - ml_enhanced_3m_metalabel.pkl (Meta-Label LGB)")
    if hmm_model:
        print("  - ml_enhanced_3m_hmm.pkl (Regime HMM)")
    print("\nDone!")


if __name__ == "__main__":
    main()

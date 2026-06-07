"""ML Scalping Enhanced 3m: Stacking Ensemble + Meta-Labeling + Regime HMM.

Architecture (3 layers):
  Layer 1: Stacking Ensemble
    - ExtraTrees + LightGBM base models
    - LogisticRegression meta-learner on OOF predictions
  Layer 2: Meta-Labeling Filter
    - LightGBM binary classifier: "will this signal win?"
    - 20+ context features at signal time
  Layer 3: Regime HMM
    - 3-state GaussianHMM on [atr%, adx, vol_ratio]
    - Adjusts confidence threshold per regime state

Based on ml_scalping_sol_3m (56.9% WR) — expected improvement: +1-5pp WR.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config import StrategyConfig
from engine.events import Direction, ExitRequest, Signal, Urgency
from strategies import register_strategy
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
RETRAIN_INTERVAL = 160  # candles (8h at 3m)
TRAIN_WINDOW = 14400  # candles (30d at 3m)
MIN_TRAIN_SAMPLES = 2000

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

META_FEATURE_COLS = [
    "prob_et", "prob_lgb", "prob_final", "model_disagreement",
    "adx", "atr_pct", "bb_width", "vol_ratio", "rsi_14",
    "hour_sin", "hour_cos", "is_us_session",
    "ema_spread", "ema8_dist", "atr_ratio",
    "ret_1", "ret_3", "ret_5",
    "stoch_k", "buy_pressure",
]

PREFIX = "mle3_"


@register_strategy
class MLScalpingEnhanced3m(BaseStrategy):
    name = "ml_scalping_enhanced_3m"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self._et_model = None
        self._lgb_model = None
        self._meta_learner = None
        self._meta_label_model = None
        self._hmm = None
        self._model_ready = False
        self._candle_count = 0
        self._last_train_idx = 0
        self._regime_map = {0: "calm", 1: "normal", 2: "volatile"}

    def _on_init_hook(self) -> None:
        self._try_load_models()

    def _try_load_models(self):
        try:
            import joblib
        except ImportError:
            return

        models = {
            "et": MODEL_DIR / "ml_enhanced_3m_et.pkl",
            "lgb": MODEL_DIR / "ml_enhanced_3m_lgb.pkl",
            "meta_learner": MODEL_DIR / "ml_enhanced_3m_stacker.pkl",
            "meta_label": MODEL_DIR / "ml_enhanced_3m_metalabel.pkl",
            "hmm": MODEL_DIR / "ml_enhanced_3m_hmm.pkl",
        }

        try:
            if all(p.exists() for p in models.values()):
                self._et_model = joblib.load(models["et"])
                self._lgb_model = joblib.load(models["lgb"])
                self._meta_learner = joblib.load(models["meta_learner"])
                self._meta_label_model = joblib.load(models["meta_label"])
                self._hmm = joblib.load(models["hmm"])
                self._model_ready = True
                logger.info("ml_scalping_enhanced_3m: loaded all pre-trained models")
            else:
                missing = [k for k, p in models.items() if not p.exists()]
                logger.info(f"ml_scalping_enhanced_3m: missing models {missing}, will retrain")
        except Exception as e:
            logger.warning(f"ml_scalping_enhanced_3m: failed to load models ({e}), will retrain")

    def _train_all_models(self, df: pd.DataFrame):
        try:
            from sklearn.ensemble import ExtraTreesClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import TimeSeriesSplit
            import lightgbm as lgb
            from hmmlearn import hmm
        except ImportError as e:
            logger.warning(f"ml_scalping_enhanced_3m: missing dependency ({e})")
            return

        feat_cols = [f"{PREFIX}{c}" for c in FEATURE_COLS]
        if not all(c in df.columns for c in feat_cols):
            return

        window = df.tail(TRAIN_WINDOW).copy()
        if len(window) < MIN_TRAIN_SAMPLES:
            return

        c = window["close"].values
        atr_col = f"{PREFIX}atr_14_raw"
        if atr_col not in window.columns:
            return
        atr = window[atr_col].values

        # --- Label: fixed-horizon (6 bars forward) ---
        horizon = 6
        labels = np.zeros(len(window), dtype=int)
        n = len(c)
        for i in range(n - horizon):
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue
            fwd_ret = (c[i + horizon] - c[i]) / c[i]
            thresh = max(0.0006, 0.3 * atr[i] / c[i])
            if fwd_ret > thresh:
                labels[i] = 1
            elif fwd_ret < -thresh:
                labels[i] = -1

        X_all = window[feat_cols].values
        valid_mask = (labels != 0) & np.all(np.isfinite(X_all), axis=1)
        X = X_all[valid_mask]
        y = (labels[valid_mask] == 1).astype(int)

        if len(X) < MIN_TRAIN_SAMPLES:
            return

        # === LAYER 1: Stacking Ensemble ===
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

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            et = ExtraTreesClassifier(**et_params)
            et.fit(X_tr, y_tr)
            oof_et[val_idx] = et.predict_proba(X_val)[:, 1]

            lgb_model = lgb.LGBMClassifier(**lgb_params)
            lgb_model.fit(X_tr, y_tr)
            oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]

        # Train final base models on all data
        self._et_model = ExtraTreesClassifier(**et_params)
        self._et_model.fit(X, y)

        self._lgb_model = lgb.LGBMClassifier(**lgb_params)
        self._lgb_model.fit(X, y)

        # Train meta-learner on full-model predictions
        full_et = self._et_model.predict_proba(X)[:, 1]
        full_lgb = self._lgb_model.predict_proba(X)[:, 1]
        stack_features = np.column_stack([full_et, full_lgb])
        self._meta_learner = LogisticRegression(C=1.0, max_iter=1000)
        self._meta_learner.fit(stack_features, y)

        logger.info(
            f"ml_scalping_enhanced_3m: stacking trained on {len(X)} samples "
            f"(long={y.sum()}, short={len(y)-y.sum()})"
        )

        # === LAYER 2: Meta-Labeling ===
        # Use full-model predictions (not OOF) to match inference calibration
        full_et_probs = self._et_model.predict_proba(X)[:, 1]
        full_lgb_probs = self._lgb_model.predict_proba(X)[:, 1]
        full_stack = self._meta_learner.predict_proba(
            np.column_stack([full_et_probs, full_lgb_probs])
        )[:, 1]
        base_threshold = self.config.entry.get("confidence_threshold", 0.58)

        signal_mask = (full_stack > base_threshold) | (full_stack < (1 - base_threshold))
        if signal_mask.sum() > 200:
            signal_indices = np.where(valid_mask)[0][signal_mask]
            meta_labels = np.zeros(signal_mask.sum(), dtype=int)

            for j, idx in enumerate(signal_indices):
                if idx + horizon >= len(c):
                    continue
                fwd = (c[idx + horizon] - c[idx]) / c[idx]
                is_long = full_stack[signal_mask][j] > 0.5
                meta_labels[j] = 1 if (is_long and fwd > 0) or (not is_long and fwd < 0) else 0

            # Build meta-features
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
                self._meta_label_model = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=3, num_leaves=8,
                    min_child_samples=50, learning_rate=0.05,
                    class_weight="balanced", verbose=-1, n_jobs=-1,
                )
                self._meta_label_model.fit(meta_X[valid_meta], meta_labels[valid_meta])
                ml_acc = (self._meta_label_model.predict(meta_X[valid_meta]) == meta_labels[valid_meta]).mean()
                logger.info(f"ml_scalping_enhanced_3m: meta-label trained, acc={ml_acc:.3f}")
            else:
                self._meta_label_model = None
        else:
            self._meta_label_model = None

        # === LAYER 3: Regime HMM ===
        hmm_features = np.column_stack([
            window[f"{PREFIX}atr_pct"].values,
            window[f"{PREFIX}adx"].values / 100.0,
            window[f"{PREFIX}vol_ratio"].values / 5.0,
        ])
        hmm_clean = hmm_features[np.all(np.isfinite(hmm_features), axis=1)]
        if len(hmm_clean) > 500:
            try:
                self._hmm = hmm.GaussianHMM(
                    n_components=3, covariance_type="full",
                    n_iter=100, random_state=42,
                )
                self._hmm.fit(hmm_clean)
                logger.info("ml_scalping_enhanced_3m: HMM regime model fitted")
            except Exception as e:
                logger.warning(f"ml_scalping_enhanced_3m: HMM fit failed ({e})")
                self._hmm = None
        else:
            self._hmm = None

        self._model_ready = True

        # Save all models
        try:
            import joblib
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._et_model, MODEL_DIR / "ml_enhanced_3m_et.pkl")
            joblib.dump(self._lgb_model, MODEL_DIR / "ml_enhanced_3m_lgb.pkl")
            joblib.dump(self._meta_learner, MODEL_DIR / "ml_enhanced_3m_stacker.pkl")
            if self._meta_label_model is not None:
                joblib.dump(self._meta_label_model, MODEL_DIR / "ml_enhanced_3m_metalabel.pkl")
            else:
                (MODEL_DIR / "ml_enhanced_3m_metalabel.pkl").unlink(missing_ok=True)
            if self._hmm is not None:
                joblib.dump(self._hmm, MODEL_DIR / "ml_enhanced_3m_hmm.pkl")
            else:
                (MODEL_DIR / "ml_enhanced_3m_hmm.pkl").unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"ml_scalping_enhanced_3m: model save failed ({e})")

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        o = dataframe["open"]
        v = dataframe["volume"].astype(float)

        dataframe[f"{PREFIX}body_pct"] = (c - o) / (c + 1e-10)
        dataframe[f"{PREFIX}range_pct"] = (h - lo) / (c + 1e-10)
        dataframe[f"{PREFIX}upper_wick"] = (h - np.maximum(c, o)) / (h - lo + 1e-10)
        dataframe[f"{PREFIX}lower_wick"] = (np.minimum(c, o) - lo) / (h - lo + 1e-10)

        for lb in [1, 2, 3, 5, 8, 13, 21]:
            dataframe[f"{PREFIX}ret_{lb}"] = c.pct_change(lb)

        tr = np.maximum(h - lo, np.maximum(abs(h - c.shift(1)), abs(lo - c.shift(1))))
        atr5 = tr.rolling(5).mean()
        atr14 = tr.rolling(14).mean()
        dataframe[f"{PREFIX}atr_5"] = atr5
        dataframe[f"{PREFIX}atr_14"] = atr14
        dataframe[f"{PREFIX}atr_14_raw"] = atr14
        dataframe[f"{PREFIX}atr_ratio"] = atr5 / (atr14 + 1e-10)
        dataframe[f"{PREFIX}atr_pct"] = atr14 / (c + 1e-10)

        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        dataframe[f"{PREFIX}bb_pos"] = (c - sma20) / (2 * std20 + 1e-10)
        dataframe[f"{PREFIX}bb_width"] = (4 * std20) / (sma20 + 1e-10)

        for p in [3, 7, 14]:
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(p).mean()
            loss = (-delta.clip(upper=0)).rolling(p).mean()
            dataframe[f"{PREFIX}rsi_{p}"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        dataframe[f"{PREFIX}rsi_delta"] = dataframe[f"{PREFIX}rsi_7"] - dataframe[f"{PREFIX}rsi_7"].shift(3)

        low14 = lo.rolling(14).min()
        high14 = h.rolling(14).max()
        dataframe[f"{PREFIX}stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-10)
        dataframe[f"{PREFIX}stoch_d"] = dataframe[f"{PREFIX}stoch_k"].rolling(3).mean()

        for p in [8, 21, 50]:
            ema = c.ewm(span=p, adjust=False).mean()
            dataframe[f"{PREFIX}ema{p}_dist"] = (c - ema) / (c + 1e-10)
        dataframe[f"{PREFIX}ema_spread"] = dataframe[f"{PREFIX}ema8_dist"] - dataframe[f"{PREFIX}ema50_dist"]

        vol_ema = v.rolling(20).mean()
        dataframe[f"{PREFIX}vol_ratio"] = v / (vol_ema + 1e-10)
        dataframe[f"{PREFIX}vol_ratio_3"] = v.rolling(3).mean() / (vol_ema + 1e-10)
        obv = (np.sign(c.diff()) * v).cumsum()
        dataframe[f"{PREFIX}obv_slope"] = obv.diff(5) / (c * 5 + 1e-10)
        dataframe[f"{PREFIX}buy_pressure"] = (c - lo) / (h - lo + 1e-10)

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        dataframe[f"{PREFIX}macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (c + 1e-10)

        plus_dm = (h.diff()).clip(lower=0)
        minus_dm = (-lo.diff()).clip(lower=0)
        atr_adx = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr_adx + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr_adx + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        dataframe[f"{PREFIX}adx"] = dx.rolling(14).mean()

        dt = pd.to_datetime(dataframe["date"])
        dataframe[f"{PREFIX}hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        dataframe[f"{PREFIX}hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        dataframe[f"{PREFIX}is_us_session"] = ((dt.dt.hour >= 13) & (dt.dt.hour <= 21)).astype(float)
        dataframe[f"{PREFIX}is_asia_session"] = ((dt.dt.hour >= 0) & (dt.dt.hour <= 8)).astype(float)

        tick_dir = np.sign(c.diff())
        dataframe[f"{PREFIX}tick_direction"] = tick_dir
        ticks = tick_dir.values
        runs = np.zeros(len(dataframe))
        for i in range(1, len(ticks)):
            if ticks[i] == ticks[i - 1] and ticks[i] != 0:
                runs[i] = runs[i - 1] + 1
        dataframe[f"{PREFIX}tick_run"] = runs
        dataframe[f"{PREFIX}spread_proxy"] = (h - lo) / (c + 1e-10)

        return dataframe

    def on_tick(self, dataframe: pd.DataFrame, pair: str, current_time: datetime) -> None:
        self._candle_count += 1
        if self._candle_count % RETRAIN_INTERVAL == 0 or not self._model_ready:
            self._train_all_models(dataframe)

        signals = self.detect_entries(dataframe, pair)
        for sig in signals:
            self.emit_signal(
                pair=sig.pair, direction=sig.direction,
                strength=sig.strength, tag=sig.tag, metadata=sig.metadata,
            )

    def detect_entries(self, dataframe: pd.DataFrame, pair: str) -> list[Signal]:
        signals: list[Signal] = []
        if not self._model_ready:
            if len(dataframe) > MIN_TRAIN_SAMPLES:
                self._train_all_models(dataframe)
            if not self._model_ready:
                return signals

        if len(dataframe) < 60:
            return signals

        feat_cols = [f"{PREFIX}{c}" for c in FEATURE_COLS]
        last = dataframe.iloc[-1:]

        if not all(c in last.columns for c in feat_cols):
            return signals

        X = last[feat_cols].values
        if np.any(~np.isfinite(X)):
            return signals

        # === LAYER 1: Stacking Ensemble ===
        prob_et = self._et_model.predict_proba(X)[0]
        prob_et_long = prob_et[1] if len(prob_et) > 1 else 0.5

        prob_lgb = self._lgb_model.predict_proba(X)[0]
        prob_lgb_long = prob_lgb[1] if len(prob_lgb) > 1 else 0.5

        stack_input = np.array([[prob_et_long, prob_lgb_long]])
        prob_final = self._meta_learner.predict_proba(stack_input)[0]
        prob_long = prob_final[1] if len(prob_final) > 1 else 0.5

        # === LAYER 3: Regime HMM (applied before threshold check) ===
        base_threshold = self.config.entry.get("confidence_threshold", 0.58)
        regime_label = "normal"

        if self._hmm is not None:
            hmm_obs = np.array([[
                float(last.iloc[0].get(f"{PREFIX}atr_pct", 0.01)),
                float(last.iloc[0].get(f"{PREFIX}adx", 25)) / 100.0,
                float(last.iloc[0].get(f"{PREFIX}vol_ratio", 1.0)) / 5.0,
            ]])
            if np.all(np.isfinite(hmm_obs)):
                try:
                    regime_state = self._hmm.predict(hmm_obs)[0]
                    regime_label = self._regime_map.get(regime_state, "normal")
                except Exception:
                    pass

        offsets = self.config.entry.get("regime_offsets", {"calm": -0.02, "normal": 0.0, "volatile": 0.03})
        threshold = base_threshold + offsets.get(regime_label, 0.0)

        # Determine direction
        direction = None
        strength = 0.0
        if prob_long > threshold:
            direction = Direction.LONG
            strength = float(prob_long)
        elif prob_long < (1 - threshold):
            direction = Direction.SHORT
            strength = float(1 - prob_long)

        if direction is None:
            return signals

        # === LAYER 2: Meta-Labeling Filter ===
        use_meta = self.config.entry.get("use_meta_label", True)
        if use_meta and self._meta_label_model is not None:
            meta_features = np.array([[
                prob_et_long,
                prob_lgb_long,
                prob_long,
                abs(prob_et_long - prob_lgb_long),
                float(last.iloc[0].get(f"{PREFIX}adx", 25)),
                float(last.iloc[0].get(f"{PREFIX}atr_pct", 0.01)),
                float(last.iloc[0].get(f"{PREFIX}bb_width", 0.05)),
                float(last.iloc[0].get(f"{PREFIX}vol_ratio", 1.0)),
                float(last.iloc[0].get(f"{PREFIX}rsi_14", 50)),
                float(last.iloc[0].get(f"{PREFIX}hour_sin", 0)),
                float(last.iloc[0].get(f"{PREFIX}hour_cos", 0)),
                float(last.iloc[0].get(f"{PREFIX}is_us_session", 0)),
                float(last.iloc[0].get(f"{PREFIX}ema_spread", 0)),
                float(last.iloc[0].get(f"{PREFIX}ema8_dist", 0)),
                float(last.iloc[0].get(f"{PREFIX}atr_ratio", 1.0)),
                float(last.iloc[0].get(f"{PREFIX}ret_1", 0)),
                float(last.iloc[0].get(f"{PREFIX}ret_3", 0)),
                float(last.iloc[0].get(f"{PREFIX}ret_5", 0)),
                float(last.iloc[0].get(f"{PREFIX}stoch_k", 50)),
                float(last.iloc[0].get(f"{PREFIX}buy_pressure", 0.5)),
            ]])

            if np.all(np.isfinite(meta_features)):
                meta_prob = self._meta_label_model.predict_proba(meta_features)[0]
                meta_score = meta_prob[1] if len(meta_prob) > 1 else 0.5
                meta_threshold = self.config.entry.get("meta_label_threshold", 0.55)

                if meta_score < meta_threshold:
                    return signals

                strength *= meta_score

        # All layers passed — emit signal
        tag_suffix = "long" if direction == Direction.LONG else "short"
        signals.append(Signal(
            strategy_name=self.name, pair=pair,
            direction=direction, strength=strength,
            tag=f"{self.name}_{tag_suffix}",
            timestamp=datetime.utcnow(),
            metadata={
                "prob_et": float(prob_et_long),
                "prob_lgb": float(prob_lgb_long),
                "prob_final": float(prob_long),
                "regime": regime_label,
            },
        ))

        return signals

    def detect_exits(
        self, dataframe: pd.DataFrame, pair: str, trade_info: dict | None
    ) -> ExitRequest | None:
        if trade_info is None:
            return None

        current_profit = trade_info.get("current_profit", 0)
        current_time = trade_info.get("current_time", datetime.utcnow())
        entry_time = trade_info.get("entry_time", current_time)
        open_rate = trade_info.get("entry_rate", 0)

        last = dataframe.iloc[-1]
        atr = float(last.get(f"{PREFIX}atr_14_raw", 0))
        if atr <= 0 or open_rate <= 0:
            return None

        tp_pct = self.tp_atr_mult * atr / open_rate
        if current_profit >= tp_pct:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="TP_HIT", urgency=Urgency.IMMEDIATE,
                timestamp=datetime.utcnow(),
            )

        minutes = (current_time - entry_time).total_seconds() / 60
        exit_cfg = self.config.exit

        if minutes >= 9 and current_profit < exit_cfg.get("time_cut_9m", -0.006):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_9m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if minutes >= 18 and current_profit < exit_cfg.get("time_cut_18m", -0.003):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_18m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if minutes >= 30 and current_profit < exit_cfg.get("time_cut_30m", -0.001):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_30m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if minutes >= 45 and current_profit < exit_cfg.get("time_cut_45m", 0.0):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_45m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

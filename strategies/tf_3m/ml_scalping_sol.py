"""ML Scalping SOL 3m: ExtraTrees-based directional prediction with online retraining.

Trained on research/ml_scalping_test.py results showing:
  SOL_3m_ExtraTrees_fixed @0.58: 541 trades, 56.9% WR, +24% PnL (taker fees), Sharpe 5.8
  SOL_3m_ExtraTrees_fixed @0.60: 305 trades, 56.7% WR, +15% PnL (taker fees), Sharpe 7.3

Uses fixed-horizon labeling (6-bar forward return vs adaptive threshold).
Retrains every 8h (160 candles) on a rolling 30d window.
Works with both maker (limit) and taker (market) orders.
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


@register_strategy
class MLScalpingSOL3m(BaseStrategy):
    name = "ml_scalping_sol_3m"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self._model = None
        self._last_train_idx = 0
        self._candle_count = 0
        self._model_ready = False

    def _on_init_hook(self) -> None:
        self._try_load_model()

    def _try_load_model(self):
        """Try loading a pre-trained model from disk."""
        try:
            import joblib
        except ImportError:
            return
        model_path = MODEL_DIR / "ml_scalping_sol_3m_model.pkl"
        if model_path.exists():
            try:
                self._model = joblib.load(model_path)
                self._model_ready = True
                logger.info("ml_scalping_sol_3m: loaded pre-trained model")
            except Exception as e:
                logger.warning(f"ml_scalping_sol_3m: failed to load model ({e}), will retrain")

    def _train_model(self, df: pd.DataFrame):
        """Train ExtraTrees on rolling window of labeled data."""
        try:
            from sklearn.ensemble import ExtraTreesClassifier
        except ImportError:
            logger.warning("ml_scalping_sol_3m: sklearn not available")
            return

        feat_cols_prefixed = [f"mls3_{c}" for c in FEATURE_COLS]
        if not all(c in df.columns for c in feat_cols_prefixed):
            return

        window = df.tail(TRAIN_WINDOW).copy()
        if len(window) < MIN_TRAIN_SAMPLES:
            return

        c = window["close"].values
        atr = window["mls3_atr_14_raw"].values if "mls3_atr_14_raw" in window.columns else None
        if atr is None:
            return

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

        X = window[feat_cols_prefixed].values
        valid_mask = (labels != 0) & np.all(np.isfinite(X), axis=1)
        X_train = X[valid_mask]
        y_train = (labels[valid_mask] == 1).astype(int)

        if len(X_train) < MIN_TRAIN_SAMPLES:
            return

        model = ExtraTreesClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            max_features="sqrt", class_weight="balanced", n_jobs=-1,
        )
        model.fit(X_train, y_train)
        self._model = model
        self._model_ready = True
        logger.info(
            f"ml_scalping_sol_3m: trained on {len(X_train)} samples "
            f"(long={y_train.sum()}, short={len(y_train)-y_train.sum()})"
        )

        try:
            import joblib
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODEL_DIR / "ml_scalping_sol_3m_model.pkl")
        except Exception:
            pass

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        o = dataframe["open"]
        v = dataframe["volume"].astype(float)

        dataframe["mls3_body_pct"] = (c - o) / (c + 1e-10)
        dataframe["mls3_range_pct"] = (h - lo) / (c + 1e-10)
        dataframe["mls3_upper_wick"] = (h - np.maximum(c, o)) / (h - lo + 1e-10)
        dataframe["mls3_lower_wick"] = (np.minimum(c, o) - lo) / (h - lo + 1e-10)

        for lb in [1, 2, 3, 5, 8, 13, 21]:
            dataframe[f"mls3_ret_{lb}"] = c.pct_change(lb)

        tr = np.maximum(h - lo, np.maximum(abs(h - c.shift(1)), abs(lo - c.shift(1))))
        atr5 = tr.rolling(5).mean()
        atr14 = tr.rolling(14).mean()
        dataframe["mls3_atr_5"] = atr5
        dataframe["mls3_atr_14"] = atr14
        dataframe["mls3_atr_14_raw"] = atr14
        dataframe["mls3_atr_ratio"] = atr5 / (atr14 + 1e-10)
        dataframe["mls3_atr_pct"] = atr14 / (c + 1e-10)

        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        dataframe["mls3_bb_pos"] = (c - sma20) / (2 * std20 + 1e-10)
        dataframe["mls3_bb_width"] = (4 * std20) / (sma20 + 1e-10)

        for p in [3, 7, 14]:
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(p).mean()
            loss = (-delta.clip(upper=0)).rolling(p).mean()
            dataframe[f"mls3_rsi_{p}"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        dataframe["mls3_rsi_delta"] = dataframe["mls3_rsi_7"] - dataframe["mls3_rsi_7"].shift(3)

        low14 = lo.rolling(14).min()
        high14 = h.rolling(14).max()
        dataframe["mls3_stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-10)
        dataframe["mls3_stoch_d"] = dataframe["mls3_stoch_k"].rolling(3).mean()

        for p in [8, 21, 50]:
            ema = c.ewm(span=p, adjust=False).mean()
            dataframe[f"mls3_ema{p}_dist"] = (c - ema) / (c + 1e-10)
        dataframe["mls3_ema_spread"] = dataframe["mls3_ema8_dist"] - dataframe["mls3_ema50_dist"]

        vol_ema = v.rolling(20).mean()
        dataframe["mls3_vol_ratio"] = v / (vol_ema + 1e-10)
        dataframe["mls3_vol_ratio_3"] = v.rolling(3).mean() / (vol_ema + 1e-10)
        obv = (np.sign(c.diff()) * v).cumsum()
        dataframe["mls3_obv_slope"] = obv.diff(5) / (c * 5 + 1e-10)
        dataframe["mls3_buy_pressure"] = (c - lo) / (h - lo + 1e-10)

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        dataframe["mls3_macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (c + 1e-10)

        plus_dm = (h.diff()).clip(lower=0)
        minus_dm = (-lo.diff()).clip(lower=0)
        atr_adx = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr_adx + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr_adx + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        dataframe["mls3_adx"] = dx.rolling(14).mean()

        dt = pd.to_datetime(dataframe["date"])
        dataframe["mls3_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        dataframe["mls3_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        dataframe["mls3_is_us_session"] = ((dt.dt.hour >= 13) & (dt.dt.hour <= 21)).astype(float)
        dataframe["mls3_is_asia_session"] = ((dt.dt.hour >= 0) & (dt.dt.hour <= 8)).astype(float)

        tick_dir = np.sign(c.diff())
        dataframe["mls3_tick_direction"] = tick_dir
        ticks = tick_dir.values
        runs = np.zeros(len(dataframe))
        for i in range(1, len(ticks)):
            if ticks[i] == ticks[i - 1] and ticks[i] != 0:
                runs[i] = runs[i - 1] + 1
        dataframe["mls3_tick_run"] = runs
        dataframe["mls3_spread_proxy"] = (h - lo) / (c + 1e-10)

        return dataframe

    def on_tick(self, dataframe: pd.DataFrame, pair: str, current_time: datetime) -> None:
        self._candle_count += 1
        if self._candle_count % RETRAIN_INTERVAL == 0 or not self._model_ready:
            self._train_model(dataframe)

        signals = self.detect_entries(dataframe, pair)
        for sig in signals:
            self.emit_signal(
                pair=sig.pair, direction=sig.direction,
                strength=sig.strength, tag=sig.tag, metadata=sig.metadata,
            )

    def detect_entries(self, dataframe: pd.DataFrame, pair: str) -> list[Signal]:
        signals: list[Signal] = []
        if not self._model_ready or self._model is None:
            if len(dataframe) > MIN_TRAIN_SAMPLES:
                self._train_model(dataframe)
            if not self._model_ready:
                return signals

        if len(dataframe) < 60:
            return signals

        feat_cols_prefixed = [f"mls3_{c}" for c in FEATURE_COLS]
        last = dataframe.iloc[-1:]

        if not all(c in last.columns for c in feat_cols_prefixed):
            return signals

        X = last[feat_cols_prefixed].values
        if np.any(~np.isfinite(X)):
            return signals

        proba = self._model.predict_proba(X)[0]
        prob_long = proba[1] if len(proba) > 1 else 0.5

        threshold = self.config.entry.get("confidence_threshold", 0.58)

        if prob_long > threshold:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=float(prob_long),
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"prob": float(prob_long)},
            ))
        elif prob_long < (1 - threshold):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=float(1 - prob_long),
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"prob": float(1 - prob_long)},
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
        atr = float(last.get("mls3_atr_14_raw", 0))
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

"""ML Ensemble 5m Strategy: LightGBM-based directional prediction.

Uses pre-trained models to predict 6-bar (30min) forward returns.
Only signals when model confidence exceeds threshold (selective trading).
Designed for limit-order entry (maker fees) but works with taker too.

Edge: ETH +32.86% (866 trades, WR 60.3%) at threshold 0.63 with maker fees.
BTC +13.61% (581 trades, WR 55.4%) at threshold 0.67 with maker fees.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

from engine.config import StrategyConfig
from engine.events import Direction, ExitRequest, Signal, Urgency
from strategies import register_strategy
from strategies.base import BaseStrategy

MODEL_DIR = Path(__file__).parent.parent.parent / "models"


@register_strategy
class MLEnsemble5mStrategy(BaseStrategy):
    name = "ml_ensemble_5m"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self._models_loaded = False
        self._model_long = None
        self._model_short = None
        self._feature_names = None

    def _load_models(self, pair: str):
        """Lazy-load models for the given pair."""
        if self._models_loaded:
            return

        try:
            import lightgbm as lgb
            import joblib
        except ImportError:
            return

        pair_clean = pair.replace("/", "_").replace(":", "_")
        long_path = MODEL_DIR / f"ml_5m_{pair_clean}_long.txt"
        short_path = MODEL_DIR / f"ml_5m_{pair_clean}_short.txt"
        feat_path = MODEL_DIR / f"ml_5m_{pair_clean}_features.pkl"

        if long_path.exists() and short_path.exists() and feat_path.exists():
            self._model_long = lgb.Booster(model_file=str(long_path))
            self._model_short = lgb.Booster(model_file=str(short_path))
            self._feature_names = joblib.load(feat_path)
            self._models_loaded = True

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        o = dataframe["open"]
        v = dataframe["volume"].astype(float)

        # ATR for exits
        dataframe["ml5_atr"] = ta.atr(h, lo, c, length=14)

        # All ML features
        dataframe["ml5_body_pct"] = (c - o) / (c + 1e-10)
        dataframe["ml5_range_pct"] = (h - lo) / (c + 1e-10)
        max_co = pd.concat([c, o], axis=1).max(axis=1)
        min_co = pd.concat([c, o], axis=1).min(axis=1)
        dataframe["ml5_upper_wick_ratio"] = (h - max_co) / (h - lo + 1e-10)
        dataframe["ml5_lower_wick_ratio"] = (min_co - lo) / (h - lo + 1e-10)

        for lb in [1, 3, 6, 12, 24, 36]:
            dataframe[f"ml5_ret_{lb}"] = c.pct_change(lb)

        dataframe["ml5_rsi_3"] = ta.rsi(c, length=3)
        dataframe["ml5_rsi_9"] = ta.rsi(c, length=9)
        dataframe["ml5_rsi_14"] = ta.rsi(c, length=14)
        dataframe["ml5_rsi_9_delta"] = dataframe["ml5_rsi_9"] - dataframe["ml5_rsi_9"].shift(3)
        dataframe["ml5_cci"] = ta.cci(h, lo, c, length=14)

        stoch = ta.stoch(h, lo, c, k=14, d=3)
        if stoch is not None:
            dataframe["ml5_stoch_k"] = stoch.iloc[:, 0]
            dataframe["ml5_stoch_d"] = stoch.iloc[:, 1]
        else:
            dataframe["ml5_stoch_k"] = 50.0
            dataframe["ml5_stoch_d"] = 50.0

        macd = ta.macd(c, fast=12, slow=26, signal=9)
        if macd is not None:
            dataframe["ml5_macd_hist"] = macd.iloc[:, 2]
        else:
            dataframe["ml5_macd_hist"] = 0.0

        ema8 = ta.ema(c, length=8)
        ema21 = ta.ema(c, length=21)
        ema50 = ta.ema(c, length=50)
        dataframe["ml5_price_vs_ema8"] = (c - ema8) / (ema8 + 1e-10)
        dataframe["ml5_price_vs_ema21"] = (c - ema21) / (ema21 + 1e-10)
        dataframe["ml5_ema_spread"] = (ema8 - ema21) / (ema21 + 1e-10)

        adx_r = ta.adx(h, lo, c, length=14)
        if adx_r is not None:
            dataframe["ml5_adx"] = adx_r.iloc[:, 0]
            dataframe["ml5_di_diff"] = adx_r.iloc[:, 1] - adx_r.iloc[:, 2]
        else:
            dataframe["ml5_adx"] = 25.0
            dataframe["ml5_di_diff"] = 0.0

        atr14 = dataframe["ml5_atr"]
        atr5 = ta.atr(h, lo, c, length=5)
        dataframe["ml5_atr_pct"] = atr14 / (c + 1e-10)
        dataframe["ml5_atr_ratio"] = atr5 / (atr14 + 1e-10)
        dataframe["ml5_range_vs_atr"] = (h - lo) / (atr14 + 1e-10)

        bb = ta.bbands(c, length=20, std=2.0)
        if bb is not None:
            dataframe["ml5_bb_pos"] = (c - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2] + 1e-10)
            dataframe["ml5_bb_width"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / (bb.iloc[:, 1] + 1e-10)
        else:
            dataframe["ml5_bb_pos"] = 0.5
            dataframe["ml5_bb_width"] = 0.0

        vol_ema = ta.ema(v, length=20)
        dataframe["ml5_vol_ratio"] = v / (vol_ema + 1e-10)
        dataframe["ml5_vol_ratio_3"] = v.rolling(3).mean() / (vol_ema + 1e-10)
        dataframe["ml5_buy_pressure"] = (c - lo) / (h - lo + 1e-10)

        obv = ta.obv(c, v)
        if obv is not None:
            dataframe["ml5_obv_slope"] = (obv - obv.shift(5)) / (obv.abs().rolling(20).mean() + 1e-10)
        else:
            dataframe["ml5_obv_slope"] = 0.0

        dataframe["ml5_ret_15m"] = c.pct_change(3)
        dataframe["ml5_range_15m_pct"] = (h.rolling(3).max() - lo.rolling(3).min()) / (c + 1e-10)
        dataframe["ml5_ret_1h"] = c.pct_change(12)
        dataframe["ml5_range_1h_pct"] = (h.rolling(12).max() - lo.rolling(12).min()) / (c + 1e-10)
        dataframe["ml5_ret_4h"] = c.pct_change(48)

        dt = pd.to_datetime(dataframe["date"])
        dataframe["ml5_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        dataframe["ml5_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        dataframe["ml5_is_us_session"] = ((dt.dt.hour >= 13) & (dt.dt.hour < 21)).astype(float)

        dataframe["ml5_pos_in_day_range"] = (
            (c - lo.rolling(288).min()) / (h.rolling(288).max() - lo.rolling(288).min() + 1e-10)
        )

        return dataframe

    def _get_feature_row(self, dataframe: pd.DataFrame) -> pd.DataFrame | None:
        """Extract feature row matching model's expected feature names."""
        if self._feature_names is None:
            return None

        last = dataframe.iloc[-1:]
        row = pd.DataFrame(index=[0])

        feature_map = {
            "body_pct": "ml5_body_pct",
            "range_pct": "ml5_range_pct",
            "upper_wick_ratio": "ml5_upper_wick_ratio",
            "lower_wick_ratio": "ml5_lower_wick_ratio",
            "rsi_3": "ml5_rsi_3",
            "rsi_9": "ml5_rsi_9",
            "rsi_14": "ml5_rsi_14",
            "rsi_9_delta": "ml5_rsi_9_delta",
            "cci": "ml5_cci",
            "stoch_k": "ml5_stoch_k",
            "stoch_d": "ml5_stoch_d",
            "macd_hist": "ml5_macd_hist",
            "price_vs_ema8": "ml5_price_vs_ema8",
            "price_vs_ema21": "ml5_price_vs_ema21",
            "ema_spread": "ml5_ema_spread",
            "adx": "ml5_adx",
            "di_diff": "ml5_di_diff",
            "atr_pct": "ml5_atr_pct",
            "atr_ratio": "ml5_atr_ratio",
            "range_vs_atr": "ml5_range_vs_atr",
            "bb_pos": "ml5_bb_pos",
            "bb_width": "ml5_bb_width",
            "vol_ratio": "ml5_vol_ratio",
            "vol_ratio_3": "ml5_vol_ratio_3",
            "buy_pressure": "ml5_buy_pressure",
            "obv_slope": "ml5_obv_slope",
            "ret_15m": "ml5_ret_15m",
            "range_15m_pct": "ml5_range_15m_pct",
            "ret_1h": "ml5_ret_1h",
            "range_1h_pct": "ml5_range_1h_pct",
            "ret_4h": "ml5_ret_4h",
            "hour_sin": "ml5_hour_sin",
            "hour_cos": "ml5_hour_cos",
            "is_us_session": "ml5_is_us_session",
            "pos_in_day_range": "ml5_pos_in_day_range",
        }

        for lb in [1, 3, 6, 12, 24, 36]:
            feature_map[f"ret_{lb}"] = f"ml5_ret_{lb}"

        for feat_name in self._feature_names:
            col_name = feature_map.get(feat_name, f"ml5_{feat_name}")
            if col_name in last.columns:
                val = float(last[col_name].iloc[0])
                row[feat_name] = val
            else:
                row[feat_name] = 0.0

        if row.isna().any(axis=1).iloc[0]:
            return None

        return row

    def on_tick(self, dataframe: pd.DataFrame, pair: str, current_time: datetime) -> None:
        signals = self.detect_entries(dataframe, pair)
        for sig in signals:
            self.emit_signal(
                pair=sig.pair, direction=sig.direction,
                strength=sig.strength, tag=sig.tag, metadata=sig.metadata,
            )

    def detect_entries(self, dataframe: pd.DataFrame, pair: str) -> list[Signal]:
        signals: list[Signal] = []
        if len(dataframe) < 60:
            return signals

        self._load_models(pair)
        if not self._models_loaded:
            return signals

        row = self._get_feature_row(dataframe)
        if row is None:
            return signals

        prob_long = float(self._model_long.predict(row)[0])
        prob_short = float(self._model_short.predict(row)[0])

        threshold = self.config.entry.get("confidence_threshold", 0.65)
        # Per-pair threshold overrides
        pair_key = pair.replace("/", "_").replace(":", "_").lower()
        if "btc" in pair_key:
            threshold = self.config.entry.get("confidence_threshold_btc", threshold)
        elif "eth" in pair_key:
            threshold = self.config.entry.get("confidence_threshold_eth", threshold)
        elif "spx" in pair_key:
            threshold = self.config.entry.get("confidence_threshold_spx", threshold)
        direction_gap = self.config.entry.get("direction_gap", 0.05)

        if prob_long > threshold and prob_long > prob_short + direction_gap:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=prob_long,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"prob_long": prob_long, "prob_short": prob_short},
            ))
        elif prob_short > threshold and prob_short > prob_long + direction_gap:
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=prob_short,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"prob_long": prob_long, "prob_short": prob_short},
            ))

        return signals

    def detect_exits(
        self, dataframe: pd.DataFrame, pair: str, trade_info: dict | None
    ) -> ExitRequest | None:
        if trade_info is None:
            return None

        last = dataframe.iloc[-1]
        current_profit = trade_info.get("current_profit", 0)
        current_time = trade_info.get("current_time", datetime.utcnow())
        entry_time = trade_info.get("entry_time", current_time)
        open_rate = trade_info.get("entry_rate", 0)

        atr = float(last.get("ml5_atr", 0))
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

        if minutes >= 15 and current_profit < exit_cfg.get("time_cut_15m", -0.004):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_15m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if minutes >= 30 and current_profit < exit_cfg.get("time_cut_30m", -0.002):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_30m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if minutes >= 45:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_45m", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

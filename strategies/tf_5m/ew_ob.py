"""Elliott Wave + Order Block strategy (HTF trend, LTF entry).

15m Elliott Wave determines trend direction (bullish/bearish).
15m Order Blocks identify institutional support/resistance zones.
Enters on LTF (5m or 1m) when price touches an OB zone aligned with EW trend,
confirmed by a reversal candle (green at bullish OB, red at bearish OB).

Validated: WF 4/4 windows OOS pass, MC p=0.000 for both 5m and 1m.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from engine.config import StrategyConfig
from engine.events import Direction, ExitRequest, Signal, Urgency
from strategies import register_strategy
from strategies.base import BaseStrategy


def _compute_zigzag(highs: np.ndarray, lows: np.ndarray, pct: float) -> list[tuple[int, float, str]]:
    """Percentage-based zigzag. Returns [(index, price, 'high'|'low'), ...]."""
    if len(highs) < 5:
        return []

    points: list[tuple[int, float, str]] = []
    last_high = highs[0]
    last_high_idx = 0
    last_low = lows[0]
    last_low_idx = 0
    direction = 0

    for i in range(1, len(highs)):
        h, lo = highs[i], lows[i]

        if direction >= 0:
            if h > last_high:
                last_high, last_high_idx = h, i
            if lo < last_high * (1 - pct):
                if direction != 0 or last_high > highs[0] * (1 + pct / 2):
                    points.append((last_high_idx, last_high, "high"))
                last_low, last_low_idx = lo, i
                direction = -1

        if direction <= 0:
            if lo < last_low:
                last_low, last_low_idx = lo, i
            if h > last_low * (1 + pct):
                if direction != 0 or last_low < lows[0] * (1 - pct / 2):
                    points.append((last_low_idx, last_low, "low"))
                last_high, last_high_idx = h, i
                direction = 1

    return points


def _wave_trend(points: list[tuple[int, float, str]]) -> str:
    """Determine wave trend from zigzag points: 'bullish', 'bearish', 'neutral'."""
    if len(points) < 4:
        return "neutral"

    last = points[-6:] if len(points) >= 6 else points[-4:]
    highs = [p for p in last if p[2] == "high"]
    lows = [p for p in last if p[2] == "low"]

    if len(highs) >= 2 and len(lows) >= 2:
        hh = all(highs[i][1] < highs[i + 1][1] for i in range(len(highs) - 1))
        hl = all(lows[i][1] < lows[i + 1][1] for i in range(len(lows) - 1))
        lh = all(highs[i][1] > highs[i + 1][1] for i in range(len(highs) - 1))
        ll = all(lows[i][1] > lows[i + 1][1] for i in range(len(lows) - 1))

        if hh and hl:
            return "bullish"
        if lh and ll:
            return "bearish"

    if len(points) >= 2:
        p_last, p_prev = points[-1], points[-2]
        if p_last[2] == "high" and p_last[1] > p_prev[1]:
            return "bullish"
        if p_last[2] == "low" and p_last[1] < p_prev[1]:
            return "bearish"

    return "neutral"


def _detect_order_blocks(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    atr: np.ndarray, min_impulse: int = 3, min_impulse_atr: float = 1.5,
) -> list[dict]:
    """Detect bullish and bearish order blocks."""
    obs: list[dict] = []
    n = len(closes)

    for i in range(min_impulse + 1, n - 1):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        # Bullish OB: red candle followed by 3+ green impulse
        if closes[i] < opens[i]:
            ok = True
            move = 0.0
            for j in range(1, min_impulse + 1):
                if i + j >= n:
                    ok = False
                    break
                if closes[i + j] <= opens[i + j]:
                    ok = False
                    break
                move += closes[i + j] - opens[i + j]
            if ok and move / atr[i] >= min_impulse_atr:
                obs.append({"type": "bullish", "high": highs[i], "low": lows[i],
                           "idx": i, "strength": move / atr[i]})

        # Bearish OB: green candle followed by 3+ red impulse
        if closes[i] > opens[i]:
            ok = True
            move = 0.0
            for j in range(1, min_impulse + 1):
                if i + j >= n:
                    ok = False
                    break
                if closes[i + j] >= opens[i + j]:
                    ok = False
                    break
                move += opens[i + j] - closes[i + j]
            if ok and move / atr[i] >= min_impulse_atr:
                obs.append({"type": "bearish", "high": highs[i], "low": lows[i],
                           "idx": i, "strength": move / atr[i]})

    # Deduplicate within 0.5%
    filtered: list[dict] = []
    for ob in obs:
        mid = (ob["high"] + ob["low"]) / 2
        dup = False
        for ex in filtered:
            ex_mid = (ex["high"] + ex["low"]) / 2
            if abs(mid - ex_mid) / (ex_mid + 1e-10) < 0.005:
                if ob["strength"] > ex["strength"]:
                    filtered.remove(ex)
                    filtered.append(ob)
                dup = True
                break
        if not dup:
            filtered.append(ob)

    return filtered


def _compute_htf_features(df_15m: pd.DataFrame, window: int = 120, zz_pct: float = 0.015):
    """Compute EW trend + OB zones on 15m data, return per-timestamp lookup."""
    tr = pd.concat([
        df_15m["high"] - df_15m["low"],
        (df_15m["high"] - df_15m["close"].shift()).abs(),
        (df_15m["low"] - df_15m["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_arr = tr.rolling(14).mean().values
    opens = df_15m["open"].values
    highs = df_15m["high"].values
    lows = df_15m["low"].values
    closes = df_15m["close"].values
    dates = df_15m["date"].values

    results = []
    step = 4  # update every 4 bars (1 hour) for speed

    for end_idx in range(window, len(df_15m), step):
        s = end_idx - window
        chunk_h = highs[s:end_idx]
        chunk_l = lows[s:end_idx]
        chunk_o = opens[s:end_idx]
        chunk_c = closes[s:end_idx]
        chunk_atr = atr_arr[s:end_idx]
        price = closes[end_idx - 1]

        points = _compute_zigzag(chunk_h, chunk_l, zz_pct)
        trend = _wave_trend(points)
        obs = _detect_order_blocks(chunk_o, chunk_h, chunk_l, chunk_c, chunk_atr)

        bull_obs = sorted([o for o in obs if o["type"] == "bullish" and price > o["low"]],
                         key=lambda x: x["strength"], reverse=True)[:3]
        bear_obs = sorted([o for o in obs if o["type"] == "bearish" and price < o["high"]],
                         key=lambda x: x["strength"], reverse=True)[:3]

        results.append({
            "date": dates[end_idx - 1],
            "ew_trend": trend,
            "bull_ob_highs": [o["high"] for o in bull_obs],
            "bull_ob_lows": [o["low"] for o in bull_obs],
            "bear_ob_highs": [o["high"] for o in bear_obs],
            "bear_ob_lows": [o["low"] for o in bear_obs],
        })

    return pd.DataFrame(results)


@register_strategy
class EWOrderBlockStrategy(BaseStrategy):
    """Elliott Wave trend (15m) + Order Block entry on LTF."""

    name = "ew_ob"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self._htf_features: pd.DataFrame | None = None
        self._htf_computed_for: str = ""

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata.get("pair", "")
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"].astype(float)

        # LTF ATR
        tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
        dataframe["eo_atr"] = tr.rolling(14).mean()

        # Candle properties
        dataframe["eo_is_green"] = c > dataframe["open"]
        dataframe["eo_is_red"] = c < dataframe["open"]

        # Volume ratio
        dataframe["eo_vol_sma"] = v.rolling(20).mean()
        dataframe["eo_vol_ratio"] = v / (dataframe["eo_vol_sma"] + 1e-10)

        # Fetch 15m data for EW + OB computation
        htf_df = None
        if self._dp is not None:
            htf_df = self._dp.get_pair_dataframe(pair=pair, timeframe="15m")

        if htf_df is not None and len(htf_df) >= 120:
            pair_key = f"{pair}_{len(htf_df)}"
            if self._htf_computed_for != pair_key:
                self._htf_features = _compute_htf_features(htf_df, window=120, zz_pct=0.015)
                self._htf_computed_for = pair_key

        if self._htf_features is not None and len(self._htf_features) > 0:
            htf = self._htf_features.copy()
            htf["date"] = pd.to_datetime(htf["date"], utc=True)
            dataframe["date_merge"] = pd.to_datetime(dataframe["date"], utc=True)

            # Ensure matching datetime resolution
            htf["date_merge"] = htf["date"].dt.as_unit("ns")
            dataframe["date_merge"] = dataframe["date_merge"].dt.as_unit("ns")

            merged = pd.merge_asof(
                dataframe[["date_merge"]].reset_index(),
                htf.rename(columns={"date": "date_orig"}).sort_values("date_merge"),
                on="date_merge", direction="backward",
            ).set_index("index")

            dataframe["eo_ew_trend"] = merged["ew_trend"].values
            for col in ["bull_ob_highs", "bull_ob_lows", "bear_ob_highs", "bear_ob_lows"]:
                dataframe[f"eo_{col}"] = merged[col].values
        else:
            dataframe["eo_ew_trend"] = "neutral"
            for col in ["bull_ob_highs", "bull_ob_lows", "bear_ob_highs", "bear_ob_lows"]:
                dataframe[f"eo_{col}"] = None

        # Dedup signal tracking
        dedup = self.config.entry.get("dedup_bars", 15)
        dataframe["eo_dedup_block"] = False

        return dataframe

    def _check_ob_touch(self, row, side: str, touch_pct: float) -> bool:
        """Check if price touches an OB zone."""
        price = row.get("close", 0)
        low = row.get("low", 0)
        high = row.get("high", 0)

        if side == "long":
            ob_highs = row.get("eo_bull_ob_highs")
            ob_lows = row.get("eo_bull_ob_lows")
            if not isinstance(ob_highs, list) or not ob_highs:
                return False
            for oh, ol in zip(ob_highs, ob_lows):
                ob_top = oh * (1 + touch_pct)
                if low <= ob_top and price >= ol * (1 - touch_pct):
                    return True
        else:
            ob_highs = row.get("eo_bear_ob_highs")
            ob_lows = row.get("eo_bear_ob_lows")
            if not isinstance(ob_highs, list) or not ob_highs:
                return False
            for oh, ol in zip(ob_highs, ob_lows):
                ob_bot = ol * (1 - touch_pct)
                if high >= ob_bot and price <= oh * (1 + touch_pct):
                    return True
        return False

    def on_tick(self, dataframe: pd.DataFrame, pair: str, current_time: datetime) -> None:
        for sig in self.detect_entries(dataframe, pair):
            self.emit_signal(
                pair=sig.pair, direction=sig.direction, strength=sig.strength,
                tag=sig.tag, metadata=sig.metadata,
            )

    def detect_entries(self, dataframe: pd.DataFrame, pair: str) -> list[Signal]:
        signals: list[Signal] = []
        if len(dataframe) < self.startup_candle_count:
            return signals

        last = dataframe.iloc[-1]
        entry_cfg = self.config.get_entry(pair)
        touch_pct = entry_cfg.get("ob_touch_pct", 0.003)

        trend = last.get("eo_ew_trend", "neutral")
        if not isinstance(trend, str) or trend == "neutral":
            return signals

        # LONG: EW bullish + price touches bullish OB + green candle
        if trend == "bullish" and bool(last.get("eo_is_green", False)):
            if self._check_ob_touch(last, "long", touch_pct):
                signals.append(Signal(
                    strategy_name=self.name, pair=pair,
                    direction=Direction.LONG, strength=1.0,
                    tag=f"{self.name}_long",
                    timestamp=datetime.utcnow(),
                    metadata={"ew_trend": trend},
                ))

        # SHORT: EW bearish + price touches bearish OB + red candle
        if trend == "bearish" and bool(last.get("eo_is_red", False)):
            if self._check_ob_touch(last, "short", touch_pct):
                signals.append(Signal(
                    strategy_name=self.name, pair=pair,
                    direction=Direction.SHORT, strength=1.0,
                    tag=f"{self.name}_short",
                    timestamp=datetime.utcnow(),
                    metadata={"ew_trend": trend},
                ))

        return signals

    def populate_entry_columns(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        startup = self.startup_candle_count
        entry_cfg = self.config.get_entry(pair)
        touch_pct = entry_cfg.get("ob_touch_pct", 0.003)
        dedup = entry_cfg.get("dedup_bars", 15)

        trend = dataframe["eo_ew_trend"]
        is_green = dataframe["eo_is_green"]
        is_red = dataframe["eo_is_red"]
        close = dataframe["close"]
        low = dataframe["low"]
        high = dataframe["high"]

        enter_long = pd.Series(False, index=dataframe.index)
        enter_short = pd.Series(False, index=dataframe.index)

        last_entry_idx = -999

        for i in range(startup, len(dataframe)):
            if i - last_entry_idx < dedup:
                continue

            t = trend.iloc[i]
            if not isinstance(t, str) or t == "neutral":
                continue

            row = dataframe.iloc[i]

            if t == "bullish" and bool(is_green.iloc[i]):
                if self._check_ob_touch(row, "long", touch_pct):
                    enter_long.iloc[i] = True
                    last_entry_idx = i

            elif t == "bearish" and bool(is_red.iloc[i]):
                if self._check_ob_touch(row, "short", touch_pct):
                    enter_short.iloc[i] = True
                    last_entry_idx = i

        dataframe.loc[enter_long, "enter_long"] = 1
        dataframe.loc[enter_long, "enter_tag"] = f"{self.name}_long"
        dataframe.loc[enter_short, "enter_short"] = 1
        dataframe.loc[enter_short, "enter_tag"] = f"{self.name}_short"

        return dataframe

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
        is_short = trade_info.get("is_short", False)
        leverage = float(trade_info.get("leverage", 1.0))

        atr = float(last.get("eo_atr", 0))
        if atr <= 0 or open_rate <= 0:
            return None

        exit_cfg = self.config.get_exit(pair)
        tp_atr = exit_cfg.get("price_tp_atr", 3.0)
        sl_atr = exit_cfg.get("price_sl_atr", 3.0)

        # TP: close-based (current_profit already reflects close price)
        tp_pct = tp_atr * atr / open_rate * leverage
        if current_profit >= tp_pct:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="TP_HIT", urgency=Urgency.IMMEDIATE,
                timestamp=datetime.utcnow(),
            )

        # SL: high/low-based (catches intra-bar moves)
        bar_high = float(last.get("high", 0))
        bar_low = float(last.get("low", 0))

        if is_short:
            sl_price = open_rate + sl_atr * atr
            hit_sl = bar_high >= sl_price
        else:
            sl_price = open_rate - sl_atr * atr
            hit_sl = bar_low <= sl_price

        if hit_sl:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="price_sl", urgency=Urgency.IMMEDIATE,
                timestamp=datetime.utcnow(),
            )

        hours = (current_time - entry_time).total_seconds() / 3600
        if hours >= 8:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_8h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

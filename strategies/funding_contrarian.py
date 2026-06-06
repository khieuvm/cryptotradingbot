"""Funding Contrarian strategy: fade extreme funding rates.

CONCEPT: When funding rate is extremely positive (>0.05%/8h), the market is
heavily long-biased (crowded trade). This creates mean-reversion pressure.
Conversely, extreme negative funding = crowded shorts → fade with longs.

This is a crypto-native structural edge that doesn't exist in traditional markets.
Settlement times (00:00, 08:00, 16:00 UTC) create predictable pressure points.

Entry: Extreme funding + RSI divergence + price at BB extreme
Exit: Funding normalizes, BB-mid target, time cuts
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pandas_ta as ta

from engine.config import StrategyConfig
from engine.events import Direction, ExitRequest, Signal, Urgency
from strategies import register_strategy
from strategies.base import BaseStrategy


@register_strategy
class FundingContrarianStrategy(BaseStrategy):
    name = "funding_contrarian"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"]

        # ATR
        dataframe["fc_atr"] = ta.atr(h, lo, c, length=14)

        # RSI
        dataframe["fc_rsi"] = ta.rsi(c, length=14)

        # Bollinger Bands for price extreme detection
        bb = ta.bbands(c, length=20, std=2.0)
        if bb is not None:
            dataframe["fc_bb_upper"] = bb.iloc[:, 2]
            dataframe["fc_bb_mid"] = bb.iloc[:, 1]
            dataframe["fc_bb_lower"] = bb.iloc[:, 0]
            dataframe["fc_bb_pct"] = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0] + 1e-10)
        else:
            dataframe["fc_bb_upper"] = c * 1.02
            dataframe["fc_bb_mid"] = c
            dataframe["fc_bb_lower"] = c * 0.98
            dataframe["fc_bb_pct"] = 0.5

        # Volume
        dataframe["fc_vol_ema"] = ta.ema(v.astype(float), length=20)
        dataframe["fc_vol_ratio"] = v.astype(float) / (dataframe["fc_vol_ema"] + 1e-10)

        # Funding rate analysis (from market_data indicator)
        # Assuming funding_rate column exists from orchestrator populate_indicators
        if "funding_rate" not in dataframe.columns:
            dataframe["funding_rate"] = 0.0

        funding_extreme_pos = self.config.entry.get("funding_extreme_pos", 0.0005)
        funding_extreme_neg = self.config.entry.get("funding_extreme_neg", -0.0005)

        dataframe["fc_funding_extreme_long"] = dataframe["funding_rate"] < funding_extreme_neg
        dataframe["fc_funding_extreme_short"] = dataframe["funding_rate"] > funding_extreme_pos

        # Funding rate momentum (is it getting more extreme or normalizing?)
        dataframe["fc_funding_ma"] = dataframe["funding_rate"].rolling(8).mean()
        dataframe["fc_funding_increasing"] = dataframe["funding_rate"] > dataframe["fc_funding_ma"]

        return dataframe

    def on_tick(self, dataframe: pd.DataFrame, pair: str, current_time: datetime) -> None:
        signals = self.detect_entries(dataframe, pair)
        for sig in signals:
            self.emit_signal(
                pair=sig.pair,
                direction=sig.direction,
                strength=sig.strength,
                tag=sig.tag,
                metadata=sig.metadata,
            )

    def detect_entries(self, dataframe: pd.DataFrame, pair: str) -> list[Signal]:
        signals: list[Signal] = []
        if len(dataframe) < 20:
            return signals

        last = dataframe.iloc[-1]
        prev = dataframe.iloc[-2]

        rsi = float(last.get("fc_rsi", 50))
        bb_pct = float(last.get("fc_bb_pct", 0.5))
        vol_ratio = float(last.get("fc_vol_ratio", 0))
        funding = float(last.get("funding_rate", 0))

        vol_min = self.config.entry.get("vol_min", 0.5)
        if vol_ratio < vol_min:
            return signals

        rsi_ob = self.config.entry.get("rsi_ob_thr", 65)
        rsi_os = self.config.entry.get("rsi_os_thr", 35)

        # LONG: Extreme negative funding (shorts crowded) + RSI oversold + price near BB lower
        if (
            bool(last.get("fc_funding_extreme_long", False))
            and rsi < rsi_os
            and bb_pct < 0.2
        ):
            strength = min(1.0, abs(funding) / 0.001)
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=strength,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
                metadata={"funding_rate": funding, "rsi": rsi},
            ))

        # SHORT: Extreme positive funding (longs crowded) + RSI overbought + price near BB upper
        if (
            bool(last.get("fc_funding_extreme_short", False))
            and rsi > rsi_ob
            and bb_pct > 0.8
        ):
            strength = min(1.0, abs(funding) / 0.001)
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=strength,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
                metadata={"funding_rate": funding, "rsi": rsi},
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
        is_short = trade_info.get("is_short", False)

        atr = float(last.get("fc_atr", 0))
        if atr <= 0 or open_rate <= 0:
            return None

        # ATR-based TP
        tp_pct = self.tp_atr_mult * atr / open_rate
        if current_profit >= tp_pct:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="TP_HIT", urgency=Urgency.IMMEDIATE,
                timestamp=datetime.utcnow(),
            )

        # BB-mid target (mean reversion target)
        close = float(last.get("close", 0))
        bb_mid = float(last.get("fc_bb_mid", 0))
        if not is_short and close > bb_mid and current_profit > 0:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="bb_mid_target", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if is_short and close < bb_mid and current_profit > 0:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="bb_mid_target", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        # Funding normalizes → edge gone
        funding = float(last.get("funding_rate", 0))
        if not is_short and funding > 0:
            if current_profit > 0.002:
                return ExitRequest(
                    strategy_name=self.name, pair=pair,
                    reason="funding_normalized", urgency=Urgency.NEXT_CANDLE,
                    timestamp=datetime.utcnow(),
                )
        if is_short and funding < 0:
            if current_profit > 0.002:
                return ExitRequest(
                    strategy_name=self.name, pair=pair,
                    reason="funding_normalized", urgency=Urgency.NEXT_CANDLE,
                    timestamp=datetime.utcnow(),
                )

        # Time cuts (funding plays are medium-term, 8-24h)
        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.exit

        if hours >= 8 and current_profit < exit_cfg.get("time_cut_8h", -0.01):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_8h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 24 and current_profit < exit_cfg.get("time_cut_24h", 0.0):
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_24h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 48:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_48h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

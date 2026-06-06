"""Mean-reversion confluence strategy: pullback scalper in trend direction.

Entry: RSI extreme + BB touch + volume spike + EMA trend alignment
Exit: BB-mid reversion, ATR-based TP/SL, time cuts (3h/6h/12h)
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
class MeanRevConfluenceStrategy(BaseStrategy):
    name = "meanrev_confluence"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def compute_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"]

        ema_fast_len = self.config.entry.get("ema_fast", 20)
        ema_slow_len = self.config.entry.get("ema_slow", 50)
        ema_macro = self.config.entry.get("ema_macro", 100)
        bb_period = self.config.entry.get("bb_period", 20)
        bb_std = self.config.entry.get("bb_std", 2.0)

        dataframe["mr_atr"] = ta.atr(h, lo, c, length=14)
        dataframe["mr_rsi"] = ta.rsi(c, length=14)

        bb = ta.bbands(c, length=bb_period, std=bb_std)
        if bb is not None:
            dataframe["mr_bb_upper"] = bb.iloc[:, 2]
            dataframe["mr_bb_mid"] = bb.iloc[:, 1]
            dataframe["mr_bb_lower"] = bb.iloc[:, 0]
        else:
            dataframe["mr_bb_upper"] = c
            dataframe["mr_bb_mid"] = c
            dataframe["mr_bb_lower"] = c

        dataframe["mr_ema_fast"] = ta.ema(c, length=ema_fast_len)
        dataframe["mr_ema_slow"] = ta.ema(c, length=ema_slow_len)
        dataframe["mr_ema_macro"] = ta.ema(c, length=ema_macro)

        dataframe["mr_vol_ema"] = ta.ema(v.astype(float), length=20)
        dataframe["mr_vol_ratio"] = v.astype(float) / (dataframe["mr_vol_ema"] + 1e-10)

        return dataframe

    def on_tick(self, dataframe: pd.DataFrame, pair: str, current_time: datetime) -> None:
        signals = self.detect_entries(dataframe, pair)
        for sig in signals:
            self.emit_signal(
                pair=sig.pair,
                direction=sig.direction,
                strength=sig.strength,
                tag=sig.tag,
            )

    def detect_entries(self, dataframe: pd.DataFrame, pair: str) -> list[Signal]:
        signals: list[Signal] = []
        last = dataframe.iloc[-1]

        rsi_buy = self.config.entry.get("rsi_buy", 30)
        rsi_sell = self.config.entry.get("rsi_sell", 70)
        vol_mult = self.config.entry.get("vol_mult", 1.2)

        vol_ratio = float(last.get("mr_vol_ratio", 0))
        if vol_ratio < vol_mult:
            return signals

        rsi = float(last.get("mr_rsi", 50))
        ema_fast = float(last.get("mr_ema_fast", 0))
        ema_slow = float(last.get("mr_ema_slow", 0))
        ema_macro = float(last.get("mr_ema_macro", 0))
        close = float(last.get("close", 0))
        low = float(last.get("low", 0))
        high = float(last.get("high", 0))
        bb_lower = float(last.get("mr_bb_lower", 0))
        bb_upper = float(last.get("mr_bb_upper", 0))

        # Long: trend up + RSI oversold + BB touch
        if (
            ema_fast > ema_slow
            and ema_slow > ema_macro
            and rsi <= rsi_buy
            and low <= bb_lower
        ):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.LONG, strength=1.0,
                tag=f"{self.name}_long",
                timestamp=datetime.utcnow(),
            ))

        # Short: trend down + RSI overbought + BB touch
        if (
            ema_fast < ema_slow
            and ema_slow < ema_macro
            and rsi >= rsi_sell
            and high >= bb_upper
        ):
            signals.append(Signal(
                strategy_name=self.name, pair=pair,
                direction=Direction.SHORT, strength=1.0,
                tag=f"{self.name}_short",
                timestamp=datetime.utcnow(),
            ))

        return signals

    def detect_exits(
        self, dataframe: pd.DataFrame, pair: str, trade_info: dict | None
    ) -> ExitRequest | None:
        if trade_info is None:
            return None

        last = dataframe.iloc[-1]
        current_profit = trade_info.get("current_profit", 0)
        is_short = trade_info.get("is_short", False)
        current_time = trade_info.get("current_time", datetime.utcnow())
        entry_time = trade_info.get("entry_time", current_time)
        open_rate = trade_info.get("entry_rate", 0)

        # ATR-based TP
        atr = float(last.get("mr_atr", 0))
        if atr > 0 and open_rate > 0:
            tp_pct = self.tp_atr_mult * atr / open_rate
            if current_profit >= tp_pct:
                return ExitRequest(
                    strategy_name=self.name, pair=pair,
                    reason="TP_HIT", urgency=Urgency.IMMEDIATE,
                    timestamp=datetime.utcnow(),
                )

        # BB-mid reversion exit
        rsi = float(last.get("mr_rsi", 50))
        close = float(last.get("close", 0))
        bb_mid = float(last.get("mr_bb_mid", 0))

        is_long = not is_short
        if is_long and rsi > 50 and close > bb_mid:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="bb_mid_exit", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if not is_long and rsi < 50 and close < bb_mid:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="bb_mid_exit", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        # Cascading time cuts
        hours = (current_time - entry_time).total_seconds() / 3600
        exit_cfg = self.config.exit
        time_cut_3h = exit_cfg.get("time_cut_3h", -0.005)
        time_cut_6h = exit_cfg.get("time_cut_6h", 0.002)
        time_cut_12h = exit_cfg.get("time_cut_12h", 0.0)

        if hours >= 3 and current_profit < time_cut_3h:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_3h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 6 and current_profit < time_cut_6h:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_6h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )
        if hours >= 12:
            return ExitRequest(
                strategy_name=self.name, pair=pair,
                reason="time_cut_12h", urgency=Urgency.NEXT_CANDLE,
                timestamp=datetime.utcnow(),
            )

        return None

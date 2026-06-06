"""CryptoEngine: Freqtrade IStrategy adapter bridging to the event-driven engine.

This is the ONLY IStrategy class in the system. It translates freqtrade's
callback model into the engine's event-driven model.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from freqtrade.strategy import IStrategy

from engine.config import AppConfig
from engine.orchestrator import Orchestrator


class CryptoEngine(IStrategy):
    """Freqtrade IStrategy that delegates all logic to the orchestrator."""

    INTERFACE_VERSION = 3
    timeframe = "15m"
    startup_candle_count = 220
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": 0.99}
    stoploss = -0.15
    use_custom_stoploss = True
    trailing_stop = False

    protections = [
        {"method": "CooldownPeriod", "stop_duration_candles": 5},
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 96,
            "trade_limit": 3,
            "stop_duration_candles": 40,
            "only_per_pair": True,
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 96,
            "max_allowed_drawdown": 0.15,
            "stop_duration_candles": 120,
            "trade_limit": 5,
        },
    ]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        env = config.get("engine_env", "dryrun")
        self._app_config = AppConfig(env=env)
        self._engine = Orchestrator(self._app_config)
        self.startup_candle_count = self._engine.get_max_startup_candles()

    def bot_start(self, **kwargs) -> None:
        self._engine.initialize(dp=self.dp)

    def informative_pairs(self):
        return self._engine.get_informative_pairs()

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        return self._engine.populate_indicators(dataframe, metadata)

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        pair = metadata["pair"]
        signals = self._engine.get_pending_entries(pair, dataframe)
        for signal in signals:
            if not hasattr(signal, "bar_index"):
                idx = dataframe.index[-1]
            else:
                idx = signal.bar_index

            if signal.direction.value == "long":
                dataframe.loc[idx, "enter_long"] = 1
                dataframe.loc[idx, "enter_tag"] = signal.tag
            else:
                dataframe.loc[idx, "enter_short"] = 1
                dataframe.loc[idx, "enter_tag"] = signal.tag
        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        return dataframe

    def custom_entry_price(
        self, pair, trade, current_time, proposed_rate, entry_tag, side, **kwargs
    ) -> float:
        if not entry_tag:
            return proposed_rate
        strategy_name = entry_tag.rsplit("_", 1)[0]
        strat = self._engine._strategies.get(strategy_name)
        if strat is None or strat.entry_atr_fraction == 0:
            return proposed_rate

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return proposed_rate

        atr = self._engine._get_atr(df.iloc[-1], strategy_name)
        if atr <= 0:
            return proposed_rate

        offset = strat.entry_atr_fraction * atr
        return proposed_rate - offset if side == "long" else proposed_rate + offset

    def custom_stake_amount(
        self, pair, current_time, current_rate, proposed_stake,
        min_stake, max_stake, leverage, entry_tag, side, **kwargs
    ) -> float:
        if not entry_tag:
            return proposed_stake
        return self._engine.calculate_stake(
            pair, entry_tag, side, proposed_stake, min_stake, max_stake
        )

    def custom_stoploss(
        self, pair, trade, current_time, current_rate, current_profit, **kwargs
    ) -> float:
        return self._engine.get_stoploss(pair, trade, current_rate, current_profit)

    def custom_exit(
        self, pair, trade, current_time, current_rate, current_profit, **kwargs
    ) -> str | None:
        return self._engine.check_exit(
            pair, trade, current_time, current_rate, current_profit
        )

    def confirm_trade_entry(
        self, pair, order_type, amount, rate, time_in_force,
        current_time, entry_tag, side, **kwargs
    ) -> bool:
        return self._engine.confirm_entry(pair, entry_tag or "", side, rate, current_time)

    def confirm_trade_exit(
        self, pair, trade, order_type, amount, rate,
        time_in_force, exit_reason, current_time, **kwargs
    ) -> bool:
        return self._engine.confirm_exit(pair, trade, rate, exit_reason)

    def leverage(
        self, pair, current_time, current_rate, proposed_leverage,
        max_leverage, entry_tag, side, **kwargs
    ) -> float:
        return self._engine.get_leverage(pair, entry_tag or "")

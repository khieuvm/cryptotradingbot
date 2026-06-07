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

        if "active_strategies" in config:
            self._app_config._active_override = config["active_strategies"]
        if "timeframe" in config:
            self.timeframe = config["timeframe"]
            self._app_config._merged.setdefault("market", {})["timeframe"] = config["timeframe"]

        self._engine = Orchestrator(self._app_config)
        active = self._app_config.get_active_strategies()
        max_startup = max(
            (self._app_config.get_strategy_config(name).startup_candle_count
             for name in active),
            default=220,
        )
        self.startup_candle_count = max_startup

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
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        for strat in self._engine._pair_strategies.get(pair, []):
            if strat.grade not in ("A", "B"):
                continue
            if not self._engine.state.is_strategy_active(strat.name):
                continue
            dataframe = strat.populate_entry_columns(dataframe, pair)

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
        strategy_name = self._engine._parse_strategy_name(entry_tag)
        strat = self._engine._strategies.get(strategy_name)
        if strat is None:
            return proposed_rate

        entry_opt = strat.get_entry_optimization(pair)
        method = entry_opt.get("method", "market")
        if method == "market":
            return proposed_rate

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return proposed_rate

        atr = self._engine._get_atr(df.iloc[-1], strategy_name)
        if atr <= 0:
            return proposed_rate

        if method == "limit_atr":
            offset = entry_opt.get("atr_offset", 0.1) * atr
            return proposed_rate - offset if side == "long" else proposed_rate + offset
        elif method == "ema8_retest":
            ema8_col = next(
                (c for c in df.columns if "ema" in c.lower() and "8" in c),
                None,
            )
            if ema8_col and not df[ema8_col].empty:
                ema8_val = float(df.iloc[-1][ema8_col])
                if ema8_val > 0:
                    return ema8_val
            return proposed_rate - 0.3 * atr if side == "long" else proposed_rate + 0.3 * atr

        return proposed_rate

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
        return self._engine.get_stoploss(
            pair, trade, current_rate, current_profit, current_time
        )

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

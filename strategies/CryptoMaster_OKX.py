"""
CryptoMaster_OKX — Unified Strategy Dispatcher
================================================
Single IStrategy that delegates to the combo registry.
All entry/exit logic lives in combos/; this file handles:
- Indicator population (calls each active combo)
- Signal merging (enter_tag identifies which combo fired)
- Custom exit (dispatches by enter_tag to combo-specific logic)
- Custom stoploss (ATR-based, combo-aware)
- Position sizing (combo/regime-aware)
- Entry confirmation (regime filter, signal tracker, BTC sentiment, funding)
"""

from datetime import datetime

import pandas as pd
import pandas_ta as ta

from freqtrade.strategy import IStrategy, DecimalParameter, merge_informative_pair
from freqtrade.constants import CandleType

from combos import get_active_combos, get_combo
from src.strategy_config import get_combo_config
from src.signal_tracker import SignalTracker


class CryptoMaster_OKX(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"

    startup_candle_count = 220
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": 0.99}
    stoploss = -0.10
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

    ENTRY_ATR_FRACTION = DecimalParameter(0.0, 0.3, default=0.15, decimals=2, space="buy", optimize=True)

    _signal_tracker = SignalTracker()

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        inf = [("BTC/USDT:USDT", "1h")]
        for pair in pairs:
            inf.append((pair, "1h", CandleType.FUNDING_RATE))
        return inf

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = self._add_btc_sentiment(dataframe)
        dataframe = self._add_funding_rate(dataframe, metadata)

        for combo in get_active_combos(metadata["pair"]):
            dataframe = combo.populate_indicators(dataframe, metadata)

        return dataframe

    def _add_btc_sentiment(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self.dp is None:
            dataframe["btc_rsi_1h"] = 50.0
            return dataframe
        btc_df = self.dp.get_pair_dataframe(pair="BTC/USDT:USDT", timeframe="1h")
        if btc_df.empty:
            dataframe["btc_rsi_1h"] = 50.0
            return dataframe
        btc_df = btc_df.copy()
        btc_df["btc_rsi"] = ta.rsi(btc_df["close"], length=14)
        btc_df = btc_df[["date", "btc_rsi"]].copy()
        dataframe = merge_informative_pair(dataframe, btc_df, self.timeframe, "1h", ffill=True)
        if "btc_rsi_1h" not in dataframe.columns:
            dataframe["btc_rsi_1h"] = 50.0
        return dataframe

    def _add_funding_rate(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if self.dp is None:
            dataframe["funding_rate"] = 0.0
            return dataframe
        try:
            fr_df = self.dp.get_pair_dataframe(
                pair=metadata["pair"], timeframe="1h", candle_type=CandleType.FUNDING_RATE
            )
            if fr_df.empty:
                dataframe["funding_rate"] = 0.0
                return dataframe
            fr_df = fr_df[["date", "open"]].copy().rename(columns={"open": "fr"})
            dataframe = merge_informative_pair(dataframe, fr_df, self.timeframe, "1h", ffill=True)
            dataframe["funding_rate"] = dataframe.get("fr_1h", 0.0)
        except Exception:
            dataframe["funding_rate"] = 0.0
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        for combo in get_active_combos(metadata["pair"]):
            if combo.grade not in ("A", "B"):
                continue

            long_mask = combo.detect_long(dataframe, metadata)
            short_mask = combo.detect_short(dataframe, metadata)

            dataframe.loc[long_mask, ["enter_long", "enter_tag"]] = (1, f"{combo.name}_long")
            dataframe.loc[short_mask, ["enter_short", "enter_tag"]] = (1, f"{combo.name}_short")

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    def custom_entry_price(self, pair, trade, current_time, proposed_rate,
                           entry_tag, side, **kwargs) -> float:
        if self.ENTRY_ATR_FRACTION.value == 0:
            return proposed_rate
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return proposed_rate
        atr = float(df.iloc[-1].get("ra_atr", 0) or df.iloc[-1].get("tc_atr", 0) or df.iloc[-1].get("mr_atr", 0))
        if atr <= 0:
            return proposed_rate
        offset = self.ENTRY_ATR_FRACTION.value * atr
        return proposed_rate - offset if side == "long" else proposed_rate + offset

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float:
        if not entry_tag:
            return proposed_stake

        combo_name = entry_tag.rsplit("_", 1)[0]
        cfg = get_combo_config(combo_name)
        stake_cfg = cfg.get("stake", {})

        if not stake_cfg:
            return proposed_stake

        if "trend" in entry_tag and "_long" in entry_tag:
            factor = stake_cfg.get("trend_long", 1.0)
        elif "trend" in entry_tag and "_short" in entry_tag:
            factor = stake_cfg.get("trend_short", 1.0)
        elif "range" in entry_tag and "_long" in entry_tag:
            factor = stake_cfg.get("range_long", 1.0)
        elif "range" in entry_tag and "_short" in entry_tag:
            factor = stake_cfg.get("range_short", 1.0)
        else:
            factor = 1.0

        stake = proposed_stake * factor
        if min_stake is not None:
            stake = max(stake, float(min_stake))
        return min(stake, max_stake)

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return self.stoploss

        enter_tag = trade.enter_tag or ""
        combo_name = enter_tag.rsplit("_", 1)[0]
        cfg = get_combo_config(combo_name)
        exit_cfg = cfg.get("exit", {})

        sl_mult = exit_cfg.get("sl_atr_mult", 2.0)
        tp_mult = exit_cfg.get("tp_atr_mult", 3.0)

        last = df.iloc[-1]
        atr = float(last.get("ra_atr", 0) or last.get("tc_atr", 0) or last.get("mr_atr", 0))
        if atr <= 0 or trade.open_rate <= 0:
            return self.stoploss

        sl_pct = sl_mult * atr / trade.open_rate
        tp_pct = tp_mult * atr / trade.open_rate
        fee = trade.fee_open + trade.fee_close

        # 3-phase: initial → break-even → trail-lock
        if current_profit >= tp_pct:
            if not trade.is_short:
                lock_price = trade.open_rate * (1 + tp_pct * 0.60)
                return max((lock_price / current_rate) - 1, -0.005)
            else:
                lock_price = trade.open_rate * (1 - tp_pct * 0.60)
                return max(1 - (lock_price / current_rate), -0.005)

        if current_profit >= tp_pct * 0.5:
            if not trade.is_short:
                be_price = trade.open_rate * (1 + fee)
                return max((be_price / current_rate) - 1, -sl_pct)
            else:
                be_price = trade.open_rate * (1 - fee)
                return max(1 - (be_price / current_rate), -sl_pct)

        return max(-sl_pct, -0.20)

    def custom_exit(self, pair, trade, current_time, current_rate,
                    current_profit, **kwargs) -> str | None:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return None

        enter_tag = trade.enter_tag or ""
        combo_name = enter_tag.rsplit("_", 1)[0]
        cfg = get_combo_config(combo_name)
        exit_cfg = cfg.get("exit", {})

        last = df.iloc[-1]
        atr = float(last.get("ra_atr", 0) or last.get("tc_atr", 0) or last.get("mr_atr", 0))

        # ATR-based TP
        if atr > 0 and trade.open_rate > 0:
            tp_mult = exit_cfg.get("tp_atr_mult", 3.0)
            tp_pct = tp_mult * atr / trade.open_rate
            if current_profit >= tp_pct:
                return "TP_HIT"

        # Combo-specific signal exits
        is_long = not trade.is_short
        if combo_name == "regime_adaptive":
            if "trend" in enter_tag:
                sig = "ra_sig_trend_exit_long" if is_long else "ra_sig_trend_exit_short"
                if int(last.get(sig, 0)) == 1 and current_profit >= 0.0:
                    return "trend_signal_exit"
            elif "range" in enter_tag:
                sig = "ra_sig_range_exit_long" if is_long else "ra_sig_range_exit_short"
                if int(last.get(sig, 0)) == 1:
                    return "range_target_exit"

        elif combo_name == "trend_composite":
            exit_sig = "tc_cross_down" if is_long else "tc_cross_up"
            if last.get(exit_sig, False):
                return "ema_reversal_exit"

        elif combo_name == "meanrev_confluence":
            if is_long and last.get("mr_rsi", 50) > 50 and float(last.get("close", 0)) > float(last.get("mr_bb_mid", 0)):
                return "bb_mid_exit"
            if not is_long and last.get("mr_rsi", 50) < 50 and float(last.get("close", 0)) < float(last.get("mr_bb_mid", 0)):
                return "bb_mid_exit"

        # Cascading time-loss cuts
        hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        time_cut_2h = exit_cfg.get("time_cut_2h", -0.015)
        time_cut_8h = exit_cfg.get("time_cut_8h", -0.008)
        time_cut_3h = exit_cfg.get("time_cut_3h")
        time_cut_6h = exit_cfg.get("time_cut_6h")

        if time_cut_3h is not None:
            if hours >= 3 and current_profit < time_cut_3h:
                return "time_cut_3h"
            if time_cut_6h is not None and hours >= 6 and current_profit < time_cut_6h:
                return "time_cut_6h"
            if hours >= 12:
                return "time_cut_12h"
        else:
            if hours >= 2 and current_profit < time_cut_2h:
                return "time_cut_2h"
            if hours >= 8 and current_profit < time_cut_8h:
                return "time_cut_8h"
            if hours >= 24 and current_profit < 0.0:
                return "time_cut_24h"
            if hours >= 48 and current_profit < 0.005:
                return "time_cut_48h"

        return None

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return True
        last = df.iloc[-1]

        # Signal tracker auto-disable check
        combo_name = (entry_tag or "").rsplit("_", 1)[0]
        if self._signal_tracker.is_disabled(combo_name, pair):
            return False

        # Volume sanity
        if float(last.get("ra_vol_ratio", last.get("mr_vol_ratio", last.get("tc_vol_ratio", 1)))) < 0.4:
            return False

        # ATR spike protection
        atr = float(last.get("ra_atr", 0) or last.get("tc_atr", 0) or last.get("mr_atr", 0))
        close = float(last.get("close", 1))
        if atr > 0 and (atr / close) > 0.04:
            return False

        # BTC sentiment gate
        btc_rsi = float(last.get("btc_rsi_1h", 50))
        if side == "long" and btc_rsi < 35:
            return False
        if side == "short" and btc_rsi > 65:
            return False

        # Funding rate extreme filter
        funding = float(last.get("funding_rate", 0.0))
        if side == "long" and funding > 0.00008:
            return False
        if side == "short" and funding < -0.00007:
            return False

        return True

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate,
                           time_in_force, exit_reason, current_time, **kwargs) -> bool:
        # Record trade outcome for signal tracker
        enter_tag = trade.enter_tag or ""
        combo_name = enter_tag.rsplit("_", 1)[0]
        is_win = (rate - trade.open_rate > 0) if not trade.is_short else (trade.open_rate - rate > 0)
        self._signal_tracker.record_trade(combo_name, pair, is_win)
        return True

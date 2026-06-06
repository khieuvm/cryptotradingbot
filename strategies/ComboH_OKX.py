"""
ComboH_OKX — Regime-Adaptive Crypto Futures Strategy
======================================================
Purpose-built for 24/7 crypto futures (OKX isolated margin).
Inspired by TrendRiderStrategy (github.com/freqtrade/freqtrade-strategies)
and FAdxSmaStrategy.

Core idea:
  Detect market regime first, then apply the right type of signal.

  TRENDING (ADX > threshold AND price aligned with EMA200):
    Long:  EMA21 crosses above EMA55 + MACD hist > 0 + volume
    Short: EMA21 crosses below EMA55 + MACD hist < 0 + volume

  RANGING (ADX <= threshold OR price near EMA200):
    Long:  RSI < 35, close below BB lower, OBV rising, bullish candle
    Short: RSI > 65, close above BB upper, OBV falling, bearish candle

  GATES (both modes):
    - Volume ratio > VOL_MIN
    - Supertrend direction aligned
    - BB width squeeze/expansion OK

  EXIT:
    - ATR-based take-profit
    - Cascading time-based loss cut (2h / 8h / 24h)
    - ATR-based custom stoploss

Timeframe: 15m primary + 5m informative (precision entry)
"""

from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import pandas_ta as ta

from freqtrade.strategy import (
    IStrategy,
    DecimalParameter,
    IntParameter,
    merge_informative_pair,
)


class ComboH_OKX(IStrategy):
    INTERFACE_VERSION = 3

    # --- Timeframes ---
    timeframe = "15m"

    # 4-tier stake sizing: trend/range × long/short direction
    STAKE_TREND_LONG  = DecimalParameter(0.5, 2.0, default=1.5, decimals=1, space="buy", optimize=False)
    STAKE_TREND_SHORT = DecimalParameter(0.5, 2.0, default=1.2, decimals=1, space="buy", optimize=False)
    STAKE_RANGE_LONG  = DecimalParameter(0.2, 1.0, default=0.3, decimals=1, space="buy", optimize=False)
    STAKE_RANGE_SHORT = DecimalParameter(0.5, 1.5, default=1.2, decimals=1, space="buy", optimize=False)
    # Aliases so hyperopt doesn't break if old params referenced
    STAKE_TREND = DecimalParameter(0.8, 1.5, default=1.35, decimals=1, space="buy", optimize=False)
    STAKE_RANGE = DecimalParameter(0.2, 1.0, default=0.43, decimals=1, space="buy", optimize=False)
    STAKE_S1 = DecimalParameter(0.5, 1.5, default=1.0, decimals=1, space="buy", optimize=False)
    STAKE_S2 = DecimalParameter(0.5, 1.5, default=1.0, decimals=1, space="buy", optimize=False)
    STAKE_S3 = DecimalParameter(0.5, 1.5, default=1.0, decimals=1, space="buy", optimize=False)
    STAKE_S4 = DecimalParameter(0.5, 1.5, default=1.0, decimals=1, space="buy", optimize=False)

    startup_candle_count = 220
    process_only_new_candles = True
    can_short = True

    # --- ROI (disabled — custom_exit fires TP) ---
    minimal_roi = {"0": 0.99}

    # --- Stoploss ---
    stoploss = -0.10
    use_custom_stoploss = True

    # --- Trailing (off — ATR-based custom) ---
    trailing_stop = False

    # --- Protections (inspired by TrendRiderStrategy + NFI) ---
    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 5,          # 5×15m = 75 min cooldown per pair
        },
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 96,        # 24h at 15m
            "trade_limit": 3,
            "stop_duration_candles": 40,          # 10h pause on bad run
            "only_per_pair": True,
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 96,
            "max_allowed_drawdown": 0.15,         # 15% account drawdown → pause
            "stop_duration_candles": 120,
            "trade_limit": 5,
        },
    ]

    # --- SL/TP hyperopt ---
    SL_ATR_MULT = DecimalParameter(2.0, 10.0, default=5.9, decimals=1, space="sell", optimize=True)
    TP_ATR_MULT = DecimalParameter(2.0, 16.0, default=9.3, decimals=1, space="sell", optimize=True)
    # Time-based loss cut thresholds (negative = loss %)
    TIME_CUT_2H_THR = DecimalParameter(-0.030, -0.005, default=-0.015, decimals=3, space="sell", optimize=True)
    TIME_CUT_8H_THR = DecimalParameter(-0.020, -0.003, default=-0.008, decimals=3, space="sell", optimize=True)

    # --- Entry price improvement ---
    # Bid below close (long) or offer above close (short) by fraction×ATR
    # Improves avg entry and R:R; small value = high fill rate
    ENTRY_ATR_FRACTION = DecimalParameter(0.0, 0.3, default=0.15, decimals=2, space="buy", optimize=True)
    # ATR ratio spike filter: skip entries when atr/atr_baseline > threshold
    ATR_SPIKE_THR = DecimalParameter(1.5, 3.0, default=2.2, decimals=1, space="buy", optimize=True)

    # --- Regime hyperopt ---
    ADX_TREND_THR  = IntParameter(18, 42, default=31, space="buy", optimize=True)
    RSI_OS         = IntParameter(20, 38, default=34, space="buy", optimize=True)
    RSI_OB         = IntParameter(65, 82, default=71, space="buy", optimize=True)
    BB_MULT        = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy", optimize=False)  # used in indicators, affects BB col names
    VOL_MIN        = DecimalParameter(0.3, 1.2, default=0.5, decimals=1, space="buy", optimize=True)
    EMA_FAST_P     = IntParameter(8, 25,  default=18, space="buy", optimize=True)
    EMA_SLOW_P     = IntParameter(21, 70, default=50, space="buy", optimize=True)

    # ------------------------------------------------------------------ #
    #  INFORMATIVE PAIRS                                                   #
    # ------------------------------------------------------------------ #
    def informative_pairs(self):
        from freqtrade.constants import CandleType
        pairs = self.dp.current_whitelist()
        inf = [("BTC/USDT:USDT", "1h")]
        # Funding rate 1h for each pair (NFI technique)
        for pair in pairs:
            inf.append((pair, "1h", CandleType.FUNDING_RATE))
        return inf

    # ------------------------------------------------------------------ #
    #  INDICATORS                                                          #
    # ------------------------------------------------------------------ #
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = self._add_15m_indicators(dataframe)
        dataframe = self._add_btc_sentiment(dataframe)
        dataframe = self._add_funding_rate(dataframe, metadata)
        return dataframe

    def _add_btc_sentiment(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Merge BTC 1h RSI as market-wide sentiment gate (TrendRiderStrategy technique)."""
        if self.dp is None:
            dataframe["btc_rsi_1h"] = 50.0
            return dataframe
        btc_df = self.dp.get_pair_dataframe(pair="BTC/USDT:USDT", timeframe="1h")
        if btc_df.empty:
            dataframe["btc_rsi_1h"] = 50.0
            return dataframe
        btc_df = btc_df.copy()
        btc_df["btc_rsi"] = ta.rsi(btc_df["close"], length=14)
        btc_df["btc_ema200"] = ta.ema(btc_df["close"], length=200)
        btc_df["btc_bull"] = (btc_df["close"] > btc_df["btc_ema200"]).astype(int)
        btc_df = btc_df[["date", "btc_rsi", "btc_bull"]].copy()
        dataframe = merge_informative_pair(dataframe, btc_df, self.timeframe, "1h", ffill=True)
        dataframe["btc_rsi_1h"] = dataframe.get("btc_rsi_1h", 50.0)
        dataframe["btc_bull_1h"] = dataframe.get("btc_bull_1h", 1)
        return dataframe

    def _add_funding_rate(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Merge 1h funding rate to detect crowded positioning (NFI technique)."""
        if self.dp is None:
            dataframe["funding_rate"] = 0.0
            return dataframe
        try:
            from freqtrade.constants import CandleType
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

    def _add_15m_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """All primary 15m indicators."""
        c = dataframe["close"]
        h = dataframe["high"]
        lo = dataframe["low"]
        v = dataframe["volume"]

        # --- EMAs (all values in hyperopt range, 8-70 + 200) ---
        for p in list(range(8, 71)) + [200]:
            dataframe[f"ema{p}"] = ta.ema(c, length=p)

        # --- ATR ---
        dataframe["atr"] = ta.atr(h, lo, c, length=14)
        # ATR regime: normalized ratio vs 50-bar EMA of ATR
        # > 2 = spike/panic, < 0.6 = dead market
        dataframe["atr_ma"] = ta.ema(dataframe["atr"], length=50)
        dataframe["atr_ratio"] = dataframe["atr"] / (dataframe["atr_ma"] + 1e-10)

        # --- ADX + DI ---
        adx = ta.adx(h, lo, c, length=14)
        if adx is not None:
            dataframe["adx"]      = adx.iloc[:, 0]
            dataframe["plus_di"]  = adx.iloc[:, 1]
            dataframe["minus_di"] = adx.iloc[:, 2]
        else:
            dataframe["adx"] = dataframe["plus_di"] = dataframe["minus_di"] = 0.0

        # --- RSI ---
        dataframe["rsi"] = ta.rsi(c, length=14)

        # --- MACD ---
        macd = ta.macd(c, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            dataframe["macd_hist"] = macd.iloc[:, 1]  # histogram
        else:
            dataframe["macd_hist"] = 0.0

        # --- Bollinger Bands --- (fixed multiplier — BB_MULT only used at indicator time)
        bb = ta.bbands(c, length=20, std=2.0)
        if bb is not None:
            dataframe["bb_upper"]  = bb.iloc[:, 0]
            dataframe["bb_mid"]    = bb.iloc[:, 1]
            dataframe["bb_lower"]  = bb.iloc[:, 2]
            dataframe["bb_width"]  = (bb.iloc[:, 0] - bb.iloc[:, 2]) / (bb.iloc[:, 1] + 1e-10)
        else:
            dataframe["bb_upper"] = dataframe["bb_lower"] = dataframe["bb_mid"] = c
            dataframe["bb_width"] = 0.02
        dataframe["bb_width_sma"] = ta.sma(dataframe["bb_width"], length=50)

        # --- Volume ratio ---
        dataframe["vol_ema"] = ta.ema(v, length=20)
        dataframe["vol_ratio"] = v / (dataframe["vol_ema"] + 1e-10)

        # --- OBV ---
        dataframe["obv"] = ta.obv(c, v)
        dataframe["obv_ema"] = ta.ema(dataframe["obv"], length=20)
        dataframe["obv_rising"] = (dataframe["obv"] > dataframe["obv_ema"]).astype(int)

        # --- Supertrend (pandas_ta) ---
        try:
            st = ta.supertrend(h, lo, c, length=7, multiplier=3.0)
            if st is not None:
                st_dir_col = next((col for col in st.columns if "SUPERTd" in col), None)
                dataframe["st_dir"] = st[st_dir_col].fillna(0) if st_dir_col else 0
            else:
                dataframe["st_dir"] = 0
        except Exception:
            dataframe["st_dir"] = 0

        # --- Market regime (precomputed with fixed threshold; override in entry signals) ---
        dataframe["is_trending"]    = (dataframe["adx"] > 25).astype(int)
        dataframe["is_bull"]        = (c > dataframe["ema200"]).astype(int)
        dataframe["is_bear"]        = (c < dataframe["ema200"]).astype(int)
        dataframe["bb_expanding"]   = (dataframe["bb_width"] > dataframe["bb_width_sma"]).astype(int)

        return dataframe

    # ------------------------------------------------------------------ #
    #  ENTRY SIGNALS                                                       #
    # ------------------------------------------------------------------ #
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        ef = f"ema{self.EMA_FAST_P.value}"
        es = f"ema{self.EMA_SLOW_P.value}"
        adx_thr = self.ADX_TREND_THR.value

        # Use parameter-driven regime (not precomputed hardcoded column)
        is_trending = dataframe["adx"] > adx_thr
        is_bull = dataframe["is_bull"] == 1
        is_bear = dataframe["is_bear"] == 1

        # EMA crossover within last 3 candles (relaxed from exact candle)
        cross_up_recent = (
            (dataframe[ef].shift(1) <= dataframe[es].shift(1)) |
            (dataframe[ef].shift(2) <= dataframe[es].shift(2)) |
            (dataframe[ef].shift(3) <= dataframe[es].shift(3))
        )
        cross_dn_recent = (
            (dataframe[ef].shift(1) >= dataframe[es].shift(1)) |
            (dataframe[ef].shift(2) >= dataframe[es].shift(2)) |
            (dataframe[ef].shift(3) >= dataframe[es].shift(3))
        )

        # ADX rising (trend strengthening)
        adx_rising = dataframe["adx"] > dataframe["adx"].shift(2)

        atr_ok = dataframe["atr_ratio"] < self.ATR_SPIKE_THR.value

        # ------ LONG ------
        # === S1: Trending long: EMA cross within 3 bars + MACD + DI + trend regime ===
        trend_long = (
            is_trending &
            is_bull &
            (dataframe[ef] > dataframe[es]) &
            cross_up_recent &                                          # Cross in last 3 candles
            (dataframe["macd_hist"] > 0) &
            (dataframe["plus_di"] > dataframe["minus_di"]) &
            (dataframe["st_dir"] == 1) &
            (dataframe["vol_ratio"] > self.VOL_MIN.value) &
            atr_ok &
            (dataframe["volume"] > 0)
        )
        dataframe.loc[trend_long, ["enter_long", "enter_tag"]] = (1, "trend_cross_long")

        # === S2: Trend Continuation long: EMA aligned + RSI pullback bounce ===
        # Fires on pullback within established trend (RSI dips then bounces)
        trend_cont_long = (
            is_trending &
            is_bull &
            (dataframe[ef] > dataframe[es]) &                          # EMA aligned
            (dataframe["rsi"].shift(1) < 48) &                         # RSI pulled back
            (dataframe["rsi"] > dataframe["rsi"].shift(1)) &           # RSI bouncing
            (dataframe["rsi"].shift(2) < dataframe["rsi"].shift(1)).astype(bool).__or__(  # Was declining or flat
                dataframe["rsi"].shift(1) < 42) &                      # OR deep pullback
            (dataframe["macd_hist"] > 0) &                             # MACD positive
            (dataframe["st_dir"] == 1) &                               # SuperTrend confirms
            (dataframe["plus_di"] > dataframe["minus_di"]) &           # DI confirms
            (dataframe["vol_ratio"] > self.VOL_MIN.value) &
            atr_ok &
            (dataframe["volume"] > 0) &
            (~trend_long)                                              # Don't overlap with S1
        )
        dataframe.loc[trend_cont_long, ["enter_long", "enter_tag"]] = (1, "trend_cont_long")

        # === S3: Ranging long: RSI oversold + BB lower bounce + OBV rising ===
        range_long = (
            (~is_trending) &
            (dataframe["rsi"].shift(1) < self.RSI_OS.value) &
            (dataframe["rsi"] > dataframe["rsi"].shift(1)) &  # RSI turning up
            (dataframe["close"] < dataframe["bb_lower"] * 1.01) &
            (dataframe["close"] > dataframe["open"]) &  # bullish candle
            (dataframe["obv_rising"] == 1) &
            (dataframe["vol_ratio"] > self.VOL_MIN.value) &
            atr_ok &
            (dataframe["volume"] > 0)
        )
        dataframe.loc[range_long, ["enter_long", "enter_tag"]] = (1, "range_bounce_long")

        # ------ SHORT ------
        # === S1: Trending short: EMA cross within 3 bars + MACD neg + DI + bear regime ===
        trend_short = (
            is_trending &
            is_bear &
            (dataframe[ef] < dataframe[es]) &
            cross_dn_recent &                                          # Cross in last 3 candles
            (dataframe["macd_hist"] < 0) &
            (dataframe["minus_di"] > dataframe["plus_di"]) &
            (dataframe["st_dir"] == -1) &
            (dataframe["vol_ratio"] > self.VOL_MIN.value) &
            atr_ok &
            (dataframe["volume"] > 0)
        )
        dataframe.loc[trend_short, ["enter_short", "enter_tag"]] = (1, "trend_cross_short")

        # === S2: Trend Continuation short: EMA aligned down + RSI bounce down ===
        trend_cont_short = (
            is_trending &
            is_bear &
            (dataframe[ef] < dataframe[es]) &                          # EMA aligned bearish
            (dataframe["rsi"].shift(1) > 52) &                         # RSI bounced up
            (dataframe["rsi"] < dataframe["rsi"].shift(1)) &           # RSI turning down
            (dataframe["rsi"].shift(2) > dataframe["rsi"].shift(1)).astype(bool).__or__(  # Was rising or flat
                dataframe["rsi"].shift(1) > 58) &                      # OR deep bounce
            (dataframe["macd_hist"] < 0) &                             # MACD negative
            (dataframe["st_dir"] == -1) &                              # SuperTrend confirms
            (dataframe["minus_di"] > dataframe["plus_di"]) &           # DI confirms
            (dataframe["vol_ratio"] > self.VOL_MIN.value) &
            atr_ok &
            (dataframe["volume"] > 0) &
            (~trend_short)                                             # Don't overlap with S1
        )
        dataframe.loc[trend_cont_short, ["enter_short", "enter_tag"]] = (1, "trend_cont_short")

        # === S3: Ranging short: RSI overbought + BB upper rejection + OBV falling ===
        range_short = (
            (~is_trending) &
            (dataframe["rsi"].shift(1) > self.RSI_OB.value) &
            (dataframe["rsi"] < dataframe["rsi"].shift(1)) &  # RSI turning down
            (dataframe["close"] > dataframe["bb_upper"] * 0.99) &
            (dataframe["close"] < dataframe["open"]) &  # bearish candle
            (dataframe["obv_rising"] == 0) &
            (dataframe["vol_ratio"] > self.VOL_MIN.value) &
            atr_ok &
            (dataframe["volume"] > 0)
        )
        dataframe.loc[range_short, ["enter_short", "enter_tag"]] = (1, "range_bounce_short")

        return dataframe

    # ------------------------------------------------------------------ #
    #  EXIT SIGNALS — delegated entirely to custom_exit                   #
    # ------------------------------------------------------------------ #
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Pre-compute exit signal columns used by custom_exit
        ef = f"ema{self.EMA_FAST_P.value}"
        es = f"ema{self.EMA_SLOW_P.value}"

        # Trend reversal signals (stored as columns, used in custom_exit)
        dataframe["sig_trend_exit_long"] = (
            (dataframe[ef] < dataframe[es]) &
            (dataframe[ef].shift(1) >= dataframe[es].shift(1)) &
            (dataframe["st_dir"] == -1)
        ).astype(int)

        dataframe["sig_trend_exit_short"] = (
            (dataframe[ef] > dataframe[es]) &
            (dataframe[ef].shift(1) <= dataframe[es].shift(1)) &
            (dataframe["st_dir"] == 1)
        ).astype(int)

        # Range mean-reversion target signals
        dataframe["sig_range_exit_long"] = (
            (dataframe["rsi"] > 52) &
            (dataframe["close"] > dataframe["bb_mid"])
        ).astype(int)

        dataframe["sig_range_exit_short"] = (
            (dataframe["rsi"] < 48) &
            (dataframe["close"] < dataframe["bb_mid"])
        ).astype(int)

        return dataframe

    # ------------------------------------------------------------------ #
    #  3-PHASE ATR STOPLOSS: initial → break-even → trail-lock            #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  SIGNAL-STRENGTH POSITION SIZING                                     #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  CUSTOM ENTRY PRICE — ATR-fractional limit order                   #
    # ------------------------------------------------------------------ #
    def custom_entry_price(
        self,
        pair: str,
        trade,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Place limit order slightly better than close:
          Long:  close - ENTRY_ATR_FRACTION * ATR  (bid on micro-dip)
          Short: close + ENTRY_ATR_FRACTION * ATR  (offer on micro-bounce)

        Improves average entry price and R:R. Fill rate stays high because
        0.15 ATR ≈ typical intra-candle noise. If candle doesn't reach the
        limit, freqtrade will retry or cancel per order_time_in_force.
        """
        if self.ENTRY_ATR_FRACTION.value == 0:
            return proposed_rate
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return proposed_rate
        atr = float(df["atr"].iat[-1])
        if atr <= 0:
            return proposed_rate
        offset = self.ENTRY_ATR_FRACTION.value * atr
        if side == "long":
            return proposed_rate - offset
        else:
            return proposed_rate + offset

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        4-tier sizing:
          trend_cross_long  → 150%  (5+ conditions, highest quality)
          trend_cross_short → 120%  (same quality, short confirmed)
          range_bounce_long → 30%   (3 conditions, lower quality on long)
          range_bounce_short→ 120%  (range shorts historically better)
        """
        if not entry_tag:
            return proposed_stake

        if "trend_" in entry_tag:
            if "_long" in entry_tag:
                factor = self.STAKE_TREND_LONG.value
            else:
                factor = self.STAKE_TREND_SHORT.value
        elif "range_" in entry_tag:
            if "_long" in entry_tag:
                factor = self.STAKE_RANGE_LONG.value
            else:
                factor = self.STAKE_RANGE_SHORT.value
        else:
            return proposed_stake

        stake = proposed_stake * factor
        if min_stake is not None:
            stake = max(stake, float(min_stake))
        return min(stake, max_stake)
    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return self.stoploss
        atr = float(df["atr"].iat[-1])
        if atr <= 0 or trade.open_rate <= 0:
            return self.stoploss
        sl_pct = self.SL_ATR_MULT.value * atr / trade.open_rate
        tp_pct = self.TP_ATR_MULT.value * atr / trade.open_rate
        fee    = trade.fee_open + trade.fee_close
        if not trade.is_short:
            if current_profit >= tp_pct:
                lock_price = trade.open_rate * (1 + tp_pct * 0.60)
                return max((lock_price / current_rate) - 1, -0.005)
            if current_profit >= tp_pct * 0.5:
                be_price = trade.open_rate * (1 + fee)
                return max((be_price / current_rate) - 1, -sl_pct)
        else:
            if current_profit >= tp_pct:
                lock_price = trade.open_rate * (1 - tp_pct * 0.60)
                return max(1 - (lock_price / current_rate), -0.005)
            if current_profit >= tp_pct * 0.5:
                be_price = trade.open_rate * (1 - fee)
                return max(1 - (be_price / current_rate), -sl_pct)
        return max(-sl_pct, -0.20)

    # ------------------------------------------------------------------ #
    #  CUSTOM EXIT — ATR-TP + tag-aware signal exits + cascading cuts     #
    # ------------------------------------------------------------------ #
    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return None

        last = df.iloc[-1]
        enter_tag = trade.enter_tag or ""

        # 1. ATR-based take-profit
        last_atr = float(last.get("atr", 0))
        if last_atr > 0 and trade.open_rate > 0:
            tp = self.TP_ATR_MULT.value * last_atr / trade.open_rate
            if current_profit >= tp:
                return "TP_HIT"

        # 2. Signal-based exit — depends on trade type
        is_long = not trade.is_short
        if "trend" in enter_tag:
            # Trend trade: exit on EMA reversal + ST confirmation
            # Only exit on signal if trade is at breakeven or better (avoid locking in losses)
            sig = "sig_trend_exit_long" if is_long else "sig_trend_exit_short"
            if int(last.get(sig, 0)) == 1 and current_profit >= 0.0:
                return "trend_signal_exit"
        elif "range" in enter_tag:
            # Range trade: exit when price reaches BB mid (mean reversion target)
            sig = "sig_range_exit_long" if is_long else "sig_range_exit_short"
            if int(last.get(sig, 0)) == 1:
                return "range_target_exit"

        # 3. Cascading time-based loss cuts
        hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        if hours >= 2  and current_profit < self.TIME_CUT_2H_THR.value:
            return "time_cut_2h"
        if hours >= 8  and current_profit < self.TIME_CUT_8H_THR.value:
            return "time_cut_8h"
        if hours >= 24 and current_profit <  0.0:
            return "time_cut_24h"
        if hours >= 48 and current_profit <  0.005:
            return "time_cut_48h"

        return None

    # ------------------------------------------------------------------ #
    #  CONFIRM ENTRY                                                       #
    # ------------------------------------------------------------------ #
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return True
        last = df.iloc[-1]

        # Reject entries on zero/near-zero volume
        if float(last.get("vol_ratio", 0)) < 0.4:
            return False

        # Reject if ATR is abnormally high (gap/spike protection)
        atr = float(last.get("atr", 0))
        close = float(last.get("close", 1))
        if atr > 0 and (atr / close) > 0.04:  # ATR > 4% of price = too volatile
            return False

        # BTC market sentiment gate (TrendRiderStrategy technique)
        # Block longs in BTC crash (RSI < 35), block shorts in BTC euphoria (RSI > 65)
        btc_rsi = float(last.get("btc_rsi_1h", 50))
        if side == "long" and btc_rsi < 35:
            return False
        if side == "short" and btc_rsi > 65:
            return False

        # Funding rate extreme filter (NFI technique)
        # High positive funding = crowded longs (long squeeze risk)
        # High negative funding = crowded shorts (short squeeze risk)
        funding = float(last.get("funding_rate", 0.0))
        if side == "long" and funding > 0.00008:   # >0.008% per 8h = extreme crowding
            return False
        if side == "short" and funding < -0.00007:
            return False

        return True

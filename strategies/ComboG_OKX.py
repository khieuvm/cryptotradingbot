# -*- coding: utf-8 -*-
"""
ComboG_OKX - Port from VN30F1M signals.py -> Freqtrade IStrategy
=================================================================
Combo G: Multi-Oscillator Reversal
  PRIMARY : cci_extreme + williams_extreme + mfi_confirm  (>= 1 must fire)
  CONFIRM : stoch_cross + bb_bounce + rsi_div + obv_confirm  (each +1 confidence)
  GATE    : volume_ratio_gate  (ALL must pass)

Combo G+: Enhanced Multi-Oscillator
  PRIMARY : cci_extreme + williams_extreme + mfi_confirm  (same as G)
  CONFIRM : + vol_price_divergence  (extra confirm)
  GATE    : volume_ratio_gate + momentum_ratio_gate  (stricter gate)

SL/TP    : ATR-based (SL 1.5x, TP 4.0x) -- from combo_risk in strategy_config.yaml
Timeframe: 5m primary + 15m informative
Exchange : OKX Futures (USDT-M perpetual)
"""

from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import pandas_ta as ta

from freqtrade.strategy import IStrategy, merge_informative_pair, DecimalParameter, IntParameter


class ComboG_OKX(IStrategy):
    # ------------------------------------------------------------------ #
    #  FREQTRADE PARAMS                                                    #
    # ------------------------------------------------------------------ #
    INTERFACE_VERSION = 3
    timeframe = "5m"
    inf_timeframe = "15m"

    # Fixed -3% SL: R:R = ~2.25% TP / 3% SL = 0.75, breakeven at 57% win rate
    stoploss = -0.03
    minimal_roi = {"0": 0.15}  # fallback; custom_exit ATR-TP fires first in practice

    # Position
    can_short = True
    use_custom_stoploss = True
    # --- Protections (TrendRiderStrategy + NFI pattern) ---
    protections = [
        {"method": "CooldownPeriod", "stop_duration_candles": 5},
        {"method": "StoplossGuard", "lookback_period_candles": 96,
         "trade_limit": 3, "stop_duration_candles": 40, "only_per_pair": True},
        {"method": "MaxDrawdown", "lookback_period_candles": 96,
         "max_allowed_drawdown": 0.15, "stop_duration_candles": 120, "trade_limit": 5},
    ]

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ATR multipliers -- hyperopt search space
    SL_ATR_MULT = DecimalParameter(1.0, 5.0, default=3.0, decimals=1, space="sell", optimize=True)
    TP_ATR_MULT = DecimalParameter(2.0, 10.0, default=6.0, decimals=1, space="sell", optimize=True)

    # Combo G thresholds -- hyperopt buy space
    CCI_OVERSOLD_P   = IntParameter(-130, -80, default=-100, space="buy", optimize=True)
    CCI_OVERBOUGHT_P = IntParameter(80, 130,   default=100,  space="buy", optimize=True)
    WILLR_OVERSOLD_P = IntParameter(-90, -60,  default=-80,  space="buy", optimize=True)
    WILLR_OB_P       = IntParameter(-40, -10,  default=-20,  space="buy", optimize=True)
    MFI_OVERSOLD_P   = IntParameter(20, 40,    default=30,   space="buy", optimize=True)
    MFI_OB_P         = IntParameter(60, 80,    default=70,   space="buy", optimize=True)
    VOL_RATIO_P      = DecimalParameter(0.5, 1.2, default=0.8, decimals=1, space="buy", optimize=True)
    # Stake sizing: G mode (reversal — fewer confirms = better) vs Gp mode
    STAKE_S1    = DecimalParameter(0.5, 1.5, default=1.2, decimals=1, space='buy', optimize=False)
    STAKE_S2    = DecimalParameter(0.2, 0.8, default=0.3, decimals=1, space='buy', optimize=False)
    STAKE_S3    = DecimalParameter(0.7, 1.5, default=1.1, decimals=1, space='buy', optimize=False)
    STAKE_S4    = DecimalParameter(0.7, 1.5, default=1.2, decimals=1, space='buy', optimize=False)
    STAKE_GP_S1 = DecimalParameter(0.7, 1.5, default=1.2, decimals=1, space='buy', optimize=False)
    STAKE_GP_S2 = DecimalParameter(0.5, 1.2, default=1.0, decimals=1, space='buy', optimize=False)
    STAKE_GP_S3 = DecimalParameter(1.0, 1.5, default=1.5, decimals=1, space='buy', optimize=False)


    # ------------------------------------------------------------------ #
    #  INFORMATIVE PAIRS (15m for multi-TF like COMBO_TF_MAP G:[5m,15m])  #
    # ------------------------------------------------------------------ #
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        pairs = self.dp.current_whitelist()
        inf = [(pair, self.inf_timeframe) for pair in pairs]
        # BTC 1h always needed (separate timeframe from 5m trading pair)
        if ("BTC/USDT:USDT", "1h") not in inf:
            inf.append(("BTC/USDT:USDT", "1h"))
        return inf

    # ------------------------------------------------------------------ #
    #  INDICATORS                                                          #
    # ------------------------------------------------------------------ #
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # --- Merge 15m indicators ---
        dataframe = self._add_informative_indicators(dataframe, metadata)

        # --- Moving Averages ---
        dataframe["ema8"]  = ta.ema(dataframe["close"], length=8)
        dataframe["ema12"] = ta.ema(dataframe["close"], length=12)
        dataframe["ema21"] = ta.ema(dataframe["close"], length=21)
        dataframe["ema55"] = ta.ema(dataframe["close"], length=55)
        dataframe["sma_f"] = ta.sma(dataframe["close"], length=10)
        dataframe["sma_s"] = ta.sma(dataframe["close"], length=20)
        dataframe["tema9"] = ta.tema(dataframe["close"], length=9)

        # --- ATR ---
        dataframe["atr"] = ta.atr(dataframe["high"], dataframe["low"], dataframe["close"], length=14)

        # --- RSI ---
        dataframe["rsi"] = ta.rsi(dataframe["close"], length=7)

        # --- CCI (22. cci_extreme) ---
        dataframe["cci"] = ta.cci(dataframe["high"], dataframe["low"], dataframe["close"], length=14)

        # --- Williams %R (24. williams_extreme) ---
        dataframe["willr"] = ta.willr(dataframe["high"], dataframe["low"], dataframe["close"], length=14)

        # --- MFI (31. mfi_confirm) ---
        try:
            dataframe["mfi"] = ta.mfi(
                dataframe["high"], dataframe["low"],
                dataframe["close"], dataframe["volume"].astype(float), length=14
            )
        except Exception:
            dataframe["mfi"] = 50.0

        # --- Stochastic (7. stoch_cross) ---
        stoch = ta.stoch(dataframe["high"], dataframe["low"], dataframe["close"], k=14, d=3)
        if stoch is not None:
            dataframe["stoch_k"] = stoch.iloc[:, 0]
            dataframe["stoch_d"] = stoch.iloc[:, 1]
        else:
            dataframe["stoch_k"] = 50.0
            dataframe["stoch_d"] = 50.0

        # --- Bollinger Bands (8. bb_bounce) ---
        bb = ta.bbands(dataframe["close"], length=20, std=2.0)
        if bb is not None:
            dataframe["bb_upper"] = bb.iloc[:, 2]
            dataframe["bb_mid"]   = bb.iloc[:, 1]
            dataframe["bb_lower"] = bb.iloc[:, 0]
        else:
            dataframe["bb_upper"] = dataframe["close"]
            dataframe["bb_mid"]   = dataframe["close"]
            dataframe["bb_lower"] = dataframe["close"]

        # --- OBV (32. obv_confirm) ---
        try:
            dataframe["obv"] = ta.obv(dataframe["close"], dataframe["volume"].astype(float))
        except Exception:
            dataframe["obv"] = 0.0
        dataframe["obv_ema"]   = ta.ema(dataframe["obv"], length=20)
        dataframe["obv_rising"] = (dataframe["obv"] > dataframe["obv_ema"]).astype(int)

        # --- Volume Ratio (33. volume_ratio_gate) ---
        dataframe["vol_ema20"]     = ta.ema(dataframe["volume"].astype(float), length=20)
        dataframe["volume_ratio"]  = dataframe["volume"].astype(float) / (dataframe["vol_ema20"] + 1e-10)

        # --- Trend helpers ---
        dataframe["trend_bull"] = (dataframe["sma_f"] > dataframe["sma_s"]).astype(int)
        dataframe["trend_bear"] = (dataframe["sma_f"] < dataframe["sma_s"]).astype(int)

        # --- Momentum ratio (36. momentum_ratio_gate, used by G+) ---
        up_bars = (dataframe["close"] > dataframe["close"].shift(1)).astype(float)
        dataframe["up_ratio_12"] = up_bars.rolling(12, min_periods=12).mean()

        # --- Vol-Price Divergence (37. vol_price_divergence, used by G+) ---
        vol_chg   = dataframe["volume"].astype(float).pct_change()
        price_chg = dataframe["close"].pct_change()
        dataframe["vol_price_div"] = np.sign(vol_chg) * (-1 * np.sign(price_chg))

        # --- TEMA direction ---
        dataframe["tema_rising"]  = (dataframe["tema9"] > dataframe["tema9"].shift(1)).astype(int)
        dataframe["tema_falling"] = (dataframe["tema9"] < dataframe["tema9"].shift(1)).astype(int)

        dataframe = self._add_btc_sentiment(dataframe, metadata)
        return dataframe

    def _add_informative_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Add 15m CCI + Williams %R + MFI for multi-TF confirmation (COMBO_TF_MAP G:[5m,15m])."""
        inf_tf = self.inf_timeframe
        inf_df = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=inf_tf)
        if inf_df.empty:
            dataframe["cci_15m"]   = np.nan
            dataframe["willr_15m"] = np.nan
            dataframe["mfi_15m"]   = np.nan
            return dataframe

        inf_df["cci_15m"]   = ta.cci(inf_df["high"], inf_df["low"], inf_df["close"], length=14)
        inf_df["willr_15m"] = ta.willr(inf_df["high"], inf_df["low"], inf_df["close"], length=14)
        try:
            inf_df["mfi_15m"] = ta.mfi(
                inf_df["high"], inf_df["low"],
                inf_df["close"], inf_df["volume"].astype(float), length=14
            )
        except Exception:
            inf_df["mfi_15m"] = 50.0

        dataframe = merge_informative_pair(
            dataframe, inf_df, self.timeframe, inf_tf,
            ffill=True
        )
        # merge_informative_pair appends "_15m" -> cci_15m_15m, willr_15m_15m, mfi_15m_15m
        return dataframe

    # ------------------------------------------------------------------ #
    #  COMBO G CONDITIONS (private helpers)                               #
    # ------------------------------------------------------------------ #

    def _cci_extreme_buy(self, df: pd.DataFrame) -> pd.Series:
        """CCI crosses back above oversold threshold (exits oversold zone)."""
        thr = self.CCI_OVERSOLD_P.value
        return (df["cci"] > thr) & (df["cci"].shift(1) <= thr)

    def _cci_extreme_sell(self, df: pd.DataFrame) -> pd.Series:
        """CCI crosses back below overbought threshold (exits overbought zone)."""
        thr = self.CCI_OVERBOUGHT_P.value
        return (df["cci"] < thr) & (df["cci"].shift(1) >= thr)

    def _williams_extreme_buy(self, df: pd.DataFrame) -> pd.Series:
        """Williams %R crosses back above oversold threshold."""
        thr = self.WILLR_OVERSOLD_P.value
        return (df["willr"] > thr) & (df["willr"].shift(1) <= thr)

    def _williams_extreme_sell(self, df: pd.DataFrame) -> pd.Series:
        """Williams %R crosses back below overbought threshold."""
        thr = self.WILLR_OB_P.value
        return (df["willr"] < thr) & (df["willr"].shift(1) >= thr)

    def _mfi_confirm_buy(self, df: pd.DataFrame) -> pd.Series:
        """MFI oversold AND turning up."""
        thr = self.MFI_OVERSOLD_P.value
        return (df["mfi"] < thr) & (df["mfi"] > df["mfi"].shift(1))

    def _mfi_confirm_sell(self, df: pd.DataFrame) -> pd.Series:
        """MFI overbought AND turning down."""
        thr = self.MFI_OB_P.value
        return (df["mfi"] > thr) & (df["mfi"] < df["mfi"].shift(1))

    def _stoch_cross_buy(self, df: pd.DataFrame) -> pd.Series:
        """Stoch K crosses D when leaving oversold (<20)."""
        return (
            (df["stoch_k"] > df["stoch_d"]) &
            (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) &
            (df["stoch_k"].shift(1) < 20)
        )

    def _stoch_cross_sell(self, df: pd.DataFrame) -> pd.Series:
        """Stoch K crosses D when leaving overbought (>80)."""
        return (
            (df["stoch_k"] < df["stoch_d"]) &
            (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) &
            (df["stoch_k"].shift(1) > 80)
        )

    def _bb_bounce_buy(self, df: pd.DataFrame) -> pd.Series:
        """Price touches lower BB in uptrend + RSI < 35."""
        return (df["low"] <= df["bb_lower"]) & (df["trend_bull"] == 1) & (df["rsi"] < 35)

    def _bb_bounce_sell(self, df: pd.DataFrame) -> pd.Series:
        """Price touches upper BB in downtrend + RSI > 65."""
        return (df["high"] >= df["bb_upper"]) & (df["trend_bear"] == 1) & (df["rsi"] > 65)

    def _rsi_div_buy(self, df: pd.DataFrame) -> pd.Series:
        """Bullish RSI divergence: price lower low but RSI higher low (< 45)."""
        lookback = 10
        price_ll = df["close"] < df["close"].rolling(lookback).min().shift(1)
        rsi_hl   = df["rsi"]   > df["rsi"].rolling(lookback).min().shift(1)
        return price_ll & rsi_hl & (df["rsi"] < 45)

    def _rsi_div_sell(self, df: pd.DataFrame) -> pd.Series:
        """Bearish RSI divergence: price higher high but RSI lower high (> 55)."""
        lookback = 10
        price_hh = df["close"] > df["close"].rolling(lookback).max().shift(1)
        rsi_lh   = df["rsi"]   < df["rsi"].rolling(lookback).max().shift(1)
        return price_hh & rsi_lh & (df["rsi"] > 55)

    def _obv_confirm_buy(self, df: pd.DataFrame) -> pd.Series:
        """OBV > OBV_EMA = accumulation."""
        return df["obv_rising"] == 1

    def _obv_confirm_sell(self, df: pd.DataFrame) -> pd.Series:
        """OBV < OBV_EMA = distribution."""
        return df["obv_rising"] == 0

    def _volume_ratio_gate(self, df: pd.DataFrame) -> pd.Series:
        """Volume ratio >= 0.8 (not dead market)."""
        return df["volume_ratio"] >= self.VOL_RATIO_P.value

    def _momentum_ratio_gate_buy(self, df: pd.DataFrame) -> pd.Series:
        """Up-bar ratio > 55% in last 12 bars (G+ only)."""
        return df["up_ratio_12"] > 0.55

    def _momentum_ratio_gate_sell(self, df: pd.DataFrame) -> pd.Series:
        """Up-bar ratio < 45% in last 12 bars (G+ only)."""
        return df["up_ratio_12"] < 0.45

    def _vol_price_div_buy(self, df: pd.DataFrame) -> pd.Series:
        """Volume up + price down divergence (bullish reversal, G+ only)."""
        return df["vol_price_div"] > 0

    def _vol_price_div_sell(self, df: pd.DataFrame) -> pd.Series:
        """Volume up + price up divergence (bearish, G+ only)."""
        return df["vol_price_div"] < 0

    def _combo_g_signal(self, df: pd.DataFrame, long: bool, plus_mode: bool = False):
        """
        Returns boolean Series for G or G+ entry signal.

        Primary (>= 1 must fire):
          cci_extreme + williams_extreme + mfi_confirm

        Confirm (>= 1 needed for minimum confidence):
          stoch_cross + bb_bounce + rsi_div + obv_confirm
          [G+ also adds: vol_price_divergence]

        Gate (ALL must pass):
          volume_ratio_gate
          [G+ also adds: momentum_ratio_gate]

        Plus, same conditions checked on 15m for multi-TF confluence.
        """
        if long:
            # --- Primary conditions (TIGHTENED: need >= 2 of 3) ---
            cci_ok   = self._cci_extreme_buy(df)
            willr_ok = self._williams_extreme_buy(df)
            mfi_ok   = self._mfi_confirm_buy(df)
            primary_count = cci_ok.astype(int) + willr_ok.astype(int) + mfi_ok.astype(int)
            primary = primary_count >= 2

            # 15m confluence (COMBO_TF_MAP G:[5m,15m]): at least one 15m primary
            cci_15m  = (
                df.get("cci_15m_15m",   pd.Series(np.nan, index=df.index)).notna() &
                (df.get("cci_15m_15m",  pd.Series(0, index=df.index)) > self.CCI_OVERSOLD_P.value) &
                (df.get("cci_15m_15m",  pd.Series(0, index=df.index)).shift(1) <= self.CCI_OVERSOLD_P.value)
            )
            willr_15m = (
                df.get("willr_15m_15m", pd.Series(np.nan, index=df.index)).notna() &
                (df.get("willr_15m_15m", pd.Series(-50, index=df.index)) > self.WILLR_OVERSOLD_P.value) &
                (df.get("willr_15m_15m", pd.Series(-50, index=df.index)).shift(1) <= self.WILLR_OVERSOLD_P.value)
            )
            mtf_confluence = cci_15m | willr_15m  # bonus: 15m agrees

            # --- Confirm conditions ---
            stoch_c = self._stoch_cross_buy(df)
            bb_c    = self._bb_bounce_buy(df)
            rsi_c   = self._rsi_div_buy(df)
            obv_c   = self._obv_confirm_buy(df)

            confirm_signals = [stoch_c, bb_c, rsi_c, obv_c]
            if plus_mode:
                confirm_signals.append(self._vol_price_div_buy(df))

            confirm_count = reduce(lambda a, b: a.astype(int) + b.astype(int), confirm_signals)
            has_confirm   = confirm_count >= 1

            # --- Gate conditions ---
            vol_gate  = df["volume_ratio"] >= self.VOL_RATIO_P.value
            # EMA trend filter: buy pullbacks in uptrend only (ema21 > ema55)
            ema_trend = df["ema21"] > df["ema55"]
            if plus_mode:
                mom_gate = self._momentum_ratio_gate_buy(df)
                gates    = vol_gate & ema_trend & mom_gate
            else:
                gates = vol_gate & ema_trend

        else:  # short
            # --- Primary conditions (TIGHTENED: need >= 2 of 3) ---
            cci_ok   = self._cci_extreme_sell(df)
            willr_ok = self._williams_extreme_sell(df)
            mfi_ok   = self._mfi_confirm_sell(df)
            primary_count = cci_ok.astype(int) + willr_ok.astype(int) + mfi_ok.astype(int)
            primary = primary_count >= 2

            cci_15m  = (
                df.get("cci_15m_15m",  pd.Series(np.nan, index=df.index)).notna() &
                (df.get("cci_15m_15m", pd.Series(0, index=df.index)) < self.CCI_OVERBOUGHT_P.value) &
                (df.get("cci_15m_15m", pd.Series(0, index=df.index)).shift(1) >= self.CCI_OVERBOUGHT_P.value)
            )
            willr_15m = (
                df.get("willr_15m_15m", pd.Series(np.nan, index=df.index)).notna() &
                (df.get("willr_15m_15m", pd.Series(-50, index=df.index)) < self.WILLR_OB_P.value) &
                (df.get("willr_15m_15m", pd.Series(-50, index=df.index)).shift(1) >= self.WILLR_OB_P.value)
            )
            mtf_confluence = cci_15m | willr_15m

            # --- Confirm conditions ---
            stoch_c = self._stoch_cross_sell(df)
            bb_c    = self._bb_bounce_sell(df)
            rsi_c   = self._rsi_div_sell(df)
            obv_c   = self._obv_confirm_sell(df)

            confirm_signals = [stoch_c, bb_c, rsi_c, obv_c]
            if plus_mode:
                confirm_signals.append(self._vol_price_div_sell(df))

            confirm_count = reduce(lambda a, b: a.astype(int) + b.astype(int), confirm_signals)
            has_confirm   = confirm_count >= 1

            # --- Gate conditions ---
            vol_gate  = df["volume_ratio"] >= self.VOL_RATIO_P.value
            # EMA trend filter: short bounces in downtrend only (ema21 < ema55)
            ema_trend = df["ema21"] < df["ema55"]
            if plus_mode:
                mom_gate = self._momentum_ratio_gate_sell(df)
                gates    = vol_gate & ema_trend & mom_gate
            else:
                gates = vol_gate & ema_trend

        # Final signal:
        return primary & has_confirm & gates, confirm_count

    # ------------------------------------------------------------------ #
    #  ENTRY / EXIT TREND                                                  #
    # ------------------------------------------------------------------ #
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Combo G long/short — (signal, confirm_count) tuples
        g_long,  g_long_cnt  = self._combo_g_signal(dataframe, long=True,  plus_mode=False)
        gp_long, gp_long_cnt = self._combo_g_signal(dataframe, long=True,  plus_mode=True)

        # G+ overrides G when both fire (G+ is stricter subset → higher quality)
        long_sig = g_long | gp_long
        long_tags = "G_s"  + g_long_cnt.clip(1, 4).astype(int).astype(str)
        long_tags[gp_long] = ("Gp_s" + gp_long_cnt[gp_long].clip(1, 4).astype(int).astype(str))

        dataframe.loc[long_sig, "enter_long"] = 1
        dataframe.loc[long_sig, "enter_tag"]  = long_tags[long_sig]

        # Combo G short
        g_short,  g_short_cnt  = self._combo_g_signal(dataframe, long=False, plus_mode=False)
        gp_short, gp_short_cnt = self._combo_g_signal(dataframe, long=False, plus_mode=True)

        short_sig = g_short | gp_short
        short_tags = "G_s"  + g_short_cnt.clip(1, 4).astype(int).astype(str)
        short_tags[gp_short] = ("Gp_s" + gp_short_cnt[gp_short].clip(1, 4).astype(int).astype(str))

        dataframe.loc[short_sig, "enter_short"] = 1
        dataframe.loc[short_sig, "enter_tag"]   = short_tags[short_sig]

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exit handled by custom_stoploss + custom_exit
        # No hard exit signal needed
        dataframe["exit_long"]  = 0
        dataframe["exit_short"] = 0
        return dataframe

    # ------------------------------------------------------------------ #
    #  3-PHASE ATR STOPLOSS: initial → break-even → trail-lock            #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  SIGNAL-STRENGTH POSITION SIZING                                     #
    # ------------------------------------------------------------------ #

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
        G mode:  s1=1.2x (reversal, fewer confirms = better), s2=0.3x
        Gp mode: Gp_s1=1.2x, Gp_s2=1.0x, Gp_s3=1.5x (more filters = better)
        """
        if not entry_tag:
            return proposed_stake

        if entry_tag.startswith("Gp_s"):
            gp_factors = {1: self.STAKE_GP_S1.value, 2: self.STAKE_GP_S2.value, 3: self.STAKE_GP_S3.value}
            try:
                n = int(entry_tag.split("Gp_s")[-1])
                factor = gp_factors.get(min(n, 3), self.STAKE_GP_S1.value)
            except (ValueError, IndexError):
                factor = 1.0
        elif "_s" in entry_tag:
            s_factors = {1: self.STAKE_S1.value, 2: self.STAKE_S2.value,
                         3: self.STAKE_S3.value, 4: self.STAKE_S4.value}
            try:
                n = int(entry_tag.split("_s")[-1])
                factor = s_factors.get(min(n, 4), 1.0)
            except (ValueError, IndexError):
                factor = 1.0
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
        """
        Phase 1 — ATR-based initial SL.
        Phase 2 — Break-even: SL moves to open+fees when profit >= 50% of TP.
        Phase 3 — Trail-lock: SL locks 60% of TP gain once TP is reached.
        """
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
                # Phase 3: lock in 60% of TP gain
                lock_price = trade.open_rate * (1 + tp_pct * 0.60)
                return max((lock_price / current_rate) - 1, -0.005)
            if current_profit >= tp_pct * 0.5:
                # Phase 2: break-even
                be_price = trade.open_rate * (1 + fee)
                return max((be_price / current_rate) - 1, -sl_pct)
        else:
            if current_profit >= tp_pct:
                lock_price = trade.open_rate * (1 - tp_pct * 0.60)
                return max(1 - (lock_price / current_rate), -0.005)
            if current_profit >= tp_pct * 0.5:
                be_price = trade.open_rate * (1 - fee)
                return max(1 - (be_price / current_rate), -sl_pct)

        # Phase 1: initial ATR SL
        return max(-sl_pct, -0.30)

    # ------------------------------------------------------------------ #
    #  ATR-BASED TAKE PROFIT  (TP_ATR_MULT = 4.0)                         #
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
        """TP = ATR-based target. Cascading time-cuts for losing trades."""
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if not df.empty:
            last_atr = float(df["atr"].iat[-1])
            if last_atr > 0 and trade.open_rate > 0:
                tp_ratio = self.TP_ATR_MULT.value * last_atr / trade.open_rate
                if current_profit >= tp_ratio:
                    return "TP_HIT"

        # Cascading time-based exit — cut losers before they bleed out
        # Inspired by TrendRiderStrategy (github.com/freqtrade/freqtrade-strategies)
        hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        if hours >= 4 and current_profit < -0.02:
            return "time_cut_4h"
        if hours >= 12 and current_profit < -0.01:
            return "time_cut_12h"
        if hours >= 24 and current_profit < 0.0:
            return "time_cut_24h"
        if hours >= 48 and current_profit < 0.005:
            return "time_cut_48h"

        return None

    # ------------------------------------------------------------------ #
    #  CONFIRM ENTRY (optional: add extra filters before placing order)    #
    # ------------------------------------------------------------------ #

    def _add_btc_sentiment(self, dataframe, metadata):
        """BTC 1h RSI as macro sentiment gate (TrendRiderStrategy technique)."""
        if self.dp is None:
            dataframe["btc_rsi_1h"] = 50.0
            return dataframe
        import pandas_ta as ta_local
        btc_df = self.dp.get_pair_dataframe(pair="BTC/USDT:USDT", timeframe="1h")
        if btc_df.empty:
            dataframe["btc_rsi_1h"] = 50.0
            return dataframe
        btc_df = btc_df.copy()
        btc_df["btc_rsi"] = ta_local.rsi(btc_df["close"], length=14)
        btc_df = btc_df[["date", "btc_rsi"]].copy()
        from freqtrade.strategy import merge_informative_pair
        dataframe = merge_informative_pair(dataframe, btc_df, self.timeframe, "1h", ffill=True)
        if "btc_rsi_1h" not in dataframe.columns:
            dataframe["btc_rsi_1h"] = 50.0
        return dataframe

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
        """
        Block entries when:
        - ATR = 0 (no real market data)
        - Volume too thin
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return False

        last = df.iloc[-1]
        if float(last.get("atr", 0)) <= 0:
            return False
        if float(last.get("volume_ratio", 0)) < self.VOL_RATIO_P.value:
            return False

        # BTC market sentiment gate (TrendRiderStrategy technique)
        btc_rsi = float(last.get("btc_rsi_1h", 50))
        if side == "long" and btc_rsi < 35:
            return False
        if side == "short" and btc_rsi > 65:
            return False

        return True
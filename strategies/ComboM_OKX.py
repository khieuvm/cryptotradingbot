# -*- coding: utf-8 -*-

"""

ComboM_OKX - Port from VN30F1M signals.py -> Freqtrade IStrategy

=================================================================

Combo M: Trend Breakout

  PRIMARY : trend_composite + supertrend_flip + sr_breakout  (>= 1 must fire)

  CONFIRM : adx_di + ema_ribbon + macd_cross + obv_confirm  (each +1 confidence, >= 1 needed)

  GATE    : trend_strength_gate + kb_squeeze + volume_ratio_gate  (ALL must pass)



SL/TP    : ATR-based (SL 1.5x, TP 3.0x) -- from combo_risk in strategy_config.yaml

Timeframe: 5m primary + 15m informative

Exchange : OKX Futures (USDT-M perpetual)



Source: tradingbot/src/signals.py conditions #45 (trend_composite), #19 (supertrend_flip),

        #14 (sr_breakout), #13 (adx_di), #10 (ema_ribbon), #2 (macd_cross),

        #32 (obv_confirm), #46 (trend_strength_gate), #41 (kb_squeeze),

        #33 (volume_ratio_gate)



Trend Composite = linreg_slope_norm + r2 * sign(slope) + aroon_osc/100

  (from Superalgos Masters/Zeus approach in signals.py lines ~900-950)

Trend Strength Gate: trend_composite > 0.5 AND linreg_r2 > 0.3 for BUY

                     trend_composite < -0.5 AND linreg_r2 > 0.3 for SELL

KB Squeeze Gate: BB inside Keltner channel (compressed) or just released

"""



from datetime import datetime

from functools import reduce



import numpy as np

import pandas as pd

import pandas_ta as ta

from numpy.lib.stride_tricks import sliding_window_view



from freqtrade.strategy import IStrategy, merge_informative_pair, DecimalParameter, IntParameter





class ComboM_OKX(IStrategy):

    # ------------------------------------------------------------------ #

    #  FREQTRADE PARAMS                                                    #

    # ------------------------------------------------------------------ #

    INTERFACE_VERSION = 3

    timeframe = "5m"

    inf_timeframe = "15m"



    stoploss = -0.04   # fallback; custom_stoploss fires first

    minimal_roi = {"0": 0.15}



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



    # ATR multipliers (hyperopt search space)

    SL_ATR_MULT = DecimalParameter(1.0, 4.0, default=1.5, decimals=1, space="sell", optimize=True)

    TP_ATR_MULT = DecimalParameter(1.5, 6.0, default=3.0, decimals=1, space="sell", optimize=True)



    # Gate thresholds

    VOL_RATIO_MIN = 0.8

    # Buy-space hyperopt: trend quality thresholds
    TREND_THR   = DecimalParameter(0.3, 2.0, default=1.0, decimals=1, space="buy", optimize=True)
    R2_MIN      = DecimalParameter(0.1, 0.6, default=0.3, decimals=1, space="buy", optimize=True)
    TREND_GATE  = DecimalParameter(0.2, 1.0, default=0.5, decimals=1, space="buy", optimize=True)
    VOL_RATIO_P = DecimalParameter(0.5, 1.2, default=0.8, decimals=1, space="buy", optimize=True)
    MIN_PRIMARY = IntParameter(1, 3,         default=2,   space="buy", optimize=True)

    # Stake sizing: direction × confirm count
    STAKE_S1         = DecimalParameter(0.5, 1.5, default=1.0, decimals=1, space='buy', optimize=False)
    STAKE_S2         = DecimalParameter(0.5, 1.5, default=1.0, decimals=1, space='buy', optimize=False)
    STAKE_S3         = DecimalParameter(0.5, 1.5, default=1.0, decimals=1, space='buy', optimize=False)
    STAKE_S4         = DecimalParameter(0.5, 1.5, default=1.0, decimals=1, space='buy', optimize=False)
    STAKE_LONG_MULT  = DecimalParameter(0.2, 2.0, default=0.3, decimals=1, space='buy', optimize=False)
    STAKE_SHORT_MULT = DecimalParameter(0.2, 2.0, default=1.5, decimals=1, space='buy', optimize=False)




    # ------------------------------------------------------------------ #

    #  INFORMATIVE PAIRS (15m for trend quality confirmation)             #

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

        dataframe["atr"] = ta.atr(

            dataframe["high"], dataframe["low"], dataframe["close"], length=14

        )



        # --- RSI ---

        dataframe["rsi"] = ta.rsi(dataframe["close"], length=7)



        # --- MACD ---

        macd = ta.macd(dataframe["close"], fast=12, slow=26, signal=9)

        if macd is not None:

            dataframe["macd_line"] = macd.iloc[:, 0]

            dataframe["macd_hist"] = macd.iloc[:, 1]

            dataframe["macd_sig"]  = macd.iloc[:, 2]

        else:

            dataframe["macd_line"] = 0.0

            dataframe["macd_hist"] = 0.0

            dataframe["macd_sig"]  = 0.0



        # --- ADX + DI ---

        adx = ta.adx(dataframe["high"], dataframe["low"], dataframe["close"], length=14)

        if adx is not None:

            dataframe["adx"]      = adx.iloc[:, 0]

            dataframe["plus_di"]  = adx.iloc[:, 1]

            dataframe["minus_di"] = adx.iloc[:, 2]

        else:

            dataframe["adx"]      = 0.0

            dataframe["plus_di"]  = 0.0

            dataframe["minus_di"] = 0.0



        # --- Supertrend ---

        try:

            st = ta.supertrend(

                dataframe["high"], dataframe["low"], dataframe["close"],

                length=7, multiplier=3.0

            )

            if st is not None:

                st_dir_col = next((c for c in st.columns if "SUPERTd" in c), None)

                dataframe["supertrend_dir"] = (

                    st[st_dir_col].fillna(0) if st_dir_col else 0

                )

            else:

                dataframe["supertrend_dir"] = 0

        except Exception:

            dataframe["supertrend_dir"] = 0



        # --- OBV ---

        try:

            dataframe["obv"] = ta.obv(dataframe["close"], dataframe["volume"].astype(float))

        except Exception:

            dataframe["obv"] = 0.0

        dataframe["obv_ema"]    = ta.ema(dataframe["obv"], length=20)

        dataframe["obv_rising"] = (dataframe["obv"] > dataframe["obv_ema"]).astype(int)



        # --- Volume Ratio ---

        dataframe["vol_ema20"]    = ta.ema(dataframe["volume"].astype(float), length=20)

        dataframe["volume_ratio"] = (

            dataframe["volume"].astype(float) / (dataframe["vol_ema20"] + 1e-10)

        )



        # --- Support / Resistance (20-bar rolling, shifted to avoid look-ahead) ---

        dataframe["resistance_20"] = dataframe["high"].rolling(20).max().shift(1)

        dataframe["support_20"]    = dataframe["low"].rolling(20).min().shift(1)



        # --- Trend helpers ---

        dataframe["trend_bull"] = (dataframe["sma_f"] > dataframe["sma_s"]).astype(int)

        dataframe["trend_bear"] = (dataframe["sma_f"] < dataframe["sma_s"]).astype(int)



        # --- Bollinger Bands (for KB Squeeze) ---

        bb = ta.bbands(dataframe["close"], length=20, std=2.0)

        if bb is not None:

            dataframe["bb_upper"] = bb.iloc[:, 2]

            dataframe["bb_mid"]   = bb.iloc[:, 1]

            dataframe["bb_lower"] = bb.iloc[:, 0]

        else:

            dataframe["bb_upper"] = dataframe["close"]

            dataframe["bb_mid"]   = dataframe["close"]

            dataframe["bb_lower"] = dataframe["close"]



        # --- Keltner Channels (for KB Squeeze) ---

        _kc_ema  = ta.ema(dataframe["close"], length=20)

        _kc_atr  = ta.atr(dataframe["high"], dataframe["low"], dataframe["close"], length=10)

        if _kc_atr is None:

            _kc_atr = dataframe["atr"]

        dataframe["kc_upper"] = _kc_ema + 1.5 * _kc_atr

        dataframe["kc_lower"] = _kc_ema - 1.5 * _kc_atr



        # KB Squeeze: BB fully inside Keltner (compressed volatility)

        dataframe["kb_squeeze_on"] = (

            (dataframe["bb_lower"] > dataframe["kc_lower"]) &

            (dataframe["bb_upper"] < dataframe["kc_upper"])

        )

        # Squeeze release: was squeezing last bar, now released

        dataframe["kb_squeeze_off"] = (

            ~dataframe["kb_squeeze_on"] &

            dataframe["kb_squeeze_on"].shift(1).fillna(False)

        )



        # --- Trend Composite (LinReg slope + R + Aroon, signals.py lines ~900-950) ---

        dataframe = self._compute_trend_composite(dataframe)



        dataframe = self._add_btc_sentiment(dataframe, metadata)
        return dataframe



    def _compute_trend_composite(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        """

        Trend Composite = clip(linreg_slope_norm, }1.5) + R * sign(slope) + aroon_osc/100



        linreg_slope_norm = (linear regression slope over 20 bars) / ATR

        R                = coefficient of determination of that regression

        aroon_osc         = (Aroon Up - Aroon Down) / 100, range -1 to +1

        """

        _close = dataframe["close"].values.astype(float)

        n = len(_close)

        period = 20



        _linreg_slope = np.zeros(n)

        _linreg_r2    = np.zeros(n)



        if n >= period:

            _x      = np.arange(period, dtype=float)

            _x_mean = _x.mean()

            _x_var  = np.sum((_x - _x_mean) ** 2)



            _y_windows = sliding_window_view(_close, period)

            _y_means   = _y_windows.mean(axis=1)

            _y_c       = _y_windows - _y_means[:, np.newaxis]

            _x_c       = _x - _x_mean

            _cov       = np.sum(_y_c * _x_c, axis=1)

            _slopes    = _cov / _x_var

            _y_var     = np.sum(_y_c ** 2, axis=1)

            _r_sq      = np.where(_y_var > 0, (_cov ** 2) / (_x_var * _y_var), 0.0)



            _linreg_slope[period - 1:] = _slopes

            _linreg_r2[period - 1:]    = _r_sq



        dataframe["linreg_slope"]      = _linreg_slope

        dataframe["linreg_r2"]         = _linreg_r2

        dataframe["linreg_slope_norm"] = (

            pd.Series(_linreg_slope, index=dataframe.index) /

            dataframe["atr"].replace(0, np.nan)

        )



        # Aroon oscillator

        try:

            _aroon = ta.aroon(dataframe["high"], dataframe["low"], length=25)

            if _aroon is not None:

                _up_col = next((c for c in _aroon.columns if "AROONU" in c), None)

                _dn_col = next((c for c in _aroon.columns if "AROOND" in c), None)

                dataframe["aroon_up"]   = _aroon[_up_col]  if _up_col else 50.0

                dataframe["aroon_down"] = _aroon[_dn_col]  if _dn_col else 50.0

            else:

                dataframe["aroon_up"]   = 50.0

                dataframe["aroon_down"] = 50.0

        except Exception:

            dataframe["aroon_up"]   = 50.0

            dataframe["aroon_down"] = 50.0

        dataframe["aroon_osc"] = dataframe["aroon_up"] - dataframe["aroon_down"]



        # Composite score

        _slope_score = dataframe["linreg_slope_norm"].clip(-1.5, 1.5)

        _r2_score    = dataframe["linreg_r2"]

        _aroon_score = dataframe["aroon_osc"] / 100.0

        dataframe["trend_composite"] = (

            _slope_score +

            _r2_score * np.sign(_slope_score) +

            _aroon_score

        )



        return dataframe



    def _add_informative_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        """Merge 15m trend composite for multi-TF trend quality check."""

        inf_df = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_timeframe)

        if inf_df.empty:

            dataframe["trend_composite_15m"] = np.nan

            dataframe["linreg_r2_15m"]       = np.nan

            return dataframe



        # Compute trend composite on 15m frame

        inf_df["atr"] = ta.atr(inf_df["high"], inf_df["low"], inf_df["close"], length=14)

        # Minimal linreg on 15m

        _close15 = inf_df["close"].values.astype(float)

        n15 = len(_close15)

        period = 20

        _slope15 = np.zeros(n15)

        _r2_15   = np.zeros(n15)

        if n15 >= period:

            _x     = np.arange(period, dtype=float)

            _xm    = _x.mean()

            _xv    = np.sum((_x - _xm) ** 2)

            _yw    = sliding_window_view(_close15, period)

            _ym    = _yw.mean(axis=1)

            _yc    = _yw - _ym[:, np.newaxis]

            _xc    = _x - _xm

            _cov   = np.sum(_yc * _xc, axis=1)

            _yvar  = np.sum(_yc ** 2, axis=1)

            _slope15[period - 1:] = _cov / _xv

            _r2_15[period - 1:]   = np.where(_yvar > 0, (_cov ** 2) / (_xv * _yvar), 0.0)



        _snorm15 = pd.Series(_slope15, index=inf_df.index) / inf_df["atr"].replace(0, np.nan)

        try:

            _aroon15 = ta.aroon(inf_df["high"], inf_df["low"], length=25)

            if _aroon15 is not None:

                _up = next((c for c in _aroon15.columns if "AROONU" in c), None)

                _dn = next((c for c in _aroon15.columns if "AROOND" in c), None)

                _aosc15 = (_aroon15[_up] - _aroon15[_dn]) / 100.0 if (_up and _dn) else 0.0

            else:

                _aosc15 = 0.0

        except Exception:

            _aosc15 = 0.0



        inf_df["trend_composite_15m"] = (

            _snorm15.clip(-1.5, 1.5) +

            pd.Series(_r2_15, index=inf_df.index) * np.sign(_snorm15) +

            _aosc15

        )

        inf_df["linreg_r2_15m"] = pd.Series(_r2_15, index=inf_df.index)



        dataframe = merge_informative_pair(

            dataframe, inf_df, self.timeframe, self.inf_timeframe, ffill=True

        )

        return dataframe



    # ------------------------------------------------------------------ #

    #  CONDITION HELPERS                                                   #

    # ------------------------------------------------------------------ #



    # --- PRIMARY: Trend Composite Cross ---

    def _trend_composite_buy(self, df) -> 'pd.Series':
        thr = self.TREND_THR.value
        return (df["trend_composite"] > thr) & (df["trend_composite"].shift(1) <= thr)



    def _trend_composite_sell(self, df: pd.DataFrame) -> pd.Series:
        """Trend composite crosses below -TREND_THR."""
        thr = self.TREND_THR.value
        return (df["trend_composite"] < -thr) & (df["trend_composite"].shift(1) >= -thr)



    # --- PRIMARY: Supertrend Flip ---

    def _supertrend_flip_buy(self, df: pd.DataFrame) -> pd.Series:

        return (df["supertrend_dir"] == 1) & (df["supertrend_dir"].shift(1) == -1)



    def _supertrend_flip_sell(self, df: pd.DataFrame) -> pd.Series:

        return (df["supertrend_dir"] == -1) & (df["supertrend_dir"].shift(1) == 1)



    # --- PRIMARY: S/R Breakout (20-bar) ---

    def _sr_breakout_buy(self, df: pd.DataFrame) -> pd.Series:

        """Close breaks above 20-bar high + RSI between 50-75."""

        return (

            (df["close"] > df["resistance_20"]) &

            (df["rsi"] > 50) & (df["rsi"] < 75)

        )



    def _sr_breakout_sell(self, df: pd.DataFrame) -> pd.Series:

        """Close breaks below 20-bar low + RSI between 25-50."""

        return (

            (df["close"] < df["support_20"]) &

            (df["rsi"] < 50) & (df["rsi"] > 25)

        )



    # --- CONFIRM: ADX + DI ---

    def _adx_di_buy(self, df: pd.DataFrame) -> pd.Series:

        di_cross_up = (df["plus_di"] > df["minus_di"]) & (df["plus_di"].shift(1) <= df["minus_di"].shift(1))

        return di_cross_up & (df["adx"] > 25)



    def _adx_di_sell(self, df: pd.DataFrame) -> pd.Series:

        di_cross_dn = (df["minus_di"] > df["plus_di"]) & (df["minus_di"].shift(1) <= df["plus_di"].shift(1))

        return di_cross_dn & (df["adx"] > 25)



    # --- CONFIRM: EMA Ribbon ---

    def _ema_ribbon_buy(self, df: pd.DataFrame) -> pd.Series:

        ribbon_bull  = (df["ema8"] > df["ema21"]) & (df["ema21"] > df["ema55"])

        pull_to_21   = (df["low"] <= df["ema21"] * 1.005) & (df["close"] > df["ema21"])

        return ribbon_bull & pull_to_21



    def _ema_ribbon_sell(self, df: pd.DataFrame) -> pd.Series:

        ribbon_bear  = (df["ema8"] < df["ema21"]) & (df["ema21"] < df["ema55"])

        pull_to_21   = (df["high"] >= df["ema21"] * 0.995) & (df["close"] < df["ema21"])

        return ribbon_bear & pull_to_21



    # --- CONFIRM: MACD Cross ---

    def _macd_cross_buy(self, df: pd.DataFrame) -> pd.Series:

        cross = (df["macd_line"] > df["macd_sig"]) & (df["macd_line"].shift(1) <= df["macd_sig"].shift(1))

        valid = (df["trend_bull"] == 1) | (df["macd_line"] <= 0)

        return cross & valid



    def _macd_cross_sell(self, df: pd.DataFrame) -> pd.Series:

        cross = (df["macd_line"] < df["macd_sig"]) & (df["macd_line"].shift(1) >= df["macd_sig"].shift(1))

        valid = (df["trend_bear"] == 1) | (df["macd_line"] >= 0)

        return cross & valid



    # --- CONFIRM: OBV ---

    def _obv_confirm_buy(self, df: pd.DataFrame) -> pd.Series:

        return df["obv_rising"] == 1



    def _obv_confirm_sell(self, df: pd.DataFrame) -> pd.Series:

        return df["obv_rising"] == 0



    # --- GATE: Trend Strength Gate ---

    def _trend_strength_gate_buy(self, df: pd.DataFrame) -> pd.Series:
        """Trend composite above TREND_GATE AND R2 above R2_MIN (clean uptrend)."""
        return (df["trend_composite"] > self.TREND_GATE.value) & (df["linreg_r2"] > self.R2_MIN.value)



    def _trend_strength_gate_sell(self, df: pd.DataFrame) -> pd.Series:
        """Trend composite below -TREND_GATE AND R2 above R2_MIN (clean downtrend)."""
        return (df["trend_composite"] < -self.TREND_GATE.value) & (df["linreg_r2"] > self.R2_MIN.value)



    # --- GATE: KB Squeeze ---

    def _kb_squeeze_gate(self, df: pd.DataFrame) -> pd.Series:

        """Squeeze is active or just released (volatility compression/expansion)."""

        return df["kb_squeeze_on"] | df["kb_squeeze_off"]



    # --- GATE: Volume Ratio ---

    def _volume_ratio_gate(self, df: pd.DataFrame) -> pd.Series:

        return df["volume_ratio"] >= self.VOL_RATIO_P.value



    # ------------------------------------------------------------------ #

    #  COMBO M SIGNAL                                                      #

    # ------------------------------------------------------------------ #

    def _combo_m_signal(self, df: pd.DataFrame, long: bool) -> pd.Series:

        """

        Combo M signal logic:

          Primary (>= 1 must fire): trend_composite | supertrend_flip | sr_breakout

          Confirm (>= 1 needed):    adx_di | ema_ribbon | macd_cross | obv_confirm

          Gate    (ALL must pass):  trend_strength_gate & kb_squeeze & volume_ratio_gate

        """

        if long:

            primary_signals = [

                self._trend_composite_buy(df),

                self._supertrend_flip_buy(df),

                self._sr_breakout_buy(df),

            ]

            primary_count = reduce(lambda a, b: a.astype(int) + b.astype(int), primary_signals)

            has_primary   = primary_count >= self.MIN_PRIMARY.value



            confirm_signals = [

                self._adx_di_buy(df),

                self._ema_ribbon_buy(df),

                self._macd_cross_buy(df),

                self._obv_confirm_buy(df),

            ]

            confirm_count = reduce(lambda a, b: a.astype(int) + b.astype(int), confirm_signals)

            has_confirm   = confirm_count >= 1



            gates = (

                self._trend_strength_gate_buy(df) &

                self._kb_squeeze_gate(df) &

                self._volume_ratio_gate(df)

            )

        else:

            primary_signals = [

                self._trend_composite_sell(df),

                self._supertrend_flip_sell(df),

                self._sr_breakout_sell(df),

            ]

            primary_count = reduce(lambda a, b: a.astype(int) + b.astype(int), primary_signals)

            has_primary   = primary_count >= self.MIN_PRIMARY.value



            confirm_signals = [

                self._adx_di_sell(df),

                self._ema_ribbon_sell(df),

                self._macd_cross_sell(df),

                self._obv_confirm_sell(df),

            ]

            confirm_count = reduce(lambda a, b: a.astype(int) + b.astype(int), confirm_signals)

            has_confirm   = confirm_count >= 1



            gates = (

                self._trend_strength_gate_sell(df) &

                self._kb_squeeze_gate(df) &

                self._volume_ratio_gate(df)

            )



        return has_primary & has_confirm & gates, confirm_count



    # ------------------------------------------------------------------ #

    #  ENTRY / EXIT TREND                                                  #

    # ------------------------------------------------------------------ #

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        long_sig,  long_cnt  = self._combo_m_signal(dataframe, long=True)

        short_sig, short_cnt = self._combo_m_signal(dataframe, long=False)



        long_tags  = "M_long_s"  + long_cnt.clip(1, 4).astype(int).astype(str)

        short_tags = "M_short_s" + short_cnt.clip(1, 4).astype(int).astype(str)



        dataframe.loc[long_sig,  "enter_long"]  = 1

        dataframe.loc[long_sig,  "enter_tag"]   = long_tags[long_sig]

        dataframe.loc[short_sig, "enter_short"] = 1

        dataframe.loc[short_sig, "enter_tag"]   = short_tags[short_sig]

        return dataframe
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe["exit_long"]  = 0

        dataframe["exit_short"] = 0

        return dataframe



    # ------------------------------------------------------------------ #

    #  ATR-BASED STOPLOSS  (SL_ATR_MULT default = 1.5)                    #

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  SIGNAL-STRENGTH POSITION SIZING                                     #
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
        Two-axis sizing: confirm-count (_s1.._s4) × direction (long/short mult).
        """
        factors_s = {
            1: self.STAKE_S1.value,
            2: self.STAKE_S2.value,
            3: self.STAKE_S3.value,
            4: self.STAKE_S4.value,
        }
        base_factor = 1.0
        if entry_tag and "_s" in entry_tag:
            try:
                strength = int(entry_tag.split("_s")[-1])
                base_factor = factors_s.get(min(strength, 4), 1.0)
            except (ValueError, IndexError):
                pass

        if side == "long" or (entry_tag and "_long_" in entry_tag):
            dir_factor = self.STAKE_LONG_MULT.value
        else:
            dir_factor = self.STAKE_SHORT_MULT.value

        stake = proposed_stake * base_factor * dir_factor
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



        last_atr = float(df["atr"].iat[-1])

        if last_atr <= 0 or current_rate <= 0:

            return self.stoploss



        sl_distance = self.SL_ATR_MULT.value * last_atr / current_rate

        return max(-sl_distance, -0.20)



    # ------------------------------------------------------------------ #

    #  ATR-BASED TAKE PROFIT  (TP_ATR_MULT default = 3.0)                 #

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



        last_atr = float(df["atr"].iat[-1])

        if last_atr <= 0 or trade.open_rate <= 0:

            return None



        tp_ratio = self.TP_ATR_MULT.value * last_atr / trade.open_rate

        if current_profit >= tp_ratio:

            return "TP_HIT"



        return None



    # ------------------------------------------------------------------ #

    #  CONFIRM ENTRY                                                       #

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

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if df.empty:

            return False



        last = df.iloc[-1]

        if float(last.get("atr", 0)) <= 0:

            return False

        if float(last.get("volume_ratio", 0)) < self.VOL_RATIO_MIN:

            return False



        # BTC market sentiment gate (TrendRiderStrategy technique)
        btc_rsi = float(last.get("btc_rsi_1h", 50))
        if side == "long" and btc_rsi < 35:
            return False
        if side == "short" and btc_rsi > 65:
            return False

        return True
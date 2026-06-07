"""Coin Scanner: Find pairs that match our strategy characteristics.

Analyzes available 15m data for fitness with:
1. regime_adaptive: Clear trending/ranging regimes, ADX cycles, EMA cross quality
2. volume_spike_rev: Volume spike frequency, reversal candle quality
3. cb_adx_breakout: Compression frequency, breakout quality after squeeze

Also checks: liquidity, spread, correlation with existing pairs, funding cost.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"

ALL_PAIRS = [
    "BTC_USDT_USDT",
    "ETH_USDT_USDT",
    "SOL_USDT_USDT",
    "SPX_USDT_USDT",
    "NVDA_USDT_USDT",
    "XAU_USDT_USDT",
]


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp)
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


def compute_indicators(df):
    """Compute all indicators needed for strategy fitness scoring."""
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"].astype(float)
    o = df["open"]

    # Trend indicators
    df["ema21"] = ta.ema(c, length=21)
    df["ema60"] = ta.ema(c, length=60)
    df["ema200"] = ta.ema(c, length=200)

    # ATR
    df["atr"] = ta.atr(h, lo, c, length=14)
    df["atr_ma"] = ta.ema(df["atr"], length=50)
    df["atr_ratio"] = df["atr"] / (df["atr_ma"] + 1e-10)
    df["atr_pct"] = df["atr"] / (c + 1e-10)

    # ADX
    adx = ta.adx(h, lo, c, length=14)
    if adx is not None:
        df["adx"] = adx.iloc[:, 0]
        df["plus_di"] = adx.iloc[:, 1]
        df["minus_di"] = adx.iloc[:, 2]
    else:
        df["adx"] = df["plus_di"] = df["minus_di"] = 0

    # RSI
    df["rsi"] = ta.rsi(c, length=14)

    # Bollinger Bands
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        df["bb_upper"] = bb.iloc[:, 0]
        df["bb_mid"] = bb.iloc[:, 1]
        df["bb_lower"] = bb.iloc[:, 2]
        df["bb_width"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / (bb.iloc[:, 1] + 1e-10)
    else:
        df["bb_width"] = 0

    # Volume
    df["vol_ema"] = ta.ema(v, length=20)
    df["vol_ratio"] = v / (df["vol_ema"] + 1e-10)

    # MACD
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd_hist"] = macd.iloc[:, 1]
    else:
        df["macd_hist"] = 0

    # SuperTrend
    try:
        st = ta.supertrend(h, lo, c, length=7, multiplier=3.0)
        if st is not None:
            st_dir_col = next((col for col in st.columns if "SUPERTd" in col), None)
            df["st_dir"] = st[st_dir_col].fillna(0) if st_dir_col else 0
        else:
            df["st_dir"] = 0
    except Exception:
        df["st_dir"] = 0

    # Candle metrics
    df["body_pct"] = abs(c - o) / (c + 1e-10)
    df["range_pct"] = (h - lo) / (c + 1e-10)
    df["upper_shadow"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (h - lo + 1e-10)
    df["lower_shadow"] = (pd.concat([c, o], axis=1).min(axis=1) - lo) / (h - lo + 1e-10)

    return df


def score_regime_adaptive(df):
    """Score pair fitness for regime_adaptive strategy.

    Good fit = clear regime cycles, clean EMA crosses, sustained trends.
    """
    scores = {}

    # 1. ADX cycle clarity: % time in trending (>25) vs ranging (<20)
    adx = df["adx"].dropna()
    pct_trending = (adx > 25).mean()
    pct_ranging = (adx < 20).mean()
    scores["pct_trending"] = pct_trending
    scores["pct_ranging"] = pct_ranging
    # Best: 40-60% trending, clear cycle
    regime_balance = 1 - abs(pct_trending - 0.45)
    scores["regime_balance"] = regime_balance

    # 2. EMA cross quality: how many crosses, avg profit after cross
    ema_fast = df["ema21"].values
    ema_slow = df["ema60"].values
    close = df["close"].values
    crosses_up = 0
    crosses_down = 0
    cross_profits = []

    for i in range(1, len(df) - 48):
        if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
            crosses_up += 1
            fwd = (close[min(i+48, len(close)-1)] - close[i]) / close[i]
            cross_profits.append(fwd)
        elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
            crosses_down += 1
            fwd = (close[i] - close[min(i+48, len(close)-1)]) / close[i]
            cross_profits.append(fwd)

    total_crosses = crosses_up + crosses_down
    scores["crosses_per_month"] = total_crosses / (len(df) / 96 / 30)
    scores["avg_cross_profit"] = np.mean(cross_profits) if cross_profits else 0

    # 3. Trend follow-through: when ADX > 25, avg bars until reversal
    in_trend = adx > 25
    trend_lengths = []
    current = 0
    for val in in_trend:
        if val:
            current += 1
        elif current > 0:
            trend_lengths.append(current)
            current = 0
    scores["avg_trend_length"] = np.mean(trend_lengths) if trend_lengths else 0

    # 4. Noise ratio: avg range vs ATR
    scores["noise_ratio"] = df["range_pct"].mean() / (df["atr_pct"].mean() + 1e-10)

    # Final score (0-100)
    score = 0
    score += min(regime_balance * 30, 30)  # max 30
    score += min(scores["avg_cross_profit"] * 500, 25)  # max 25
    score += min(scores["avg_trend_length"] / 20 * 20, 20)  # max 20
    score += 15 if 3 <= scores["crosses_per_month"] <= 8 else 5  # max 15
    score += 10 if scores["noise_ratio"] < 1.2 else 0  # max 10

    scores["total_score"] = min(score, 100)
    return scores


def score_volume_spike_rev(df):
    """Score pair fitness for volume_spike_rev strategy.

    Good fit = frequent volume spikes with clear reversal candles.
    """
    scores = {}

    # 1. Volume spike frequency (vol > 2x EMA)
    vol_spikes = df["vol_ratio"] > 2.0
    scores["spike_freq_per_day"] = vol_spikes.sum() / (len(df) / 96)

    # 2. Reversal candle quality at spikes
    spike_idx = df[vol_spikes].index
    reversals = 0
    reversal_profits = []

    for idx in spike_idx:
        if idx + 24 >= len(df):
            continue
        row = df.iloc[idx]
        c = float(row["close"])
        o = float(row["open"])
        h = float(row["high"])
        lo_val = float(row["low"])
        rsi = float(row.get("rsi", 50))

        body = abs(c - o)
        full_range = h - lo_val
        if full_range <= 0:
            continue

        shadow_ratio = (full_range - body) / (body + 1e-10)

        # Hammer (long reversal) or shooting star (short reversal)
        if shadow_ratio > 2.0:
            reversals += 1
            # Check 24-bar forward return
            fwd_price = float(df.iloc[min(idx + 24, len(df) - 1)]["close"])
            if c > o:  # bullish candle
                pnl = (fwd_price - c) / c
            else:  # bearish candle
                pnl = (c - fwd_price) / c
            reversal_profits.append(pnl)

    scores["reversal_freq_per_day"] = reversals / (len(df) / 96)
    scores["avg_reversal_profit"] = np.mean(reversal_profits) if reversal_profits else 0
    scores["reversal_wr"] = (sum(1 for p in reversal_profits if p > 0) / len(reversal_profits) * 100) if reversal_profits else 0

    # 3. RSI extreme frequency
    rsi = df["rsi"].dropna()
    scores["rsi_extreme_freq"] = ((rsi < 30) | (rsi > 70)).mean()

    # Final score
    score = 0
    score += min(scores["spike_freq_per_day"] * 5, 25)  # max 25
    score += min(scores["reversal_freq_per_day"] * 10, 25)  # max 25
    score += min(scores["avg_reversal_profit"] * 300, 25)  # max 25
    score += min(scores["reversal_wr"] / 4, 25)  # max 25

    scores["total_score"] = min(score, 100)
    return scores


def score_cb_adx_breakout(df):
    """Score pair fitness for cb_adx_breakout strategy.

    Good fit = frequent compression periods + clean breakouts when ADX rises.
    """
    scores = {}

    # 1. Compression frequency (3-bar range < 0.8x ATR)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values

    compressions = 0
    breakout_profits = []

    for i in range(3, len(df) - 24):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        range_3 = max(high[i-2:i+1]) - min(low[i-2:i+1])
        if range_3 < 0.8 * atr[i]:
            adx_val = float(df.iloc[i].get("adx", 30))
            if adx_val < 22:
                compressions += 1

                # Check breakout quality
                fwd_12 = (close[min(i+12, len(close)-1)] - close[i]) / close[i]
                breakout_profits.append(abs(fwd_12))

    scores["compression_per_day"] = compressions / (len(df) / 96)
    scores["avg_breakout_move"] = np.mean(breakout_profits) if breakout_profits else 0

    # 2. ADX recovery speed (bars from <20 to >25)
    adx = df["adx"].values
    recovery_speeds = []
    i = 0
    while i < len(adx) - 1:
        if adx[i] < 20:
            for j in range(i + 1, min(i + 48, len(adx))):
                if adx[j] > 25:
                    recovery_speeds.append(j - i)
                    i = j
                    break
        i += 1

    scores["avg_recovery_bars"] = np.mean(recovery_speeds) if recovery_speeds else 48
    scores["recovery_freq_per_month"] = len(recovery_speeds) / (len(df) / 96 / 30)

    # 3. BB width compression-expansion cycles
    bb_width = df["bb_width"].dropna()
    if len(bb_width) > 0:
        bb_narrow = bb_width < bb_width.quantile(0.25)
        scores["bb_squeeze_pct"] = bb_narrow.mean()
    else:
        scores["bb_squeeze_pct"] = 0

    # Final score
    score = 0
    score += min(scores["compression_per_day"] * 3, 30)  # max 30
    score += min(scores["avg_breakout_move"] * 1000, 25)  # max 25
    score += 20 if scores["avg_recovery_bars"] < 15 else (10 if scores["avg_recovery_bars"] < 25 else 0)  # max 20
    score += min(scores["recovery_freq_per_month"] * 3, 15)  # max 15
    score += min(scores["bb_squeeze_pct"] * 40, 10)  # max 10

    scores["total_score"] = min(score, 100)
    return scores


def compute_correlation(pair_returns):
    """Compute correlation matrix between pair returns."""
    returns_df = pd.DataFrame(pair_returns)
    return returns_df.corr()


def compute_liquidity_metrics(df, pair):
    """Estimate liquidity from available data."""
    metrics = {}

    # Average volume in USD
    avg_vol = (df["volume"] * df["close"]).mean()
    metrics["avg_daily_volume_usd"] = avg_vol * 96  # 96 bars per day on 15m

    # Spread proxy: high-low / close as fraction
    metrics["avg_spread_proxy"] = ((df["high"] - df["low"]) / df["close"]).median()

    # Volume consistency (coefficient of variation)
    vol = df["volume"]
    metrics["vol_cv"] = vol.std() / (vol.mean() + 1e-10)

    return metrics


def main():
    print("=" * 90)
    print("COIN SCANNER: Strategy Fitness Analysis")
    print("Finding pairs that match regime_adaptive / volume_spike_rev / cb_adx_breakout")
    print("=" * 90)

    pair_data = {}
    pair_returns = {}

    for pair in ALL_PAIRS:
        df = load_15m(pair)
        if df.empty or len(df) < 1000:
            print(f"  {pair}: insufficient data, skipping")
            continue

        df = compute_indicators(df)
        pair_data[pair] = df

        # Daily returns for correlation
        daily = df.set_index("date")["close"].resample("1D").last().pct_change().dropna()
        pair_returns[pair] = daily

    # Correlation matrix
    print(f"\n{'=' * 90}")
    print("CORRELATION MATRIX (daily returns)")
    print(f"{'=' * 90}")
    corr_df = pd.DataFrame(pair_returns).corr()
    print(f"\n{'':>16}", end="")
    for p in corr_df.columns:
        print(f"{p[:8]:>10}", end="")
    print()
    for p1 in corr_df.index:
        print(f"  {p1[:14]:<14}", end="")
        for p2 in corr_df.columns:
            val = corr_df.loc[p1, p2]
            print(f"{val:>10.3f}", end="")
        print()

    # Score each pair
    print(f"\n{'=' * 90}")
    print("STRATEGY FITNESS SCORES (0-100)")
    print(f"{'=' * 90}")

    results = []

    for pair in ALL_PAIRS:
        if pair not in pair_data:
            continue
        df = pair_data[pair]

        ra_scores = score_regime_adaptive(df)
        vsr_scores = score_volume_spike_rev(df)
        cb_scores = score_cb_adx_breakout(df)
        liq = compute_liquidity_metrics(df, pair)

        results.append({
            "pair": pair,
            "regime_adaptive": ra_scores["total_score"],
            "volume_spike_rev": vsr_scores["total_score"],
            "cb_adx_breakout": cb_scores["total_score"],
            "combined": (ra_scores["total_score"] + vsr_scores["total_score"] + cb_scores["total_score"]) / 3,
            "ra_details": ra_scores,
            "vsr_details": vsr_scores,
            "cb_details": cb_scores,
            "liquidity": liq,
        })

    # Sort by combined score
    results.sort(key=lambda x: x["combined"], reverse=True)

    print(f"\n  {'Pair':<18} {'Regime':<10} {'VolSpike':<10} {'CB_ADX':<10} {'Combined':<10} {'Daily Vol $':<14} {'Status'}")
    print(f"  {'-' * 90}")

    for r in results:
        pair = r["pair"]
        # Check if currently active
        active_pairs = ["SOL_USDT_USDT", "SPX_USDT_USDT"]
        status = "ACTIVE" if pair in active_pairs else "CANDIDATE"

        daily_vol = r["liquidity"]["avg_daily_volume_usd"]
        vol_str = f"${daily_vol/1e6:.1f}M" if daily_vol > 1e6 else f"${daily_vol/1e3:.0f}K"

        print(f"  {pair:<18} {r['regime_adaptive']:<10.1f} {r['volume_spike_rev']:<10.1f} "
              f"{r['cb_adx_breakout']:<10.1f} {r['combined']:<10.1f} {vol_str:<14} {status}")

    # Detailed breakdown for each pair
    for r in results:
        pair = r["pair"]
        print(f"\n  {'=' * 70}")
        print(f"  {pair} — Detailed Analysis")
        print(f"  {'=' * 70}")

        ra = r["ra_details"]
        print(f"\n  regime_adaptive (score={ra['total_score']:.1f}/100):")
        print(f"    Trending: {ra['pct_trending']*100:.1f}% | Ranging: {ra['pct_ranging']*100:.1f}%")
        print(f"    EMA crosses/month: {ra['crosses_per_month']:.1f}")
        print(f"    Avg cross profit (48 bars): {ra['avg_cross_profit']*100:.3f}%")
        print(f"    Avg trend length: {ra['avg_trend_length']:.1f} bars")
        print(f"    Noise ratio: {ra['noise_ratio']:.2f}")

        vsr = r["vsr_details"]
        print(f"\n  volume_spike_rev (score={vsr['total_score']:.1f}/100):")
        print(f"    Vol spikes/day (>2x): {vsr['spike_freq_per_day']:.1f}")
        print(f"    Reversals/day: {vsr['reversal_freq_per_day']:.1f}")
        print(f"    Reversal WR: {vsr['reversal_wr']:.1f}%")
        print(f"    Avg reversal profit: {vsr['avg_reversal_profit']*100:.3f}%")
        print(f"    RSI extreme freq: {vsr['rsi_extreme_freq']*100:.1f}%")

        cb = r["cb_details"]
        print(f"\n  cb_adx_breakout (score={cb['total_score']:.1f}/100):")
        print(f"    Compressions/day: {cb['compression_per_day']:.1f}")
        print(f"    Avg breakout move: {cb['avg_breakout_move']*100:.3f}%")
        print(f"    ADX recovery bars: {cb['avg_recovery_bars']:.1f}")
        print(f"    Recovery freq/month: {cb['recovery_freq_per_month']:.1f}")
        print(f"    BB squeeze %: {cb['bb_squeeze_pct']*100:.1f}%")

        liq = r["liquidity"]
        daily_vol = liq["avg_daily_volume_usd"]
        print(f"\n  Liquidity:")
        print(f"    Daily volume: ${daily_vol/1e6:.2f}M")
        print(f"    Spread proxy: {liq['avg_spread_proxy']*100:.3f}%")
        print(f"    Volume CV: {liq['vol_cv']:.2f}")

    # Recommendations
    print(f"\n{'=' * 90}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 90}")

    # Pairs ranked by strategy
    print("\n  Best pairs per strategy:")
    for strat in ["regime_adaptive", "volume_spike_rev", "cb_adx_breakout"]:
        ranked = sorted(results, key=lambda x: x[strat], reverse=True)
        top = ranked[:3]
        print(f"\n  {strat}:")
        for i, r in enumerate(top, 1):
            print(f"    {i}. {r['pair']:<18} score={r[strat]:.1f}")

    # Low correlation candidates
    print("\n  Low-correlation candidates (diversification):")
    for pair in ALL_PAIRS:
        if pair in ["SOL_USDT_USDT", "SPX_USDT_USDT"] or pair not in corr_df.index:
            continue
        max_corr_with_active = max(
            abs(corr_df.loc[pair, "SOL_USDT_USDT"]) if "SOL_USDT_USDT" in corr_df.columns else 0,
            abs(corr_df.loc[pair, "SPX_USDT_USDT"]) if "SPX_USDT_USDT" in corr_df.columns else 0,
        )
        print(f"    {pair:<18} max corr with active: {max_corr_with_active:.3f} "
              f"{'[LOW - good diversifier]' if max_corr_with_active < 0.4 else ''}")

    # Additional pairs to download
    print(f"\n  {'=' * 70}")
    print("  ADDITIONAL PAIRS TO CONSIDER (not yet downloaded)")
    print(f"  {'=' * 70}")
    print("""
  OKX Perpetual Futures - potential candidates:

  HIGH LIQUIDITY (>$500M daily):
    - DOGE/USDT:USDT  — meme coin, high volatility regime cycles
    - XRP/USDT:USDT   — ranging/breakout patterns, low corr with SOL
    - AVAX/USDT:USDT  — DeFi, strong trend cycles
    - LINK/USDT:USDT  — oracle narrative, volume spikes on news

  MEDIUM LIQUIDITY ($50-500M daily):
    - SUI/USDT:USDT   — new L1, high ATR, clear regimes
    - PEPE/USDT:USDT  — extreme vol spikes, mean-reversion
    - WIF/USDT:USDT   — meme, regime cycles
    - ARB/USDT:USDT   — L2, trending patterns
    - OP/USDT:USDT    — L2, moderate correlation with ETH

  STOCK TOKENS (low corr with crypto):
    - AAPL/USDT:USDT  — tech stock, different regime from crypto
    - TSLA/USDT:USDT  — high volatility, clear trends
    - COIN/USDT:USDT  — crypto-correlated but equity patterns
    - AMZN/USDT:USDT  — tech stock, diverse exposure

  COMMODITIES (lowest corr):
    - XAU/USDT:USDT   — ALREADY HAVE DATA (analyze above)

  Recommended download command:
    python ft_run.py download-data --exchange okx \\
      --pairs DOGE/USDT:USDT XRP/USDT:USDT AVAX/USDT:USDT LINK/USDT:USDT \\
              SUI/USDT:USDT ARB/USDT:USDT TSLA/USDT:USDT AAPL/USDT:USDT \\
      -t 15m 1h --days 180 --trading-mode futures
    """)


if __name__ == "__main__":
    main()

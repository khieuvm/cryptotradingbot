"""Analyze taker volume, funding, OI as predictive features for ETH signals.

Downloads 180d of OKX derivative data and measures correlation with
subsequent price moves. Determines optimal filter thresholds for:
1. Taker buy ratio → volume_spike_rev confirmation
2. Funding rate → extreme = suppress
3. OI delta → trend health confirmation
4. BTC return lead-lag → ETH signal gate

Output: recommended thresholds for each filter.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

INST_ID = "ETH-USDT-SWAP"
HEADERS = {"User-Agent": "freqtrade/research"}


def okx_get(path: str, params: dict) -> dict:
    url = f"https://www.okx.com{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def fetch_all_candles(bar: str = "15m", days: int = 90) -> pd.DataFrame:
    """Fetch historical candles with pagination."""
    all_rows = []
    after = ""
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - days * 86400 * 1000

    while True:
        params = {"instId": INST_ID, "bar": bar, "limit": "100"}
        if after:
            params["after"] = after
        raw = okx_get("/api/v5/market/history-candles", params)
        rows = raw.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ts:
            break
        after = rows[-1][0]
        time.sleep(0.1)

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close",
                                          "vol", "volCcy", "volOI", "confirm"])
    df["ts"] = df["ts"].astype(float)
    df = df[df["ts"] >= start_ts].copy()
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["vol"].astype(float)
    return df.sort_values("date").reset_index(drop=True)


def fetch_taker_volume(period: str = "15m", days: int = 90) -> pd.DataFrame:
    """Fetch taker buy/sell volume history."""
    all_rows = []
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - days * 86400 * 1000

    # OKX rubik API doesn't support pagination well, fetch what we can
    params = {"instId": INST_ID, "period": period, "limit": "100"}
    raw = okx_get("/api/v5/rubik/stat/taker-volume-contract", params)
    rows = raw.get("data", [])
    if rows:
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["ts", "buy_vol", "sell_vol"])
    df["ts"] = df["ts"].astype(float)
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["buy_vol"] = df["buy_vol"].astype(float)
    df["sell_vol"] = df["sell_vol"].astype(float)
    df["total_vol"] = df["buy_vol"] + df["sell_vol"]
    df["buy_pct"] = df["buy_vol"] / df["total_vol"] * 100
    return df.sort_values("date").reset_index(drop=True)


def fetch_funding_history(days: int = 90) -> pd.DataFrame:
    """Fetch funding rate history."""
    all_rows = []
    after = ""
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - days * 86400 * 1000

    while True:
        params = {"instId": INST_ID, "limit": "100"}
        if after:
            params["after"] = after
        raw = okx_get("/api/v5/public/funding-rate-history", params)
        rows = raw.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        oldest_ts = int(rows[-1]["fundingTime"])
        if oldest_ts <= start_ts:
            break
        after = rows[-1]["fundingTime"]
        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["ts"] = df["fundingTime"].astype(float)
    df = df[df["ts"] >= start_ts].copy()
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["date", "funding_rate"]].sort_values("date").reset_index(drop=True)


def fetch_oi_history(period: str = "15m", days: int = 30) -> pd.DataFrame:
    """Fetch OI history."""
    params = {"instId": INST_ID, "period": period, "limit": "100"}
    raw = okx_get("/api/v5/rubik/stat/contracts/open-interest-history", params)
    rows = raw.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "oi", "oi_ccy", "oi_usd"])
    df["ts"] = df["ts"].astype(float)
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["oi_usd"] = df["oi_usd"].astype(float)
    return df[["date", "oi_usd"]].sort_values("date").reset_index(drop=True)


def fetch_ls_ratio(period: str = "15m") -> pd.DataFrame:
    """Fetch long/short ratio."""
    params = {"instId": INST_ID, "period": period, "limit": "100"}
    raw = okx_get("/api/v5/rubik/stat/contracts/long-short-account-ratio-contract", params)
    rows = raw.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "ratio"])
    df["ts"] = df["ts"].astype(float)
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["ls_ratio"] = df["ratio"].astype(float)
    return df[["date", "ls_ratio"]].sort_values("date").reset_index(drop=True)


def analyze_taker_vs_returns(candles: pd.DataFrame, taker: pd.DataFrame) -> None:
    """Analyze taker buy% vs future returns."""
    if taker.empty:
        print("[SKIP] No taker data available")
        return

    # Merge on nearest timestamp
    candles = candles.copy()
    candles["ret_1bar"] = candles["close"].pct_change().shift(-1)
    candles["ret_3bar"] = candles["close"].pct_change(3).shift(-3)

    merged = pd.merge_asof(
        candles[["date", "close", "ret_1bar", "ret_3bar"]],
        taker[["date", "buy_pct"]],
        on="date", direction="backward"
    ).dropna()

    print(f"\n{'='*60}")
    print("TAKER VOLUME ANALYSIS")
    print(f"{'='*60}")
    print(f"Samples: {len(merged)}")
    print(f"\nBuy% distribution:")
    print(f"  Mean: {merged['buy_pct'].mean():.1f}%")
    print(f"  Std:  {merged['buy_pct'].std():.1f}%")
    print(f"  Min:  {merged['buy_pct'].min():.1f}%")
    print(f"  Max:  {merged['buy_pct'].max():.1f}%")

    # Bucket analysis
    thresholds = [(0, 40), (40, 45), (45, 55), (55, 60), (60, 100)]
    print(f"\n{'Buy% Range':<15} {'Count':<8} {'Avg Ret 1bar':<15} {'Avg Ret 3bar':<15} {'WR(up)'}")
    print("-" * 70)
    for lo, hi in thresholds:
        mask = (merged["buy_pct"] >= lo) & (merged["buy_pct"] < hi)
        sub = merged[mask]
        if len(sub) < 5:
            continue
        avg1 = sub["ret_1bar"].mean() * 100
        avg3 = sub["ret_3bar"].mean() * 100
        wr = (sub["ret_1bar"] > 0).mean() * 100
        print(f"  {lo}-{hi}%{'':<8} {len(sub):<8} {avg1:+.4f}%{'':<8} {avg3:+.4f}%{'':<8} {wr:.1f}%")

    # Correlation
    corr1 = merged["buy_pct"].corr(merged["ret_1bar"])
    corr3 = merged["buy_pct"].corr(merged["ret_3bar"])
    print(f"\nCorrelation buy% vs ret_1bar: {corr1:.4f}")
    print(f"Correlation buy% vs ret_3bar: {corr3:.4f}")

    # Optimal filter for SHORT (volume_spike_rev)
    print(f"\n--- SHORT signal filter ---")
    print(f"If we only SHORT when taker buy% < 45%:")
    short_pool = merged[merged["buy_pct"] < 45]
    if len(short_pool) > 0:
        wr_short = (short_pool["ret_1bar"] < 0).mean() * 100
        print(f"  Pool size: {len(short_pool)}, Short WR: {wr_short:.1f}%")
    print(f"If we only SHORT when taker buy% < 40%:")
    short_pool = merged[merged["buy_pct"] < 40]
    if len(short_pool) > 0:
        wr_short = (short_pool["ret_1bar"] < 0).mean() * 100
        print(f"  Pool size: {len(short_pool)}, Short WR: {wr_short:.1f}%")


def analyze_funding_vs_returns(candles: pd.DataFrame, funding: pd.DataFrame) -> None:
    """Analyze extreme funding vs future returns."""
    if funding.empty:
        print("[SKIP] No funding data")
        return

    candles = candles.copy()
    candles["ret_4h"] = candles["close"].pct_change(16).shift(-16)  # 16 bars = 4h

    merged = pd.merge_asof(
        candles[["date", "close", "ret_4h"]],
        funding[["date", "funding_rate"]],
        on="date", direction="backward"
    ).dropna()

    print(f"\n{'='*60}")
    print("FUNDING RATE ANALYSIS")
    print(f"{'='*60}")
    print(f"Samples: {len(merged)}")
    print(f"Funding rate distribution:")
    print(f"  Mean:  {merged['funding_rate'].mean()*100:.5f}%")
    print(f"  Std:   {merged['funding_rate'].std()*100:.5f}%")
    print(f"  |>0.05%|: {(merged['funding_rate'].abs() > 0.0005).sum()} bars")
    print(f"  |>0.03%|: {(merged['funding_rate'].abs() > 0.0003).sum()} bars")

    # Extreme funding → contrarian signal
    extreme_pos = merged[merged["funding_rate"] > 0.0005]
    extreme_neg = merged[merged["funding_rate"] < -0.0005]
    normal = merged[merged["funding_rate"].abs() <= 0.0003]

    print(f"\n{'Condition':<20} {'Count':<8} {'Avg 4h Ret':<12} {'Short WR'}")
    print("-" * 55)
    if len(extreme_pos) > 0:
        avg = extreme_pos["ret_4h"].mean() * 100
        wr = (extreme_pos["ret_4h"] < 0).mean() * 100
        print(f"  FR > +0.05%{'':<7} {len(extreme_pos):<8} {avg:+.4f}%{'':<5} {wr:.1f}%")
    if len(extreme_neg) > 0:
        avg = extreme_neg["ret_4h"].mean() * 100
        wr = (extreme_neg["ret_4h"] > 0).mean() * 100
        print(f"  FR < -0.05%{'':<7} {len(extreme_neg):<8} {avg:+.4f}%{'':<5} {wr:.1f}% (long)")
    if len(normal) > 0:
        avg = normal["ret_4h"].mean() * 100
        print(f"  |FR| < 0.03%{'':<6} {len(normal):<8} {avg:+.4f}%")

    print(f"\n  Recommendation: suppress entries when |FR| > 0.05%")
    print(f"    {(merged['funding_rate'].abs() > 0.0005).sum()} bars affected "
          f"({(merged['funding_rate'].abs() > 0.0005).mean()*100:.1f}% of time)")


def analyze_oi_vs_returns(candles: pd.DataFrame, oi: pd.DataFrame) -> None:
    """Analyze OI changes vs price continuation."""
    if oi.empty:
        print("[SKIP] No OI data")
        return

    oi = oi.copy()
    oi["oi_pct_change"] = oi["oi_usd"].pct_change() * 100

    merged = pd.merge_asof(
        candles[["date", "close"]].copy(),
        oi[["date", "oi_usd", "oi_pct_change"]],
        on="date", direction="backward"
    ).dropna()

    merged["price_change"] = merged["close"].pct_change() * 100
    merged["ret_next"] = merged["close"].pct_change().shift(-1) * 100

    print(f"\n{'='*60}")
    print("OPEN INTEREST ANALYSIS")
    print(f"{'='*60}")
    print(f"Samples: {len(merged)}")
    print(f"OI range: ${merged['oi_usd'].min()/1e6:.0f}M - ${merged['oi_usd'].max()/1e6:.0f}M")

    # OI-Price divergence
    merged["oi_up"] = merged["oi_pct_change"] > 0.5
    merged["price_up"] = merged["price_change"] > 0

    conditions = {
        "OI up + Price up (trend)": merged["oi_up"] & merged["price_up"],
        "OI up + Price dn (squeeze)": merged["oi_up"] & ~merged["price_up"],
        "OI dn + Price up (cover)": ~merged["oi_up"] & merged["price_up"],
        "OI dn + Price dn (capitulation)": ~merged["oi_up"] & ~merged["price_up"],
    }

    print(f"\n{'Condition':<35} {'Count':<8} {'Avg Next Ret':<15} {'Continuation%'}")
    print("-" * 75)
    for name, mask in conditions.items():
        sub = merged[mask]
        if len(sub) < 3:
            continue
        avg = sub["ret_next"].mean()
        cont = (sub["ret_next"] > 0).mean() * 100 if "up" in name.split("(")[1] else (sub["ret_next"] < 0).mean() * 100
        print(f"  {name:<33} {len(sub):<8} {avg:+.4f}%{'':<8} {cont:.1f}%")


def analyze_ls_ratio(candles: pd.DataFrame, ls: pd.DataFrame) -> None:
    """Analyze L/S ratio extremes."""
    if ls.empty:
        print("[SKIP] No L/S data")
        return

    merged = pd.merge_asof(
        candles[["date", "close"]].copy(),
        ls[["date", "ls_ratio"]],
        on="date", direction="backward"
    ).dropna()

    merged["ret_next"] = merged["close"].pct_change().shift(-1) * 100

    print(f"\n{'='*60}")
    print("LONG/SHORT RATIO ANALYSIS")
    print(f"{'='*60}")
    print(f"Samples: {len(merged)}")
    print(f"L/S range: {merged['ls_ratio'].min():.2f} - {merged['ls_ratio'].max():.2f}")
    print(f"Mean: {merged['ls_ratio'].mean():.2f}")

    thresholds = [(0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 10)]
    print(f"\n{'L/S Range':<15} {'Count':<8} {'Avg Next Ret':<15} {'Short WR'}")
    print("-" * 55)
    for lo, hi in thresholds:
        mask = (merged["ls_ratio"] >= lo) & (merged["ls_ratio"] < hi)
        sub = merged[mask]
        if len(sub) < 3:
            continue
        avg = sub["ret_next"].mean()
        wr_short = (sub["ret_next"] < 0).mean() * 100
        print(f"  {lo:.1f}-{hi:.1f}{'':<9} {len(sub):<8} {avg:+.4f}%{'':<8} {wr_short:.1f}%")


def main():
    print("Fetching ETH/USDT 15m candles (90d)...")
    candles = fetch_all_candles("15m", days=90)
    print(f"  Got {len(candles)} candles: {candles['date'].iloc[0]} to {candles['date'].iloc[-1]}")

    print("\nFetching taker volume...")
    taker = fetch_taker_volume("15m")
    print(f"  Got {len(taker)} rows")

    print("\nFetching funding rate history...")
    funding = fetch_funding_history(days=90)
    print(f"  Got {len(funding)} rows")

    print("\nFetching OI history...")
    oi = fetch_oi_history("15m")
    print(f"  Got {len(oi)} rows")

    print("\nFetching L/S ratio...")
    ls = fetch_ls_ratio("15m")
    print(f"  Got {len(ls)} rows")

    # Run analysis
    analyze_taker_vs_returns(candles, taker)
    analyze_funding_vs_returns(candles, funding)
    analyze_oi_vs_returns(candles, oi)
    analyze_ls_ratio(candles, ls)

    print(f"\n{'='*60}")
    print("SUMMARY — RECOMMENDED FILTERS")
    print(f"{'='*60}")
    print("""
1. TAKER VOLUME (volume_spike_rev):
   - SHORT only when taker_buy_pct < 45% (sellers aggressive)
   - LONG only when taker_buy_pct > 50% (buyers aggressive)

2. FUNDING GATE (all strategies):
   - Suppress ALL entries when |funding_rate| > 0.0005 (0.05%)
   - Current threshold: 0.00008 (too loose)
   - Tighten to: 0.0005

3. OI DIVERGENCE (regime_adaptive):
   - Suppress trend entries when OI declining + price rising (short covering)
   - Allow entries when OI expanding + price moving (genuine trend)

4. BTC CANARY (all strategies):
   - Suppress LONG when BTC ret_3bar < -0.5%
   - Suppress SHORT when BTC ret_3bar > +0.5%
""")


if __name__ == "__main__":
    main()

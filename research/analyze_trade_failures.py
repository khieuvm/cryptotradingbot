"""Analyze WHY trades fail at 0.58 threshold — find filterable patterns."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

DATA_DIR = Path("user_data/data/okx/futures")

FEATURE_COLS = [
    "body_pct", "range_pct", "upper_wick", "lower_wick",
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_8", "ret_13", "ret_21",
    "atr_5", "atr_14", "atr_ratio", "atr_pct", "bb_pos", "bb_width",
    "rsi_3", "rsi_7", "rsi_14", "rsi_delta", "stoch_k", "stoch_d",
    "ema8_dist", "ema21_dist", "ema50_dist", "ema_spread",
    "vol_ratio", "vol_ratio_3", "obv_slope", "buy_pressure",
    "macd_hist", "adx",
    "hour_sin", "hour_cos", "is_us_session", "is_asia_session",
    "tick_direction", "tick_run", "spread_proxy",
    "sweep_long", "sweep_short", "pullback_long", "pullback_short",
    "sweep_wick_depth", "roll_low_dist", "roll_high_dist",
    "ema8_touch_dist", "body_dir",
]


def compute_features(df):
    c, h, lo, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"].astype(float)
    df["body_pct"] = (c - o) / (c + 1e-10)
    df["range_pct"] = (h - lo) / (c + 1e-10)
    df["upper_wick"] = (h - np.maximum(c, o)) / (h - lo + 1e-10)
    df["lower_wick"] = (np.minimum(c, o) - lo) / (h - lo + 1e-10)
    for lb in [1, 2, 3, 5, 8, 13, 21]:
        df[f"ret_{lb}"] = c.pct_change(lb)
    tr = np.maximum(h - lo, np.maximum(abs(h - c.shift(1)), abs(lo - c.shift(1))))
    df["atr_5"] = tr.rolling(5).mean()
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr_5"] / (df["atr_14"] + 1e-10)
    df["atr_pct"] = df["atr_14"] / (c + 1e-10)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_pos"] = (c - sma20) / (2 * std20 + 1e-10)
    df["bb_width"] = (4 * std20) / (sma20 + 1e-10)
    for p in [3, 7, 14]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        df[f"rsi_{p}"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df["rsi_delta"] = df["rsi_7"] - df["rsi_7"].shift(3)
    low14, high14 = lo.rolling(14).min(), h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    for p in [8, 21, 50]:
        ema = c.ewm(span=p, adjust=False).mean()
        df[f"ema{p}_dist"] = (c - ema) / (c + 1e-10)
    df["ema_spread"] = df["ema8_dist"] - df["ema50_dist"]
    vol_ema = v.rolling(20).mean()
    df["vol_ratio"] = v / (vol_ema + 1e-10)
    df["vol_ratio_3"] = v.rolling(3).mean() / (vol_ema + 1e-10)
    obv = (np.sign(c.diff()) * v).cumsum()
    df["obv_slope"] = obv.diff(5) / (c * 5 + 1e-10)
    df["buy_pressure"] = (c - lo) / (h - lo + 1e-10)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (c + 1e-10)
    plus_dm = (h.diff()).clip(lower=0)
    minus_dm = (-lo.diff()).clip(lower=0)
    atr_adx = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr_adx + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr_adx + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df["adx"] = dx.rolling(14).mean()
    df["hour_sin"] = np.sin(2 * np.pi * df["date"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["date"].dt.hour / 24)
    df["is_us_session"] = ((df["date"].dt.hour >= 13) & (df["date"].dt.hour <= 21)).astype(float)
    df["is_asia_session"] = ((df["date"].dt.hour >= 0) & (df["date"].dt.hour <= 8)).astype(float)
    df["tick_direction"] = np.sign(c.diff())
    ticks = df["tick_direction"].values
    runs = np.zeros(len(df))
    for i in range(1, len(ticks)):
        if ticks[i] == ticks[i-1] and ticks[i] != 0:
            runs[i] = runs[i-1] + 1
    df["tick_run"] = runs
    df["spread_proxy"] = (h - lo) / (c + 1e-10)

    # Pattern features
    ema8 = c.ewm(span=8, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    roll_hi_5 = h.rolling(5).max().shift(1)
    roll_lo_5 = lo.rolling(5).min().shift(1)
    l_wick = np.minimum(c, o) - lo
    u_wick = h - np.maximum(c, o)
    atr14 = df["atr_14"]
    df["sweep_long"] = ((lo < roll_lo_5) & (c > roll_lo_5) & (l_wick > 0.3 * atr14) & (l_wick > 1.5 * u_wick)).astype(float)
    df["sweep_short"] = ((h > roll_hi_5) & (c < roll_hi_5) & (u_wick > 0.3 * atr14) & (u_wick > 1.5 * l_wick)).astype(float)
    body_dir_val = np.sign(c - o)
    df["pullback_long"] = ((ema8 > ema21) & (lo <= ema8 * 1.002) & (c > ema8) & (body_dir_val == 1)).astype(float)
    df["pullback_short"] = ((ema8 < ema21) & (h >= ema8 * 0.998) & (c < ema8) & (body_dir_val == -1)).astype(float)
    df["sweep_wick_depth"] = np.maximum(l_wick, u_wick) / (atr14 + 1e-10)
    df["roll_low_dist"] = (c - roll_lo_5) / (atr14 + 1e-10)
    df["roll_high_dist"] = (roll_hi_5 - c) / (atr14 + 1e-10)
    df["ema8_touch_dist"] = (c - ema8) / (atr14 + 1e-10)
    df["body_dir"] = body_dir_val
    return df


def simulate_trades(test_df, proba, threshold=0.58, sl_mult=1.5, tp_mult=2.0, max_bars=15):
    """Simulate trades with SL/TP path dependency."""
    fee = 0.001
    c_arr = test_df["close"].values
    h_arr = test_df["high"].values
    l_arr = test_df["low"].values
    atr_arr = test_df["atr_14"].values

    trades = []
    for i in range(len(test_df) - max_bars):
        if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
            continue

        entry_price = c_arr[i]
        atr = atr_arr[i]

        if proba[i] > threshold:
            direction = "long"
            sl_price = entry_price - sl_mult * atr
            tp_price = entry_price + tp_mult * atr
        elif proba[i] < (1 - threshold):
            direction = "short"
            sl_price = entry_price + sl_mult * atr
            tp_price = entry_price - tp_mult * atr
        else:
            continue

        # Simulate forward
        exit_reason = "time_cut"
        exit_bar = min(i + max_bars, len(c_arr) - 1)
        exit_price = c_arr[exit_bar]

        # Track MAE (max adverse excursion)
        mae = 0.0
        for j in range(1, min(max_bars + 1, len(c_arr) - i)):
            if direction == "long":
                adverse = (l_arr[i+j] - entry_price) / entry_price
                mae = min(mae, adverse)
                if l_arr[i+j] <= sl_price:
                    exit_price = sl_price
                    exit_reason = "SL"
                    break
                if h_arr[i+j] >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    break
            else:
                adverse = (entry_price - h_arr[i+j]) / entry_price
                mae = min(mae, adverse)
                if h_arr[i+j] >= sl_price:
                    exit_price = sl_price
                    exit_reason = "SL"
                    break
                if l_arr[i+j] <= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    break

        if direction == "long":
            pnl = (exit_price - entry_price) / entry_price - fee
        else:
            pnl = (entry_price - exit_price) / entry_price - fee

        trades.append({
            "idx": i,
            "date": test_df["date"].iloc[i],
            "direction": direction,
            "prob": proba[i] if direction == "long" else 1 - proba[i],
            "pnl": pnl,
            "mae": mae,
            "exit_reason": exit_reason,
            "atr_pct": atr / entry_price,
            "adx": test_df["adx"].iloc[i],
            "rsi_14": test_df["rsi_14"].iloc[i],
            "vol_ratio": test_df["vol_ratio"].iloc[i],
            "bb_pos": test_df["bb_pos"].iloc[i],
            "bb_width": test_df["bb_width"].iloc[i],
            "atr_ratio": test_df["atr_ratio"].iloc[i],
            "hour": test_df["date"].iloc[i].hour,
            "ema_spread": test_df["ema_spread"].iloc[i],
            "body_pct": test_df["body_pct"].iloc[i],
            "range_pct": test_df["range_pct"].iloc[i],
            "ret_1": test_df["ret_1"].iloc[i],
            "ret_3": test_df["ret_3"].iloc[i],
            "sweep_long": test_df["sweep_long"].iloc[i],
            "sweep_short": test_df["sweep_short"].iloc[i],
            "pullback_long": test_df["pullback_long"].iloc[i],
            "pullback_short": test_df["pullback_short"].iloc[i],
        })

    return pd.DataFrame(trades)


def main():
    print("=" * 70)
    print("TRADE FAILURE ANALYSIS @ 0.58 THRESHOLD")
    print("=" * 70)

    df = pd.read_feather(DATA_DIR / "SOL_USDT_USDT-3m-futures.feather")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    df = compute_features(df)

    model = joblib.load("models/ml_scalping_sol_3m_model.pkl")

    # OOS period
    train_end = 14400
    test_df = df.iloc[train_end:].copy().reset_index(drop=True)
    print(f"OOS: {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} bars)")

    X_test = test_df[FEATURE_COLS].values
    valid = np.all(np.isfinite(X_test), axis=1)
    test_df = test_df[valid].reset_index(drop=True)
    X_test = test_df[FEATURE_COLS].values
    proba = model.predict_proba(X_test)[:, 1]

    tdf = simulate_trades(test_df, proba, threshold=0.58)
    print(f"\nTotal trades: {len(tdf)}")
    print(f"Winners: {(tdf['pnl']>0).sum()} ({(tdf['pnl']>0).mean()*100:.1f}%)")
    print(f"Losers: {(tdf['pnl']<=0).sum()} ({(tdf['pnl']<=0).mean()*100:.1f}%)")
    print(f"Total PnL: {tdf['pnl'].sum()*100:.2f}%")
    print(f"Avg win: +{tdf[tdf['pnl']>0]['pnl'].mean()*100:.3f}%")
    print(f"Avg loss: {tdf[tdf['pnl']<=0]['pnl'].mean()*100:.3f}%")

    print(f"\n{'='*70}")
    print("EXIT REASON BREAKDOWN")
    print(f"{'='*70}")
    for reason in ["TP", "SL", "time_cut"]:
        sub = tdf[tdf["exit_reason"] == reason]
        if len(sub) > 0:
            wr = (sub["pnl"] > 0).mean() * 100
            print(f"  {reason:10s}: {len(sub):4d} trades ({len(sub)/len(tdf)*100:.0f}%), "
                  f"WR={wr:.0f}%, avg PnL={sub['pnl'].mean()*100:.3f}%")

    print(f"\n{'='*70}")
    print("WINNERS vs LOSERS: KEY FEATURES")
    print(f"{'='*70}")
    winners = tdf[tdf["pnl"] > 0]
    losers = tdf[tdf["pnl"] <= 0]

    compare_cols = ["prob", "adx", "atr_pct", "atr_ratio", "vol_ratio",
                    "bb_width", "rsi_14", "bb_pos", "ema_spread",
                    "range_pct", "body_pct", "ret_1", "ret_3", "mae"]
    print(f"  {'Feature':15s} {'Winners':>10s} {'Losers':>10s} {'Diff':>10s} {'Signal?':>8s}")
    for col in compare_cols:
        w_mean = winners[col].mean()
        l_mean = losers[col].mean()
        diff = w_mean - l_mean
        # Simple t-test significance
        from scipy import stats
        if len(winners[col].dropna()) > 5 and len(losers[col].dropna()) > 5:
            t, p = stats.ttest_ind(winners[col].dropna(), losers[col].dropna())
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        else:
            sig = ""
        print(f"  {col:15s} {w_mean:10.4f} {l_mean:10.4f} {diff:10.4f} {sig:>8s}")

    print(f"\n{'='*70}")
    print("BY HOUR (UTC) - find bad hours")
    print(f"{'='*70}")
    hourly = tdf.groupby("hour").agg(
        trades=("pnl", "count"),
        wr=("pnl", lambda x: (x > 0).mean() * 100),
        pnl=("pnl", "sum"),
    ).reset_index()
    for _, row in hourly.iterrows():
        if row["trades"] >= 5:
            flag = " <-- BAD" if row["wr"] < 40 else " <-- GOOD" if row["wr"] > 60 else ""
            print(f"  Hour {int(row['hour']):2d}: {int(row['trades']):3d} trades, "
                  f"WR={row['wr']:.0f}%, PnL={row['pnl']*100:.2f}%{flag}")

    print(f"\n{'='*70}")
    print("BY ADX RANGE - trend strength")
    print(f"{'='*70}")
    for lo_v, hi_v in [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 100)]:
        sub = tdf[(tdf["adx"] >= lo_v) & (tdf["adx"] < hi_v)]
        if len(sub) >= 5:
            wr = (sub["pnl"] > 0).mean() * 100
            flag = " <-- BAD" if wr < 40 else " <-- GOOD" if wr > 60 else ""
            print(f"  ADX [{lo_v:2d}-{hi_v:2d}): {len(sub):3d} trades, "
                  f"WR={wr:.0f}%, PnL={sub['pnl'].sum()*100:.2f}%{flag}")

    print(f"\n{'='*70}")
    print("BY ATR RATIO (vol expanding vs contracting)")
    print(f"{'='*70}")
    for lo_v, hi_v in [(0, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 1.3), (1.3, 1.6), (1.6, 5.0)]:
        sub = tdf[(tdf["atr_ratio"] >= lo_v) & (tdf["atr_ratio"] < hi_v)]
        if len(sub) >= 5:
            wr = (sub["pnl"] > 0).mean() * 100
            flag = " <-- BAD" if wr < 40 else " <-- GOOD" if wr > 60 else ""
            print(f"  ATR_ratio [{lo_v:.1f}-{hi_v:.1f}): {len(sub):3d} trades, "
                  f"WR={wr:.0f}%, PnL={sub['pnl'].sum()*100:.2f}%{flag}")

    print(f"\n{'='*70}")
    print("BY VOLUME RATIO")
    print(f"{'='*70}")
    for lo_v, hi_v in [(0, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 3.0), (3.0, 20.0)]:
        sub = tdf[(tdf["vol_ratio"] >= lo_v) & (tdf["vol_ratio"] < hi_v)]
        if len(sub) >= 5:
            wr = (sub["pnl"] > 0).mean() * 100
            flag = " <-- BAD" if wr < 40 else " <-- GOOD" if wr > 60 else ""
            print(f"  Vol [{lo_v:.1f}-{hi_v:.1f}): {len(sub):3d} trades, "
                  f"WR={wr:.0f}%, PnL={sub['pnl'].sum()*100:.2f}%{flag}")

    print(f"\n{'='*70}")
    print("BY BB WIDTH (volatility regime)")
    print(f"{'='*70}")
    for lo_v, hi_v in [(0, 0.01), (0.01, 0.015), (0.015, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, 1.0)]:
        sub = tdf[(tdf["bb_width"] >= lo_v) & (tdf["bb_width"] < hi_v)]
        if len(sub) >= 5:
            wr = (sub["pnl"] > 0).mean() * 100
            flag = " <-- BAD" if wr < 40 else " <-- GOOD" if wr > 60 else ""
            print(f"  BB_w [{lo_v:.3f}-{hi_v:.3f}): {len(sub):3d} trades, "
                  f"WR={wr:.0f}%, PnL={sub['pnl'].sum()*100:.2f}%{flag}")

    print(f"\n{'='*70}")
    print("BY DIRECTION")
    print(f"{'='*70}")
    for d in ["long", "short"]:
        sub = tdf[tdf["direction"] == d]
        if len(sub) > 0:
            wr = (sub["pnl"] > 0).mean() * 100
            print(f"  {d:5s}: {len(sub):3d} trades, WR={wr:.0f}%, PnL={sub['pnl'].sum()*100:.2f}%")
            # SL rate
            sl_rate = (sub["exit_reason"] == "SL").mean() * 100
            print(f"         SL rate: {sl_rate:.0f}%")

    print(f"\n{'='*70}")
    print("BY CONFIDENCE LEVEL")
    print(f"{'='*70}")
    for lo_v, hi_v in [(0.58, 0.60), (0.60, 0.63), (0.63, 0.67), (0.67, 0.75), (0.75, 1.0)]:
        sub = tdf[(tdf["prob"] >= lo_v) & (tdf["prob"] < hi_v)]
        if len(sub) >= 3:
            wr = (sub["pnl"] > 0).mean() * 100
            flag = " <-- BAD" if wr < 40 else " <-- GOOD" if wr > 60 else ""
            print(f"  Prob [{lo_v:.2f}-{hi_v:.2f}): {len(sub):3d} trades, "
                  f"WR={wr:.0f}%, PnL={sub['pnl'].sum()*100:.2f}%{flag}")

    print(f"\n{'='*70}")
    print("COMBINED FILTER CANDIDATES")
    print(f"{'='*70}")

    # Test various filter combos
    filters = {
        "baseline (no filter)": tdf,
        "prob >= 0.60": tdf[tdf["prob"] >= 0.60],
        "adx > 20": tdf[tdf["adx"] > 20],
        "atr_ratio < 1.3": tdf[tdf["atr_ratio"] < 1.3],
        "vol_ratio > 0.8": tdf[tdf["vol_ratio"] > 0.8],
        "bb_width < 0.03": tdf[tdf["bb_width"] < 0.03],
        "not (hour 21-23)": tdf[~tdf["hour"].isin([21, 22, 23])],
        "prob>=0.60 + adx>20": tdf[(tdf["prob"] >= 0.60) & (tdf["adx"] > 20)],
        "prob>=0.60 + atr_r<1.3": tdf[(tdf["prob"] >= 0.60) & (tdf["atr_ratio"] < 1.3)],
        "prob>=0.60 + vol>0.8": tdf[(tdf["prob"] >= 0.60) & (tdf["vol_ratio"] > 0.8)],
        "prob>=0.60 + bb_w<0.03": tdf[(tdf["prob"] >= 0.60) & (tdf["bb_width"] < 0.03)],
        "adx>20 + atr_r<1.3": tdf[(tdf["adx"] > 20) & (tdf["atr_ratio"] < 1.3)],
        "adx>20 + vol>0.8 + atr_r<1.3": tdf[(tdf["adx"] > 20) & (tdf["vol_ratio"] > 0.8) & (tdf["atr_ratio"] < 1.3)],
        "best combo: p60+adx20+atr<1.3": tdf[(tdf["prob"] >= 0.60) & (tdf["adx"] > 20) & (tdf["atr_ratio"] < 1.3)],
    }

    print(f"  {'Filter':40s} {'Trades':>7s} {'WR%':>6s} {'PnL%':>8s} {'Avg%':>8s}")
    for name, sub in filters.items():
        if len(sub) >= 3:
            wr = (sub["pnl"] > 0).mean() * 100
            total_pnl = sub["pnl"].sum() * 100
            avg_pnl = sub["pnl"].mean() * 100
            print(f"  {name:40s} {len(sub):7d} {wr:6.1f} {total_pnl:8.2f} {avg_pnl:8.3f}")

    # MAE analysis - how deep do losers dip?
    print(f"\n{'='*70}")
    print("MAE ANALYSIS (Max Adverse Excursion)")
    print(f"{'='*70}")
    print(f"  Winners avg MAE: {winners['mae'].mean()*100:.3f}%")
    print(f"  Losers avg MAE:  {losers['mae'].mean()*100:.3f}%")
    print(f"  SL trades MAE:   {tdf[tdf['exit_reason']=='SL']['mae'].mean()*100:.3f}%")

    # Could a tighter SL help?
    print(f"\n  --- What if tighter SL? ---")
    for sl_test in [1.0, 1.2, 1.5, 2.0, 2.5]:
        tdf2 = simulate_trades(test_df, proba, threshold=0.58, sl_mult=sl_test, tp_mult=2.0)
        if len(tdf2) > 0:
            wr = (tdf2["pnl"] > 0).mean() * 100
            pnl = tdf2["pnl"].sum() * 100
            print(f"    SL={sl_test:.1f}x ATR: {len(tdf2)} trades, WR={wr:.1f}%, PnL={pnl:.2f}%")

    # What about wider TP?
    print(f"\n  --- What if different TP? ---")
    for tp_test in [1.5, 2.0, 2.5, 3.0, 4.0]:
        tdf2 = simulate_trades(test_df, proba, threshold=0.58, sl_mult=1.5, tp_mult=tp_test)
        if len(tdf2) > 0:
            wr = (tdf2["pnl"] > 0).mean() * 100
            pnl = tdf2["pnl"].sum() * 100
            print(f"    TP={tp_test:.1f}x ATR: {len(tdf2)} trades, WR={wr:.1f}%, PnL={pnl:.2f}%")


if __name__ == "__main__":
    main()

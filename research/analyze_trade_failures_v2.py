"""Deep dive: optimized SL/TP + post-ML filters."""
import sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pathlib import Path
import joblib

DATA_DIR = Path("user_data/data/okx/futures")

exec(open("research/analyze_trade_failures.py").read().split("def simulate_trades")[0])

df = pd.read_feather(DATA_DIR / "SOL_USDT_USDT-3m-futures.feather")
df["date"] = pd.to_datetime(df["date"], utc=True)
df = df.sort_values("date").reset_index(drop=True)
df = compute_features(df)
model = joblib.load("models/ml_scalping_sol_3m_model.pkl")

train_end = 14400
test_df = df.iloc[train_end:].copy().reset_index(drop=True)
X_test = test_df[FEATURE_COLS].values
valid = np.all(np.isfinite(X_test), axis=1)
test_df = test_df[valid].reset_index(drop=True)
X_test = test_df[FEATURE_COLS].values
proba = model.predict_proba(X_test)[:, 1]

c_arr = test_df["close"].values
h_arr = test_df["high"].values
l_arr = test_df["low"].values
atr_arr = test_df["atr_14"].values
threshold = 0.58
max_bars = 15


def run_sim(sl_m, tp_m, extra_filter=None):
    """Run simulation with given SL/TP and optional post-ML filter."""
    fee = 0.001
    trades = []
    for i in range(len(test_df) - max_bars):
        if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
            continue
        if proba[i] > threshold:
            direction = "long"
        elif proba[i] < (1 - threshold):
            direction = "short"
        else:
            continue

        # Apply extra filter
        if extra_filter and not extra_filter(i):
            continue

        entry_price = c_arr[i]
        atr = atr_arr[i]
        sl_price = entry_price - sl_m * atr if direction == "long" else entry_price + sl_m * atr
        tp_price = entry_price + tp_m * atr if direction == "long" else entry_price - tp_m * atr

        exit_reason = "time_cut"
        exit_price = c_arr[min(i + max_bars, len(c_arr) - 1)]
        for j in range(1, min(max_bars + 1, len(c_arr) - i)):
            if direction == "long":
                if l_arr[i + j] <= sl_price:
                    exit_price = sl_price; exit_reason = "SL"; break
                if h_arr[i + j] >= tp_price:
                    exit_price = tp_price; exit_reason = "TP"; break
            else:
                if h_arr[i + j] >= sl_price:
                    exit_price = sl_price; exit_reason = "SL"; break
                if l_arr[i + j] <= tp_price:
                    exit_price = tp_price; exit_reason = "TP"; break

        if direction == "long":
            pnl = (exit_price - entry_price) / entry_price - fee
        else:
            pnl = (entry_price - exit_price) / entry_price - fee

        trades.append({"pnl": pnl, "exit_reason": exit_reason})

    return pd.DataFrame(trades)


print("=" * 70)
print("OPTIMIZED SL/TP + POST-ML FILTERS (OOS)")
print("=" * 70)
test_days = (test_df["date"].max() - test_df["date"].min()).days
print(f"OOS period: {test_days} days\n")

configs = [
    ("Current: SL1.5/TP2.0, no filter", 1.5, 2.0, None),
    ("SL2.0/TP2.5, no filter", 2.0, 2.5, None),
    ("SL2.0/TP3.0, no filter", 2.0, 3.0, None),
    ("SL2.0/TP2.5 + atr_r<1.3", 2.0, 2.5,
     lambda i: test_df["atr_ratio"].iloc[i] < 1.3),
    ("SL2.0/TP2.5 + atr_r<1.1", 2.0, 2.5,
     lambda i: test_df["atr_ratio"].iloc[i] < 1.1),
    ("SL2.0/TP2.5 + vol<2.0", 2.0, 2.5,
     lambda i: test_df["vol_ratio"].iloc[i] < 2.0),
    ("SL2.0/TP2.5 + atr_r<1.3 + vol<2.0", 2.0, 2.5,
     lambda i: test_df["atr_ratio"].iloc[i] < 1.3 and test_df["vol_ratio"].iloc[i] < 2.0),
    ("SL2.0/TP3.0 + atr_r<1.3", 2.0, 3.0,
     lambda i: test_df["atr_ratio"].iloc[i] < 1.3),
    ("SL2.0/TP3.0 + atr_r<1.1", 2.0, 3.0,
     lambda i: test_df["atr_ratio"].iloc[i] < 1.1),
    ("SL2.5/TP3.0 + atr_r<1.3", 2.5, 3.0,
     lambda i: test_df["atr_ratio"].iloc[i] < 1.3),
]

print(f"  {'Config':45s} {'Trades':>6s} {'T/d':>5s} {'WR%':>6s} {'PnL%':>8s} {'Avg%':>8s} {'PF':>5s}")
print(f"  {'-'*45} {'-'*6} {'-'*5} {'-'*6} {'-'*8} {'-'*8} {'-'*5}")

for name, sl_m, tp_m, filt in configs:
    tdf = run_sim(sl_m, tp_m, filt)
    if len(tdf) >= 3:
        wr = (tdf["pnl"] > 0).mean() * 100
        total_pnl = tdf["pnl"].sum() * 100
        avg_pnl = tdf["pnl"].mean() * 100
        gross_win = tdf[tdf["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(tdf[tdf["pnl"] <= 0]["pnl"].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else 99
        tpd = len(tdf) / test_days
        print(f"  {name:45s} {len(tdf):6d} {tpd:5.1f} {wr:6.1f} {total_pnl:8.2f} {avg_pnl:8.3f} {pf:5.2f}")

# RSI filter test
print(f"\n{'='*70}")
print("RSI ZONE ANALYSIS (with SL2.0/TP2.5)")
print(f"{'='*70}")

tdf_all = []
for i in range(len(test_df) - max_bars):
    if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
        continue
    if proba[i] > threshold:
        direction = "long"
    elif proba[i] < (1 - threshold):
        direction = "short"
    else:
        continue
    entry_price = c_arr[i]
    atr = atr_arr[i]
    sl_price = entry_price - 2.0 * atr if direction == "long" else entry_price + 2.0 * atr
    tp_price = entry_price + 2.5 * atr if direction == "long" else entry_price - 2.5 * atr
    exit_reason = "time_cut"
    exit_price = c_arr[min(i + max_bars, len(c_arr) - 1)]
    for j in range(1, min(max_bars + 1, len(c_arr) - i)):
        if direction == "long":
            if l_arr[i + j] <= sl_price:
                exit_price = sl_price; exit_reason = "SL"; break
            if h_arr[i + j] >= tp_price:
                exit_price = tp_price; exit_reason = "TP"; break
        else:
            if h_arr[i + j] >= sl_price:
                exit_price = sl_price; exit_reason = "SL"; break
            if l_arr[i + j] <= tp_price:
                exit_price = tp_price; exit_reason = "TP"; break
    pnl = ((exit_price - entry_price) / entry_price - 0.001) if direction == "long" else ((entry_price - exit_price) / entry_price - 0.001)
    tdf_all.append({"pnl": pnl, "rsi_14": test_df["rsi_14"].iloc[i],
                    "atr_ratio": test_df["atr_ratio"].iloc[i],
                    "vol_ratio": test_df["vol_ratio"].iloc[i],
                    "adx": test_df["adx"].iloc[i],
                    "hour": test_df["date"].iloc[i].hour})

rdf = pd.DataFrame(tdf_all)
for lo_v, hi_v in [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 50), (50, 100)]:
    sub = rdf[(rdf["rsi_14"] >= lo_v) & (rdf["rsi_14"] < hi_v)]
    if len(sub) >= 3:
        wr = (sub["pnl"] > 0).mean() * 100
        flag = " <-- BAD" if wr < 50 else " <-- GOOD" if wr > 70 else ""
        print(f"  RSI [{lo_v:2d}-{hi_v:2d}): {len(sub):3d} trades, WR={wr:.0f}%, PnL={sub['pnl'].sum()*100:.2f}%{flag}")

# Final recommended filter combo
print(f"\n{'='*70}")
print("RECOMMENDED: SL2.0/TP2.5 + atr_ratio<1.3 + avoid hours 1,6,18")
print(f"{'='*70}")

bad_hours = {1, 6, 18}
tdf_rec = run_sim(2.0, 2.5, lambda i: (
    test_df["atr_ratio"].iloc[i] < 1.3
    and test_df["date"].iloc[i].hour not in bad_hours
))
if len(tdf_rec) >= 3:
    wr = (tdf_rec["pnl"] > 0).mean() * 100
    total_pnl = tdf_rec["pnl"].sum() * 100
    avg_pnl = tdf_rec["pnl"].mean() * 100
    gross_win = tdf_rec[tdf_rec["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(tdf_rec[tdf_rec["pnl"] <= 0]["pnl"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else 99
    tpd = len(tdf_rec) / test_days
    print(f"  Trades: {len(tdf_rec)} ({tpd:.1f}/day)")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Total PnL: {total_pnl:.2f}%")
    print(f"  Avg PnL/trade: {avg_pnl:.3f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Exit breakdown:")
    for reason in ["TP", "SL", "time_cut"]:
        sub = tdf_rec[tdf_rec["exit_reason"] == reason]
        if len(sub) > 0:
            print(f"    {reason}: {len(sub)} ({len(sub)/len(tdf_rec)*100:.0f}%)")


if __name__ == "__main__":
    pass

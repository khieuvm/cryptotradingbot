"""Deep analysis of enhanced volume_spike_rev with time-cuts and per-pair breakdown."""
import pandas as pd, numpy as np, pandas_ta as ta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "user_data" / "data" / "okx" / "futures"
PAIRS = ["ETH_USDT_USDT", "SOL_USDT_USDT", "SPX_USDT_USDT", "DOGE_USDT_USDT"]
FEE = 0.001; STARTUP = 60

data = {}
for pair in PAIRS:
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    df = pd.read_feather(str(fp)).sort_values("date").reset_index(drop=True)
    c = df["close"].values.astype(float); h = df["high"].values.astype(float)
    lo = df["low"].values.astype(float); o = df["open"].values.astype(float)
    v = df["volume"].values.astype(float); n = len(df)
    atr = ta.atr(df["high"], df["low"], df["close"], length=14).values
    rsi = ta.rsi(df["close"], length=14).values
    ve = ta.ema(df["volume"].astype(float), length=20).values
    vr = v / (ve + 1e-10)
    body = np.abs(c - o); is_red = c < o
    candle_range = h - lo + 1e-10; body_ratio = body / candle_range
    upper_shadow = h - np.maximum(c, o); lower_shadow = np.minimum(c, o) - lo
    data[pair] = dict(c=c, h=h, lo=lo, o=o, n=n, atr=atr, rsi=rsi, vr=vr,
                      body_ratio=body_ratio, is_red=is_red, body=body,
                      upper_shadow=upper_shadow, lower_shadow=lower_shadow)

bars_per_day = 96
train_end = 120 * bars_per_day
val_start = train_end

def sim_with_tc(d, sig_long, sig_short, sl_m, tp_m, start, end, time_cuts=None):
    trades = []; next_ok = 0
    for idx in range(start, min(end, d["n"])):
        if idx < next_ok: continue
        il = bool(sig_long[idx]); ish = bool(sig_short[idx])
        if not il and not ish: continue
        ep = d["c"][idx]; ea = d["atr"][idx]
        if ea <= 0 or ep <= 0 or np.isnan(ea): continue
        sl_d = sl_m * ea; tp_d = tp_m * ea
        eend = min(idx + 48, min(end, d["n"] - 1))
        pnl = 0; exited = False
        for j in range(idx + 1, eend + 1):
            if il:
                if d["lo"][j] <= ep - sl_d: pnl = -(sl_d / ep) - FEE; exited = True; break
                if d["h"][j] >= ep + tp_d: pnl = (tp_d / ep) - FEE; exited = True; break
            else:
                if d["h"][j] >= ep + sl_d: pnl = -(sl_d / ep) - FEE; exited = True; break
                if d["lo"][j] <= ep - tp_d: pnl = (tp_d / ep) - FEE; exited = True; break
            if time_cuts:
                hours = (j - idx) * 15 / 60
                curr_pnl = (d["c"][j] - ep) / ep if il else (ep - d["c"][j]) / ep
                for tc_h, tc_thr in time_cuts:
                    if hours >= tc_h and curr_pnl < tc_thr:
                        pnl = curr_pnl - FEE; exited = True; break
                if exited: break
        if not exited:
            if il: pnl = (d["c"][eend] - ep) / ep - FEE
            else: pnl = (ep - d["c"][eend]) / ep - FEE
        trades.append(pnl); next_ok = idx + 6
    return trades

def report(trades, label):
    if not trades:
        print(f"    {label}: 0 trades"); return
    arr = np.array(trades)
    wr = (arr > 0).sum() / len(arr) * 100
    pf = arr[arr > 0].sum() / (abs(arr[arr < 0].sum()) + 1e-10)
    cum = np.cumsum(arr); dd = abs(np.min(cum - np.maximum.accumulate(cum)))
    print(f"    {label}: {len(arr)}t {arr.sum()*100:+.1f}% WR={wr:.0f}% PF={pf:.2f} DD={dd*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# TEST 1: P3 without time-cuts vs with time-cuts
# ═══════════════════════════════════════════════════════════════
print("=" * 90)
print("  TEST 1: P3 (spike=3.0, body>0.55, RSI 15-50, SL=3.0, TP=5.0)")
print("  Without vs With time-cuts (4h@-1%, 8h@-0.5%)")
print("=" * 90)

tc = [(4, -0.01), (8, -0.005)]
total_no_tc = {"train": 0, "oos": 0}
total_tc = {"train": 0, "oos": 0}

for pair, d in data.items():
    sig_s = (d["vr"] >= 3.0) & d["is_red"] & (d["body_ratio"] > 0.55) & \
            (d["body"] > 0.3 * d["atr"]) & (d["rsi"] < 50) & (d["rsi"] > 15)
    sig_l = (d["vr"] >= 3.0) & (d["lower_shadow"] > 2.0 * d["body"]) & \
            (d["upper_shadow"] < d["body"] * 0.5) & (d["body"] > 0) & (d["rsi"] < 25)

    print(f"\n  {pair}:")
    for label, start, end, key in [("TRAIN", STARTUP, train_end, "train"), ("OOS", val_start, d["n"], "oos")]:
        t1 = sim_with_tc(d, sig_l, sig_s, 3.0, 5.0, start, end, time_cuts=None)
        t2 = sim_with_tc(d, sig_l, sig_s, 3.0, 5.0, start, end, time_cuts=tc)
        total_no_tc[key] += sum(t1)
        total_tc[key] += sum(t2)
        report(t1, f"{label} no-TC")
        report(t2, f"{label} TC   ")

print(f"\n  TOTALS:")
print(f"    TRAIN no-TC: {total_no_tc['train']*100:+.1f}% | TC: {total_tc['train']*100:+.1f}%")
print(f"    OOS   no-TC: {total_no_tc['oos']*100:+.1f}% | TC: {total_tc['oos']*100:+.1f}%")

# ═══════════════════════════════════════════════════════════════
# TEST 2: Per-pair SL optimization on P3
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  TEST 2: Per-pair SL sweep (P3 base, OOS period only)")
print(f"{'='*90}")

for pair, d in data.items():
    sig_s = (d["vr"] >= 3.0) & d["is_red"] & (d["body_ratio"] > 0.55) & \
            (d["body"] > 0.3 * d["atr"]) & (d["rsi"] < 50) & (d["rsi"] > 15)
    sig_l = (d["vr"] >= 3.0) & (d["lower_shadow"] > 2.0 * d["body"]) & \
            (d["upper_shadow"] < d["body"] * 0.5) & (d["body"] > 0) & (d["rsi"] < 25)

    print(f"\n  {pair}:")
    for sl_m in [2.0, 2.5, 3.0, 3.5, 4.0]:
        for tp_m in [3.0, 4.0, 5.0, 6.0]:
            t = sim_with_tc(d, sig_l, sig_s, sl_m, tp_m, val_start, d["n"], time_cuts=tc)
            if t:
                arr = np.array(t)
                wr = (arr > 0).sum() / len(arr) * 100
                pf = arr[arr > 0].sum() / (abs(arr[arr < 0].sum()) + 1e-10)
                if arr.sum() > 0.02:  # only show profitable
                    print(f"    SL={sl_m} TP={tp_m}: {len(arr)}t {arr.sum()*100:+.1f}% WR={wr:.0f}% PF={pf:.2f}")

# ═══════════════════════════════════════════════════════════════
# TEST 3: Short-only vs Long+Short
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  TEST 3: SHORT-only vs LONG+SHORT (OOS period)")
print(f"{'='*90}")

total_short_only = 0; total_both = 0
for pair, d in data.items():
    sig_s = (d["vr"] >= 3.0) & d["is_red"] & (d["body_ratio"] > 0.55) & \
            (d["body"] > 0.3 * d["atr"]) & (d["rsi"] < 50) & (d["rsi"] > 15)
    sig_l = (d["vr"] >= 3.0) & (d["lower_shadow"] > 2.0 * d["body"]) & \
            (d["upper_shadow"] < d["body"] * 0.5) & (d["body"] > 0) & (d["rsi"] < 25)
    no_long = np.zeros(d["n"], dtype=bool)

    t_short = sim_with_tc(d, no_long, sig_s, 3.0, 5.0, val_start, d["n"], time_cuts=tc)
    t_both = sim_with_tc(d, sig_l, sig_s, 3.0, 5.0, val_start, d["n"], time_cuts=tc)

    total_short_only += sum(t_short)
    total_both += sum(t_both)

    print(f"  {pair}:")
    report(t_short, "SHORT-only")
    report(t_both, "LONG+SHORT")

print(f"\n  TOTAL SHORT-only: {total_short_only*100:+.1f}% | LONG+SHORT: {total_both*100:+.1f}%")

if __name__ == "__main__":
    pass

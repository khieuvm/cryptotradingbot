"""ML Experiment 1: Triple-Barrier Labeling.

Instead of "6-bar forward return > fee", labels trades as:
  +1 = price hit TP BEFORE SL within max_bars
  -1 = price hit SL (or short TP) BEFORE long TP
   0 = neither barrier hit within max_bars (time exit)

This directly matches actual execution with SL/TP orders.
Key hypothesis: if relabeling alone improves consistency, the problem
was labeling mismatch, not the model itself.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
MAKER = 0.0002
TAKER = 0.0005
PAIRS = ["ETH_USDT_USDT", "BTC_USDT_USDT", "SPX_USDT_USDT"]


def load(pair):
    fp = DATA_DIR / f"{pair}-5m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp)
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


def engineer_features(df):
    """Same feature set as v3 for fair comparison."""
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    o = df["open"]
    v = df["volume"].astype(float)

    feat = pd.DataFrame(index=df.index)

    feat["body_pct"] = (c - o) / (c + 1e-10)
    feat["range_pct"] = (h - lo) / (c + 1e-10)
    feat["upper_wick_ratio"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (h - lo + 1e-10)
    feat["lower_wick_ratio"] = (pd.concat([c, o], axis=1).min(axis=1) - lo) / (h - lo + 1e-10)

    for lb in [1, 3, 6, 12, 24, 36]:
        feat[f"ret_{lb}"] = c.pct_change(lb)

    feat["rsi_3"] = ta.rsi(c, length=3)
    feat["rsi_9"] = ta.rsi(c, length=9)
    feat["rsi_14"] = ta.rsi(c, length=14)
    feat["rsi_9_delta"] = feat["rsi_9"] - feat["rsi_9"].shift(3)
    feat["cci"] = ta.cci(h, lo, c, length=14)

    stoch = ta.stoch(h, lo, c, k=14, d=3)
    if stoch is not None:
        feat["stoch_k"] = stoch.iloc[:, 0]
        feat["stoch_d"] = stoch.iloc[:, 1]

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat["macd_hist"] = macd.iloc[:, 2]

    ema8 = ta.ema(c, length=8)
    ema21 = ta.ema(c, length=21)
    feat["price_vs_ema8"] = (c - ema8) / (ema8 + 1e-10)
    feat["price_vs_ema21"] = (c - ema21) / (ema21 + 1e-10)
    feat["ema_spread"] = (ema8 - ema21) / (ema21 + 1e-10)

    adx_r = ta.adx(h, lo, c, length=14)
    if adx_r is not None:
        feat["adx"] = adx_r.iloc[:, 0]
        feat["di_diff"] = adx_r.iloc[:, 1] - adx_r.iloc[:, 2]

    atr14 = ta.atr(h, lo, c, length=14)
    atr5 = ta.atr(h, lo, c, length=5)
    feat["atr_pct"] = atr14 / (c + 1e-10)
    feat["atr_ratio"] = atr5 / (atr14 + 1e-10)
    feat["range_vs_atr"] = (h - lo) / (atr14 + 1e-10)

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        feat["bb_pos"] = (c - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2] + 1e-10)
        feat["bb_width"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / (bb.iloc[:, 1] + 1e-10)

    vol_ema = ta.ema(v, length=20)
    feat["vol_ratio"] = v / (vol_ema + 1e-10)
    feat["vol_ratio_3"] = v.rolling(3).mean() / (vol_ema + 1e-10)
    feat["buy_pressure"] = (c - lo) / (h - lo + 1e-10)

    obv = ta.obv(c, v)
    if obv is not None:
        feat["obv_slope"] = (obv - obv.shift(5)) / (obv.abs().rolling(20).mean() + 1e-10)

    feat["ret_15m"] = c.pct_change(3)
    feat["range_15m_pct"] = (h.rolling(3).max() - lo.rolling(3).min()) / (c + 1e-10)
    feat["ret_1h"] = c.pct_change(12)
    feat["range_1h_pct"] = (h.rolling(12).max() - lo.rolling(12).min()) / (c + 1e-10)
    feat["ret_4h"] = c.pct_change(48)

    dt = pd.to_datetime(df["date"])
    feat["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    feat["is_us_session"] = ((dt.dt.hour >= 13) & (dt.dt.hour < 21)).astype(float)

    feat["pos_in_day_range"] = (c - lo.rolling(288).min()) / (
        h.rolling(288).max() - lo.rolling(288).min() + 1e-10
    )

    return feat, atr14


def triple_barrier_labels(df, atr_series, sl_mult=1.5, tp_mult=2.0, max_bars=18):
    """Label each bar using triple-barrier method.

    For each bar i, simulate:
      LONG: TP = close[i] + tp_mult * ATR[i], SL = close[i] - sl_mult * ATR[i]
      SHORT: TP = close[i] - tp_mult * ATR[i], SL = close[i] + sl_mult * ATR[i]

    Labels:
      +1 = long TP hit before long SL within max_bars
      -1 = short TP hit before short SL within max_bars
       0 = neither direction has a clear win, or time barrier hit
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = atr_series.values
    n = len(close)

    labels = np.zeros(n, dtype=int)
    long_pnl = np.zeros(n)
    short_pnl = np.zeros(n)

    for i in range(n - max_bars):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry = close[i]
        long_tp = entry + tp_mult * atr[i]
        long_sl = entry - sl_mult * atr[i]
        short_tp = entry - tp_mult * atr[i]
        short_sl = entry + sl_mult * atr[i]

        long_result = 0  # 0=timeout, 1=TP, -1=SL
        short_result = 0

        for j in range(1, max_bars + 1):
            idx = i + j
            if idx >= n:
                break

            # Long check
            if long_result == 0:
                if low[idx] <= long_sl:
                    long_result = -1
                elif high[idx] >= long_tp:
                    long_result = 1

            # Short check
            if short_result == 0:
                if high[idx] >= short_sl:
                    short_result = -1
                elif low[idx] <= short_tp:
                    short_result = 1

            if long_result != 0 and short_result != 0:
                break

        # Assign label based on which direction wins
        if long_result == 1 and short_result != 1:
            labels[i] = 1
            long_pnl[i] = tp_mult * atr[i] / entry - 2 * MAKER
        elif short_result == 1 and long_result != 1:
            labels[i] = -1
            short_pnl[i] = tp_mult * atr[i] / entry - 2 * MAKER
        elif long_result == 1 and short_result == 1:
            # Both hit TP — label as the direction where TP was hit FIRST
            long_first = _find_first_hit(high, low, i, long_tp, long_sl, max_bars, direction="long")
            short_first = _find_first_hit(high, low, i, short_tp, short_sl, max_bars, direction="short")
            if long_first < short_first:
                labels[i] = 1
                long_pnl[i] = tp_mult * atr[i] / entry - 2 * MAKER
            else:
                labels[i] = -1
                short_pnl[i] = tp_mult * atr[i] / entry - 2 * MAKER

    return labels, long_pnl, short_pnl


def _find_first_hit(high, low, start, tp, sl, max_bars, direction):
    """Find the bar index where TP is first hit."""
    n = len(high)
    for j in range(1, max_bars + 1):
        idx = start + j
        if idx >= n:
            return max_bars + 1
        if direction == "long":
            if high[idx] >= tp:
                return j
            if low[idx] <= sl:
                return max_bars + 1
        else:
            if low[idx] <= tp:
                return j
            if high[idx] >= sl:
                return max_bars + 1
    return max_bars + 1


def walk_forward_triple_barrier(features, labels, long_pnl, short_pnl, pair_name,
                                 train_days=60, test_days=30, bars_per_day=288):
    """Walk-forward with triple-barrier labels."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 100,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    train_size = train_days * bars_per_day
    test_size = test_days * bars_per_day
    n = len(features)

    fold_results = []
    all_trades = {thr: {"long": [], "short": []} for thr in [0.55, 0.60, 0.63, 0.65, 0.67, 0.70, 0.75]}

    start = 0
    fold_num = 0
    while start + train_size + test_size <= n:
        fold_num += 1
        train_end = start + train_size
        test_end = train_end + test_size

        X_train = features.iloc[start:train_end]
        y_train = labels[start:train_end]
        X_test = features.iloc[train_end:test_end]
        y_test = labels[train_end:test_end]
        long_pnl_test = long_pnl[train_end:test_end]
        short_pnl_test = short_pnl[train_end:test_end]

        train_mask = ~X_train.isna().any(axis=1)
        test_mask = ~X_test.isna().any(axis=1)

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask.values]
        X_te = X_test[test_mask]
        y_te = y_test[test_mask.values]
        lpnl_te = long_pnl_test[test_mask.values]
        spnl_te = short_pnl_test[test_mask.values]

        if len(X_tr) < 1000 or len(X_te) < 100:
            start += test_size
            continue

        # Long model: predict P(long TP hit)
        y_tr_long = (y_tr == 1).astype(int)
        pw_l = sum(y_tr_long == 0) / max(sum(y_tr_long == 1), 1)
        params_l = {**params, "scale_pos_weight": pw_l}
        model_long = lgb.train(params_l, lgb.Dataset(X_tr, label=y_tr_long), num_boost_round=300)

        # Short model: predict P(short TP hit)
        y_tr_short = (y_tr == -1).astype(int)
        pw_s = sum(y_tr_short == 0) / max(sum(y_tr_short == 1), 1)
        params_s = {**params, "scale_pos_weight": pw_s}
        model_short = lgb.train(params_s, lgb.Dataset(X_tr, label=y_tr_short), num_boost_round=300)

        prob_long = model_long.predict(X_te)
        prob_short = model_short.predict(X_te)

        # Evaluate per threshold
        fold_info = {"fold": fold_num, "train_range": f"{start}-{train_end}", "test_range": f"{train_end}-{test_end}"}

        for thr in all_trades.keys():
            long_wins = 0
            long_total = 0
            short_wins = 0
            short_total = 0
            long_pnl_sum = 0.0
            short_pnl_sum = 0.0

            cooldown = 0
            for idx in range(len(X_te)):
                if cooldown > 0:
                    cooldown -= 1
                    continue

                if prob_long[idx] > thr and prob_long[idx] > prob_short[idx] + 0.05:
                    long_total += 1
                    actual_label = y_te[idx]
                    if actual_label == 1:
                        long_wins += 1
                        long_pnl_sum += lpnl_te[idx]
                    else:
                        long_pnl_sum -= lpnl_te[idx] if lpnl_te[idx] > 0 else 0.003
                    all_trades[thr]["long"].append((fold_num, lpnl_te[idx] if actual_label == 1 else -0.003))
                    cooldown = 3

                elif prob_short[idx] > thr and prob_short[idx] > prob_long[idx] + 0.05:
                    short_total += 1
                    actual_label = y_te[idx]
                    if actual_label == -1:
                        short_wins += 1
                        short_pnl_sum += spnl_te[idx]
                    else:
                        short_pnl_sum -= spnl_te[idx] if spnl_te[idx] > 0 else 0.003
                    all_trades[thr]["short"].append((fold_num, spnl_te[idx] if actual_label == -1 else -0.003))
                    cooldown = 3

        fold_results.append(fold_info)
        start += test_size

    return all_trades, fold_results


def main():
    print("=" * 90)
    print("ML EXPERIMENT 1: Triple-Barrier Labeling")
    print("Hypothesis: SL/TP barrier labels match execution better than forward-return labels")
    print(f"Parameters: SL=1.5x ATR, TP=2.0x ATR, max_bars=18 (90min), cooldown=3 bars")
    print("=" * 90)

    # Also test wider SL/TP ratios
    sl_tp_configs = [
        (1.5, 2.0, 18, "Tight (1.5/2.0, 90m)"),
        (2.0, 3.0, 24, "Medium (2.0/3.0, 2h)"),
        (2.5, 4.0, 36, "Wide (2.5/4.0, 3h)"),
    ]

    for pair in PAIRS:
        print(f"\n{'=' * 90}")
        print(f"  {pair}")
        print(f"{'=' * 90}")

        df = load(pair)
        if df.empty:
            print(f"  No data for {pair}")
            continue

        features, atr14 = engineer_features(df)

        for sl_mult, tp_mult, max_bars, config_name in sl_tp_configs:
            print(f"\n  -- Config: {config_name} --")

            labels, long_pnl_arr, short_pnl_arr = triple_barrier_labels(
                df, atr14, sl_mult=sl_mult, tp_mult=tp_mult, max_bars=max_bars
            )

            n_long = sum(labels == 1)
            n_short = sum(labels == -1)
            n_flat = sum(labels == 0)
            total = len(labels)
            print(f"  Labels: long_tp={n_long} ({n_long/total*100:.1f}%), "
                  f"short_tp={n_short} ({n_short/total*100:.1f}%), "
                  f"neutral={n_flat} ({n_flat/total*100:.1f}%)")

            if n_long < 500 or n_short < 500:
                print(f"  [!] Too few labels, skipping")
                continue

            all_trades, fold_results = walk_forward_triple_barrier(
                features, labels, long_pnl_arr, short_pnl_arr, pair
            )

            print(f"\n  {'Thr':<6} {'L_Trades':<10} {'L_WR':<8} {'L_PnL%':<10} "
                  f"{'S_Trades':<10} {'S_WR':<8} {'S_PnL%':<10} {'Total PnL%':<10}")
            print(f"  {'-' * 80}")

            for thr in sorted(all_trades.keys()):
                long_trades = all_trades[thr]["long"]
                short_trades = all_trades[thr]["short"]

                lt = len(long_trades)
                st = len(short_trades)

                if lt > 0:
                    l_pnls = np.array([t[1] for t in long_trades])
                    l_wr = sum(1 for p in l_pnls if p > 0) / lt * 100
                    l_pnl = l_pnls.sum() * 100
                else:
                    l_wr = 0
                    l_pnl = 0

                if st > 0:
                    s_pnls = np.array([t[1] for t in short_trades])
                    s_wr = sum(1 for p in s_pnls if p > 0) / st * 100
                    s_pnl = s_pnls.sum() * 100
                else:
                    s_wr = 0
                    s_pnl = 0

                total_pnl = l_pnl + s_pnl
                marker = " <<<" if total_pnl > 0 else ""

                print(f"  {thr:<6.2f} {lt:<10} {l_wr:<7.1f}% {l_pnl:<+9.2f}% "
                      f"{st:<10} {s_wr:<7.1f}% {s_pnl:<+9.2f}% {total_pnl:<+9.2f}%{marker}")

            # Per-fold consistency for best threshold
            print(f"\n  Per-fold consistency (thr=0.63):")
            trades_063 = all_trades.get(0.63, {"long": [], "short": []})
            for direction in ["long", "short"]:
                trades = trades_063[direction]
                if not trades:
                    continue
                folds = {}
                for fold_num, pnl in trades:
                    if fold_num not in folds:
                        folds[fold_num] = []
                    folds[fold_num].append(pnl)

                for f in sorted(folds.keys()):
                    pnls = np.array(folds[f])
                    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100 if len(pnls) > 0 else 0
                    total = pnls.sum() * 100
                    print(f"    Fold {f} {direction.upper()}: "
                          f"{len(pnls)} trades, WR={wr:.1f}%, PnL={total:+.2f}%")

    # ─── COMPARISON WITH OLD METHOD ──────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("COMPARISON: Triple-Barrier vs Forward-Return Labels")
    print(f"{'=' * 90}")
    print("""
  Forward-Return (old method):
    - ETH: +$5.84 / 142d (with SL/TP execution) = basically breakeven
    - SPX: -$30.59 / 142d = loss
    - Problem: 31% of trades get stopped out because labels don't account for SL

  Triple-Barrier (this experiment):
    - Labels directly encode "will price hit TP before SL?"
    - Model learns to predict execution outcome, not just direction
    - Cooldown prevents overlapping signals
    - If this works: problem was labeling mismatch
    - If this fails: problem is fundamental (features can't predict SL/TP outcomes)
    """)


if __name__ == "__main__":
    main()

"""ML Experiment 1b: Triple-Barrier Labeling — FIXED PnL calculation.

Bug fix from v1: Loss on wrong prediction now uses actual SL distance
(sl_mult * ATR / price) instead of fixed -0.3%.

Proper PnL:
  Correct LONG prediction (label=1): +tp_mult * ATR / entry - fees
  Wrong LONG prediction (label=-1 or 0): -sl_mult * ATR / entry - fees
  (Same logic for SHORT)
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
    """Label each bar + pre-compute win/loss PnL arrays for proper evaluation."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = atr_series.values
    n = len(close)

    labels = np.zeros(n, dtype=int)
    # Pre-compute potential reward and risk per bar
    tp_reward = np.zeros(n)  # TP pnl if direction is correct
    sl_loss = np.zeros(n)    # SL pnl if direction is wrong

    for i in range(n - max_bars):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry = close[i]
        tp_reward[i] = tp_mult * atr[i] / entry
        sl_loss[i] = sl_mult * atr[i] / entry

        long_tp = entry + tp_mult * atr[i]
        long_sl = entry - sl_mult * atr[i]
        short_tp = entry - tp_mult * atr[i]
        short_sl = entry + sl_mult * atr[i]

        long_result = 0
        short_result = 0
        long_bar = max_bars + 1
        short_bar = max_bars + 1

        for j in range(1, max_bars + 1):
            idx = i + j
            if idx >= n:
                break

            if long_result == 0:
                if low[idx] <= long_sl:
                    long_result = -1
                elif high[idx] >= long_tp:
                    long_result = 1
                    long_bar = j

            if short_result == 0:
                if high[idx] >= short_sl:
                    short_result = -1
                elif low[idx] <= short_tp:
                    short_result = 1
                    short_bar = j

            if long_result != 0 and short_result != 0:
                break

        # Assign label
        if long_result == 1 and short_result != 1:
            labels[i] = 1
        elif short_result == 1 and long_result != 1:
            labels[i] = -1
        elif long_result == 1 and short_result == 1:
            labels[i] = 1 if long_bar <= short_bar else -1

    return labels, tp_reward, sl_loss


def walk_forward(features, labels, tp_reward, sl_loss, pair_name,
                 train_days=60, test_days=30, bars_per_day=288):
    """Walk-forward with CORRECT PnL: win = +tp_reward, loss = -sl_loss."""
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

    thresholds = [0.55, 0.58, 0.60, 0.63, 0.65, 0.67, 0.70, 0.75]
    results = {thr: {"long": [], "short": []} for thr in thresholds}

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
        tp_test = tp_reward[train_end:test_end]
        sl_test = sl_loss[train_end:test_end]

        train_mask = ~X_train.isna().any(axis=1)
        test_mask = ~X_test.isna().any(axis=1)

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask.values]
        X_te = X_test[test_mask]
        y_te = y_test[test_mask.values]
        tp_te = tp_test[test_mask.values]
        sl_te = sl_loss[train_end:test_end][test_mask.values]

        if len(X_tr) < 1000 or len(X_te) < 100:
            start += test_size
            continue

        # Long model
        y_tr_long = (y_tr == 1).astype(int)
        pw_l = sum(y_tr_long == 0) / max(sum(y_tr_long == 1), 1)
        model_long = lgb.train(
            {**params, "scale_pos_weight": pw_l},
            lgb.Dataset(X_tr, label=y_tr_long),
            num_boost_round=300,
        )

        # Short model
        y_tr_short = (y_tr == -1).astype(int)
        pw_s = sum(y_tr_short == 0) / max(sum(y_tr_short == 1), 1)
        model_short = lgb.train(
            {**params, "scale_pos_weight": pw_s},
            lgb.Dataset(X_tr, label=y_tr_short),
            num_boost_round=300,
        )

        prob_long = model_long.predict(X_te)
        prob_short = model_short.predict(X_te)

        for thr in thresholds:
            cooldown = 0
            for idx in range(len(X_te)):
                if cooldown > 0:
                    cooldown -= 1
                    continue

                trade_pnl = None
                direction = None

                if prob_long[idx] > thr and prob_long[idx] > prob_short[idx] + 0.05:
                    direction = "long"
                    if y_te[idx] == 1:
                        # CORRECT: hit TP
                        trade_pnl = tp_te[idx] - 2 * MAKER
                    else:
                        # WRONG: hit SL (or timeout with small loss)
                        if y_te[idx] == -1:
                            trade_pnl = -(sl_te[idx] + 2 * MAKER)
                        else:
                            # Timeout (label=0): assume small loss (half SL)
                            trade_pnl = -(sl_te[idx] * 0.3 + 2 * MAKER)
                    cooldown = 3

                elif prob_short[idx] > thr and prob_short[idx] > prob_long[idx] + 0.05:
                    direction = "short"
                    if y_te[idx] == -1:
                        trade_pnl = tp_te[idx] - 2 * MAKER
                    else:
                        if y_te[idx] == 1:
                            trade_pnl = -(sl_te[idx] + 2 * MAKER)
                        else:
                            trade_pnl = -(sl_te[idx] * 0.3 + 2 * MAKER)
                    cooldown = 3

                if trade_pnl is not None:
                    results[thr][direction].append((fold_num, trade_pnl))

        start += test_size

    return results, fold_num


def print_results(results, num_folds, pair_name, config_name):
    """Print formatted results with per-fold breakdown."""
    print(f"\n  -- Config: {config_name} --")
    print(f"  {'Thr':<6} {'L_Trades':<9} {'L_WR':<8} {'L_PnL%':<10} "
          f"{'S_Trades':<9} {'S_WR':<8} {'S_PnL%':<10} {'NET PnL%':<10} {'PF':<6}")
    print(f"  {'-' * 85}")

    best_net = -999
    best_thr = None

    for thr in sorted(results.keys()):
        long_trades = results[thr]["long"]
        short_trades = results[thr]["short"]

        lt = len(long_trades)
        st = len(short_trades)

        if lt > 0:
            l_pnls = np.array([t[1] for t in long_trades])
            l_wr = sum(1 for p in l_pnls if p > 0) / lt * 100
            l_pnl = l_pnls.sum() * 100
            l_wins_sum = l_pnls[l_pnls > 0].sum()
            l_loss_sum = abs(l_pnls[l_pnls < 0].sum())
        else:
            l_wr = l_pnl = l_wins_sum = l_loss_sum = 0

        if st > 0:
            s_pnls = np.array([t[1] for t in short_trades])
            s_wr = sum(1 for p in s_pnls if p > 0) / st * 100
            s_pnl = s_pnls.sum() * 100
            s_wins_sum = s_pnls[s_pnls > 0].sum()
            s_loss_sum = abs(s_pnls[s_pnls < 0].sum())
        else:
            s_wr = s_pnl = s_wins_sum = s_loss_sum = 0

        net_pnl = l_pnl + s_pnl
        total_wins = l_wins_sum + s_wins_sum
        total_losses = l_loss_sum + s_loss_sum
        pf = total_wins / total_losses if total_losses > 0 else 0

        marker = " <<<" if net_pnl > 0 and pf > 1.0 else ""
        if net_pnl > best_net:
            best_net = net_pnl
            best_thr = thr

        print(f"  {thr:<6.2f} {lt:<9} {l_wr:<7.1f}% {l_pnl:<+9.2f}% "
              f"{st:<9} {s_wr:<7.1f}% {s_pnl:<+9.2f}% {net_pnl:<+9.2f}% {pf:<5.2f}{marker}")

    # Per-fold consistency for best threshold
    if best_thr is not None:
        print(f"\n  Per-fold consistency (thr={best_thr}):")
        for direction in ["long", "short"]:
            trades = results[best_thr][direction]
            if not trades:
                continue
            folds = {}
            for fold_num, pnl in trades:
                folds.setdefault(fold_num, []).append(pnl)

            consistent = True
            for f in sorted(folds.keys()):
                pnls = np.array(folds[f])
                wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                total = pnls.sum() * 100
                status = "+" if total > 0 else "-"
                if total < 0:
                    consistent = False
                print(f"    Fold {f} {direction.upper()}: "
                      f"{len(pnls)} trades, WR={wr:.1f}%, PnL={total:+.2f}% [{status}]")
            if consistent and len(folds) >= 2:
                print(f"    >>> {direction.upper()} CONSISTENT across all folds!")


def main():
    print("=" * 90)
    print("ML EXPERIMENT 1b: Triple-Barrier Labeling (FIXED PnL)")
    print("FIX: Loss = -sl_mult * ATR / price (not fixed -0.3%)")
    print("=" * 90)

    sl_tp_configs = [
        (1.5, 2.0, 18, "Tight (SL=1.5, TP=2.0, 90m)"),
        (2.0, 3.0, 24, "Medium (SL=2.0, TP=3.0, 2h)"),
        (2.5, 4.0, 36, "Wide (SL=2.5, TP=4.0, 3h)"),
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

        # Show ATR stats for context
        atr_pct = (atr14 / df["close"]).dropna()
        print(f"  ATR stats: mean={atr_pct.mean()*100:.3f}%, "
              f"median={atr_pct.median()*100:.3f}%, "
              f"p95={atr_pct.quantile(0.95)*100:.3f}%")

        for sl_mult, tp_mult, max_bars, config_name in sl_tp_configs:
            labels, tp_reward, sl_loss_arr = triple_barrier_labels(
                df, atr14, sl_mult=sl_mult, tp_mult=tp_mult, max_bars=max_bars
            )

            n_long = sum(labels == 1)
            n_short = sum(labels == -1)
            n_flat = sum(labels == 0)
            total = len(labels)

            # Effective R:R ratio
            avg_tp_reward = tp_reward[tp_reward > 0].mean() * 100
            avg_sl_loss = sl_loss_arr[sl_loss_arr > 0].mean() * 100
            rr = avg_tp_reward / avg_sl_loss if avg_sl_loss > 0 else 0
            be_wr = 1 / (1 + rr) * 100 if rr > 0 else 50

            print(f"\n  Labels: L={n_long}({n_long/total*100:.1f}%) "
                  f"S={n_short}({n_short/total*100:.1f}%) "
                  f"N={n_flat}({n_flat/total*100:.1f}%) | "
                  f"Avg TP={avg_tp_reward:.2f}%, SL={avg_sl_loss:.2f}%, "
                  f"R:R={rr:.2f}, BE WR={be_wr:.1f}%")

            if n_long < 500 or n_short < 500:
                print(f"  [!] Too few labels, skipping")
                continue

            results, num_folds = walk_forward(
                features, labels, tp_reward, sl_loss_arr, pair
            )
            print_results(results, num_folds, pair, config_name)

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print("""
  Key metrics to evaluate:
  1. Is NET PnL positive? (after proper SL losses)
  2. Is PF > 1.2? (sustainable edge, not noise)
  3. Are BOTH folds profitable? (consistency)
  4. Is WR > breakeven WR for the R:R ratio?

  If YES to all 4 -> Triple-barrier labeling WORKS, implement in production
  If NO -> Move to Experiment 2 (Meta-labeling on 15m signals)
    """)


if __name__ == "__main__":
    main()

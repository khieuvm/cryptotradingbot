"""ML 5m Production Backtest — Walk-forward with trained models.

Uses walk-forward methodology (train 60d, test 30d) with:
- ETH/USDT:USDT (threshold 0.63)
- SPX/USDT:USDT (threshold 0.60)
- BTC/USDT:USDT (threshold 0.63) [optional, low edge]

Reports PnL in USDT with stake weighting for portfolio integration.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
BASE_STAKE = 50
LEVERAGE = 3

PAIRS_CONFIG = {
    "ETH_USDT_USDT": {"threshold": 0.63, "stake_mult": 1.5, "direction_gap": 0.05},
    "SPX_USDT_USDT": {"threshold": 0.60, "stake_mult": 1.0, "direction_gap": 0.05},
}


def load_5m(pair):
    fp = DATA_DIR / f"{pair}-5m-futures.feather"
    if not fp.exists():
        print(f"  WARNING: {fp} not found")
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
    feat["pos_in_day_range"] = (c - lo.rolling(288).min()) / (h.rolling(288).max() - lo.rolling(288).min() + 1e-10)

    return feat


def generate_labels(df, forward_bars=6):
    c = df["close"].values
    n = len(c)
    fee = 2 * MAKER_FEE

    fwd_ret = np.zeros(n)
    for i in range(n - forward_bars):
        fwd_ret[i] = (c[i + forward_bars] - c[i]) / c[i]

    labels = np.zeros(n, dtype=int)
    min_move = fee + 0.0001

    for i in range(n - forward_bars):
        if fwd_ret[i] > min_move:
            labels[i] = 1
        elif fwd_ret[i] < -min_move:
            labels[i] = -1

    return labels, fwd_ret


def backtest_ml_production(pair, cfg):
    """Walk-forward ML backtest with ATR-based SL/TP."""
    df = load_5m(pair)
    if df.empty:
        return []

    features = engineer_features(df)
    labels, fwd_returns = generate_labels(df, forward_bars=6)

    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    atr_vals = ta.atr(df["high"], df["low"], df["close"], length=14).values

    threshold = cfg["threshold"]
    direction_gap = cfg["direction_gap"]
    stake_mult = cfg["stake_mult"]

    train_days = 60
    test_days = 30
    bars_per_day = 288
    train_size = train_days * bars_per_day
    test_size = test_days * bars_per_day
    n = len(features)

    sl_mult = 1.5
    tp_mult = 2.0
    max_bars = 18  # 18 bars = 90min max hold

    model_params = {
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

    all_trades = []
    start = 0

    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size

        X_train = features.iloc[start:train_end]
        y_train = labels[start:train_end]
        X_test = features.iloc[train_end:test_end]

        train_mask = ~X_train.isna().any(axis=1)
        test_mask = ~X_test.isna().any(axis=1)

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask.values]
        X_te = X_test[test_mask]
        test_indices = X_test.index[test_mask]

        if len(X_tr) < 1000 or len(X_te) < 100:
            start += test_size
            continue

        # Train long model
        y_tr_long = (y_tr == 1).astype(int)
        params_l = model_params.copy()
        params_l["scale_pos_weight"] = sum(y_tr_long == 0) / max(sum(y_tr_long == 1), 1)
        model_long = lgb.train(params_l, lgb.Dataset(X_tr, label=y_tr_long), num_boost_round=300)

        # Train short model
        y_tr_short = (y_tr == -1).astype(int)
        params_s = model_params.copy()
        params_s["scale_pos_weight"] = sum(y_tr_short == 0) / max(sum(y_tr_short == 1), 1)
        model_short = lgb.train(params_s, lgb.Dataset(X_tr, label=y_tr_short), num_boost_round=300)

        prob_long = model_long.predict(X_te)
        prob_short = model_short.predict(X_te)

        # Generate signals and simulate with SL/TP
        cooldown = 0
        for k in range(len(X_te)):
            if cooldown > 0:
                cooldown -= 1
                continue

            idx = test_indices[k]
            if idx + max_bars >= n:
                continue

            ea = atr_vals[idx]
            if ea <= 0 or np.isnan(ea):
                continue

            is_long = None
            if prob_long[k] > threshold and prob_long[k] > prob_short[k] + direction_gap:
                is_long = True
            elif prob_short[k] > threshold and prob_short[k] > prob_long[k] + direction_gap:
                is_long = False
            else:
                continue

            ep = c[idx]
            sl_dist = sl_mult * ea
            tp_dist = tp_mult * ea

            exit_price = ep
            exit_reason = "timeout"
            bars_held = 0

            for j in range(idx + 1, min(idx + max_bars + 1, n)):
                bars_held = j - idx
                if is_long:
                    if lo[j] <= ep - sl_dist:
                        exit_price = ep - sl_dist
                        exit_reason = "SL"
                        break
                    if h[j] >= ep + tp_dist:
                        exit_price = ep + tp_dist
                        exit_reason = "TP"
                        break
                else:
                    if h[j] >= ep + sl_dist:
                        exit_price = ep + sl_dist
                        exit_reason = "SL"
                        break
                    if lo[j] <= ep - tp_dist:
                        exit_price = ep - tp_dist
                        exit_reason = "TP"
                        break
            else:
                ei = min(idx + max_bars, n - 1)
                exit_price = c[ei]
                bars_held = ei - idx

            fee = 2 * MAKER_FEE  # limit entry
            pnl_pct = ((exit_price - ep) / ep if is_long else (ep - exit_price) / ep) - fee
            stake_usdt = BASE_STAKE * stake_mult
            pnl_usdt = pnl_pct * stake_usdt * LEVERAGE

            all_trades.append({
                "idx": idx,
                "dir": "LONG" if is_long else "SHORT",
                "entry": ep,
                "exit": exit_price,
                "exit_reason": exit_reason,
                "pnl_pct": pnl_pct,
                "pnl_usdt": pnl_usdt,
                "stake_usdt": stake_usdt,
                "bars_held": bars_held,
                "confidence": max(prob_long[k], prob_short[k]),
            })

            cooldown = 3  # min 15min between trades

        start += test_size

    return all_trades


def main():
    print("=" * 100)
    print("ML 5m PRODUCTION BACKTEST — WALK-FORWARD + SL/TP + STAKE WEIGHTED")
    print(f"Period: 2026-01-01 to 2026-05-21 (142 days, OOS portions only)")
    print(f"Base stake: {BASE_STAKE} USDT | Leverage: {LEVERAGE}x")
    print(f"SL: 1.5x ATR | TP: 2.0x ATR | Max hold: 18 bars (90min)")
    print(f"Entry: limit orders (maker fee 0.04% RT)")
    print("=" * 100)

    all_results = []
    portfolio_trades = []

    for pair, cfg in PAIRS_CONFIG.items():
        print(f"\n  Training + backtesting {pair} (threshold={cfg['threshold']}, stake={cfg['stake_mult']}x)...")
        trades = backtest_ml_production(pair, cfg)
        portfolio_trades.extend(trades)

        if not trades:
            print(f"  No trades for {pair}")
            continue

        pnls = np.array([t["pnl_pct"] for t in trades])
        usdt_pnls = np.array([t["pnl_usdt"] for t in trades])
        n_trades = len(pnls)
        wins = (pnls > 0).sum()
        wr = wins / n_trades
        pf = abs(pnls[pnls > 0].sum() / pnls[pnls <= 0].sum()) if pnls[pnls <= 0].sum() != 0 else 99

        tp_c = sum(1 for t in trades if t["exit_reason"] == "TP")
        sl_c = sum(1 for t in trades if t["exit_reason"] == "SL")
        to_c = sum(1 for t in trades if t["exit_reason"] == "timeout")

        longs = [t for t in trades if t["dir"] == "LONG"]
        shorts = [t for t in trades if t["dir"] == "SHORT"]
        long_pnl = sum(t["pnl_usdt"] for t in longs)
        short_pnl = sum(t["pnl_usdt"] for t in shorts)

        equity = np.cumsum(usdt_pnls)
        peak = np.maximum.accumulate(equity)
        max_dd_usdt = (peak - equity).max()

        avg_conf = np.mean([t["confidence"] for t in trades])
        avg_bars = np.mean([t["bars_held"] for t in trades])

        print(f"\n  {'=' * 90}")
        print(f"  ML_ENSEMBLE_5M | {pair} | Stake: {cfg['stake_mult']}x ({BASE_STAKE*cfg['stake_mult']:.0f} USDT)")
        print(f"  Threshold: {cfg['threshold']} | SL: 1.5x ATR | TP: 2.0x ATR")
        print(f"  {'=' * 90}")
        print(f"  Trades: {n_trades} | WR: {wr*100:.1f}% | PF: {pf:.2f}")
        print(f"  PnL: {pnls.sum()*100:+.2f}% = ${usdt_pnls.sum():+.2f} USDT ({LEVERAGE}x lev)")
        print(f"  Max DD: ${max_dd_usdt:.2f}")
        print(f"  Avg confidence: {avg_conf:.3f} | Avg bars held: {avg_bars:.1f} ({avg_bars*5:.0f}min)")
        print(f"  Exit: TP={tp_c} | SL={sl_c} | Timeout={to_c}")
        print(f"  LONG:  {len(longs)} trades, WR={sum(1 for t in longs if t['pnl_pct']>0)/max(len(longs),1)*100:.0f}%, ${long_pnl:+.2f}")
        print(f"  SHORT: {len(shorts)} trades, WR={sum(1 for t in shorts if t['pnl_pct']>0)/max(len(shorts),1)*100:.0f}%, ${short_pnl:+.2f}")

        all_results.append({
            "pair": pair, "trades": n_trades, "wr": wr, "pf": pf,
            "pnl_pct": pnls.sum(), "pnl_usdt": usdt_pnls.sum(),
            "max_dd_usdt": max_dd_usdt, "stake_mult": cfg["stake_mult"],
        })

    # Portfolio summary
    if portfolio_trades:
        all_usdt = np.array([t["pnl_usdt"] for t in portfolio_trades])
        total_pnl = all_usdt.sum()
        total_trades = len(portfolio_trades)
        overall_wr = (all_usdt > 0).sum() / total_trades

        eq = np.cumsum(all_usdt)
        pk = np.maximum.accumulate(eq)
        portfolio_dd = (pk - eq).max()

        print(f"\n\n{'=' * 100}")
        print("ML 5m PORTFOLIO SUMMARY")
        print(f"{'=' * 100}")
        print(f"""
  Period:              ~60 OOS days (walk-forward, non-overlapping)
  Pairs:              ETH, SPX
  Base Stake:          {BASE_STAKE} USDT per trade
  Leverage:            {LEVERAGE}x

  Total Trades:        {total_trades}
  Overall Win Rate:    {overall_wr*100:.1f}%
  Total PnL:           ${total_pnl:+,.2f} USDT

  Portfolio Max DD:    ${portfolio_dd:,.2f}
  Per-Trade Avg:       ${total_pnl/total_trades:+,.2f}/trade
  Per-Day Avg:         ${total_pnl/60:+,.2f}/day (OOS days)
""")

        # Combined with 15m strategies
        print(f"  {'=' * 90}")
        print(f"  FULL PORTFOLIO PROJECTION (15m + 5m ML combined)")
        print(f"  {'=' * 90}")

        # 15m results from production backtest
        pnl_15m = 1253.43  # from previous run
        print(f"""
  15m Strategies (142d):     ${pnl_15m:+,.2f} USDT
  5m ML Ensemble (60d OOS):  ${total_pnl:+,.2f} USDT
  5m ML projected (142d):    ${total_pnl/60*142:+,.2f} USDT (linear extrapolation)

  COMBINED (142d est):       ${pnl_15m + total_pnl/60*142:+,.2f} USDT
  Monthly est:               ${(pnl_15m + total_pnl/60*142)/4.7:+,.2f}/month
""")


if __name__ == "__main__":
    main()

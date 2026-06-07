"""
ML Super-Scalping Research v2: Test ML models on 1m/3m OKX real data.
Fixed: calendar-based walk-forward windows, realistic PnL from actual returns.

Tests: LightGBM, XGBoost, CatBoost, ExtraTrees, RandomForest
Timeframes: 1m, 3m
Pairs: BTC/USDT, ETH/USDT, SOL/USDT
Walk-forward: 30d train / 10d test / 10d step
Labels: Triple-barrier + fixed-horizon
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

DATA_DIR = Path("user_data/data/okx/futures")
PAIRS = {"BTC": "BTC_USDT_USDT", "ETH": "ETH_USDT_USDT", "SOL": "SOL_USDT_USDT"}
TIMEFRAMES = ["1m", "3m"]

COST_MAKER_RT = 0.0004
COST_TAKER_RT = 0.0010


def load_data(pair_key, timeframe):
    fpath = DATA_DIR / f"{PAIRS[pair_key]}-{timeframe}-futures.feather"
    if not fpath.exists():
        return pd.DataFrame()
    df = pd.read_feather(fpath)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def compute_features(df):
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    df['body_pct'] = (c - o) / c
    df['range_pct'] = (h - l) / c
    df['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-10)
    df['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-10)

    for lb in [1, 2, 3, 5, 8, 13, 21]:
        df[f'ret_{lb}'] = c.pct_change(lb)

    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    df['atr_5'] = tr.rolling(5).mean()
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr_5'] / (df['atr_14'] + 1e-10)
    df['atr_pct'] = df['atr_14'] / c

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['bb_pos'] = (c - sma20) / (2 * std20 + 1e-10)
    df['bb_width'] = (4 * std20) / (sma20 + 1e-10)

    for p in [3, 7, 14]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        df[f'rsi_{p}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['rsi_delta'] = df['rsi_7'] - df['rsi_7'].shift(3)

    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    df['stoch_k'] = 100 * (c - low14) / (high14 - low14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    for p in [8, 21, 50]:
        ema = c.ewm(span=p, adjust=False).mean()
        df[f'ema{p}_dist'] = (c - ema) / c
    df['ema_spread'] = df['ema8_dist'] - df['ema50_dist']

    vol_ema = v.rolling(20).mean()
    df['vol_ratio'] = v / (vol_ema + 1e-10)
    df['vol_ratio_3'] = v.rolling(3).mean() / (vol_ema + 1e-10)
    obv = (np.sign(c.diff()) * v).cumsum()
    df['obv_slope'] = obv.diff(5) / (c * 5 + 1e-10)
    df['buy_pressure'] = (c - l) / (h - l + 1e-10)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df['macd_hist'] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    plus_dm = (h.diff()).clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    atr_adx = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr_adx + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr_adx + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean()

    df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)
    df['is_us_session'] = ((df['date'].dt.hour >= 13) & (df['date'].dt.hour <= 21)).astype(int)
    df['is_asia_session'] = ((df['date'].dt.hour >= 0) & (df['date'].dt.hour <= 8)).astype(int)

    df['tick_direction'] = np.sign(c.diff())
    runs = np.zeros(len(df))
    ticks = df['tick_direction'].values
    for i in range(1, len(ticks)):
        if ticks[i] == ticks[i-1] and ticks[i] != 0:
            runs[i] = runs[i-1] + 1
    df['tick_run'] = runs
    df['spread_proxy'] = (h - l) / c

    return df


FEATURE_COLS = [
    'body_pct', 'range_pct', 'upper_wick', 'lower_wick',
    'ret_1', 'ret_2', 'ret_3', 'ret_5', 'ret_8', 'ret_13', 'ret_21',
    'atr_5', 'atr_14', 'atr_ratio', 'atr_pct', 'bb_pos', 'bb_width',
    'rsi_3', 'rsi_7', 'rsi_14', 'rsi_delta', 'stoch_k', 'stoch_d',
    'ema8_dist', 'ema21_dist', 'ema50_dist', 'ema_spread',
    'vol_ratio', 'vol_ratio_3', 'obv_slope', 'buy_pressure',
    'macd_hist', 'adx',
    'hour_sin', 'hour_cos', 'is_us_session', 'is_asia_session',
    'tick_direction', 'tick_run', 'spread_proxy',
]


def compute_labels_and_returns(df, label_type, tf_minutes):
    """Compute labels AND actual forward returns for PnL calculation."""
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    atr = df['atr_14'].values
    n = len(c)

    if label_type == 'triple_barrier':
        sl_mult = 1.0
        tp_mult = 1.5
        max_bars = {1: 15, 3: 10, 5: 8}.get(tf_minutes, 10)
    else:
        horizon = {1: 8, 3: 6, 5: 4}.get(tf_minutes, 6)

    labels = np.zeros(n, dtype=int)
    actual_returns = np.zeros(n)

    if label_type == 'triple_barrier':
        for i in range(n - max_bars - 1):
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue
            entry = c[i]
            sl_d = sl_mult * atr[i]
            tp_d = tp_mult * atr[i]

            # Check LONG: TP above, SL below, use high/low
            long_result = 0
            for j in range(i + 1, min(i + max_bars + 1, n)):
                if h[j] >= entry + tp_d:
                    long_result = 1
                    break
                if l[j] <= entry - sl_d:
                    long_result = -1
                    break

            # Check SHORT: TP below, SL above
            short_result = 0
            for j in range(i + 1, min(i + max_bars + 1, n)):
                if l[j] <= entry - tp_d:
                    short_result = 1
                    break
                if h[j] >= entry + sl_d:
                    short_result = -1
                    break

            if long_result == 1 and short_result != 1:
                labels[i] = 1
                actual_returns[i] = tp_d / entry  # actual return if long TP hit
            elif short_result == 1 and long_result != 1:
                labels[i] = -1
                actual_returns[i] = tp_d / entry  # actual return if short TP hit
            elif long_result == -1 and short_result == -1:
                labels[i] = 0  # both sides SL'd
            # else: timeout or mixed → 0
    else:
        ret = pd.Series(c).pct_change(horizon).values
        # Use high/low for more realistic returns
        for i in range(n - horizon):
            if np.isnan(ret[i + horizon]):
                continue
            fwd_ret = ret[i + horizon]
            thresh = max(0.0006, 0.3 * atr[i] / c[i]) if not np.isnan(atr[i]) else 0.0006
            if fwd_ret > thresh:
                labels[i] = 1
                actual_returns[i] = abs(fwd_ret)
            elif fwd_ret < -thresh:
                labels[i] = -1
                actual_returns[i] = abs(fwd_ret)

    return labels, actual_returns


def walk_forward_calendar(df, model_fn, label_type, tf_minutes,
                          train_days=30, test_days=10,
                          thresholds=None):
    """Calendar-based walk-forward: use dates, not bar counts."""
    if thresholds is None:
        thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]

    labels, actual_returns = compute_labels_and_returns(df, label_type, tf_minutes)
    df = df.copy()
    df['label'] = labels
    df['actual_ret'] = actual_returns

    # Drop rows with NaN features or neutral labels
    valid_mask = df[FEATURE_COLS].notna().all(axis=1) & (df['label'] != 0)
    valid = df[valid_mask].copy()

    if len(valid) < 500:
        return {}

    dates = valid['date']
    min_date = dates.min()
    max_date = dates.max()

    results = {t: {'trades': 0, 'wins': 0, 'pnl_gross': 0.0, 'returns': []}
               for t in thresholds}

    train_start = min_date
    fold = 0

    while True:
        train_end = train_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)

        if test_end > max_date:
            break

        train_mask = (valid['date'] >= train_start) & (valid['date'] < train_end)
        test_mask = (valid['date'] >= train_end) & (valid['date'] < test_end)

        train_data = valid[train_mask]
        test_data = valid[test_mask]

        if len(train_data) < 500 or len(test_data) < 50:
            train_start += timedelta(days=test_days)
            continue

        X_train = train_data[FEATURE_COLS].values
        # Binary: 1 = long wins, 0 = short wins
        y_train = (train_data['label'] == 1).astype(int).values
        X_test = test_data[FEATURE_COLS].values
        y_test = (test_data['label'] == 1).astype(int).values
        test_rets = test_data['actual_ret'].values

        try:
            model = model_fn()
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            train_start += timedelta(days=test_days)
            fold += 1
            continue

        for threshold in thresholds:
            r = results[threshold]
            for j in range(len(proba)):
                trade_ret = None
                if proba[j] > threshold:
                    # Long signal — correct if label=1 (long wins)
                    is_win = y_test[j] == 1
                    trade_ret = test_rets[j] if is_win else -test_rets[j] * (1.0/1.5)  # SL is 1/1.5 of TP
                elif proba[j] < (1 - threshold):
                    # Short signal — correct if label=0 (short wins)
                    is_win = y_test[j] == 0
                    trade_ret = test_rets[j] if is_win else -test_rets[j] * (1.0/1.5)

                if trade_ret is not None:
                    r['trades'] += 1
                    if trade_ret > 0:
                        r['wins'] += 1
                    r['pnl_gross'] += trade_ret
                    r['returns'].append(trade_ret)

        fold += 1
        train_start += timedelta(days=test_days)

    # Compute summary
    summary = {}
    for threshold in thresholds:
        r = results[threshold]
        if r['trades'] == 0:
            continue
        trades = r['trades']
        wr = r['wins'] / trades * 100
        avg_ret = np.mean(r['returns'])
        std_ret = np.std(r['returns']) if len(r['returns']) > 1 else 1e-10

        bars_per_day = 1440 // tf_minutes
        sharpe = (avg_ret / (std_ret + 1e-10)) * np.sqrt(252 * bars_per_day / max(trades / fold, 1))

        pnl_maker = r['pnl_gross'] - trades * COST_MAKER_RT
        pnl_taker = r['pnl_gross'] - trades * COST_TAKER_RT

        summary[threshold] = {
            'trades': trades,
            'trades_per_day': round(trades / (fold * test_days), 1),
            'wr': round(wr, 1),
            'avg_ret_bps': round(avg_ret * 10000, 2),
            'pnl_gross_%': round(r['pnl_gross'] * 100, 2),
            'pnl_maker_%': round(pnl_maker * 100, 2),
            'pnl_taker_%': round(pnl_taker * 100, 2),
            'sharpe': round(sharpe, 1),
            'folds': fold,
        }
    return summary


# ─── MODEL FACTORIES ────────────────────────────────────────────────────────

def make_lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=200, max_depth=3, num_leaves=8,
        min_child_samples=200, learning_rate=0.03,
        colsample_bytree=0.5, subsample=0.7, subsample_freq=5,
        reg_alpha=0.5, reg_lambda=2.0,
        class_weight='balanced', verbose=-1, n_jobs=-1,
    )

def make_xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.5,
        reg_alpha=0.5, reg_lambda=2.0,
        eval_metric='auc', tree_method='hist',
        use_label_encoder=False, verbosity=0, n_jobs=-1,
    )

def make_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=150, depth=4, learning_rate=0.03,
        l2_leaf_reg=3.0, subsample=0.7,
        auto_class_weights='Balanced', verbose=0,
    )

def make_extra_trees():
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50,
        max_features='sqrt', class_weight='balanced', n_jobs=-1,
    )

def make_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50,
        max_features='sqrt', class_weight='balanced', n_jobs=-1,
    )

MODELS = {
    'LightGBM': make_lgbm,
    'XGBoost': make_xgboost,
    'CatBoost': make_catboost,
    'ExtraTrees': make_extra_trees,
    'RandomForest': make_rf,
}


def main():
    print("=" * 90)
    print("ML SUPER-SCALPING v2 — OKX Real Data (60 days)")
    print("=" * 90)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Walk-forward: 30d train / 10d test (calendar-based)")
    print(f"Labels: triple_barrier (SL=1.0×ATR, TP=1.5×ATR) + fixed_horizon")
    print(f"Cost: maker={COST_MAKER_RT*100:.2f}%, taker={COST_TAKER_RT*100:.2f}% round-trip")
    print()

    all_results = {}

    for tf in TIMEFRAMES:
        tf_min = {'1m': 1, '3m': 3}[tf]
        print(f"\n{'='*90}")
        print(f"TIMEFRAME: {tf}")
        print(f"{'='*90}")

        for pair in PAIRS:
            df = load_data(pair, tf)
            if df.empty:
                continue
            print(f"\n  {pair}/USDT: {len(df)} bars, "
                  f"{df['date'].min().strftime('%m/%d')} - {df['date'].max().strftime('%m/%d')}")
            df = compute_features(df)

            for model_name in MODELS:
                for label_type in ['triple_barrier', 'fixed_horizon']:
                    key = f"{pair}_{tf}_{model_name}_{label_type[:5]}"

                    result = walk_forward_calendar(
                        df, MODELS[model_name], label_type, tf_min)

                    if not result:
                        print(f"    [{model_name:12s}] {label_type[:5]:>5} — no trades")
                        continue

                    all_results[key] = result

                    # Show best threshold
                    best = max(result.items(), key=lambda x: x[1].get('pnl_maker_%', -999))
                    bt, bm = best
                    viable = "OK" if bm.get('pnl_maker_%', 0) > 0 else "NO"
                    print(f"    [{model_name:12s}] {label_type[:5]:>5} | "
                          f"best@{bt:.2f}: {bm['trades']:>5}t "
                          f"WR={bm['wr']:>5.1f}% "
                          f"PnL_mk={bm.get('pnl_maker_%',0):>6.2f}% "
                          f"PnL_tk={bm.get('pnl_taker_%',0):>6.2f}% "
                          f"Sharpe={bm.get('sharpe',0):>5.1f} {viable}")

    # ─── FINAL REPORT ────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("VIABLE CONFIGURATIONS (PnL after maker fees > 0, trades >= 50)")
    print("=" * 100)

    viable_list = []
    for key, result in all_results.items():
        for threshold, m in result.items():
            if m.get('pnl_maker_%', 0) > 0 and m['trades'] >= 50:
                viable_list.append({
                    'config': key, 'threshold': threshold,
                    **m
                })

    if viable_list:
        viable_list.sort(key=lambda x: x['pnl_maker_%'], reverse=True)
        print(f"\n{'Config':<35} {'Thr':>4} {'Trades':>6} {'T/day':>5} "
              f"{'WR%':>5} {'Gross%':>7} {'Maker%':>7} {'Taker%':>7} {'Sharpe':>6}")
        print("-" * 100)
        for v in viable_list[:30]:
            print(f"{v['config']:<35} {v['threshold']:>4.2f} "
                  f"{v['trades']:>6} {v.get('trades_per_day',0):>5.1f} "
                  f"{v['wr']:>4.1f}% {v.get('pnl_gross_%',0):>6.2f}% "
                  f"{v.get('pnl_maker_%',0):>6.2f}% {v.get('pnl_taker_%',0):>6.2f}% "
                  f"{v.get('sharpe',0):>5.1f}")
    else:
        print("\n  NO VIABLE CONFIGURATIONS (all negative PnL after maker fees)")
        print("\n  Top 10 by gross PnL (before fees):")
        all_configs = []
        for key, result in all_results.items():
            for threshold, m in result.items():
                if m['trades'] >= 50:
                    all_configs.append({'config': key, 'threshold': threshold, **m})
        if all_configs:
            all_configs.sort(key=lambda x: x.get('pnl_gross_%', 0), reverse=True)
            print(f"\n  {'Config':<35} {'Thr':>4} {'Trades':>6} {'WR%':>5} "
                  f"{'Gross%':>7} {'Maker%':>7} {'Taker%':>7}")
            print("  " + "-" * 90)
            for v in all_configs[:10]:
                print(f"  {v['config']:<35} {v['threshold']:>4.2f} "
                      f"{v['trades']:>6} {v['wr']:>4.1f}% "
                      f"{v.get('pnl_gross_%',0):>6.2f}% "
                      f"{v.get('pnl_maker_%',0):>6.2f}% "
                      f"{v.get('pnl_taker_%',0):>6.2f}%")

    # Save
    save_data = {}
    for k, r in all_results.items():
        save_data[k] = {str(t): v for t, v in r.items()}
    with open('research/ml_scalping_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to research/ml_scalping_results.json")


if __name__ == '__main__':
    main()

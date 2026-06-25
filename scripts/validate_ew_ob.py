"""
Full Walk-Forward + Monte Carlo validation for EW+OB strategy
3 timeframes: 1m, 5m, 15m
Multiple walk-forward windows for robustness
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')
from scripts.htf_ltf_backtest_v3 import (
    load_data, compute_rolling_ew_ob, simulate_ew_ob
)

DATA_DIR = Path("user_data/data/okx/futures")


def run_period(df_15m, df_ltf, start, end, tf, sl, tp, touch, dedup, max_hold):
    lookback = pd.Timedelta(days=30)
    df_15m_p = df_15m[(df_15m['date'] >= start - lookback) & (df_15m['date'] <= end)].reset_index(drop=True)
    df_ltf_p = df_ltf[(df_ltf['date'] >= start) & (df_ltf['date'] <= end)].reset_index(drop=True)

    if len(df_15m_p) < 120 or len(df_ltf_p) < 100:
        return None

    ew_ob = compute_rolling_ew_ob(df_15m_p, window=120, zz_pct=0.015)
    trades = simulate_ew_ob(df_ltf_p, ew_ob, tf=tf, sl_atr=sl, tp_atr=tp,
                           ob_touch_pct=touch, dedup_bars=dedup, max_hold_bars=max_hold)
    return trades


def stats(trades):
    if not trades:
        return None
    df_t = pd.DataFrame(trades)
    n = len(df_t)
    wins = (df_t['pnl'] > 0).sum()
    wr = wins / n * 100
    gp = df_t.loc[df_t['pnl'] > 0, 'pnl'].sum()
    gl = -df_t.loc[df_t['pnl'] <= 0, 'pnl'].sum()
    pf = gp / gl if gl > 0 else 99
    tot = df_t['pnl'].sum() * 100
    cum = df_t['pnl'].cumsum()
    dd = (cum - cum.cummax()).min() * 100
    grade = 'A' if pf > 1.5 and wr > 52 else 'B' if pf > 1.3 and wr > 48 else 'C' if pf > 1.1 else 'F'
    return {'n': n, 'wr': wr, 'pf': pf, 'pnl': tot, 'dd': dd, 'grade': grade}


def mc_test(trades, n_perms=200):
    if not trades or len(trades) < 10:
        return 1.0
    pnls = np.array([t['pnl'] for t in trades])
    real_pnl = pnls.sum()
    np.random.seed(42)
    count_beat = 0
    for _ in range(n_perms):
        shuffled = pnls.copy()
        signs = np.random.choice([-1, 1], size=len(shuffled))
        rand_pnl = (np.abs(shuffled) * signs).sum()
        if rand_pnl >= real_pnl:
            count_beat += 1
    return count_beat / n_perms


def main():
    print("Loading data...")
    df_15m = load_data('15m')
    df_5m = load_data('5m')
    df_1m = load_data('1m')

    print(f"1m: {df_1m.iloc[0]['date'].date()} to {df_1m.iloc[-1]['date'].date()}")
    print(f"5m: {df_5m.iloc[0]['date'].date()} to {df_5m.iloc[-1]['date'].date()}")
    print(f"15m: {df_15m.iloc[0]['date'].date()} to {df_15m.iloc[-1]['date'].date()}")

    # Configs per timeframe
    configs = {
        '1m': (1.5, 4.0, 0.003, 60, 240),   # SL1.5 TP4.0
        '5m': (1.0, 2.5, 0.003, 12, 36),     # SL1.0 TP2.5
        '15m': (1.5, 3.0, 0.003, 4, 12),     # SL1.5 TP3.0 (adapted for 15m)
    }

    # Walk-forward windows (IS 60d, OOS 30d, step 30d)
    wf_windows = [
        (pd.Timestamp('2026-01-01', tz='UTC'), pd.Timestamp('2026-03-01', tz='UTC'),
         pd.Timestamp('2026-03-01', tz='UTC'), pd.Timestamp('2026-03-31', tz='UTC')),
        (pd.Timestamp('2026-02-01', tz='UTC'), pd.Timestamp('2026-04-01', tz='UTC'),
         pd.Timestamp('2026-04-01', tz='UTC'), pd.Timestamp('2026-04-30', tz='UTC')),
        (pd.Timestamp('2026-03-01', tz='UTC'), pd.Timestamp('2026-05-01', tz='UTC'),
         pd.Timestamp('2026-05-01', tz='UTC'), pd.Timestamp('2026-05-31', tz='UTC')),
        (pd.Timestamp('2026-04-01', tz='UTC'), pd.Timestamp('2026-06-01', tz='UTC'),
         pd.Timestamp('2026-06-01', tz='UTC'), pd.Timestamp('2026-06-25', tz='UTC')),
    ]

    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION (IS 60d / OOS 30d / step 30d)")
    print("=" * 80)

    for tf_name in ['1m', '5m', '15m']:
        sl, tp, touch, dedup, max_hold = configs[tf_name]
        df_ltf = {'1m': df_1m, '5m': df_5m, '15m': df_15m}[tf_name]

        print(f"\n{'='*80}")
        print(f"  {tf_name} | SL={sl} TP={tp} touch={touch}")
        print(f"{'='*80}")

        all_oos_trades = []
        oos_results = []

        for is_start, is_end, oos_start, oos_end in wf_windows:
            # Check data availability
            ltf_start = df_ltf['date'].min()
            if is_start < ltf_start:
                is_start = ltf_start + pd.Timedelta(days=1)
            if oos_start < ltf_start:
                continue

            # IS
            trades_is = run_period(df_15m, df_ltf, is_start, is_end, tf_name, sl, tp, touch, dedup, max_hold)
            is_s = stats(trades_is)

            # OOS
            trades_oos = run_period(df_15m, df_ltf, oos_start, oos_end, tf_name, sl, tp, touch, dedup, max_hold)
            oos_s = stats(trades_oos)

            is_label = f"IS  {is_start.strftime('%m/%d')}-{is_end.strftime('%m/%d')}"
            oos_label = f"OOS {oos_start.strftime('%m/%d')}-{oos_end.strftime('%m/%d')}"

            if is_s:
                print(f"  {is_label}: [{is_s['grade']}] {is_s['n']:3d} trades WR={is_s['wr']:.1f}% PF={is_s['pf']:.2f} PnL={is_s['pnl']:+.1f}%")
            else:
                print(f"  {is_label}: INSUFFICIENT DATA")

            if oos_s:
                print(f"  {oos_label}: [{oos_s['grade']}] {oos_s['n']:3d} trades WR={oos_s['wr']:.1f}% PF={oos_s['pf']:.2f} PnL={oos_s['pnl']:+.1f}%")
                oos_results.append(oos_s)
                if trades_oos:
                    all_oos_trades.extend(trades_oos)
            else:
                print(f"  {oos_label}: INSUFFICIENT DATA")

            if is_s and oos_s:
                decay = (oos_s['pf'] - is_s['pf']) / is_s['pf'] * 100
                print(f"  {'':>26s} decay: PF {decay:+.0f}% | WR {oos_s['wr']-is_s['wr']:+.1f}pp")
            print()

        # Aggregate OOS
        if oos_results:
            n_pass = sum(1 for r in oos_results if r['grade'] in ('A', 'B'))
            avg_pf = np.mean([r['pf'] for r in oos_results])
            avg_wr = np.mean([r['wr'] for r in oos_results])
            total_trades = sum(r['n'] for r in oos_results)
            total_pnl = sum(r['pnl'] for r in oos_results)

            print(f"  --- AGGREGATE OOS ---")
            print(f"  Windows passed: {n_pass}/{len(oos_results)}")
            print(f"  Avg OOS PF: {avg_pf:.2f} | Avg OOS WR: {avg_wr:.1f}%")
            print(f"  Total OOS trades: {total_trades} | Total OOS PnL: {total_pnl:+.1f}%")

            # Monte Carlo on aggregated OOS trades
            mc_p = mc_test(all_oos_trades, n_perms=200)
            print(f"  MC p-value (aggregated OOS): {mc_p:.3f}")

            if n_pass >= len(oos_results) * 0.6 and mc_p < 0.05:
                print(f"  >>> VALIDATED <<<")
            elif n_pass >= len(oos_results) * 0.5:
                print(f"  >>> BORDERLINE <<<")
            else:
                print(f"  >>> FAILED <<<")


if __name__ == "__main__":
    main()

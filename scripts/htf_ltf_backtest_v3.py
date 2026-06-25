"""
Elliott Wave (15m) + Order Block (15m) → LTF Entry (5m/1m)
- 15m EW determines trend direction (bullish/bearish wave)
- 15m OB identifies institutional zones (support/resistance)
- 5m/1m enters when price touches OB zone in direction of EW trend
  - LONG: EW bullish + price touches bullish OB (demand zone)
  - SHORT: EW bearish + price touches bearish OB (supply zone)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

DATA_DIR = Path("user_data/data/okx/futures")


def load_data(tf):
    f = DATA_DIR / f"ETH_USDT_USDT-{tf}-futures.feather"
    df = pd.read_feather(f)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    return df.sort_values('date').reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# ELLIOTT WAVE (from eth_monitor.py logic)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ZigZagPoint:
    index: int
    price: float
    point_type: str  # "high" | "low"


def compute_zigzag(df, pct_threshold=0.015):
    """Percentage-based zigzag filter"""
    points = []
    if len(df) < 5:
        return points

    last_high_idx = 0
    last_high = df['high'].iloc[0]
    last_low_idx = 0
    last_low = df['low'].iloc[0]
    direction = 0  # 0=undecided, 1=up, -1=down

    for i in range(1, len(df)):
        h = df['high'].iloc[i]
        l = df['low'].iloc[i]

        if direction >= 0:
            if h > last_high:
                last_high = h
                last_high_idx = i
            if l < last_high * (1 - pct_threshold):
                if direction != 0 or last_high > df['high'].iloc[0] * (1 + pct_threshold / 2):
                    points.append(ZigZagPoint(last_high_idx, last_high, "high"))
                last_low = l
                last_low_idx = i
                direction = -1

        if direction <= 0:
            if l < last_low:
                last_low = l
                last_low_idx = i
            if h > last_low * (1 + pct_threshold):
                if direction != 0 or last_low < df['low'].iloc[0] * (1 - pct_threshold / 2):
                    points.append(ZigZagPoint(last_low_idx, last_low, "low"))
                last_high = h
                last_high_idx = i
                direction = 1

    return points


def detect_wave_trend(points) -> str:
    """
    Detect wave trend from zigzag points.
    Returns: 'bullish', 'bearish', or 'neutral'
    """
    if len(points) < 4:
        return 'neutral'

    last_points = points[-6:] if len(points) >= 6 else points[-4:]

    # Check impulse up: L-H-L-H-L-H (rising lows, rising highs)
    # Check impulse down: H-L-H-L-H-L (falling highs, falling lows)
    # Check correction patterns

    # Simple approach: look at last 4-6 points direction
    highs = [p for p in last_points if p.point_type == "high"]
    lows = [p for p in last_points if p.point_type == "low"]

    if len(highs) >= 2 and len(lows) >= 2:
        # Rising highs + rising lows = bullish
        hh = all(highs[i].price < highs[i + 1].price for i in range(len(highs) - 1))
        hl = all(lows[i].price < lows[i + 1].price for i in range(len(lows) - 1))
        # Falling highs + falling lows = bearish
        lh = all(highs[i].price > highs[i + 1].price for i in range(len(highs) - 1))
        ll = all(lows[i].price > lows[i + 1].price for i in range(len(lows) - 1))

        if hh and hl:
            return 'bullish'
        elif lh and ll:
            return 'bearish'

    # Fallback: last swing direction
    if len(points) >= 2:
        last = points[-1]
        prev = points[-2]
        if last.point_type == "high" and last.price > prev.price:
            return 'bullish'
        elif last.point_type == "low" and last.price < prev.price:
            return 'bearish'

    return 'neutral'


# ═══════════════════════════════════════════════════════════════════
# ORDER BLOCKS (from eth_monitor.py logic)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OrderBlock:
    ob_type: str  # "bullish" | "bearish"
    high: float
    low: float
    index: int
    strength: float  # impulse size in ATR


def compute_order_blocks(df, min_impulse_candles=3, min_impulse_atr=1.5):
    """
    Detect Order Blocks on given dataframe.
    Bullish OB: last red candle before 3+ green impulse candles
    Bearish OB: last green candle before 3+ red impulse candles
    """
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    obs = []

    for i in range(min_impulse_candles + 1, len(df) - 1):
        if pd.isna(atr.iloc[i]):
            continue

        # Check bullish OB: red candle at i, followed by 3+ green candles moving up
        if df['close'].iloc[i] < df['open'].iloc[i]:  # red candle
            impulse_ok = True
            impulse_move = 0
            for j in range(1, min_impulse_candles + 1):
                if i + j >= len(df):
                    impulse_ok = False
                    break
                if df['close'].iloc[i + j] <= df['open'].iloc[i + j]:  # not green
                    impulse_ok = False
                    break
                impulse_move += df['close'].iloc[i + j] - df['open'].iloc[i + j]

            if impulse_ok and impulse_move / atr.iloc[i] >= min_impulse_atr:
                obs.append(OrderBlock(
                    ob_type='bullish',
                    high=df['high'].iloc[i],
                    low=df['low'].iloc[i],
                    index=i,
                    strength=impulse_move / atr.iloc[i]
                ))

        # Check bearish OB: green candle at i, followed by 3+ red candles moving down
        if df['close'].iloc[i] > df['open'].iloc[i]:  # green candle
            impulse_ok = True
            impulse_move = 0
            for j in range(1, min_impulse_candles + 1):
                if i + j >= len(df):
                    impulse_ok = False
                    break
                if df['close'].iloc[i + j] >= df['open'].iloc[i + j]:  # not red
                    impulse_ok = False
                    break
                impulse_move += df['open'].iloc[i + j] - df['close'].iloc[i + j]

            if impulse_ok and impulse_move / atr.iloc[i] >= min_impulse_atr:
                obs.append(OrderBlock(
                    ob_type='bearish',
                    high=df['high'].iloc[i],
                    low=df['low'].iloc[i],
                    index=i,
                    strength=impulse_move / atr.iloc[i]
                ))

    # Deduplicate: keep strongest within 0.5% proximity
    filtered = []
    for ob in obs:
        mid = (ob.high + ob.low) / 2
        overlap = False
        for existing in filtered:
            existing_mid = (existing.high + existing.low) / 2
            if abs(mid - existing_mid) / existing_mid < 0.005:
                if ob.strength > existing.strength:
                    filtered.remove(existing)
                    filtered.append(ob)
                overlap = True
                break
        if not overlap:
            filtered.append(ob)

    return filtered


# ═══════════════════════════════════════════════════════════════════
# SIMULATION: EW + OB → LTF Entry
# ═══════════════════════════════════════════════════════════════════

def compute_rolling_ew_ob(df_15m, window=120, zz_pct=0.015):
    """
    For each 15m candle, compute:
    - Current EW trend (from last `window` candles)
    - Active OBs (from last `window` candles, not yet broken)
    Returns a dataframe with date, ew_trend, and list of active OBs.
    """
    results = []
    # Process every 4th candle for speed (15m = update every hour)
    step = 4

    for end_idx in range(window, len(df_15m), step):
        chunk = df_15m.iloc[end_idx - window:end_idx].reset_index(drop=True)
        date = df_15m.iloc[end_idx - 1]['date']
        price = df_15m.iloc[end_idx - 1]['close']

        # EW trend
        points = compute_zigzag(chunk, pct_threshold=zz_pct)
        trend = detect_wave_trend(points)

        # OBs (only keep untouched ones relative to current price)
        obs = compute_order_blocks(chunk)
        active_obs = []
        for ob in obs:
            # Bullish OB only valid if price is above it (hasn't broken below)
            if ob.ob_type == 'bullish' and price > ob.low:
                active_obs.append(ob)
            # Bearish OB only valid if price is below it (hasn't broken above)
            elif ob.ob_type == 'bearish' and price < ob.high:
                active_obs.append(ob)

        # Keep top 3 by strength for each type
        bull_obs = sorted([o for o in active_obs if o.ob_type == 'bullish'],
                         key=lambda x: x.strength, reverse=True)[:3]
        bear_obs = sorted([o for o in active_obs if o.ob_type == 'bearish'],
                         key=lambda x: x.strength, reverse=True)[:3]

        results.append({
            'date': date,
            'ew_trend': trend,
            'bull_obs': bull_obs,
            'bear_obs': bear_obs,
            'price_15m': price
        })

    return results


def simulate_ew_ob(df_ltf, ew_ob_data, tf='5m', sl_atr=1.5, tp_atr=3.0,
                   ob_touch_pct=0.002, dedup_bars=20, max_hold_bars=60):
    """
    Entry: price enters OB zone (within ob_touch_pct) + EW confirms direction
    - LONG: EW=bullish + price touches bullish OB zone
    - SHORT: EW=bearish + price touches bearish OB zone
    """
    df = df_ltf.copy()
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']

    # Build lookup: for each LTF bar, find the most recent EW+OB state
    # Convert ew_ob_data to dataframe for merge_asof
    ew_df = pd.DataFrame([{'date': r['date'], 'ew_trend': r['ew_trend'],
                           'bull_obs_data': r['bull_obs'], 'bear_obs_data': r['bear_obs']}
                          for r in ew_ob_data])
    ew_df['date'] = pd.to_datetime(ew_df['date'], utc=True)

    df = pd.merge_asof(df.sort_values('date'), ew_df.sort_values('date'),
                       on='date', direction='backward')

    trades = []
    position = None
    last_entry_idx = -999

    for i in range(50, len(df)):
        row = df.iloc[i]

        # Exit
        if position is not None:
            if position['side'] == 'short':
                hit_tp = row['low'] <= position['tp']
                hit_sl = row['high'] >= position['sl']
            else:
                hit_tp = row['high'] >= position['tp']
                hit_sl = row['low'] <= position['sl']

            bars = i - position['entry_idx']
            if hit_sl and hit_tp:
                if abs(row['open'] - position['sl']) < abs(row['open'] - position['tp']):
                    hit_tp = False
                else:
                    hit_sl = False

            if hit_sl:
                trades.append({'pnl': -sl_atr * position['atr_pct'] - 0.001,
                             'side': position['side'], 'exit': 'sl', 'bars': bars,
                             'date': row['date']})
                position = None
            elif hit_tp:
                trades.append({'pnl': tp_atr * position['atr_pct'] - 0.001,
                             'side': position['side'], 'exit': 'tp', 'bars': bars,
                             'date': row['date']})
                position = None
            elif bars >= max_hold_bars:
                pnl = ((row['close'] - position['entry_price']) / position['entry_price']) * \
                      (1 if position['side'] == 'long' else -1)
                trades.append({'pnl': pnl - 0.001, 'side': position['side'],
                             'exit': 'time', 'bars': bars, 'date': row['date']})
                position = None
            continue

        # Entry
        if i - last_entry_idx < dedup_bars:
            continue
        if pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        ew_trend = row.get('ew_trend', 'neutral')
        if pd.isna(ew_trend) or ew_trend == 'neutral':
            continue

        bull_obs = row.get('bull_obs_data', [])
        bear_obs = row.get('bear_obs_data', [])
        if not isinstance(bull_obs, list):
            bull_obs = []
        if not isinstance(bear_obs, list):
            bear_obs = []

        atr_pct = row['atr'] / row['close']
        price = row['close']
        low = row['low']
        high = row['high']

        # LONG: EW bullish + price touches bullish OB (demand zone)
        if ew_trend == 'bullish' and bull_obs:
            for ob in bull_obs:
                # Price entered the OB zone (low touched or close within zone)
                ob_top = ob.high * (1 + ob_touch_pct)
                ob_bot = ob.low * (1 - ob_touch_pct)
                if low <= ob_top and price >= ob_bot and row['is_green']:
                    position = {
                        'side': 'long', 'entry_price': price, 'entry_idx': i,
                        'sl': ob_bot - sl_atr * row['atr'],
                        'tp': price + tp_atr * row['atr'],
                        'atr_pct': atr_pct
                    }
                    last_entry_idx = i
                    break

        # SHORT: EW bearish + price touches bearish OB (supply zone)
        elif ew_trend == 'bearish' and bear_obs:
            for ob in bear_obs:
                ob_top = ob.high * (1 + ob_touch_pct)
                ob_bot = ob.low * (1 - ob_touch_pct)
                if high >= ob_bot and price <= ob_top and row['is_red']:
                    position = {
                        'side': 'short', 'entry_price': price, 'entry_idx': i,
                        'sl': ob_top + sl_atr * row['atr'],
                        'tp': price - tp_atr * row['atr'],
                        'atr_pct': atr_pct
                    }
                    last_entry_idx = i
                    break

    return trades


def report(trades, label):
    if not trades:
        print(f"  {label}: NO TRADES")
        return
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

    grade = "A" if pf > 1.5 and wr > 52 else "B" if pf > 1.3 and wr > 48 else "C" if pf > 1.1 else "F"

    sides = df_t.groupby('side').agg(cnt=('pnl', 'count'),
                                      wr_s=('pnl', lambda x: (x > 0).mean() * 100),
                                      tot_s=('pnl', 'sum')).reset_index()
    exits = df_t.groupby('exit').agg(cnt=('pnl', 'count'),
                                      avg=('pnl', 'mean')).reset_index()

    print(f"\n  {label}")
    print(f"  [{grade}] Trades={n} | WR={wr:.1f}% | PF={pf:.2f} | PnL={tot:.1f}% | DD={dd:.1f}%")
    for _, r in sides.iterrows():
        print(f"    {r['side']:6s}: {int(r['cnt']):3d} trades, WR={r['wr_s']:.1f}%, tot={r['tot_s']*100:.1f}%")
    for _, r in exits.iterrows():
        print(f"    exit {r['exit']:5s}: {int(r['cnt']):3d}, avg={r['avg']*100:.3f}%")


def main():
    print("=" * 70)
    print("ELLIOTT WAVE (15m) + ORDER BLOCK (15m) -> LTF ENTRY")
    print("=" * 70)

    print("\nLoading data...")
    df_15m = load_data('15m')
    df_5m = load_data('5m')
    df_1m = load_data('1m')

    print("Computing 15m EW + OB (this takes a minute)...")
    ew_ob_data = compute_rolling_ew_ob(df_15m, window=120, zz_pct=0.015)

    # Stats
    trends = [r['ew_trend'] for r in ew_ob_data]
    print(f"  EW trend distribution: { {t: trends.count(t) for t in set(trends)} }")
    avg_bull_obs = np.mean([len(r['bull_obs']) for r in ew_ob_data])
    avg_bear_obs = np.mean([len(r['bear_obs']) for r in ew_ob_data])
    print(f"  Avg active OBs: {avg_bull_obs:.1f} bullish, {avg_bear_obs:.1f} bearish")

    print("\n" + "-" * 70)
    print("Testing multiple configs...")
    print("-" * 70)

    configs = [
        # (tf, sl, tp, ob_touch, dedup, max_hold, label)
        ('5m', 1.5, 3.0, 0.002, 15, 48, "5m | SL1.5 TP3.0 | touch=0.2%"),
        ('5m', 1.5, 3.0, 0.004, 15, 48, "5m | SL1.5 TP3.0 | touch=0.4%"),
        ('5m', 2.0, 4.0, 0.003, 20, 60, "5m | SL2.0 TP4.0 | touch=0.3%"),
        ('5m', 1.0, 2.5, 0.003, 12, 36, "5m | SL1.0 TP2.5 | touch=0.3%"),
        ('5m', 1.5, 4.0, 0.003, 15, 60, "5m | SL1.5 TP4.0 | touch=0.3% (1:2.7)"),
        ('1m', 1.5, 3.0, 0.002, 60, 180, "1m | SL1.5 TP3.0 | touch=0.2%"),
        ('1m', 1.5, 3.0, 0.004, 60, 180, "1m | SL1.5 TP3.0 | touch=0.4%"),
        ('1m', 2.0, 4.0, 0.003, 90, 240, "1m | SL2.0 TP4.0 | touch=0.3%"),
        ('1m', 1.5, 4.0, 0.003, 60, 240, "1m | SL1.5 TP4.0 | touch=0.3% (1:2.7)"),
    ]

    for tf, sl, tp, touch, dedup, max_hold, label in configs:
        df_ltf = df_5m if tf == '5m' else df_1m
        trades = simulate_ew_ob(df_ltf, ew_ob_data, tf=tf, sl_atr=sl, tp_atr=tp,
                               ob_touch_pct=touch, dedup_bars=dedup, max_hold_bars=max_hold)
        report(trades, label)


if __name__ == "__main__":
    main()

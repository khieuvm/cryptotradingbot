"""Backtest candidates: Run regime_adaptive, volume_spike_rev, cb_adx_breakout
on XAU and NVDA using their 15m data. Simulate actual SL/TP execution.

Focus: Can our existing strategies generate edge on these new pairs?
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

DATA_DIR = Path(__file__).parent.parent / "data" / "okx" / "futures"
MAKER = 0.0002
TAKER = 0.0005


def load_15m(pair):
    fp = DATA_DIR / f"{pair}-15m-futures.feather"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_feather(fp)
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-21")].reset_index(drop=True)
    return df


def compute_indicators(df):
    """Compute all indicators for all 3 strategies."""
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"].astype(float)
    o = df["open"]

    # EMA
    df["ema21"] = ta.ema(c, length=21)
    df["ema60"] = ta.ema(c, length=60)
    df["ema200"] = ta.ema(c, length=200)
    df["ema8"] = ta.ema(c, length=8)

    # ATR
    df["atr"] = ta.atr(h, lo, c, length=14)
    df["atr_ma"] = ta.ema(df["atr"], length=50)
    df["atr_ratio"] = df["atr"] / (df["atr_ma"] + 1e-10)

    # ADX
    adx = ta.adx(h, lo, c, length=14)
    if adx is not None:
        df["adx"] = adx.iloc[:, 0]
        df["plus_di"] = adx.iloc[:, 1]
        df["minus_di"] = adx.iloc[:, 2]
    else:
        df["adx"] = df["plus_di"] = df["minus_di"] = 0

    # RSI
    df["rsi"] = ta.rsi(c, length=14)

    # Bollinger Bands
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        df["bb_upper"] = bb.iloc[:, 0]
        df["bb_mid"] = bb.iloc[:, 1]
        df["bb_lower"] = bb.iloc[:, 2]
    else:
        df["bb_upper"] = df["bb_mid"] = df["bb_lower"] = c

    # Volume
    df["vol_ema"] = ta.ema(v, length=20)
    df["vol_ratio"] = v / (df["vol_ema"] + 1e-10)

    # MACD
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd_hist"] = macd.iloc[:, 1]
    else:
        df["macd_hist"] = 0

    # OBV
    df["obv"] = ta.obv(c, v)
    df["obv_ema"] = ta.ema(df["obv"], length=20)
    df["obv_rising"] = (df["obv"] > df["obv_ema"]).astype(int)

    # SuperTrend
    try:
        st = ta.supertrend(h, lo, c, length=7, multiplier=3.0)
        if st is not None:
            st_dir_col = next((col for col in st.columns if "SUPERTd" in col), None)
            df["st_dir"] = st[st_dir_col].fillna(0) if st_dir_col else 0
        else:
            df["st_dir"] = 0
    except Exception:
        df["st_dir"] = 0

    # EMA cross freshness
    cross_up = (df["ema21"] > df["ema60"]) & (df["ema21"].shift(1) <= df["ema60"].shift(1))
    cross_down = (df["ema21"] < df["ema60"]) & (df["ema21"].shift(1) >= df["ema60"].shift(1))
    df["cross_up_recent"] = cross_up.rolling(5).max().fillna(0).astype(int)
    df["cross_down_recent"] = cross_down.rolling(5).max().fillna(0).astype(int)

    df["is_bull"] = (c > df["ema200"]).astype(int)
    df["is_bear"] = (c < df["ema200"]).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_regime_adaptive(df, i):
    """Detect regime_adaptive signals at bar i."""
    row = df.iloc[i]
    signals = []

    if float(row.get("atr_ratio", 0)) >= 2.2:
        return signals
    if float(row.get("vol_ratio", 0)) < 1.0:
        return signals
    if float(row.get("volume", 0)) <= 0:
        return signals

    is_trending = float(row.get("adx", 0)) > 25
    is_bull = int(row.get("is_bull", 0)) == 1
    is_bear = int(row.get("is_bear", 0)) == 1

    if is_trending:
        if (is_bull
            and int(row.get("cross_up_recent", 0)) == 1
            and float(row.get("ema21", 0)) > float(row.get("ema60", 0))
            and float(row.get("macd_hist", 0)) > 0
            and float(row.get("plus_di", 0)) > float(row.get("minus_di", 0))
            and int(row.get("st_dir", 0)) == 1):
            signals.append(("long", "ra_trend"))

        if (is_bear
            and int(row.get("cross_down_recent", 0)) == 1
            and float(row.get("ema21", 0)) < float(row.get("ema60", 0))
            and float(row.get("macd_hist", 0)) < 0
            and float(row.get("minus_di", 0)) > float(row.get("plus_di", 0))
            and int(row.get("st_dir", 0)) == -1):
            signals.append(("short", "ra_trend"))
    else:
        rsi = float(row.get("rsi", 50))
        prev_rsi = float(df.iloc[i-1].get("rsi", 50)) if i > 0 else 50

        if (prev_rsi < 28 and rsi > prev_rsi
            and float(row["close"]) < float(row.get("bb_lower", 0)) * 1.01
            and float(row["close"]) > float(row["open"])
            and int(row.get("obv_rising", 0)) == 1):
            signals.append(("long", "ra_range"))

        if (prev_rsi > 67 and rsi < prev_rsi
            and float(row["close"]) > float(row.get("bb_upper", 0)) * 0.99
            and float(row["close"]) < float(row["open"])
            and int(row.get("obv_rising", 0)) == 0):
            signals.append(("short", "ra_range"))

    return signals


def detect_volume_spike_rev(df, i):
    """Detect volume_spike_rev signals at bar i."""
    row = df.iloc[i]
    signals = []

    vol_ratio = float(row.get("vol_ratio", 0))
    if vol_ratio < 2.0:
        return signals

    c = float(row["close"])
    o = float(row["open"])
    h = float(row["high"])
    lo_val = float(row["low"])
    rsi = float(row.get("rsi", 50))

    body = abs(c - o)
    full_range = h - lo_val
    if full_range <= 0 or body <= 0:
        return signals

    shadow_ratio = (full_range - body) / body

    if shadow_ratio >= 3.0:
        # Hammer (bullish reversal)
        lower_shadow = min(c, o) - lo_val
        if lower_shadow > full_range * 0.5 and rsi < 33:
            signals.append(("long", "vsr"))

        # Shooting star (bearish reversal)
        upper_shadow = h - max(c, o)
        if upper_shadow > full_range * 0.5 and rsi > 72:
            signals.append(("short", "vsr"))

    return signals


def detect_cb_adx_breakout(df, i):
    """Detect cb_adx_breakout signals at bar i."""
    if i < 3:
        return []
    row = df.iloc[i]
    signals = []

    # 3-bar compression check
    h3 = max(float(df.iloc[j]["high"]) for j in range(i-2, i+1))
    l3 = min(float(df.iloc[j]["low"]) for j in range(i-2, i+1))
    atr = float(row.get("atr", 0))
    if atr <= 0:
        return signals

    range_3 = h3 - l3
    if range_3 >= 0.8 * atr:
        return signals

    adx = float(row.get("adx", 30))
    if adx >= 22:
        return signals

    # ADX rising (breakout starting)
    prev_adx = float(df.iloc[i-2].get("adx", 30))
    if adx <= prev_adx:
        return signals

    vol_ratio = float(row.get("vol_ratio", 0))
    if vol_ratio < 0.8:
        return signals

    rsi = float(row.get("rsi", 50))
    c = float(row["close"])
    o = float(row["open"])

    if rsi < 72 and c > o:
        signals.append(("long", "cb_adx"))
    if rsi > 22 and c < o:
        signals.append(("short", "cb_adx"))

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_trades(df, signals, sl_mult, tp_mult, max_bars=96, cooldown=5):
    """Simulate trades with SL/TP on 15m bars."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values

    trades = []
    last_trade_bar = -cooldown

    for bar_idx, direction, tag in signals:
        if bar_idx - last_trade_bar < cooldown:
            continue
        if bar_idx + max_bars >= len(close):
            continue

        entry = close[bar_idx]
        a = atr[bar_idx]
        if np.isnan(a) or a <= 0:
            continue

        if direction == "long":
            tp_price = entry + tp_mult * a
            sl_price = entry - sl_mult * a
        else:
            tp_price = entry - tp_mult * a
            sl_price = entry + sl_mult * a

        result = "timeout"
        exit_bar = bar_idx + max_bars
        exit_price = close[min(exit_bar, len(close) - 1)]

        for j in range(1, max_bars + 1):
            idx = bar_idx + j
            if idx >= len(close):
                break

            if direction == "long":
                if low[idx] <= sl_price:
                    result = "sl"
                    exit_price = sl_price
                    exit_bar = idx
                    break
                if high[idx] >= tp_price:
                    result = "tp"
                    exit_price = tp_price
                    exit_bar = idx
                    break
            else:
                if high[idx] >= sl_price:
                    result = "sl"
                    exit_price = sl_price
                    exit_bar = idx
                    break
                if low[idx] <= tp_price:
                    result = "tp"
                    exit_price = tp_price
                    exit_bar = idx
                    break

        if direction == "long":
            pnl = (exit_price - entry) / entry - 2 * TAKER
        else:
            pnl = (entry - exit_price) / entry - 2 * TAKER

        trades.append({
            "bar": bar_idx,
            "direction": direction,
            "tag": tag,
            "entry": entry,
            "exit": exit_price,
            "result": result,
            "pnl": pnl,
            "hold_bars": exit_bar - bar_idx,
            "date": str(df.iloc[bar_idx]["date"]),
        })
        last_trade_bar = bar_idx

    return trades


def analyze_trades(trades, pair, strategy):
    """Analyze trade results."""
    if not trades:
        return {"n": 0}

    pnls = np.array([t["pnl"] for t in trades])
    n = len(trades)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100
    total_pnl = pnls.sum() * 100
    avg_pnl = pnls.mean() * 100
    max_dd = 0
    running = 0
    for p in pnls:
        running += p
        if running < max_dd:
            max_dd = running

    # PF
    gross_win = pnls[pnls > 0].sum() if wins > 0 else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (n - wins) > 0 else 1e-10
    pf = gross_win / gross_loss if gross_loss > 0 else 0

    # Per direction
    long_trades = [t for t in trades if t["direction"] == "long"]
    short_trades = [t for t in trades if t["direction"] == "short"]

    long_wr = sum(1 for t in long_trades if t["pnl"] > 0) / len(long_trades) * 100 if long_trades else 0
    short_wr = sum(1 for t in short_trades if t["pnl"] > 0) / len(short_trades) * 100 if short_trades else 0

    # Result breakdown
    tp_count = sum(1 for t in trades if t["result"] == "tp")
    sl_count = sum(1 for t in trades if t["result"] == "sl")
    to_count = sum(1 for t in trades if t["result"] == "timeout")

    return {
        "n": n, "wr": wr, "pnl": total_pnl, "avg_pnl": avg_pnl,
        "pf": pf, "max_dd": max_dd * 100,
        "long_n": len(long_trades), "long_wr": long_wr,
        "short_n": len(short_trades), "short_wr": short_wr,
        "tp": tp_count, "sl": sl_count, "timeout": to_count,
        "avg_hold": np.mean([t["hold_bars"] for t in trades]),
    }


def main():
    print("=" * 90)
    print("CANDIDATE BACKTEST: XAU & NVDA with Active Strategies")
    print("Simulating actual SL/TP execution on 15m data (Jan-May 2026)")
    print("=" * 90)

    # Strategy configs (SL/TP multipliers)
    strategy_configs = {
        "regime_adaptive": {"sl_mult": 10.0, "tp_mult": 11.0, "max_bars": 96, "cooldown": 5},
        "volume_spike_rev": {"sl_mult": 3.0, "tp_mult": 5.5, "max_bars": 48, "cooldown": 5},
        "cb_adx_breakout": {"sl_mult": 5.0, "tp_mult": 5.0, "max_bars": 48, "cooldown": 6},
    }

    detectors = {
        "regime_adaptive": detect_regime_adaptive,
        "volume_spike_rev": detect_volume_spike_rev,
        "cb_adx_breakout": detect_cb_adx_breakout,
    }

    candidates = ["XAU_USDT_USDT", "NVDA_USDT_USDT"]
    # Also backtest active pairs for comparison
    active = ["SOL_USDT_USDT", "SPX_USDT_USDT"]
    all_pairs = candidates + active

    all_results = {}

    for pair in all_pairs:
        print(f"\n{'=' * 90}")
        print(f"  {pair}")
        print(f"{'=' * 90}")

        df = load_15m(pair)
        if df.empty or len(df) < 500:
            print(f"  Insufficient data")
            continue

        df = compute_indicators(df)
        print(f"  Data: {len(df)} bars ({len(df)/96:.0f} days)")
        print(f"  ATR: mean={df['atr'].mean():.4f}, "
              f"pct={((df['atr']/df['close'])*100).mean():.3f}%")

        pair_results = {}

        for strat_name, config in strategy_configs.items():
            detector = detectors[strat_name]

            # Detect signals
            signals = []
            for i in range(220, len(df)):
                sigs = detector(df, i)
                for direction, tag in sigs:
                    signals.append((i, direction, tag))

            if not signals:
                pair_results[strat_name] = {"n": 0}
                continue

            # Simulate trades
            trades = simulate_trades(
                df, signals,
                sl_mult=config["sl_mult"],
                tp_mult=config["tp_mult"],
                max_bars=config["max_bars"],
                cooldown=config["cooldown"],
            )

            stats = analyze_trades(trades, pair, strat_name)
            pair_results[strat_name] = stats

            if stats["n"] > 0:
                print(f"\n  {strat_name} (SL={config['sl_mult']}x, TP={config['tp_mult']}x):")
                print(f"    Signals: {len(signals)} | Trades: {stats['n']} "
                      f"(cooldown filtered {len(signals) - stats['n']})")
                print(f"    WR: {stats['wr']:.1f}% | PnL: {stats['pnl']:+.2f}% | PF: {stats['pf']:.2f}")
                print(f"    Long: {stats['long_n']} trades, WR={stats['long_wr']:.1f}%")
                print(f"    Short: {stats['short_n']} trades, WR={stats['short_wr']:.1f}%")
                print(f"    TP: {stats['tp']} | SL: {stats['sl']} | Timeout: {stats['timeout']}")
                print(f"    Max DD: {stats['max_dd']:.2f}% | Avg hold: {stats['avg_hold']:.0f} bars")
            else:
                print(f"\n  {strat_name}: No trades")

        all_results[pair] = pair_results

    # ─── COMPARISON TABLE ────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("COMPARISON TABLE")
    print(f"{'=' * 90}")
    print(f"\n  {'Pair':<18} {'Strategy':<20} {'Trades':<8} {'WR':<8} {'PnL%':<10} {'PF':<8} {'MaxDD%':<8}")
    print(f"  {'-' * 80}")

    for pair in all_pairs:
        if pair not in all_results:
            continue
        for strat_name in strategy_configs:
            stats = all_results[pair].get(strat_name, {"n": 0})
            if stats["n"] == 0:
                continue
            is_candidate = pair in candidates
            marker = " *NEW*" if is_candidate else ""
            print(f"  {pair:<18} {strat_name:<20} {stats['n']:<8} "
                  f"{stats['wr']:<7.1f}% {stats['pnl']:<+9.2f}% "
                  f"{stats['pf']:<7.2f} {stats['max_dd']:<7.2f}%{marker}")

    # ─── RECOMMENDATIONS ─────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 90}")

    for pair in candidates:
        if pair not in all_results:
            continue
        print(f"\n  {pair}:")
        profitable = []
        for strat_name, stats in all_results[pair].items():
            if stats["n"] > 0 and stats["pnl"] > 0 and stats["pf"] > 1.1:
                profitable.append((strat_name, stats))

        if profitable:
            print(f"    VIABLE! Profitable strategies:")
            for name, s in profitable:
                print(f"      - {name}: PnL={s['pnl']:+.2f}%, PF={s['pf']:.2f}, WR={s['wr']:.1f}%")
            print(f"    Action: Add to config/base.yaml active pairs")
        else:
            best = max(all_results[pair].items(), key=lambda x: x[1].get("pnl", -999))
            if best[1]["n"] > 0:
                print(f"    NOT VIABLE with current params. Best: {best[0]} PnL={best[1]['pnl']:+.2f}%")
                print(f"    Action: Needs parameter optimization (hyperopt)")
            else:
                print(f"    No trades generated. Strategy conditions too strict.")


if __name__ == "__main__":
    main()

"""Offline backtester: test strategies against local data without exchange connection.

Validates that:
1. All strategies compute indicators without error
2. Signal detection produces reasonable trade counts
3. Exit logic triggers correctly
4. Basic metrics (WR, PF, DD) are within expected ranges

This is NOT a replacement for freqtrade's full backtest — it's a quick verification
that the strategy code works correctly on real data.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.config import AppConfig, StrategyConfig
from engine.events import Direction
from strategies import get_strategy_class


@dataclass
class Trade:
    pair: str
    direction: Direction
    entry_time: datetime
    entry_rate: float
    strategy: str
    tag: str
    exit_time: datetime | None = None
    exit_rate: float | None = None
    exit_reason: str | None = None
    profit_pct: float = 0.0


@dataclass
class BacktestResult:
    strategy: str
    pair: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit_pct: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_duration_h: float = 0.0
    trades: list = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / max(1, self.total_trades)

    @property
    def profit_factor(self) -> float:
        gross_wins = sum(t.profit_pct for t in self.trades if t.profit_pct > 0)
        gross_losses = abs(sum(t.profit_pct for t in self.trades if t.profit_pct < 0))
        return gross_wins / max(0.0001, gross_losses)


TAKER_FEE = 0.0005
MAKER_FEE = 0.0002


def load_pair_data(pair: str, timeframe: str = "15m") -> pd.DataFrame:
    """Load OHLCV data for a pair from local feather files."""
    data_dir = Path(__file__).parent.parent / "data" / "okx" / "futures"
    filename = pair.replace("/", "_").replace(":", "_") + f"-{timeframe}-futures.feather"
    filepath = data_dir / filename
    if not filepath.exists():
        print(f"  WARNING: No data for {pair} at {filepath}")
        return pd.DataFrame()
    df = pd.read_feather(filepath)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def run_strategy_backtest(
    strategy_name: str,
    config: StrategyConfig,
    pair: str,
    timerange: tuple[str, str] | None = None,
) -> BacktestResult:
    """Run a single strategy on a single pair."""
    result = BacktestResult(strategy=strategy_name, pair=pair)

    df = load_pair_data(pair)
    if df.empty:
        return result

    if timerange:
        start, end = timerange
        df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    if len(df) < config.startup_candle_count + 50:
        print(f"  WARNING: Not enough data for {strategy_name}/{pair}")
        return result

    # Initialize strategy
    StrategyClass = get_strategy_class(strategy_name)
    strategy = StrategyClass(config)

    # Compute indicators on full dataframe
    df = strategy.compute_indicators(df, {"pair": pair})

    # Simulate bar-by-bar
    open_trade: Trade | None = None
    trades: list[Trade] = []
    equity_curve = [0.0]

    start_idx = config.startup_candle_count
    sl_mult = config.exit.get("sl_atr_mult", 2.0)
    tp_mult = config.exit.get("tp_atr_mult", 3.0)

    for i in range(start_idx, len(df)):
        current_bar = df.iloc[i]
        current_time = pd.Timestamp(current_bar["date"])
        close = float(current_bar["close"])
        high = float(current_bar["high"])
        low = float(current_bar["low"])

        # Check exits for open trade
        if open_trade is not None:
            entry_rate = open_trade.entry_rate
            hours_held = (current_time - open_trade.entry_time).total_seconds() / 3600

            # Calculate current profit
            if open_trade.direction == Direction.LONG:
                current_profit = (close - entry_rate) / entry_rate - 2 * TAKER_FEE
            else:
                current_profit = (entry_rate - close) / entry_rate - 2 * TAKER_FEE

            # Check strategy-specific exit
            trade_info = {
                "current_profit": current_profit,
                "current_time": current_time.to_pydatetime(),
                "entry_time": open_trade.entry_time,
                "entry_rate": entry_rate,
                "is_short": open_trade.direction == Direction.SHORT,
            }

            sub_df = df.iloc[: i + 1]
            exit_req = strategy.detect_exits(sub_df, pair, trade_info)

            # ATR-based stoploss check
            atr_col = None
            for col in df.columns:
                if col.endswith("_atr") and not col.endswith("_atr_ma"):
                    atr_col = col
                    break
            atr = float(current_bar.get(atr_col, 0)) if atr_col else 0

            sl_pct = sl_mult * atr / entry_rate if atr > 0 and entry_rate > 0 else 0.05

            # Check SL hit via high/low
            should_exit = False
            exit_reason = ""
            exit_price = close

            if open_trade.direction == Direction.LONG:
                sl_price = entry_rate * (1 - sl_pct)
                if low <= sl_price:
                    should_exit = True
                    exit_reason = "SL_HIT"
                    exit_price = sl_price
            else:
                sl_price = entry_rate * (1 + sl_pct)
                if high >= sl_price:
                    should_exit = True
                    exit_reason = "SL_HIT"
                    exit_price = sl_price

            if not should_exit and exit_req is not None:
                should_exit = True
                exit_reason = exit_req.reason
                exit_price = close

            if should_exit:
                if open_trade.direction == Direction.LONG:
                    profit = (exit_price - entry_rate) / entry_rate - 2 * TAKER_FEE
                else:
                    profit = (entry_rate - exit_price) / entry_rate - 2 * TAKER_FEE

                open_trade.exit_time = current_time.to_pydatetime()
                open_trade.exit_rate = exit_price
                open_trade.exit_reason = exit_reason
                open_trade.profit_pct = profit
                trades.append(open_trade)
                equity_curve.append(equity_curve[-1] + profit)
                open_trade = None

        # Check entries (only if no open trade for this pair)
        if open_trade is None:
            sub_df = df.iloc[: i + 1]
            signals = strategy.detect_entries(sub_df, pair)

            if signals:
                sig = signals[0]
                open_trade = Trade(
                    pair=pair,
                    direction=sig.direction,
                    entry_time=current_time.to_pydatetime(),
                    entry_rate=close,
                    strategy=strategy_name,
                    tag=sig.tag,
                )

    # Close any remaining open trade at last close
    if open_trade is not None:
        last_bar = df.iloc[-1]
        close = float(last_bar["close"])
        if open_trade.direction == Direction.LONG:
            profit = (close - open_trade.entry_rate) / open_trade.entry_rate - 2 * TAKER_FEE
        else:
            profit = (open_trade.entry_rate - close) / open_trade.entry_rate - 2 * TAKER_FEE
        open_trade.exit_time = pd.Timestamp(last_bar["date"]).to_pydatetime()
        open_trade.exit_rate = close
        open_trade.exit_reason = "end_of_data"
        open_trade.profit_pct = profit
        trades.append(open_trade)

    # Calculate metrics
    result.trades = trades
    result.total_trades = len(trades)
    result.winning_trades = sum(1 for t in trades if t.profit_pct > 0)
    result.losing_trades = sum(1 for t in trades if t.profit_pct <= 0)
    result.total_profit_pct = sum(t.profit_pct for t in trades)

    # Max drawdown
    peak = 0.0
    dd = 0.0
    cum = 0.0
    for t in trades:
        cum += t.profit_pct
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    result.max_drawdown = abs(dd)

    # Average duration
    durations = []
    for t in trades:
        if t.entry_time and t.exit_time:
            durations.append((t.exit_time - t.entry_time).total_seconds() / 3600)
    result.avg_trade_duration_h = sum(durations) / max(1, len(durations))

    return result


def main():
    print("=" * 70)
    print("OFFLINE BACKTEST — Strategy Verification")
    print("=" * 70)

    cfg = AppConfig("backtest")
    active = cfg.get_active_strategies()
    # Also include new research strategies
    all_strategies = list(set(active + [
        "nr7_breakout", "nr4_breakout", "donchian_breakout",
        "cb_adx_breakout", "cb_nr7_breakout",
        "vwap_meanrev", "micro_pullback", "volume_spike_rev",
    ]))
    pairs = cfg.get_pairs()
    timerange = ("2026-01-01", "2026-05-22")

    print(f"\nStrategies: {active}")
    print(f"Pairs: {pairs}")
    print(f"Timerange: {timerange[0]} to {timerange[1]}")
    print()

    all_results: list[BacktestResult] = []

    for strat_name in all_strategies:
        strat_cfg = cfg.get_strategy_config(strat_name)
        strat_pairs = [p for p in strat_cfg.pairs if p in pairs]

        print(f"\n{'-' * 70}")
        print(f"Strategy: {strat_name} (grade {strat_cfg.grade})")
        print(f"{'-' * 70}")

        for pair in strat_pairs:
            result = run_strategy_backtest(strat_name, strat_cfg, pair, timerange)
            all_results.append(result)

            if result.total_trades > 0:
                print(
                    f"  {pair:20s} | Trades: {result.total_trades:3d} | "
                    f"WR: {result.win_rate:.1%} | PF: {result.profit_factor:.2f} | "
                    f"Profit: {result.total_profit_pct:+.2%} | "
                    f"MaxDD: {result.max_drawdown:.2%} | "
                    f"AvgDur: {result.avg_trade_duration_h:.1f}h"
                )
            else:
                print(f"  {pair:20s} | No trades")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for strat_name in all_strategies:
        strat_results = [r for r in all_results if r.strategy == strat_name]
        total_trades = sum(r.total_trades for r in strat_results)
        total_wins = sum(r.winning_trades for r in strat_results)
        total_profit = sum(r.total_profit_pct for r in strat_results)
        max_dd = max((r.max_drawdown for r in strat_results), default=0)

        wr = total_wins / max(1, total_trades)
        all_trades = [t for r in strat_results for t in r.trades]
        gross_wins = sum(t.profit_pct for t in all_trades if t.profit_pct > 0)
        gross_losses = abs(sum(t.profit_pct for t in all_trades if t.profit_pct < 0))
        pf = gross_wins / max(0.0001, gross_losses)

        status = "OK" if total_trades >= 10 else "LOW TRADES"
        print(
            f"  {strat_name:25s} | {total_trades:3d} trades | "
            f"WR: {wr:.1%} | PF: {pf:.2f} | "
            f"Total: {total_profit:+.2%} | MaxDD: {max_dd:.2%} | {status}"
        )

    grand_total = sum(r.total_trades for r in all_results)
    grand_profit = sum(r.total_profit_pct for r in all_results)
    print(f"\n  {'TOTAL':25s} | {grand_total:3d} trades | Total profit: {grand_profit:+.2%}")


if __name__ == "__main__":
    main()

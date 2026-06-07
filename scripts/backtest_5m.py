"""5m vs 15m Timeframe Comparison Backtest.

Tests the 3 active Grade A/B strategies on 5m scalping data and compares
performance to the 15m baseline. Uses identical parameters from config/base.yaml.

Cost model applied:
  - Taker fee: 0.05% each side (entry + exit)
  - Round-trip: ~0.10% minimum
  - Funding: NOT applied here (simulation does not track hold duration vs settlement)

Strategies tested:
  - regime_adaptive  (Grade A on 15m)
  - volume_spike_rev (Grade B on 15m)
  - cb_adx_breakout  (Grade B on 15m)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.config import AppConfig, StrategyConfig
from engine.events import Direction
from strategies import get_strategy_class

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_STRATEGIES = ["regime_adaptive", "volume_spike_rev", "cb_adx_breakout"]
TIMERANGE = ("2026-01-01", "2026-05-21")
TAKER_FEE = 0.0005
DATA_DIR = ROOT / "data" / "okx" / "futures"

# 15m grade reference (established on the 15m data by same backtest engine)
# Populated at runtime — used for comparison table
_BASELINE_15M: dict[str, dict] = {}


# ── Data Classes ──────────────────────────────────────────────────────────────

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
    timeframe: str
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

    @property
    def avg_profit_per_trade(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_profit_pct / len(self.trades)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_pair_data(pair: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV feather file for the given pair and timeframe."""
    filename = pair.replace("/", "_").replace(":", "_") + f"-{timeframe}-futures.feather"
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return pd.DataFrame()
    df = pd.read_feather(filepath)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Core Simulation ───────────────────────────────────────────────────────────

def run_strategy_backtest(
    strategy_name: str,
    config: StrategyConfig,
    pair: str,
    timeframe: str,
    timerange: tuple[str, str] | None = None,
) -> BacktestResult:
    """Simulate bar-by-bar execution of a strategy on a single pair."""
    result = BacktestResult(strategy=strategy_name, pair=pair, timeframe=timeframe)

    df = load_pair_data(pair, timeframe)
    if df.empty:
        return result

    if timerange:
        start, end = timerange
        df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    if len(df) < config.startup_candle_count + 50:
        print(f"    WARNING: insufficient data for {strategy_name}/{pair} on {timeframe} "
              f"({len(df)} candles, need {config.startup_candle_count + 50})")
        return result

    StrategyClass = get_strategy_class(strategy_name)
    strategy = StrategyClass(config)

    # Compute all indicators upfront (full-dataframe vectorised pass)
    df = strategy.compute_indicators(df, {"pair": pair})

    # Find ATR column name used by this strategy
    atr_col = _find_atr_column(df, strategy_name)

    sl_mult = config.exit.get("sl_atr_mult", 2.0)
    tp_mult = config.exit.get("tp_atr_mult", 3.0)

    open_trade: Trade | None = None
    trades: list[Trade] = []
    equity_curve = [0.0]

    start_idx = config.startup_candle_count

    for i in range(start_idx, len(df)):
        bar = df.iloc[i]
        current_time = pd.Timestamp(bar["date"])
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])

        # ── Exit check ────────────────────────────────────────────────────────
        if open_trade is not None:
            entry_rate = open_trade.entry_rate
            hours_held = (current_time - open_trade.entry_time).total_seconds() / 3600

            if open_trade.direction == Direction.LONG:
                current_profit = (close - entry_rate) / entry_rate - 2 * TAKER_FEE
            else:
                current_profit = (entry_rate - close) / entry_rate - 2 * TAKER_FEE

            trade_info = {
                "current_profit": current_profit,
                "current_time": current_time.to_pydatetime(),
                "entry_time": open_trade.entry_time,
                "entry_rate": entry_rate,
                "is_short": open_trade.direction == Direction.SHORT,
                "enter_tag": open_trade.tag,
            }

            sub_df = df.iloc[: i + 1]
            exit_req = strategy.detect_exits(sub_df, pair, trade_info)

            # ATR stoploss check
            atr = float(bar.get(atr_col, 0)) if atr_col else 0
            sl_pct = sl_mult * atr / entry_rate if (atr > 0 and entry_rate > 0) else 0.05

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

        # ── Entry check ───────────────────────────────────────────────────────
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

    # Force-close any open position at end of data
    if open_trade is not None:
        last_bar = df.iloc[-1]
        last_close = float(last_bar["close"])
        if open_trade.direction == Direction.LONG:
            profit = (last_close - open_trade.entry_rate) / open_trade.entry_rate - 2 * TAKER_FEE
        else:
            profit = (open_trade.entry_rate - last_close) / open_trade.entry_rate - 2 * TAKER_FEE
        open_trade.exit_time = pd.Timestamp(last_bar["date"]).to_pydatetime()
        open_trade.exit_rate = last_close
        open_trade.exit_reason = "end_of_data"
        open_trade.profit_pct = profit
        trades.append(open_trade)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    result.trades = trades
    result.total_trades = len(trades)
    result.winning_trades = sum(1 for t in trades if t.profit_pct > 0)
    result.losing_trades = sum(1 for t in trades if t.profit_pct <= 0)
    result.total_profit_pct = sum(t.profit_pct for t in trades)

    peak, dd, cum = 0.0, 0.0, 0.0
    for t in trades:
        cum += t.profit_pct
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    result.max_drawdown = abs(dd)

    durations = [
        (t.exit_time - t.entry_time).total_seconds() / 3600
        for t in trades
        if t.entry_time and t.exit_time
    ]
    result.avg_trade_duration_h = sum(durations) / max(1, len(durations))

    return result


def _find_atr_column(df: pd.DataFrame, strategy_name: str) -> str | None:
    """Find strategy-specific ATR column by prefix convention."""
    prefixes = {
        "regime_adaptive": "ra_atr",
        "volume_spike_rev": "vs_atr",
        "cb_adx_breakout": "cba_atr",
    }
    preferred = prefixes.get(strategy_name)
    if preferred and preferred in df.columns:
        return preferred
    # Fallback: first column ending in _atr (not _atr_ma)
    for col in df.columns:
        if col.endswith("_atr") and not col.endswith("_atr_ma"):
            return col
    return None


# ── Aggregation Helpers ───────────────────────────────────────────────────────

def aggregate_strategy_results(results: list[BacktestResult]) -> dict:
    """Collapse per-pair results into a single strategy summary."""
    all_trades = [t for r in results for t in r.trades]
    total = len(all_trades)
    wins = sum(1 for t in all_trades if t.profit_pct > 0)
    total_profit = sum(t.profit_pct for t in all_trades)
    gross_wins = sum(t.profit_pct for t in all_trades if t.profit_pct > 0)
    gross_losses = abs(sum(t.profit_pct for t in all_trades if t.profit_pct < 0))
    pf = gross_wins / max(0.0001, gross_losses)
    max_dd = max((r.max_drawdown for r in results), default=0.0)
    durations = [
        (t.exit_time - t.entry_time).total_seconds() / 3600
        for r in results for t in r.trades
        if t.entry_time and t.exit_time
    ]
    avg_dur = sum(durations) / max(1, len(durations))

    exit_counts: dict[str, int] = {}
    for t in all_trades:
        reason = t.exit_reason or "unknown"
        exit_counts[reason] = exit_counts.get(reason, 0) + 1

    return {
        "trades": total,
        "wr": wins / max(1, total),
        "pf": pf,
        "profit": total_profit,
        "max_dd": max_dd,
        "avg_dur_h": avg_dur,
        "exit_counts": exit_counts,
        "avg_per_trade": total_profit / max(1, total),
    }


# ── Printing Helpers ──────────────────────────────────────────────────────────

def _grade(pf: float, wr: float) -> str:
    if pf > 1.5 and wr > 0.52:
        return "A"
    if pf > 1.3 and wr > 0.48:
        return "B"
    if pf > 1.1 and wr > 0.45:
        return "C"
    return "F"


def print_per_pair_block(results: list[BacktestResult], timeframe: str) -> None:
    print(f"\n  Per-pair ({timeframe}):")
    print(f"  {'Pair':<22} {'Trades':>6} {'WR':>7} {'PF':>6} {'Profit%':>9} {'MaxDD%':>8} {'AvgDur':>8}")
    print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*6} {'-'*9} {'-'*8} {'-'*8}")
    for r in results:
        if r.total_trades == 0:
            print(f"  {r.pair:<22} {'--':>6}")
            continue
        print(
            f"  {r.pair:<22} {r.total_trades:>6} {r.win_rate:>7.1%} "
            f"{r.profit_factor:>6.2f} {r.total_profit_pct:>+9.2%} "
            f"{r.max_drawdown:>8.2%} {r.avg_trade_duration_h:>7.1f}h"
        )


def print_exit_breakdown(exit_counts: dict[str, int], total: int) -> None:
    if not exit_counts or total == 0:
        return
    print(f"\n  Exit reasons:")
    for reason, cnt in sorted(exit_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:<25} {cnt:>4} ({cnt/total:>5.1%})")


def print_comparison_table(
    strat_summaries_15m: dict[str, dict],
    strat_summaries_5m: dict[str, dict],
) -> None:
    """Print the final side-by-side 5m vs 15m comparison table."""
    divider = "=" * 100

    print(f"\n\n{divider}")
    print("5m vs 15m COMPARISON TABLE")
    print(divider)
    print(
        f"{'Strategy':<22} {'TF':>4} {'Trades':>7} {'WR':>7} {'PF':>6} "
        f"{'Profit%':>9} {'MaxDD%':>8} {'AvgDur':>8} {'Grade':>6}"
    )
    print(f"{'-'*22} {'-'*4} {'-'*7} {'-'*7} {'-'*6} {'-'*9} {'-'*8} {'-'*8} {'-'*6}")

    for strat in TARGET_STRATEGIES:
        for tf_label, summaries in [("15m", strat_summaries_15m), ("5m", strat_summaries_5m)]:
            s = summaries.get(strat)
            if s is None:
                print(f"  {strat:<22} {tf_label:>4}  (no data)")
                continue
            g = _grade(s["pf"], s["wr"])
            profit_str = f"{s['profit']:+.2%}"
            print(
                f"  {strat:<22} {tf_label:>4} {s['trades']:>7} {s['wr']:>7.1%} "
                f"{s['pf']:>6.2f} {profit_str:>9} {s['max_dd']:>8.2%} "
                f"{s['avg_dur_h']:>7.1f}h {g:>6}"
            )
        # delta row
        s15 = strat_summaries_15m.get(strat)
        s5 = strat_summaries_5m.get(strat)
        if s15 and s5 and s15["trades"] > 0 and s5["trades"] > 0:
            d_wr = s5["wr"] - s15["wr"]
            d_pf = s5["pf"] - s15["pf"]
            d_profit = s5["profit"] - s15["profit"]
            d_dd = s5["max_dd"] - s15["max_dd"]
            d_dur = s5["avg_dur_h"] - s15["avg_dur_h"]
            sign_wr = "+" if d_wr >= 0 else ""
            sign_pf = "+" if d_pf >= 0 else ""
            sign_profit = "+" if d_profit >= 0 else ""
            sign_dd = "+" if d_dd >= 0 else ""
            sign_dur = "+" if d_dur >= 0 else ""
            print(
                f"  {'  delta (5m-15m)':<22} {'':>4} {'':>7} "
                f"{sign_wr}{d_wr:>6.1%} {sign_pf}{d_pf:>5.2f} "
                f"{sign_profit}{d_profit:>8.2%} {sign_dd}{d_dd:>7.2%} "
                f"{sign_dur}{d_dur:>6.1f}h"
            )
        print()

    print(divider)


# ── Verdict Generator ─────────────────────────────────────────────────────────

def print_verdict(strat: str, s15: dict | None, s5: dict | None) -> None:
    if s15 is None or s5 is None:
        return

    grade_5m = _grade(s5["pf"], s5["wr"])
    grade_15m = _grade(s15["pf"], s15["wr"])

    # Scalping suitability: higher trade frequency, shorter duration
    freq_ratio = s5["trades"] / max(1, s15["trades"])
    dur_ratio = s5["avg_dur_h"] / max(0.001, s15["avg_dur_h"])

    verdict_lines = [f"  Strategy: {strat}"]
    verdict_lines.append(f"    15m grade: {grade_15m} | 5m grade: {grade_5m}")
    verdict_lines.append(
        f"    5m trade count vs 15m: {freq_ratio:.1f}x  "
        f"(avg duration: {s5['avg_dur_h']:.1f}h vs {s15['avg_dur_h']:.1f}h)"
    )

    if grade_5m in ("A", "B"):
        if s5["max_dd"] <= 0.15:
            verdict_lines.append("    VERDICT: 5m holds up — suitable for scalping deployment")
        else:
            verdict_lines.append(
                f"    VERDICT: 5m grade acceptable but MaxDD={s5['max_dd']:.1%} is elevated — optimize SL"
            )
    elif grade_5m == "C":
        verdict_lines.append("    VERDICT: 5m marginal — further parameter tuning required before deploy")
    else:
        verdict_lines.append("    VERDICT: 5m FAILS grading — strategy does not translate to 5m")

    # Regime-adaptive specific note
    if strat == "regime_adaptive":
        verdict_lines.append(
            "    NOTE: EMA 18/50 on 5m covers 90m/250m (vs 270m/750m on 15m). "
            "Regime detection will be noisier."
        )
    elif strat == "cb_adx_breakout":
        verdict_lines.append(
            "    NOTE: 3-bar compression on 5m = 15-min zone (vs 45-min on 15m). "
            "Higher false-breakout risk expected."
        )
    elif strat == "volume_spike_rev":
        verdict_lines.append(
            "    NOTE: Volume spikes on 5m are more frequent and noisier. "
            "Spike multiplier may need tightening."
        )

    print("\n".join(verdict_lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 80)
    print("5m SCALPING BACKTEST — Active Strategy Evaluation")
    print("=" * 80)
    print(f"Timerange : {TIMERANGE[0]} to {TIMERANGE[1]}")
    print(f"Strategies: {TARGET_STRATEGIES}")
    print(f"Cost model: taker {TAKER_FEE:.2%} per side (round-trip ~{2*TAKER_FEE:.2%})")
    print(f"Data dir  : {DATA_DIR}")
    print()

    cfg = AppConfig("backtest")
    all_pairs = cfg.get_pairs()

    summaries_15m: dict[str, dict] = {}
    summaries_5m: dict[str, dict] = {}

    for timeframe in ("15m", "5m"):
        print()
        print(f"{'#' * 80}")
        print(f"# TIMEFRAME: {timeframe}")
        print(f"{'#' * 80}")

        tf_summaries: dict[str, dict] = {}

        for strat_name in TARGET_STRATEGIES:
            strat_cfg = cfg.get_strategy_config(strat_name)
            # Use only pairs that exist for this strategy and that have data available
            strat_pairs_raw = [p for p in strat_cfg.pairs if p in all_pairs]

            # Check which pairs actually have data files for this timeframe
            strat_pairs = []
            for p in strat_pairs_raw:
                fname = p.replace("/", "_").replace(":", "_") + f"-{timeframe}-futures.feather"
                if (DATA_DIR / fname).exists():
                    strat_pairs.append(p)
                else:
                    print(f"  SKIP {p}: no {timeframe} data file found")

            print(f"\n{'─' * 80}")
            print(f"Strategy : {strat_name}  (grade={strat_cfg.grade}, "
                  f"startup={strat_cfg.startup_candle_count} candles)")
            print(f"Pairs    : {strat_pairs}")
            print(f"SL/TP    : {strat_cfg.exit.get('sl_atr_mult')}x ATR / "
                  f"{strat_cfg.exit.get('tp_atr_mult')}x ATR")
            print(f"{'─' * 80}")

            strat_results: list[BacktestResult] = []
            for pair in strat_pairs:
                result = run_strategy_backtest(
                    strat_name, strat_cfg, pair, timeframe, TIMERANGE
                )
                strat_results.append(result)

                if result.total_trades > 0:
                    print(
                        f"  {pair:<22} | Trades: {result.total_trades:>3} | "
                        f"WR: {result.win_rate:.1%} | PF: {result.profit_factor:.2f} | "
                        f"Profit: {result.total_profit_pct:>+.2%} | "
                        f"MaxDD: {result.max_drawdown:.2%} | "
                        f"AvgDur: {result.avg_trade_duration_h:.1f}h"
                    )
                else:
                    print(f"  {pair:<22} | No trades generated")

            # Per-pair block
            if strat_results:
                print_per_pair_block(strat_results, timeframe)

            # Aggregate
            if any(r.total_trades > 0 for r in strat_results):
                summary = aggregate_strategy_results(strat_results)
                tf_summaries[strat_name] = summary
                g = _grade(summary["pf"], summary["wr"])
                print(
                    f"\n  TOTAL [{strat_name}] {timeframe}: "
                    f"{summary['trades']} trades | "
                    f"WR={summary['wr']:.1%} | PF={summary['pf']:.2f} | "
                    f"Profit={summary['profit']:+.2%} | "
                    f"MaxDD={summary['max_dd']:.2%} | "
                    f"AvgDur={summary['avg_dur_h']:.1f}h | Grade={g}"
                )
                print_exit_breakdown(summary["exit_counts"], summary["trades"])
            else:
                print(f"\n  TOTAL [{strat_name}] {timeframe}: NO TRADES")

        if timeframe == "15m":
            summaries_15m = tf_summaries
        else:
            summaries_5m = tf_summaries

    # ── Final comparison & verdicts ───────────────────────────────────────────
    print_comparison_table(summaries_15m, summaries_5m)

    print("\nINDIVIDUAL VERDICTS")
    print("=" * 80)
    for strat in TARGET_STRATEGIES:
        print_verdict(strat, summaries_15m.get(strat), summaries_5m.get(strat))
        print()

    print("=" * 80)
    print("SCALPING RECOMMENDATION SUMMARY")
    print("=" * 80)
    deployable = []
    optimize = []
    reject = []
    for strat in TARGET_STRATEGIES:
        s = summaries_5m.get(strat)
        if s is None:
            reject.append(strat)
            continue
        g = _grade(s["pf"], s["wr"])
        if g in ("A", "B") and s["max_dd"] <= 0.20:
            deployable.append(f"{strat} (Grade {g})")
        elif g == "C" or s["max_dd"] > 0.20:
            optimize.append(f"{strat} (Grade {g}, DD={s['max_dd']:.1%})")
        else:
            reject.append(f"{strat} (Grade {g})")

    if deployable:
        print(f"  Deploy to 5m  : {', '.join(deployable)}")
    if optimize:
        print(f"  Optimize first: {', '.join(optimize)}")
    if reject:
        print(f"  Reject / skip : {', '.join(reject)}")
    print()


if __name__ == "__main__":
    main()

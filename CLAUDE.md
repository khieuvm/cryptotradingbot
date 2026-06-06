# Freqtrade Crypto Trading Bot — OKX Futures

## Overview

Automated cryptocurrency futures trading system on OKX using freqtrade as the execution engine. Event-driven architecture with self-contained strategies, centralized risk management, and multi-agent research pipeline.

## Architecture

```
engine/              Central orchestrator, event bus, config, state
strategies/          Self-contained strategy units (BaseStrategy subclasses)
adapters/            Freqtrade IStrategy bridge (CryptoEngine)
risk/                Position sizing, stoploss, circuit breaker, exposure
indicators/          Shared indicator library (trend, volatility, volume, momentum, market_data)
config/              base.yaml + env overlays (backtest/dryrun/live)
scripts/             Deployment helpers (run_master, daily_report)
tests/               pytest test suite
research/            Research scripts and analysis
```

### Key Components

- **CryptoEngine** (`adapters/ft_strategy.py`): Single IStrategy that bridges freqtrade to the Orchestrator
- **Orchestrator** (`engine/orchestrator.py`): Coordinates strategy lifecycle, signal routing, risk checks
- **EventBus** (`engine/event_bus.py`): Thread-safe typed pub/sub for internal communication
- **BaseStrategy** (`strategies/base.py`): Enhanced base class with lifecycle hooks (on_tick, on_entry, on_exit)
- **AppConfig** (`engine/config.py`): Loads `config/base.yaml` + env overlay, generates freqtrade JSON on-the-fly

### Execution Paths

- **Backtest**: Column-based (populate_indicators → populate_entry_trend → custom_exit)
- **Live**: Event-driven (on_tick → emit_signal → EventBus → Orchestrator confirms)

## Market & Exchange

- **Exchange**: OKX futures, isolated margin
- **Pairs**: BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT, SPX/USDT:USDT, NVDA/USDT:USDT
- **Timeframe**: 15m primary
- **Leverage**: 3x default, 5x max
- **Trading**: 24/7, both LONG and SHORT

## Cost Model

- Maker fee: 0.02%
- Taker fee: 0.05%
- Funding rate: ~0.01% per 8h (settles at 00:00, 08:00, 16:00 UTC)
- Round-trip (taker both sides): ~0.10%
- With 24h hold including funding: ~0.13%

## Active Strategies

| Strategy | Grade | Description |
|----------|-------|-------------|
| regime_adaptive | A | ADX regime detection, trending/ranging signals, EMA cross freshness |
| volume_spike_rev | B | Volume spike + reversal candle (hammer/shooting star) + RSI extreme |
| cb_adx_breakout | B | 3-bar compression + low ADX breakout |
| meanrev_confluence | C | RSI + BB + volume in trend direction |
| volatility_compression | C | BB/KC squeeze fire, price-action breakout (ETH/SPX only) |

## Strategy Framework

Each strategy in `strategies/<name>.py` inherits `BaseStrategy`:

```python
@register_strategy
class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def compute_indicators(self, df, metadata) -> df
    def detect_entries(self, df, pair) -> list[Signal]
    def detect_exits(self, df, pair, trade_info) -> ExitRequest | None
```

Parameters loaded from `config/base.yaml` under `strategies.<name>`.

## Risk Management

- **Circuit Breaker** (`risk/circuit_breaker.py`): WR tracking, drawdown halt, consecutive losses, auto-disable
- **Position Sizer** (`risk/position_sizer.py`): ATR-risk-based sizing, portfolio-heat-capped
- **Stoploss** (`risk/stoploss.py`): 3-phase (initial ATR → break-even → trail-lock)
- **Exposure** (`risk/exposure.py`): Max portfolio heat 15%, correlation limits, max 6 trades

### Auto-Disable Rules

- WR < 40% over last 10 trades → 24h cooldown (per strategy per pair)
- WR < 35% over last 20 trades → halt + alert
- Daily DD > 8% → halt all trading 4h
- Weekly DD > 12% → halt + manual review
- 5 consecutive losses → halt affected strategy

## Grading System

| Grade | OOS PF | OOS WR | MC p-value | Max DD |
|-------|--------|--------|------------|--------|
| A | > 1.5 | > 52% | < 0.03 | < 10% |
| B | > 1.3 | > 48% | < 0.05 | < 15% |
| C | > 1.1 | > 45% | < 0.10 | < 20% |
| F | < 1.0 | any | > 0.10 | > 20% |

Only Grade A and B strategies are active in live trading. Grade C requires further validation.

## Validation Requirements

- Minimum 90 days of data, 50+ trades
- Walk-forward: 60d in-sample / 30d out-of-sample, 30d step
- Monte Carlo: 100 signal-time permutations, p < 0.05
- Must pass on each pair independently

## Key Commands

```bash
# Run backtest
python ft_run.py backtesting --strategy CryptoEngine --timerange 20260101-

# Run hyperopt
python ft_run.py hyperopt --strategy CryptoEngine --hyperopt-loss SharpeHyperOptLoss -e 500

# Dry-run trading
python ft_run.py trade --strategy CryptoEngine --env dryrun

# Download data
python ft_run.py download-data --exchange okx --pairs BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT SPX/USDT:USDT NVDA/USDT:USDT -t 15m 1h --days 180
```

## Config

- **Single source of truth**: `config/base.yaml` — all strategy parameters, risk settings, market config
- **Env overlays**: `config/env/{backtest,dryrun,live}.yaml` — environment-specific overrides
- **Generated**: Freqtrade JSON config generated on-the-fly by `AppConfig.get_freqtrade_config()`
- **Never hardcode** parameters in strategy files

## File Conventions

- Strategy code: `strategies/<name>.py`
- Strategy parameters: `config/base.yaml` under `strategies.<name>`
- Research scripts: `research/analyze_<name>.py`
- Backtest results: `backtest_results/`
- External research: `E:\Trading\research\` (VN30F1M adapted strategies)

## Environment

- Python 3.12, Windows 11
- Corporate proxy: F-Soft (SSL bypass via `ft_run.py`)
- Telegram notifications enabled
- FreqUI on localhost:8080

## Research Pipeline

```
researcher agent    -> candidate discovery (web, papers, E:\Trading\research)
analyst agent       -> data-driven measurement against random baseline
backtester agent    -> freqtrade backtest + walk-forward + hyperopt
backtest-validator  -> Monte Carlo + grading (A/B/C/F)
reviewer agent      -> code review (safety, config, integration)
qa_tester agent     -> end-to-end QA (runtime, regression, smoke tests)
circuit_breaker     -> live performance monitoring + auto-disable
```

## Code Change Workflow

When modifying or adding strategy code, follow this pipeline:

1. **Backtest** — Run offline backtest (`scripts/backtest_offline.py` or `scripts/hyperopt_fast.py`)
2. **Analyst** — Measure signal quality against random baseline, check for over-trading
3. **Review** — Static code review (safety, look-ahead bias, config consistency)
4. **QA Test** — End-to-end runtime tests (imports, indicators, signals, regressions)
5. **Deploy** — Only if QA passes; Grade A/B to live, Grade C to dry-run only

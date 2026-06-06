# Freqtrade Crypto Trading Bot — OKX Futures

## Overview

Automated cryptocurrency futures trading system on OKX using freqtrade as the execution engine, with a multi-agent pipeline for strategy research, validation, and deployment.

## Architecture

- **Engine**: Freqtrade (exchange connectivity, backtesting, live execution, FreqUI)
- **Strategies**: Modular combo system — `combos/` contains strategy logic, `strategies/CryptoMaster_OKX.py` is the unified IStrategy dispatcher
- **Config**: `config/strategy_config.yaml` is the single source of truth for all strategy parameters
- **Agents**: `.claude/agents/` — researcher, backtester, analyst, reviewer
- **Skills**: `skills/` — edge-researcher, backtest-validator, regime-detector, trade-postmortem

## Market & Exchange

- **Exchange**: OKX futures, isolated margin
- **Pairs**: BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT
- **Timeframe**: 15m primary
- **Trading**: 24/7 (crypto), both LONG and SHORT

## Cost Model

- Maker fee: 0.02%
- Taker fee: 0.05%
- Funding rate: ~0.01% per 8h (settles at 00:00, 08:00, 16:00 UTC)
- Round-trip (taker both sides): ~0.10%
- With 24h hold including funding: ~0.13%

## Strategy Pipeline

```
researcher agent → candidate discovery
analyst agent → data-driven measurement
backtester agent → freqtrade backtest + walk-forward
backtest-validator skill → Monte Carlo + grading (A/B/C/F)
reviewer agent → code review before deployment
signal_tracker → live performance monitoring + auto-disable
trade-postmortem skill → outcome classification + feedback
```

## Combo Framework

Each strategy lives in `combos/<name>.py` inheriting `BaseCryptoCombo`:
- `populate_indicators(df, metadata)` → add technical indicators
- `detect_long(df, metadata)` → boolean Series for long entries
- `detect_short(df, metadata)` → boolean Series for short entries
- Parameters loaded from `config/strategy_config.yaml`

## Grading System

| Grade | OOS PF | OOS WR | MC p-value | Max DD |
|-------|--------|--------|------------|--------|
| A | > 1.5 | > 52% | < 0.03 | < 10% |
| B | > 1.3 | > 48% | < 0.05 | < 15% |
| C | > 1.1 | > 45% | < 0.10 | < 20% |
| F | < 1.0 | any | > 0.10 | > 20% |

Only Grade A and B combos are active in live trading.

## Auto-Disable Rules

- WR < 40% over last 10 trades → 24h cooldown (per combo per pair)
- WR < 35% over last 20 trades → manual review required
- Max DD > 15% intraday → pause all combos 4h
- 3 consecutive SL hits on same pair → 2h cooldown

## Validation Requirements

- Minimum 90 days of data, 50+ trades
- Walk-forward: 60d in-sample / 30d out-of-sample, 30d step
- Monte Carlo: 100 signal-time permutations, p < 0.05
- Must pass on each pair independently

## Key Commands

```bash
# Run backtest
python ft_run.py backtesting --strategy CryptoMaster_OKX --timerange 20260101-

# Run hyperopt
python ft_run.py hyperopt --strategy CryptoMaster_OKX --hyperopt-loss SharpeHyperOptLoss -e 500

# Dry-run trading
python ft_run.py trade --strategy CryptoMaster_OKX -c config_master.json

# Download data
python ft_run.py download-data --exchange okx --pairs BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT -t 15m 1h --days 180
```

## File Conventions

- Strategy parameters: ONLY in `config/strategy_config.yaml` (never hardcoded)
- Research scripts: `research/analyze_<name>.py`
- Backtest results: `backtest_results/`
- Agent outputs: structured markdown in `research/` or `combos/CATALOG.md`

## Environment

- Python 3.12, Windows
- Corporate proxy: F-Soft (SSL bypass via `ft_run.py`)
- Telegram notifications enabled
- FreqUI on localhost:8080

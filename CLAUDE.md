# Freqtrade Crypto Trading Bot — OKX Futures

## Quick Setup (New Machine)

```bash
# 1. Clone
git clone <repo-url>
cd freqtrade

# 2. Python 3.12 + dependencies
pip install -r requirements.txt
# TA-Lib requires C library: https://github.com/TA-Lib/ta-lib-python#dependencies
# FinBERT (for news monitor): pip install transformers torch

# 3. Config — copy templates and fill in credentials
cp config/env/dryrun.yaml.example config/env/dryrun.yaml
cp config/env/live.yaml.example config/env/live.yaml
# Edit dryrun.yaml: set telegram token + chat_id
# Edit live.yaml: set telegram + OKX API key/secret/password

# 4. Download market data
python ft_run.py download-data --exchange okx --pairs ETH/USDT:USDT -t 15m 1h --days 180

# 5. Run
python ft_run.py trade --strategy CryptoEngine --env dryrun    # bot trading
python scripts/eth_monitor.py --verbose                        # market monitor (separate process)
```

## Overview

Automated ETH futures trading on OKX via freqtrade. Event-driven architecture with self-contained strategies, centralized risk management.

## Architecture

```
engine/              Orchestrator, event bus, config, state
strategies/          Self-contained strategy units (BaseStrategy subclasses)
adapters/            Freqtrade IStrategy bridge (CryptoEngine)
risk/                Position sizing, stoploss, circuit breaker, exposure
indicators/          Shared indicator library (trend, volatility, volume, momentum, market_data)
config/              base.yaml + env overlays (backtest/dryrun/live)
scripts/             ETH monitor, daily report, backtest/hyperopt helpers
tests/               pytest test suite
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
- **Pairs**: ETH/USDT:USDT (primary focus)
- **Timeframe**: 15m
- **Leverage**: 3x default, 5x max
- **Trading**: 24/7, both LONG and SHORT

## Cost Model

- Maker fee: 0.02%
- Taker fee: 0.05%
- Funding rate: ~0.01% per 8h (settles at 00:00, 08:00, 16:00 UTC)
- Round-trip (taker both sides): ~0.10%
- With 24h hold including funding: ~0.13%

## Active Strategies

Only Grade A and B are active. Parameters in `config/base.yaml` under `strategies.<name>`.

| Strategy | Grade | Description |
|----------|-------|-------------|
| regime_adaptive | A | ADX regime detection, trending/ranging signals, EMA cross freshness |
| volume_spike_rev | B | Volume spike + big red candle reversal SHORT; hammer + RSI<25 LONG |
| cb_adx_breakout | B | Bollinger bandwidth compression + ADX<30 declining → breakout |
| ha_stoch_rev | B | Heikin-Ashi candle color reversal + Stochastic oversold/overbought |
| stoch_trend_pullback | B | Stochastic pullback in established EMA trend |

## ETH Market Monitor (`scripts/eth_monitor.py`)

Standalone process — runs independently from the trading bot. Sends Telegram alerts only when something notable happens.

```
while True (10s):
    ├── OKX ticker → compare vs S/R levels → alert on touch/break
    │
    ├── Every 5 min:
    │   ├── Market context (funding, OI, taker ratio, L/S, liquidations)
    │   │   → alert only on extreme conditions
    │   └── News: RSS (CoinTelegraph, CoinDesk, Decrypt, Blockworks)
    │       → filter ETH keywords → dedup → FinBERT classify
    │       → alert only NEGATIVE with confidence > 75%
    │
    └── Every 24h: Fear & Greed index → alert if change > 10 points
```

```bash
python scripts/eth_monitor.py                  # continuous loop
python scripts/eth_monitor.py --once --verbose  # single cycle, print everything
```

News is informational only — NOT used as a trading filter.

## Risk Management

- **Circuit Breaker** (`risk/circuit_breaker.py`): WR tracking, drawdown halt, consecutive losses, auto-disable
- **Position Sizer** (`risk/position_sizer.py`): ATR-risk-based sizing, portfolio-heat-capped
- **Stoploss** (`risk/stoploss.py`): 3-phase (fixed ATR from entry → break-even → trail-lock). Returns account-level values (×leverage) because freqtrade divides by leverage internally. 1-candle grace period on entry candle to avoid SL trigger on volatile entries.
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

Only Grade A and B strategies are active in live trading.

## Key Commands

```bash
# Dry-run trading
python ft_run.py trade --strategy CryptoEngine --env dryrun

# Backtest
python ft_run.py backtesting --strategy CryptoEngine --timerange 20260101-

# Hyperopt
python ft_run.py hyperopt --strategy CryptoEngine --hyperopt-loss SharpeHyperOptLoss -e 500

# Download data
python ft_run.py download-data --exchange okx --pairs ETH/USDT:USDT -t 15m 1h --days 180

# ETH monitor (separate terminal)
python scripts/eth_monitor.py --verbose
```

## Config

- **Single source of truth**: `config/base.yaml` — all strategy parameters, risk settings, market config
- **Env overlays**: `config/env/{dryrun,live}.yaml` — credentials + environment overrides (gitignored, use `.example` as template)
- **Generated**: Freqtrade JSON config generated on-the-fly by `AppConfig.get_freqtrade_config()`
- **Never hardcode** parameters in strategy files

## Environment

- Python 3.12, Windows 11
- Telegram notifications enabled
- FreqUI on localhost:8080

## Code Change Workflow

1. **Backtest** — Run offline backtest (`scripts/backtest_offline.py` or `scripts/hyperopt_fast.py`)
2. **Analyst** — Measure signal quality against random baseline, check for over-trading
3. **Review** — Static code review (safety, look-ahead bias, config consistency)
4. **QA Test** — End-to-end runtime tests (imports, indicators, signals, regressions)
5. **Deploy** — Only if QA passes; Grade A/B to live, Grade C to dry-run only

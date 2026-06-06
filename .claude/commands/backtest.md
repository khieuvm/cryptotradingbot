---
description: "Run a backtest and analyze results for a combo or the full strategy"
---

# /backtest [combo_name] [timerange]

Run freqtrade backtesting and provide detailed analysis.

## Usage
```
/backtest                           # Full strategy, all pairs, all time
/backtest regime_adaptive           # Specific combo (filter by enter_tag)
/backtest --timerange 20260401-     # Custom timerange
```

## What happens
1. Backtester agent runs `python ft_run.py backtesting --strategy CryptoMaster_OKX`
2. Analyzes results: per-combo, per-pair, per-direction breakdown
3. Reports: WR, PF, Sharpe, Max DD, exit reasons, time analysis
4. Compares to baseline if available

## Requirements
- Data must be downloaded first (`ft_run.py download-data`)
- Config file `config_master.json` must exist

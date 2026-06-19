---
description: "Run freqtrade hyperopt to optimize strategy parameters"
---

# /hyperopt [strategy] [loss_fn]

Run parameter optimization on a strategy and report best params vs current config.

## Usage
```
/hyperopt regime_adaptive
/hyperopt volume_spike_rev SharpeHyperOptLoss
/hyperopt cb_adx_breakout ProfitDrawDownHyperOptLoss
/hyperopt regime_adaptive --epochs 500
```

## What happens
1. Backtester agent runs freqtrade hyperopt with the given loss function
2. Compares best params to current values in `config/base.yaml`
3. Reports: best WR, PF, Sharpe on in-sample period
4. Recommends params to update (with delta from current)
5. Reminds you to re-validate with `/validate` before deploying

## Commands run internally

```bash
# Standard hyperopt
python ft_run.py hyperopt --strategy CryptoEngine \
  --hyperopt-loss SharpeHyperOptLoss -e 300

# More epochs for thorough search
python ft_run.py hyperopt --strategy CryptoEngine \
  --hyperopt-loss ProfitDrawDownHyperOptLoss -e 500

# Show best results
python ft_run.py hyperopt-show --best --no-header
```

## Loss functions

| Loss | Use when |
|------|----------|
| `SharpeHyperOptLoss` | Default — maximize risk-adjusted return |
| `ProfitDrawDownHyperOptLoss` | Want lower drawdown with profit trade-off |
| `OnlyProfitHyperOptLoss` | Research only — ignores risk metrics |

## After hyperopt

1. Review proposed params — check for over-fitting (too narrow ranges = suspicious)
2. Update `config/base.yaml` under `strategies.<name>` if params look stable
3. Run `/validate <strategy>` on updated params before deploying
4. Only deploy if walk-forward OOS still meets grade threshold

## Requirements
- Data must be downloaded: `python ft_run.py download-data --days 180`
- Strategy must have `HyperOpt` parameters defined (space annotations)
- Run `/backtest` first to confirm the strategy produces trades

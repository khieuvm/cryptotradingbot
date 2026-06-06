---
description: "Run walk-forward validation + Monte Carlo on a combo"
---

# /validate [combo_name]

Formal validation pipeline: walk-forward + Monte Carlo significance test.

## Usage
```
/validate regime_adaptive
/validate trend_composite
```

## What happens
1. Walk-forward: 60d IS / 30d OOS, rolling 30d step
2. Monte Carlo: 100 signal-time permutations
3. Stress tests: funding, flash crash, weekend, per-pair
4. Grade assignment: A / B / C / F
5. Update `combos/CATALOG.md` with grade
6. Update `config/strategy_config.yaml` if grade changed

## Requirements
- Minimum 180 days of data
- Combo must produce 50+ trades in full period

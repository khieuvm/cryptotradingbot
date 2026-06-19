---
description: "Root cause analysis for a strategy that is underperforming"
---

# /diagnose [strategy] [pair?]

Investigate why a strategy is underperforming. Classifies the degradation type and recommends an action.

## Usage
```
/diagnose volume_spike_rev
/diagnose regime_adaptive ETH/USDT:USDT
/diagnose cb_adx_breakout SOL/USDT:USDT --days 30
```

## What happens
1. Spawns the `diagnostician` agent
2. Loads recent trades and compares to historical baseline
3. Classifies root cause (see degradation types below)
4. Reports confidence level and recommended action

## Degradation types

| Type | Symptoms | Typical action |
|------|----------|----------------|
| `regime_change` | WR declined, but entry conditions still fire | Pause, add regime filter |
| `parameter_drift` | Entry rate same, avg win/loss ratio shifted | Re-run `/hyperopt` |
| `edge_decay` | Both WR and frequency declining | Kill or deep research |
| `market_microstructure` | Specific pair degraded, others OK | Deactivate on that pair |
| `data_issue` | Sudden change in behavior, looks like a cliff | Check data quality |
| `cost_bleed` | Marginally profitable before fees, negative after | Widen entry filter |

## Output

```
## Diagnosis: <strategy> [<pair>]

**Period analyzed:** last N days (N trades)
**Baseline:** historical M days (M trades)

### Signal Quality Trend
| Window | WR | PF | Avg Win | Avg Loss |
|--------|----|----|---------|----------|
| -90d   |    |    |         |          |
| -60d   |    |    |         |          |
| -30d   |    |    |         |          |
| -14d   |    |    |         |          |

### Root Cause
**Classification:** <degradation_type>
**Confidence:** HIGH / MEDIUM / LOW
**Evidence:** [What specifically changed]

### Recommended Action
**Action:** tune / pause / kill / monitor
**Specifics:** [Exact param changes or pair to deactivate]
**Timeline:** [How long to pause / when to re-evaluate]
```

## When to run `/diagnose`

- Circuit breaker auto-disabled the strategy
- WR dropped more than 10pp vs baseline
- Profit factor below 1.0 over last 20 trades
- You see a string of consecutive losses (5+)

## Related commands
- `/performance` — Review recent trade postmortem
- `/hyperopt` — Re-optimize params after diagnosis
- `/deploy deactivate` — Deactivate if action is "kill"

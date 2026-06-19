---
description: "Show system health: strategy grades, active/inactive, circuit breaker state"
---

# /status

Print a dashboard of all strategies — grades, active state, circuit breaker, and recent performance.

## Usage
```
/status
/status --all        # Include inactive/F-grade strategies
/status --brief      # Summary line only
```

## What happens
1. Read `config/base.yaml` for all strategies, their grades, and active flags
2. Check circuit breaker state (`risk/circuit_breaker.py`) for any auto-disables
3. Print strategy table grouped by grade
4. Print system health summary line

## Output format

```
## System Status — <date>

### Active Strategies
| Strategy | Grade | Pairs | Active | CB State | Last N Trades | WR |
|----------|-------|-------|--------|----------|---------------|----|
| regime_adaptive | A | ETH/SOL/SPX/DOGE | ✓ | OK | 10 | 48% |
| volume_spike_rev | B | ETH/SOL/SPX | ✓ | OK | 10 | 62% |
| cb_adx_breakout | B | ETH/SOL/SPX/DOGE | ✓ | WARN (WR 41%) | 10 | 41% |

### Inactive / Research Strategies
| Strategy | Grade | Reason |
|----------|-------|--------|
| meanrev_confluence | F | Rejected |
| trend_composite | C | Dry-run only |
...

### Circuit Breaker Events (last 7 days)
- <event list or "None">

### System Health
✓ OK — 3 strategies active, no auto-disables
  OR
⚠ WARNING — cb_adx_breakout approaching WR threshold (41% < 48%)
  OR
✗ HALT — <strategy> auto-disabled: <reason>
```

## Circuit breaker thresholds (from CLAUDE.md)
- WR < 40% over last 10 trades → 24h cooldown
- WR < 35% over last 20 trades → halt
- Daily DD > 8% → halt all trading 4h
- Weekly DD > 12% → halt + manual review
- 5 consecutive losses → halt strategy

## Related commands
- `/performance` — Detailed postmortem of recent trades
- `/diagnose <strategy>` — RCA if a strategy is struggling
- `/deploy deactivate <strategy>` — Manually deactivate a strategy

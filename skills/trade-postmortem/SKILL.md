# Trade Postmortem Skill

Classify trade outcomes, feed signal tracker, identify systemic issues.

## When to Run

- **Daily:** Classify all trades closed in last 24h
- **Weekly:** Full performance review, identify degrading combos
- **Monthly:** Edge discovery cycle — identify gaps, research new hypotheses

## Outcome Categories

| Category | Definition | Action |
|----------|-----------|--------|
| TRUE_POSITIVE | Signal correct, profitable after all costs | None — working as intended |
| FALSE_POSITIVE | SL hit, entry direction was wrong | Review entry conditions |
| REGIME_MISMATCH | Combo fired in wrong regime (e.g., trend combo in range) | Improve regime detection |
| FUNDING_DRAG | Would be profitable without funding cost | Optimize hold duration |
| TIME_DECAY | Exited via time cut (not SL or TP) | Entry timing issue |
| VOLATILITY_CRUSH | Entered expecting expansion, got contraction | Filter low-vol periods |
| CORRELATION_BREAK | Alt trade failed because BTC diverged | Review BTC sentiment gate |
| CASCADE_STOP | SL hit by liquidation cascade, not real signal failure | Add OI/liquidation filter |

## Classification Logic

```python
def classify_trade(trade):
    profit = trade.close_profit
    exit_reason = trade.exit_reason
    hold_hours = (trade.close_date - trade.open_date).total_seconds() / 3600

    if profit > 0:
        return "TRUE_POSITIVE"

    if "time_cut" in exit_reason:
        return "TIME_DECAY"

    if exit_reason == "stoploss" or exit_reason == "stop_loss":
        # Check if BTC moved against us sharply
        btc_move = get_btc_move_during_trade(trade)
        if abs(btc_move) > 0.03:  # 3% BTC move
            return "CASCADE_STOP"
        return "FALSE_POSITIVE"

    if "funding" in get_pnl_breakdown(trade):
        return "FUNDING_DRAG"

    return "FALSE_POSITIVE"
```

## Signal Tracker Integration

After classification:
1. Record win/loss to `SignalTracker`
2. Check auto-disable rules
3. If disabled, log and send Telegram alert

## Weekly Review Template

```
## Weekly Postmortem: [Date Range]

### Per-Combo Performance
| Combo | Trades | WR | PF | Avg Hold | Status |
|-------|--------|----|----|----------|--------|

### Outcome Distribution
| Category | Count | % |
|----------|-------|---|

### Auto-Disable Events
- [combo:pair] disabled at [time] — WR [X]% over last 10 trades

### Issues Identified
1. [Issue] → [Proposed fix]

### Action Items
- [ ] [Action for next week]
```

## Monthly Edge Discovery Trigger

At month end:
1. Identify combos with WR < 50% (underperforming)
2. Identify time windows with no signals (coverage gaps)
3. Identify pairs with no profitable combo (opportunity)
4. Generate 2-3 research hypotheses for researcher agent
5. Queue in `combos/CATALOG.md` as candidates

# Workflows — Orchestration Schedule

How the agent pipeline operates on recurring schedules.

## Real-Time (Every Trade)

```
confirm_trade_entry():
  → signal_tracker.is_disabled(combo, pair)?
  → regime matches combo?
  → BTC sentiment OK?
  → funding rate OK?

confirm_trade_exit():
  → signal_tracker.record_trade(combo, pair, win/loss)
  → check auto-disable rules
```

## Daily (Every 24h)

```
1. Run daily_report.py:
   - Query trade database for closed trades
   - Per-combo/per-pair breakdown
   - Exit reason distribution
   - Signal tracker rolling WR
   - Any auto-disable events
   - Send Telegram summary

2. Trade postmortem (daily):
   - Classify all closed trades
   - Update signal tracker
   - Flag any combo with WR < 45% over last 7 days
```

## Weekly (Sunday)

```
1. Full weekly postmortem:
   - All combos, all pairs
   - Identify underperformers (PF < 1.0 over week)
   - Check for regime mismatch patterns

2. Re-validate degrading combos:
   - If WR dropped 10%+ from baseline → run backtest on last 30d
   - If still underperforming → downgrade to Grade C (dry-run only)

3. Signal tracker review:
   - Reset any expired cooldowns
   - Review disabled combos — still warranted?
```

## Monthly (1st Sunday)

```
1. Full monthly performance review:
   - Compare to baseline grades
   - Identify any Grade A/B that degraded to C/F
   - Identify coverage gaps (time windows, pairs without signals)

2. Edge discovery cycle:
   - Use researcher agent to find 2-3 new candidates
   - Add to combos/CATALOG.md
   - Prioritize based on:
     a. Fills a coverage gap
     b. Crypto-specific structural edge
     c. Complements (not duplicates) existing combos

3. Validation cycle:
   - Run backtest-validator on new candidates
   - Grade A/B → deploy to live
   - Grade C → dry-run observation
   - Grade F → reject, add to anti_patterns.md

4. Parameter refresh:
   - Run hyperopt on existing combos with last 90d data
   - Compare new params vs old on OOS (last 30d)
   - Only update if improvement > 10% in PF
```

## Quarterly

```
1. Full architecture review:
   - Are all combos still relevant?
   - Any market structure changes?
   - New exchange features to exploit?

2. Deep research:
   - Commission researcher for new strategy types
   - Explore new data sources (on-chain, order flow)
   - Consider new pairs
```

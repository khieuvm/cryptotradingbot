---
description: "Analyze recent trading performance with postmortem classification"
---

# /performance [period]

Review recent trading performance with outcome classification.

## Usage
```
/performance              # Last 7 days
/performance today        # Today only
/performance 30d          # Last 30 days
/performance 2026-05-01   # Since specific date
```

## What happens
1. Query trade database for closed trades in period
2. Classify each trade (TRUE_POS, FALSE_POS, REGIME_MISMATCH, etc.)
3. Per-combo and per-pair breakdown
4. Signal tracker status (any auto-disables?)
5. Recommendations for improvement

## Output
- Performance table (combo × pair)
- Outcome distribution
- Auto-disable events
- Action items

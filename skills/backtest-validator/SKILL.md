# Backtest Validator Skill

Formal walk-forward validation with Monte Carlo statistical significance testing for crypto futures strategies.

## When to Use

Run this BEFORE promoting any combo to Grade A/B (live deployment).

## Validation Pipeline

### Step 1: Walk-Forward Setup

```
Data requirement: minimum 180 days of 15m OHLCV
Window configuration:
  - In-sample (IS): 60 days
  - Out-of-sample (OOS): 30 days
  - Step: 30 days
  - Result: ~4 OOS windows

For each window:
  1. Run backtest on IS period (evaluate performance)
  2. Run backtest on OOS period with SAME parameters (no re-optimization)
  3. Record OOS metrics: PF, WR, max DD, Sharpe, trade count
```

### Step 2: OOS Performance Requirements

| Metric | Grade A | Grade B | Grade C | Grade F |
|--------|---------|---------|---------|---------|
| OOS PF | > 1.5 | > 1.3 | > 1.1 | < 1.0 |
| OOS WR | > 52% | > 48% | > 45% | < 45% |
| Max DD | < 10% | < 15% | < 20% | > 20% |
| MC p-value | < 0.03 | < 0.05 | < 0.10 | > 0.10 |

Additional requirements:
- No single OOS window with PF < 0.8 (catastrophic failure)
- OOS PF must be > 50% of IS PF (no severe overfitting)
- Trade count per OOS window >= 10 (statistical relevance)

### Step 3: Monte Carlo Permutation Test

```python
# Purpose: prove the signal timing matters (not just market drift)
# Method:
#   1. Record real trade entry times and durations
#   2. Shuffle entry times randomly (preserve frequency per day)
#   3. Run 100 permutations with random entry timing
#   4. Compare real PnL to distribution of random PnLs
#   5. Require: real PnL > 95th percentile of random (p < 0.05)
```

### Step 4: Crypto-Specific Stress Tests

1. **Funding rate stress:** Does strategy survive sustained negative funding (long-biased) or positive funding (short-biased)?
2. **Flash crash:** Performance during 10%+ daily drops (filter extreme days)
3. **Weekend:** Performance on weekends (low liquidity)
4. **Leverage sensitivity:** Results at 1x, 3x, 5x leverage
5. **Per-pair independence:** Pass on at least 2/3 pairs individually

### Step 5: Red Flags (Auto-Reject)

- OOS PF < 50% of IS PF → overfitting
- MC p > 0.10 → can't distinguish from random
- Max DD > 25% at 3x leverage → unacceptable risk
- All profits from single pair → not robust
- 80%+ of exits are time-cuts → entry timing is wrong
- Sharpe < 0.5 → risk-adjusted returns too low

## Grade Assignment

```
IF all OOS windows pass AND MC p < 0.03 AND max DD < 10%:
    → Grade A: Deploy to live immediately
ELIF all OOS windows pass AND MC p < 0.05 AND max DD < 15%:
    → Grade B: Deploy with monitoring, re-validate monthly
ELIF most OOS windows pass AND MC p < 0.10:
    → Grade C: Dry-run only, needs parameter refinement
ELSE:
    → Grade F: Reject, document in anti_patterns.md
```

## Output Format

```
## Walk-Forward Validation: [Combo Name]

**Data:** [timerange], [days] days
**Windows:** [N] OOS periods

### Per-Window Results
| Window | IS PF | OOS PF | OOS WR | OOS DD | Trades |
|--------|-------|--------|--------|--------|--------|

### Monte Carlo
- Permutations: 100
- Real PnL: $X
- Random 95th percentile: $Y
- p-value: Z

### Stress Tests
- Funding stress: PASS/FAIL
- Flash crash: PASS/FAIL
- Weekend: PASS/FAIL
- Per-pair: BTC [PASS/FAIL] ETH [PASS/FAIL] SOL [PASS/FAIL]

### VERDICT: Grade [A/B/C/F]
[Reasoning]
```

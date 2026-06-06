# Edge Researcher Skill

Systematic, data-driven discovery of new trading edges for crypto futures.

## Philosophy

- Measure the problem first. Don't add a filter until you've quantified what you're filtering.
- Every hypothesis must be tested against random baseline.
- Focus on VOLATILITY PREDICTION and STRUCTURAL EDGES, not direction prediction.
- Kill fast: if first measurement shows no signal, move on.

## Process

1. **Observe anomaly** — notice a pattern in data, research, or trade outcomes
2. **Form hypothesis** — "If X condition is met, next N bars will have ATR > Y"
3. **Write focused script** — `research/analyze_<hypothesis>.py`
4. **Measure** — frequency, MFE distribution, comparison to random
5. **Decide** — PROMISING (proceed to backtest) or DEAD (document in anti_patterns.md)
6. **Iterate** — if promising, refine parameters and test edge cases

## Crypto-Specific Research Areas

### Tier 1: Highest Priority
1. **Funding Rate Contrarian** — When funding > 0.05%/8h, short bias wins
2. **Volatility Compression Breakout** — BB squeeze + KC on crypto 15m
3. **BTC Sentiment Regime** — BTC RSI/trend as filter for alt trades

### Tier 2: Medium Priority
4. **OI Divergence** — Price rising + OI falling = weak rally
5. **Cross-Pair Momentum Transfer** — BTC moves first, alts follow with lag
6. **Hour-of-Day Effect** — Asian range → US breakout pattern

### Tier 3: Exploratory
7. **Liquidation Level Support** — Clusters of long liquidations as support
8. **Weekend Effect** — Low liquidity creates different patterns
9. **Correlation Breakdown** — When ETH decorrelates from BTC

## Measurement Standards

- **Minimum sample:** 50 signals for any claim
- **Baseline comparison:** Random entries at same frequency
- **Per-pair:** Must work on at least 2/3 pairs independently
- **Cost-aware:** All returns net of 0.10% round-trip
- **No data snooping:** Never fit parameters on same data used for evaluation

## Output: proven_edges.md

```
## [Edge Name]
- **Discovery date:** YYYY-MM-DD
- **Hypothesis:** [What structural inefficiency this exploits]
- **Signal:** [Detection logic in plain English]
- **Frequency:** X signals/day/pair
- **Backtest:** WR X%, PF Y, Sharpe Z (timerange)
- **Grade:** A/B (walk-forward validated)
- **Status:** LIVE / TESTING / PENDING_VALIDATION
```

## Output: anti_patterns.md

```
## [Dead End Name]
- **Tested:** YYYY-MM-DD
- **Hypothesis:** [What we thought would work]
- **Result:** WR X%, PF Y — NO EDGE
- **Why it fails:** [Root cause]
- **Don't retry:** [Why this is fundamentally broken for crypto]
```

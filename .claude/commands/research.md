---
description: "Research a new trading strategy or concept for crypto futures"
---

# /research [topic]

Research a specific trading concept, indicator, or strategy for OKX crypto futures.

## Usage
```
/research funding rate mean-reversion
/research volatility squeeze breakout crypto
/research liquidation cascade detection
```

## What happens
1. Researcher agent searches web, GitHub, papers for the topic
2. Evaluates applicability to our setup (OKX, 15m, BTC/ETH/SOL)
3. Returns structured findings with next steps
4. If promising, suggests research script to write

## After research
- If PROMISING: create `research/analyze_<name>.py` and measure
- If DEAD: add to `skills/edge-researcher/references/anti_patterns.md`
- Update `combos/CATALOG.md` with new candidate

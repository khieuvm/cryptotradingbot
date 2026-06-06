# Proven Edges — Crypto Futures (OKX)
# These have passed walk-forward validation or show strong preliminary data.

## Regime-Adaptive (regime_adaptive)
- **Discovery date:** 2026-05
- **Hypothesis:** Market alternates between trending and ranging; apply appropriate signal type
- **Signal:** ADX > 31 → EMA cross + MACD + DI; ADX <= 31 → RSI extreme + BB touch + OBV
- **Frequency:** ~2-4 signals/day across 3 pairs
- **Grade:** B (pending formal walk-forward)
- **Status:** LIVE (dry-run)

## BTC Sentiment Gate
- **Discovery date:** 2026-05
- **Hypothesis:** When BTC is in extreme state (RSI 1h < 35 or > 65), alt signals are unreliable
- **Signal:** Block longs when BTC RSI 1h < 35, block shorts when BTC RSI 1h > 65
- **Effect:** Reduces false positives by ~15-20% (estimated from dry-run)
- **Grade:** Integrated as filter, not standalone
- **Status:** LIVE (in confirm_trade_entry)

## Funding Rate Extreme Filter
- **Discovery date:** 2026-05
- **Hypothesis:** High funding = crowded positioning = squeeze risk
- **Signal:** Block longs when funding > 0.008%/8h, block shorts when funding < -0.007%/8h
- **Effect:** Avoids entries before liquidation cascades
- **Grade:** Integrated as filter
- **Status:** LIVE (in confirm_trade_entry)

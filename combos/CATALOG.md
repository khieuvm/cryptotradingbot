# Strategy Catalog — OKX Crypto Futures

## Deployed Combos

| # | Name | Grade | WR | PF | Status | Notes |
|---|------|-------|----|----|--------|-------|
| 1 | regime_adaptive | B | TBD | TBD | LIVE (dry-run) | Regime detection + appropriate signals |
| 2 | meanrev_confluence | C | TBD | TBD | LIVE (dry-run) | RSI+BB pullback in trend direction |
| 3 | trend_composite | C | TBD | TBD | LIVE (dry-run) | EMA cross + ADX + momentum |

## Candidates — Tier 1 (High Priority)

| # | Name | Type | Hypothesis | Status |
|---|------|------|-----------|--------|
| 4 | funding_contrarian | structural | Extreme funding (>0.05%/8h) = crowded positioning → fade | 📋 PENDING |
| 5 | volatility_compression | volatility | BB squeeze + KC on 15m → breakout (CB-equivalent for crypto) | 📋 PENDING |
| 6 | btc_dominance_regime | filter | BTC.D rising = alt weakness; adjust alt exposure | 📋 PENDING |
| 7 | oi_divergence | structural | Price up + OI down = weak rally → short bias | 📋 PENDING |

## Candidates — Tier 2 (Medium Priority)

| # | Name | Type | Hypothesis | Status |
|---|------|------|-----------|--------|
| 8 | cross_pair_momentum | momentum | BTC moves first, alts follow 15-60min later | 📋 PENDING |
| 9 | asian_range_breakout | time-based | Asia session (00-08 UTC) builds range → EU/US breaks out | 📋 PENDING |
| 10 | liquidation_support | structural | Clusters of long liquidations create temporary support | 📋 PENDING |
| 11 | ethbtc_meanrev | spread | ETH/BTC ratio mean-reverts on 4h+ timeframe | 📋 PENDING |

## Candidates — Tier 3 (Exploratory)

| # | Name | Type | Hypothesis | Status |
|---|------|------|-----------|--------|
| 12 | weekend_effect | time-based | Low weekend liquidity = different mean-reversion dynamics | 📋 PENDING |
| 13 | funding_settlement_scalp | structural | Position 30min before settlement, collect funding | 📋 PENDING |
| 14 | volume_profile_poc | structural | Daily POC acts as magnet/S-R for intraday | 📋 PENDING |

## Rejected

| # | Name | Reason | Date |
|---|------|--------|------|
| — | simple_ma_cross | Constant whipsaws in crypto vol, WR ~50% | Pre-project |
| — | single_indicator | RSI/MACD/Stoch alone = no edge | Pre-project |
| — | grid_bot | Unlimited downside in trends | Pre-project |

## Research Queue (Next Actions)

1. **funding_contrarian** — Researcher: find papers on funding rate mean-reversion. Analyst: measure funding → next-4h returns correlation on BTC/ETH/SOL.
2. **volatility_compression** — Analyst: implement BB squeeze detection on 15m, measure expansion magnitude and frequency.
3. **oi_divergence** — Researcher: find OI data source compatible with freqtrade/ccxt. Analyst: if data available, measure divergence → reversal correlation.

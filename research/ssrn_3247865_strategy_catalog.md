# Strategy Catalog: "151 Trading Strategies" (SSRN 3247865)
# Kakushadze & Serur (2018)
# Extracted for crypto futures adaptation (ETH/USDT, SOL/USDT on 15m)

## Filtering Key
- OHLCV: YES = can be implemented with OHLCV data only
         MAYBE = adaptable with minor compromises
         NO = requires data not available from OHLCV
- Priority: HIGH = directly testable on crypto 15m | MEDIUM = requires adaptation | LOW/SKIP = not viable

---

## CHAPTER 2: Options (ALL SKIP — requires options greeks, strikes, IVs)
Strategies 2.1–2.57 (covered call, put spreads, straddles, condors, etc.)
All require options pricing, IV surface, delta, gamma. Skip.

---

## CHAPTER 3: Stocks

| # | Strategy Name | Asset | Key Indicators | OHLCV? | Priority | Rule Summary |
|---|---------------|-------|---------------|--------|----------|--------------|
| 3.1 | Price-momentum | Stocks | Cumulative return over T months (typically 12), skip 1 month | YES | HIGH | Rank assets by 12m-1m cumulative return; buy winners, sell losers. Hold 1 month. Adapted to crypto: use 96-bar return on 15m (=1 day), or N-bar lookback. |
| 3.2 | Earnings-momentum | Stocks | SUE = (EPS - EPS_4q_ago) / stddev | NO | SKIP | Requires quarterly earnings per share data. |
| 3.3 | Value | Stocks | Book-to-Price ratio | NO | SKIP | Requires company book value / fundamentals. |
| 3.4 | Low-volatility anomaly | Stocks | Historical return volatility (126-252 day) | YES | MEDIUM | Buy low-vol assets, sell high-vol assets. Adapted: in regime detection, prefer entering when recent realized vol is LOW relative to its moving average — bet on continuation rather than mean-reversion. Single-asset: compare current vol to rolling baseline, reduce size in high-vol regimes. |
| 3.5 | Implied volatility | Stocks | Change in call IV / put IV | NO | SKIP | Requires options market data. |
| 3.6 | Multifactor portfolio | Stocks | Combination of value, momentum, quality | MAYBE | LOW | In single-asset context: combine momentum score + vol score + return skewness into composite signal. |
| 3.7 | Residual momentum | Stocks | Returns residualized vs Fama-French 3-factor model | MAYBE | LOW | Adapted: residualize crypto returns vs BTC (market factor) and trade the residual momentum. ETH and SOL residual vs BTC. |
| 3.8 | Pairs trading | Stocks | Correlated asset pairs, spread mean-reversion | YES | HIGH | Identify cointegrated crypto pairs (ETH/SOL, ETH/BTC ratio). When spread deviates >N sigma, short the rich leg / long the cheap leg. Classic Engle-Granger cointegration. |
| 3.9 | Mean-reversion (single cluster) | Stocks | Demeaned returns within a cluster | YES | HIGH | Within a basket of correlated assets (ETH, SOL, BTC), compute each asset's return minus the basket mean. Buy the underperformers, sell the outperformers. Dollar-neutral positions. |
| 3.9.1 | Mean-reversion (multiple clusters) | Stocks | Same but K>1 clusters | YES | MEDIUM | Same as 3.9, can cluster ETH/SOL/BTC by correlation regime. |
| 3.10 | Mean-reversion (weighted regression) | Stocks | Regression residuals with nonuniform weights | MAYBE | LOW | Use price-based factor loadings. Complex to implement. |
| 3.11 | Single moving average | Stocks/Any | SMA(T) or EMA(T) of price | YES | HIGH | LONG if Price > MA(T); SHORT if Price < MA(T). Long, short, or both. Classic single-MA trend filter. Works directly on any timeframe. |
| 3.12 | Two moving averages | Stocks/Any | MA(T1) and MA(T2), T1 < T2 | YES | HIGH | LONG if fast MA > slow MA; SHORT if fast MA < slow MA. Optional stop-loss: liquidate long if price drops >X% below previous bar. Common params: T1=10, T2=30 bars. |
| 3.13 | Three moving averages | Stocks/Any | MA(T1), MA(T2), MA(T3); T1 < T2 < T3 | YES | HIGH | LONG if MA(T1) > MA(T2) > MA(T3) — all aligned upward. SHORT if MA(T1) < MA(T2) < MA(T3). Liquidate long if MA(T1) falls below MA(T2). Filters false signals. |
| 3.14 | Support and resistance | Stocks/Any | Pivot point C = (H + L + C_prev)/3; R = 2C - L; S = 2C - H | YES | HIGH | Entry LONG if price > pivot C; exit at resistance R. Entry SHORT if price < pivot C; exit at support S. Computed from prior bar's H, L, C. Classic floor trader pivots. |
| 3.15 | Channel (Donchian) | Stocks/Any | Bup = max(P, T bars), Bdown = min(P, T bars) | YES | HIGH | Mean-reversion mode: BUY at channel floor (price = Bdown), SELL at channel ceiling (price = Bup). Breakout mode: reverse — buy at ceiling break, sell at floor break. Combine with volume spike for robustness. |
| 3.16 | Event-driven M&A | Stocks | Merger announcements, deal prices | NO | SKIP | Requires corporate event data. |
| 3.17 | ML - single-stock KNN | Stocks | Price/volume moving averages as features; KNN on k nearest historical states | YES | HIGH | Build feature vector from N-period price/volume MAs. Find k nearest historical analogs. Predict return as average of those analogs' subsequent returns. Trade if predicted return > threshold z1; exit if < z2. Directly applicable as ML layer on crypto OHLCV. |
| 3.18 | Statistical arbitrage (optimization) | Stocks | Expected returns + covariance matrix → Sharpe-maximizing portfolio | MAYBE | MEDIUM | Adapted: use 3 assets (ETH, SOL, BTC); compute rolling covariance; allocate based on Markowitz-optimal weights using momentum signal as expected returns. |
| 3.19 | Market-making | Stocks | Bid-ask spread, order book depth | NO | SKIP | Requires Level 2 order book data. |
| 3.20 | Alpha combos | Stocks | Hundreds of formula alphas combined | MAYBE | LOW | Can combine multiple OHLCV-based signals (momentum, reversal, vol) with Kakushadze weighting scheme. Advanced but possible. |

---

## CHAPTER 4: ETFs

| # | Strategy Name | Asset | Key Indicators | OHLCV? | Priority | Rule Summary |
|---|---------------|-------|---------------|--------|----------|--------------|
| 4.1 | Sector momentum rotation | ETFs | Cumulative return over T months (6-12) | YES | HIGH | Rank multiple assets by past return; hold top performers, exit bottom. Applied to crypto: rank ETH/SOL/BTC/others each week by 1-4 week return; hold top N. Long-only or long-short. |
| 4.1.1 | Sector momentum + MA filter | ETFs | Momentum rank + Price > MA(T0) filter | YES | HIGH | Same as 4.1 but only enter top-ranked if price is above its long-term MA. This acts as a trend filter — avoids buying relative winners in a downtrend. |
| 4.1.2 | Dual-momentum rotation | ETFs | Relative momentum (cross-section) + Absolute momentum (vs itself) | YES | HIGH | First check: is the asset in top performers (relative momentum)? Second check: is its price above its long-term MA (absolute/time-series momentum)? Only go long if BOTH pass. Otherwise hold cash or uncorrelated asset. Antonacci's Dual Momentum. |
| 4.2 | Alpha rotation | ETFs | Jensen's alpha (intercept from FF regression) | MAYBE | LOW | Requires Fama-French factors. Partially adaptable by computing alpha vs BTC as market factor. |
| 4.3 | R-squared selectivity | ETFs | Alpha + R2 from factor model regression | MAYBE | LOW | Lower R2 (less market-driven) + higher alpha = buy signal. Can adapt: compute rolling R2 of ETH returns regressed on BTC returns; trade ETH when R2 is low (idiosyncratic). |
| 4.4 | IBS mean-reversion | ETFs | IBS = (Close - Low) / (High - Low) | YES | HIGH | Internal Bar Strength. IBS near 0 = price closed near day's low (oversold relative to today's range) → BUY. IBS near 1 = price closed near day's high → SELL/SHORT. Sort multiple assets cross-sectionally by IBS and go long bottom decile, short top decile. Single-asset threshold version also works. |
| 4.5 | Leveraged ETF decay | ETFs | Short both 2x long ETF and 2x inverse ETF | NO | SKIP | Requires leveraged ETF pairs. |
| 4.6 | Multi-asset trend following | ETFs | Cumulative return + optional MA filter + volatility-adjusted weights | YES | HIGH | Build portfolio from assets with positive momentum. Weight by: w_i = Rcum_i / vol_i^2 (Sharpe-optimal allocation assuming diagonal covariance). For crypto: allocate across ETH, SOL, BTC proportional to risk-adjusted momentum. Rebalance weekly. |

---

## CHAPTER 5: Fixed Income (ALL SKIP — yield curves, bond math, duration)
Strategies 5.1–5.15. Require bond yields, coupon rates, credit spreads. Skip.

---

## CHAPTER 6: Indexes

| # | Strategy Name | Asset | Key Indicators | OHLCV? | Priority | Rule Summary |
|---|---------------|-------|---------------|--------|----------|--------------|
| 6.2 | Cash-and-carry arbitrage | Futures/Spot | Futures price vs spot price + cost of carry | NO | SKIP | Requires simultaneous spot and futures position; needs cost-of-carry model. |
| 6.3 | Dispersion trading | Equity indexes | Short index vol, long constituent vol | NO | SKIP | Requires options on index and individual components. |
| 6.4 | Intraday index ETF arbitrage | ETFs | Price discrepancies between ETF and NAV | NO | SKIP | Requires NAV data and creation/redemption access. |
| 6.5 | Index volatility targeting | Indexes | Realized vol → allocate between risky asset and risk-free | YES | MEDIUM | Compute rolling realized volatility (e.g., 20-bar). When vol is above target, reduce position size. When vol is below target, increase size. Essentially a volatility-scaled position sizer. Very applicable to crypto. |

---

## CHAPTER 7: Volatility (mostly requires options — VIX, variance swaps)

| # | Strategy Name | OHLCV? | Priority | Notes |
|---|---------------|--------|----------|-------|
| 7.2 | VIX futures basis trading | NO | SKIP | Requires VIX futures |
| 7.3 | Volatility carry (VXX) | NO | SKIP | Requires VXX/VIX instruments |
| 7.4 | Volatility risk premium | NO | SKIP | Requires realized vs implied vol |
| 7.5 | Volatility skew (risk reversal) | NO | SKIP | Requires options |
| 7.6 | Variance swaps | NO | SKIP | Requires variance swap instruments |

Note: The CONCEPT of volatility risk premium (implied > realized) can be adapted to crypto using funding rate as proxy for "implied" volatility premium.

---

## CHAPTER 8: Foreign Exchange (FX)

| # | Strategy Name | Asset | Key Indicators | OHLCV? | Priority | Rule Summary |
|---|---------------|-------|---------------|--------|----------|--------------|
| 8.1 | Moving averages with HP filter | FX/Any | HP filter (smoothed price series) + dual MA crossover | YES | HIGH | Apply Hodrick-Prescott filter to price to extract trend component; compute fast/slow MA on the smoothed series; trade the crossover. Reduces noise vs raw MA. Lambda=1600 for monthly data; for 15m crypto, tune lambda. |
| 8.2 | FX Carry trade | FX | Interest rate differential between two currencies | NO | SKIP | Requires interest rate data. Partially analogous to crypto funding rate carry (short perp when funding is positive). |
| 8.2.1 | High-minus-low carry | FX | Cross-sectional forward discount ranking | NO | SKIP | Requires multiple FX forward rates. |
| 8.3 | Dollar carry trade | FX | Average cross-sectional forward discount | NO | SKIP | Macro FX strategy. |
| 8.4 | Momentum + carry combo | FX | MA crossover (momentum) + carry signal | MAYBE | LOW | The momentum component (MA crossover) is directly applicable. Crypto analog: combine MA crossover signal with funding rate signal. Weight the two signals by minimum-variance combination formula. |
| 8.5 | FX triangular arbitrage | FX | Three-currency loop exchange rates | MAYBE | LOW | Crypto analog: ETH/BTC, BTC/USDT, ETH/USDT triangular relationship. In practice, arbitrage is near-instant at CEXs and profits are tiny. Not viable for 15m systematic trading. |

---

## CHAPTER 9: Commodities

| # | Strategy Name | Asset | Key Indicators | OHLCV? | Priority | Rule Summary |
|---|---------------|-------|---------------|--------|----------|--------------|
| 9.1 | Roll yields (backwardation/contango) | Commodity futures | Ratio of front-month to second-month futures price | NO | LOW | Requires multiple futures contract expiries. Crypto perp funding rate is a proxy for the backwardation/contango signal. |
| 9.2 | Hedging pressure (COT) | Commodity futures | CFTC Commitments of Traders report | NO | SKIP | Requires COT report data. |
| 9.3 | Portfolio diversification | Commodities | Correlation with equities | NO | SKIP | Strategic allocation, not tactical trading. |
| 9.4 | Value (commodities) | Commodities | v = P_5yr_ago / P_now | YES | MEDIUM | Buy commodities trading below their 5-year-ago price (mean reversion over long horizon). Adapted to crypto 15m: use "value" as price vs a very long-term MA (e.g., price vs 200-day MA as proxy for long-term fair value). When price is far below long MA, expect mean reversion. |
| 9.5 | Skewness premium | Commodity futures | Historical return skewness over T periods | YES | HIGH | Empirical finding: high-skewness futures have lower future returns. BUY futures in bottom quintile by historical return skewness; SELL futures in top quintile. Directly computable from OHLCV. Applied to single crypto asset: when recent return distribution is positively skewed (lottery-like), expect mean reversion or lower returns. Bet against positive skew. |
| 9.6 | Trading with pricing models (OU process) | Commodities | Ornstein-Uhlenbeck (mean-reverting) stochastic model | YES | HIGH | Fit OU parameters (kappa=speed, mu=long-run mean, sigma=vol) to log-price series. When current log-price is above model's expected value → SELL (rich). When below → BUY (cheap). Equivalent to Z-score mean reversion with time-decay. Directly applicable to crypto. |

---

## CHAPTER 10: Futures

| # | Strategy Name | Asset | Key Indicators | OHLCV? | Priority | Rule Summary |
|---|---------------|-------|---------------|--------|----------|--------------|
| 10.1 | Hedging with futures | Futures | Hedge ratio, delta | NO | SKIP | Risk management tool, not alpha strategy. |
| 10.1.1 | Cross-hedging | Futures | Regression-based hedge ratio | NO | SKIP | Requires correlated instrument with futures. |
| 10.1.2 | Interest rate risk hedging | Interest rate futures | Duration, modified duration | NO | SKIP | Fixed income specific. |
| 10.2 | Calendar spread | Commodity/Financial futures | Near-month vs deferred-month prices | NO | SKIP | Requires multiple contract expiry prices. |
| 10.3 | Contrarian (mean-reversion) | Futures | Demeaned return: each asset vs equal-weighted basket | YES | HIGH | Compute market index return = avg return across N futures. Signal for each asset = -(return_i - market_return). Buy assets that underperformed the basket; sell those that outperformed. Vol-scale positions by 1/sigma. Rebalance weekly. Applied to crypto: use ETH, SOL, BTC as the basket. |
| 10.3.1 | Contrarian + volume/OI filter | Futures | Volume change (vi) and open interest change (ui) | YES | HIGH | Same as 10.3 but filter: only apply contrarian signal to assets with HIGH recent volume change (vi > median) AND LOW recent open interest change (ui < median). Rationale: high vol change = overreaction; low OI change = less deep market → stronger snap-back. OI analog in perpetuals = can use daily volume change as main filter since OI not directly available. |
| 10.4 | Trend following (momentum) | Futures | Sign of N-period return, vol-scaled | YES | HIGH | Signal = sign(R_i) * (1 / vol_i). Buy if recent N-bar return is positive; sell if negative. Scale position inversely by volatility. Smooth signal with tanh(R/sigma_cross) to avoid instability at small returns. One of the most robust strategies known across all futures markets. |

---

## CHAPTERS 11–17: Structured Assets, Convertibles, Tax Arbitrage, Misc, Distressed, Real Estate, Cash
ALL SKIP — these require credit instruments, CDOs, MBS, municipal bonds, real estate NAV, etc.
Chapter 17.4 (REPO) and 17.5 (Pawnbroking) — irrelevant.

---

## CHAPTER 18: Cryptocurrencies

| # | Strategy Name | Asset | Key Indicators | OHLCV? | Priority | Rule Summary |
|---|---------------|-------|---------------|--------|----------|--------------|
| 18.2 | ANN on BTC price indicators | BTC/Crypto | EMA(T), EMSD(T), RSI(T) on normalized returns | YES | HIGH | Normalize returns: R_hat = (R - mean_R) / vol. Compute EMAs of normalized returns at multiple horizons (30m, 1h, 3h, 6h = 2, 4, 12, 24 bars on 15m). Compute EMSDs at same horizons. Compute RSI at 3h, 6h, 12h horizons. Feed into ANN (ReLU hidden layers, softmax output). Output = probability of return being in top/bottom K quantile. BUY if top quantile has max probability; SELL if bottom quantile. Uses explicit 15-minute intervals as designed. |
| 18.3 | Sentiment analysis (Naive Bayes) | BTC/Crypto | Twitter/social media keyword frequencies | NO | SKIP | Requires Twitter/social media data stream. Concept could be adapted with on-chain sentiment proxies (not OHLCV). |

---

## CHAPTER 19: Global Macro

| # | Strategy Name | OHLCV? | Priority | Notes |
|---|---------------|--------|----------|-------|
| 19.2 | Fundamental macro momentum | NO | SKIP | Requires GDP, CPI, interest rates, sovereign risk. |
| 19.3 | Global macro inflation hedge | NO | SKIP | Requires CPI, headline vs core inflation data. |
| 19.4 | Global fixed-income strategy | NO | SKIP | Requires government bond yields, sovereign risk. |
| 19.5 | Trading on economic announcements | MAYBE | MEDIUM | On FOMC/CPI announcement days, equities historically outperform. Crypto analog: pre-FOMC positioning. Could implement as a calendar-based filter: increase/reduce exposure around scheduled macro events (FOMC dates, CPI releases). Not OHLCV but external calendar data is trivially available. |

---

## APPENDIX A: R Backtesting Code — Two Embedded Strategies

| # | Strategy | OHLCV? | Priority | Rule |
|---|----------|--------|----------|------|
| A.1 | Overnight gap mean-reversion (DELAY-0) | YES | HIGH | Trade in the SAME DIRECTION as the overnight close-to-open gap (the gap already occurred, so this is momentum of the gap). Establish position at open, liquidate at close. Adapted to 15m crypto: "gap" = return from bar N-1 close to bar N open. If gap is up → LONG from bar N open; exit at bar N close. |
| A.2 | Intraday momentum (DELAY-1) | YES | HIGH | Use PREVIOUS day's close-to-open return as signal for TODAY's direction. Signal = sign of prior open-to-close return. Adapted to 15m: use return of the prior bar (open to close of bar N-1) as momentum signal for bar N entry at open, exit at close. |

---

## PRIORITIZED SHORTLIST FOR CRYPTO FUTURES TESTING

### Tier 1 — High Priority (directly OHLCV-testable, clear rules, proven in futures markets)

| Priority | Section | Strategy | Core Signal | Notes |
|----------|---------|----------|-------------|-------|
| 1 | 10.4 | Trend following (futures) | sign(N-bar return) * 1/vol | Most robust CTA strategy, works across all futures |
| 2 | 3.12 | Two moving averages | Fast MA > Slow MA crossover | Universal; tune T1, T2 on 15m |
| 3 | 3.14 | Support and resistance (pivot) | Price vs pivot C=(H+L+C)/3 | Pivot levels directly from prior bar H,L,C |
| 4 | 3.15 | Donchian channel | Price vs N-bar high/low | Breakout AND mean-reversion variants |
| 5 | 4.4 | IBS mean-reversion | IBS = (C-L)/(H-L) | Single indicator, cross-sectional or threshold |
| 6 | 10.3 | Contrarian (demeaned returns) | Buy underperformer vs basket | ETH/SOL/BTC basket cross-sectional signal |
| 7 | 10.3.1 | Contrarian + volume filter | Demeaned return + high vol change | Volume spike as overreaction filter |
| 8 | 3.11 | Single moving average | Price vs MA trend filter | Use as position entry gating filter |
| 9 | 9.6 | OU process mean-reversion | Z-score vs fitted mean-reversion level | Fit kappa, mu, sigma to rolling window |
| 10 | 9.5 | Skewness premium | Bet against positive-skewed return distribution | 20-100 bar rolling skewness; sell when high |
| 11 | 3.13 | Three moving averages | Triple alignment: MA(T1)>MA(T2)>MA(T3) | Stricter trend confirmation |
| 12 | 4.1.2 | Dual momentum | Relative rank + price above long MA | Absolute + relative momentum combo |
| 13 | 3.8 | Pairs trading | ETH/SOL spread Z-score | Cointegration test; trade mean-reversion of spread |
| 14 | 8.1 | MA with HP filter | HP-smoothed price + dual MA crossover | Noise reduction before MA signals |
| 15 | 18.2 | ANN on crypto indicators | EMA/EMSD/RSI → ANN classifier | Explicitly designed for 15m BTC; use our data |

### Tier 2 — Medium Priority (need adaptation but viable)

| Priority | Section | Strategy | Adaptation Needed |
|----------|---------|----------|-------------------|
| 16 | 4.6 | Multi-asset trend following | Weight ETH/SOL/BTC by momentum/vol ratio |
| 17 | 6.5 | Volatility targeting | Scale position by rolling vol vs target vol |
| 18 | 3.1 | Price momentum | Time-series momentum: N-bar return signal |
| 19 | 3.4 | Low-volatility regime | Enter only when rolling vol < rolling mean of vol |
| 20 | 3.17 | KNN ML | Price/volume MAs as features, KNN classifier |
| 21 | 3.9 | Cross-asset mean-reversion | ETH/SOL/BTC basket demeaned returns |
| 22 | 4.1.1 | Momentum with MA filter | Momentum signal gated by long-term trend |
| 23 | 19.5 | Economic announcements | Pre/post FOMC date position rule (calendar-based) |
| 24 | 9.4 | Long-term value | Price vs very long MA (200d) — mean reversion signal |
| 25 | A.1 | Gap fill | Open-to-open gap direction trade |

### Tier 3 — Low Priority or Skip

| Section | Strategy | Reason to Skip |
|---------|----------|----------------|
| 3.7 | Residual momentum | Needs Fama-French factors (can proxy vs BTC but weak rationale) |
| 3.18 | StatArb optimization | Requires many assets; with 3 pairs, degenerate |
| 4.3 | R-squared selectivity | Needs FF factor regression |
| 8.4 | Momentum + carry combo | Carry part requires funding rate data (not OHLCV) |
| 8.5 | Triangular arbitrage | Sub-second execution required; not 15m viable |
| 9.1 | Roll yield | No perpetual term structure in perps |
| 10.2 | Calendar spread | Needs two expiry contracts |

---

## STRATEGIES NOT IN BOOK BUT IMPLIED BY RELATED DISCUSSION

The book references these known strategies that have direct OHLCV implementations and would complement the catalog:

1. **Bollinger Band squeeze** (referenced in trend-following chapter notes): vol contraction → breakout
2. **RSI oversold/overbought** (explicitly used as ANN input feature in 18.2): RSI < 30 LONG, RSI > 70 SHORT
3. **MACD crossover** (standard TA, referenced as related to MA strategies): MACD line crosses signal line
4. **ATR-based trailing stop** (referenced in stop-loss discussion for 3.12): trail at N * ATR below high
5. **Volume-weighted average price (VWAP)** (referenced in code appendix): price vs VWAP mean-reversion intraday

---

## IMPLEMENTATION NOTES FOR 15m CRYPTO FUTURES

### Timeframe Translations
| Original Period | 15m Equivalent |
|----------------|----------------|
| 1 day | 96 bars |
| 1 week | 672 bars |
| 1 month | ~2880 bars |
| 12 months (momentum) | ~34,560 bars → use 96-1920 bars for crypto (1 day to 2 weeks) |
| "Skip 1 month" | Skip 96 bars |

### Strategy Priority by Current Gap in Portfolio
We already have: regime_adaptive (EMA cross + ADX), volume_spike_rev (volume + RSI), cb_adx_breakout (BB squeeze + ADX). These correspond to strategies 3.11-3.13, 10.4, and Donchian variations.

**GAPS to fill (new ideas from catalog):**
1. **IBS mean-reversion (4.4)** — pure bar-range based, no overlap with current strategies
2. **Pivot support/resistance (3.14)** — classic floor pivots, entirely different signal
3. **Cross-asset contrarian (10.3)** — ETH/SOL/BTC basket, no current cross-pair logic
4. **Skewness premium (9.5)** — return distribution-based, no current analog
5. **OU process (9.6)** — formal mean-reversion with speed-of-reversion parameter
6. **Dual momentum (4.1.2)** — relative + absolute momentum combo
7. **Pairs trading (3.8)** — ETH/SOL spread cointegration

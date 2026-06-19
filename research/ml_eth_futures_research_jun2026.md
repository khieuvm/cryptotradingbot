# ML/AI Research for ETH/USDT Crypto Futures — June 2026

**Research date:** 2026-06-19
**Researcher:** crypto_strategy_researcher agent
**Context:** OKX ETH/USDT:USDT perpetual, 15m primary, 3-5x leverage
**Prior failures:** meta-labeling (3 variants, all Grade F/INSUF), HMM + stacking ensemble, OHLCV direction prediction

---

## Starting Constraints From Memory

Before listing new approaches, the hard blockers confirmed in memory:

1. **OHLCV meta-labeling is dead**: 15m OHLCV features at signal bars contain zero residual predictive
   information over what the strategies themselves already encode (ADX, RSI, BB, volume).
   Three attempts confirmed this at IS and OOS level. Do not retry.

2. **Sample size blocker**: Grade A/B strategies generate ~150 IS trades per 60d window.
   Supervised learning on per-trade outcomes needs 500+ samples minimum. The 15m strategies
   cannot produce this with selective entries.

3. **Exit simulation mismatch**: Simulating freqtrade's 3-phase SL outside freqtrade produces
   corrupted labels. Any ML that needs correct exit simulation must use actual freqtrade
   backtest JSON, not OHLCV-derived simulation.

4. **Available OKX data NOT yet used as ML features:**
   - Funding rate (current + 8h history)
   - Open interest (15m time series)
   - Taker buy/sell volume (15m time series)
   - Long/short account ratio (15m time series)
   - Top-trader vs all-account L/S divergence
   - Liquidation order clusters

These endpoints are documented in `research/okx_public_api_reference.md`. The rule-based
strategies do not use them. No prior ML experiment has used them as features.

---

## Literature Reviewed

### Papers with Concrete Quantitative Results

| Paper | Year | Exchange | Result | Relevance |
|-------|------|----------|--------|-----------|
| arXiv:2201.04699 (Borrageiro et al.) | 2022 | BitMEX XBTUSD | 350% total / 71% from funding carry | HIGH: funding carry as dominant alpha |
| arXiv:2508.02356 | 2025 | Unspecified crypto | PF 1.15 with 0.05% costs | MEDIUM: multi-source features work marginally |
| arXiv:2509.10542 | 2025 | Binance ETH 10m | 51.36% acc / 117 from 100 USDT (1 week test) | LOW: too short test, no derivatives data |
| arXiv:2606.04574 | 2026 | Binance 1h futures | OOS beats heuristic at 10% sig | LOW: pairs trading, 1h, not 15m ETH |
| arXiv:2506.05764 | 2025 | Bybit LOB 100ms | XGBoost + Kalman beats DeepLOB | MEDIUM: needs L2 data not currently collected |
| arXiv:2010.07404 | 2020 | Unspecified | 60%+ accuracy | LOW: trade-by-trade sub-minute horizon |
| arXiv:2506.08718 | 2025 | Binance ETH | Futures lead spot in price discovery | MEDIUM: confirms futures as leading market |
| arXiv:2103.14079 | 2021 | General finance | Concept drift detectors beat continuous retraining | MEDIUM: River ADWIN applicable |

### Honest Assessment of Deep Learning for 15m ETH Futures

arXiv:2509.10542 (best current paper on ETH-USDT): 51.36% accuracy, tested over ONE WEEK.
This is not deployable evidence. The adaptive TFT barely beats the 50% random baseline and
uses only close price volatility — no order flow, no derivatives data.

arXiv:2508.02356 (CNN with on-chain + orderbook + sentiment): PF 1.15 with 0.05% costs.
Our round-trip cost is 0.10%. After our costs, PF would be approximately 1.0 (breakeven).
This approach uses orderbook data (not currently collected) and GDELT sentiment API.

arXiv:2606.04574 (PPO+LSTM pair trading): Uses 3-5 features total (Z-score + Hurst exponent),
statistical pair selection per month, 1h timeframe. Not applicable to single-pair 15m futures.

**Bottom line on deep learning for 15m ETH:** No paper reviewed demonstrates DL meaningfully
outperforms well-tuned LightGBM at the 15m timeframe without either (a) orderbook L2 data
at tick resolution, or (b) fundamental data/sentiment. For OHLCV features alone, LightGBM
is the correct and sufficient model choice (Grinsztajn et al. NeurIPS 2022 benchmark).

---

## CONFIRMED FINDING: Funding Carry is the Dominant Structural Alpha

The Borrageiro et al. (2022) echo state network result is the most actionable finding:
71% of the 350% total return over 5 years came from **funding carry**, not from direction
prediction. The agent simply held long positions when funding was positive (longs paid shorts,
so holding long meant receiving carry) and short when funding was negative.

Funding carry arithmetic for our system:
- ETH funding averages +0.01%/8h in trending markets = +0.03%/day when positive
- At 3x leverage = +0.09%/day passive carry income
- Over 143d backtest at average +0.01%/8h: +1.43% passive return from carry alone
- When funding spikes to +0.05%/8h (5x normal): longs receive +0.15%/day at 3x = major edge

This is NOT currently exploited by any active strategy. The `funding_contrarian` strategy
(Grade C) does the opposite — it fades extreme funding as a mean-reversion signal. The carry
trade would complement it: normal funding = hold with trend; extreme funding = flip (contrarian).

---

## Strategy 1: Funding Rate Regime Filter (ML-enhanced Carry)

**Source:** arXiv:2201.04699 (Borrageiro/Firoozye/Barucca, IEEE Access 2022)
**Applicability:** HIGH
**Type:** structural / carry + regime filter

**Concept:**
Classify the funding rate environment into 3 states, then apply appropriate strategy biases:
1. POSITIVE CARRY state (funding > 0.01%/8h, trending up): longs receive carry — bias towards
   long entries, suppress short entries, hold existing longs longer (funding accrues while holding)
2. NEGATIVE CARRY state (funding < -0.01%/8h): shorts receive carry — bias towards shorts
3. NEUTRAL/EXTREME state (|funding| > 0.05%/8h): contrarian signal — the crowded side is about
   to unwind, existing signals are high-risk

**ML implementation (not a direction predictor):**
Train a 3-class LightGBM classifier on 6-hour bars with features:
- funding_rate (current)
- funding_rate_8h_ago, 16h_ago, 24h_ago (trend)
- funding_rate_3d_rolling_mean (regime baseline)
- oi_pct_change_1d (open interest expanding or contracting)
- taker_buy_ratio_1h = taker_buy_vol / (taker_buy_vol + taker_sell_vol)
- mark_premium (mark - spot spread, precursor to funding)

Target: classify into CARRY_LONG / CARRY_SHORT / CROWDED_RISK
This is a REGIME classifier, not a per-trade predictor. Sample count is NOT a blocker:
funding state changes maybe 5-10 times per month = ~720 6h bars in 180d = adequate training.

**Key difference from failed meta-labeling:**
Does NOT predict trade outcomes. Predicts MARKET STATE. The model outputs a regime flag
(valid for 6-8h) that is used to gate strategy entries, not to filter individual signals.

**Crypto fit:**
- 24/7 compatible: Yes (funding settles at fixed UTC times)
- Survives 0.10% round-trip: Yes (carry trade actively receives funding, offsetting costs)
- Expected frequency: Regime shifts 5-10 times/month (not trade-by-trade)
- Works on ETH/SOL/SPX/DOGE: Yes (all have identical OKX funding rate mechanics)
- Freqtrade implementable: Yes (regime flag in populate_indicators, gate entries in populate_entry_trend)

**Evidence quality:** HIGH — 5-year confirmed result on perpetual swap with actual exchange data.
The structural funding carry mechanism is exchange-design-based, not a statistical pattern.

**Anti-pattern to avoid:** Do not train on per-trade outcomes. This must be a regime/state
classifier, not a signal filter. The funding state is a property of the market, not of a trade.

**Next step:** `research/analyze_funding_carry_regime.py`
- Download 180d of OKX funding rate history + OI + taker volume at 6h resolution
- Label regime states using threshold rules (trivially accurate: funding > 0.01% → CARRY_LONG)
- Train LightGBM to predict NEXT 6h regime from current features (lagged 1 period)
- Backtest: when CARRY_LONG active, long entries enabled, short entries suppressed; vice versa
- Measure: does regime gating improve WR on existing strategy signals?

---

## Strategy 2: Taker Volume CVD as Primary ML Feature

**Source:**
- arXiv:2010.07404 (Nakagawa et al. 2020): LSTM on trade-by-trade data achieves 60%+ accuracy
- arXiv:2209.10334 (Lu, Reinert, Cucuringu 2022): OFI correlates strongly with contemporaneous
  AND future returns for isolated trades
- arXiv:1704.08175 (Scaillet/Treccani/Trevisan 2017): OFI + spread widening predicts BTC jumps
- OKX API: `/rubik/stat/taker-volume-contract` provides 15m taker buy/sell volume (available now)

**Applicability:** HIGH
**Type:** order flow / microstructure

**Concept:**
Taker buy/sell volume imbalance at 15m (already available from OKX API) is a direct proxy
for order flow imbalance (OFI). The existing strategies use `(close-low)/(high-low)` as an
OHLCV-derived proxy. Using actual taker volume counts replaces this proxy with the real
signal. This is the MOST DIRECTLY available improvement to the existing feature set.

**Feature set (all from OKX, available now):**
- taker_buy_ratio = taker_buy_vol / (taker_buy_vol + taker_sell_vol)
  - >0.60 = aggressive buyers dominating = bullish pressure
  - <0.40 = sellers dominant = bearish
- taker_cvd_12bar = sum of (buy_vol - sell_vol) over last 12 bars (CVD proxy, 3h)
- taker_cvd_slope = linear slope of 6-bar CVD (is buying pressure accelerating or decelerating?)
- taker_vol_spike = taker_total_vol / taker_total_vol_20bar_ma (volume spike indicator)
- taker_divergence = sign(price_change_1h) != sign(cvd_change_1h) (price vs flow divergence)

**ML model:**
LightGBM binary classifier using ONLY these OFI features + existing strategy signal strength.
Target: did the strategy trade make money? (Use actual freqtrade backtest JSON labels — this
is the ONLY meta-labeling approach that works, per memory notes.)

**But there is a simpler rule-based version first:**
Before any ML, add taker_buy_ratio as a FILTER to existing signals:
- volume_spike_rev SHORT: require taker_buy_ratio_bar < 0.45 (takers were net sellers too)
- volume_spike_rev LONG: require taker_buy_ratio_bar > 0.50 (takers were net buyers)
- regime_adaptive: require taker_cvd_12bar direction matches signal direction

**Why this is different from failed meta-labeling:**
The previous meta-labeling failed because 15m OHLCV features contain no residual signal
over the strategies. Taker buy/sell volume is INDEPENDENT of OHLCV. It is not computable
from close/high/low/open/volume alone. It represents who initiated trades, which is
information not captured by any existing indicator in the active strategies.

**Evidence quality:** HIGH for OFI as a predictor (multiple papers). MEDIUM for this specific
OKX 15m application (no published study found on OKX 15m specifically, but the mechanism
is the same).

**Next step:** `research/analyze_taker_volume_features.py`
1. Download 180d of OKX 15m taker buy/sell volume for ETH/SOL/BTC
2. Compute taker_buy_ratio, taker_cvd_12bar, taker_divergence at each bar
3. Add as rule-based filters to volume_spike_rev signals
4. Run freqtrade backtest with filter and without to measure WR lift
5. If WR improves by 3pp+, integrate into strategy code directly

---

## Strategy 3: OI Divergence Regime Classifier

**Source:**
- OKX API: `/rubik/stat/contracts/open-interest-history` at 15m (available now)
- Standard futures market theory: OI divergence from price is a primary signal
- The `analyze_all_opportunities.py` research identified vol_ratio + candle_size_atr as
  top 2 features for opportunity quality — OI trend is the complementary structural signal

**Applicability:** HIGH
**Type:** structural / regime filter

**Concept:**
Open interest divergence from price movement is a well-known regime signal in futures:
- Price up + OI up = new longs entering = trend is real (continuation likely)
- Price up + OI down = short covering only = weak move (mean reversion likely)
- Price down + OI up = new shorts entering = trend is real (continuation likely)
- Price down + OI down = long liquidation only = weak move (bouncer likely)

This is not ML — it is a structural indicator. The ML angle is using OI divergence as a
FEATURE in a regime classifier rather than a hard rule, allowing it to interact with funding
rate, taker volume, and price features.

**Features:**
- oi_price_regime = categorize(sign(price_change_4h), sign(oi_change_4h)) → 4 states
- oi_expansion_ratio = oi_now / oi_20bar_ma (1 = neutral, >1.2 = expanding, <0.8 = contracting)
- oi_trend_strength = slope of OI over last 12 bars / OI_stddev (normalized acceleration)
- oi_funding_alignment = sign(oi_change_4h) == sign(funding_rate) (are they consistent?)

**Combined classifier (funding + OI + taker volume):**
Use all three data streams together in a single LightGBM regime classifier:
- Input: funding_rate, oi_expansion_ratio, oi_price_regime, taker_buy_ratio_4h, BTC_dominance_change
- Output: 3 classes → TRENDING_STRONG / TRENDING_WEAK / RANGING
- Train on 180d of 4h data (enough samples: 180 * 6 = 1080 4h bars)
- Labels: derive from ADX(14) and ATR ratio (current vs 20-bar avg) — rule-based labeling
  does NOT need per-trade outcomes, so sample size is not a blocker

This addresses the core problem from memory: "model trained on regime A only works in regime A."
The combined regime classifier tells the Orchestrator which regime we are in right now, so the
correct strategy is activated or suppressed.

**Crypto fit:**
- 24/7 compatible: Yes
- Does not need per-trade labels: Yes (uses rule-based regime labels, not trade outcomes)
- Works on ETH/SOL: Yes
- Freqtrade implementable: Yes (compute in informative_pairs, use as gate in populate_indicators)

**Evidence quality:** MEDIUM-HIGH (OI divergence is established futures theory; ML application
using it as a feature is new for our specific setup but theoretically sound).

**Next step:** `research/analyze_oi_regime_classifier.py`
Download 180d of OKX 4h OI + funding + taker volume for ETH. Label regimes by ADX/ATR rule.
Train LightGBM. Evaluate regime classifier accuracy on 30d OOS. If 65%+ regime classification
accuracy, integrate as orchestrator-level gate.

---

## Strategy 4: L/S Ratio Smart Money Divergence

**Source:**
- OKX API: both all-account L/S ratio and top-trader L/S ratio at 15m (available now)
- Standard sentiment indicator: "smart money vs dumb money" divergence

**Applicability:** MEDIUM
**Type:** sentiment / positioning signal

**Concept:**
OKX provides two L/S ratio endpoints:
1. All accounts (retail-dominated): includes all market participants
2. Top traders (presumably institutional/informed): higher-frequency, higher-capital traders

When these two diverge, the smart money is betting against retail:
- top_trader_ratio < 1 (shorting) + all_account_ratio > 2 (retail heavily long) = distribution
- top_trader_ratio > 2 (bullish) + all_account_ratio < 0.8 (retail heavily short) = accumulation

**ML feature:**
- ls_divergence = top_trader_ls_ratio - all_account_ls_ratio
- ls_extremity = abs(all_account_ls_ratio - 1) (how far from 50/50)
- ls_trend_alignment = sign(all_account_ls_ratio - 1) == sign(price_change_1h)
  (are retail traders correctly positioned or fighting the trend?)

**Usage:**
- Standalone contrarian entry signal: high ls_divergence (smart money vs retail) = counter-trade
- As ML filter: add ls_divergence as a feature to existing entry models
- As regime indicator: persistent all_account_ratio >3 = crowded long = avoid new longs

**Known limitation:** The L/S ratio can lead price by 1-3 bars at 15m (documented in OKX docs)
but the relationship is not consistent across all market regimes. Use as one of several features,
not as a standalone signal.

**Crypto fit:**
- 24/7 compatible: Yes
- Survives costs alone: No (needs to combine with directional signal)
- Expected frequency: Extreme readings ~2-3 times per week per pair
- Freqtrade implementable: Yes (informative_pairs call to OKX API per bar)

**Evidence quality:** LOW-MEDIUM (established as a sentiment indicator in crypto, but no
published paper with rigorous OOS validation found for this specific use case).

**Next step:** Rule-based first. Add to `analyze_funding_carry_regime.py`:
compute ls_divergence as an additional feature in the regime classifier.

---

## Strategy 5: Liquidation Cluster Density as Dynamic S/R

**Source:**
- OKX API: `/api/v5/public/liquidation-orders` (available now)
- Theoretical basis: liquidation clusters = price levels where forced selling/buying concentrates
- When price clears a liquidation cluster above, resistance is removed = continuation signal

**Applicability:** MEDIUM
**Type:** structural / market microstructure

**Concept:**
Poll OKX liquidation orders (last 100 liquidations) at each 15m bar:
- Cluster liquidations into 0.5% price buckets above and below current price
- liq_density_above = total USDT liquidated in (price, price*1.02) range
- liq_density_below = total USDT liquidated in (price*0.98, price) range
- When price has just passed through a dense liq zone:
  * Former resistance with cleared longs = liq_cleared_long_zone (bullish: overhead supply gone)
  * Former support with cleared shorts = liq_cleared_short_zone (bearish: below support gone)

**ML feature:**
- liq_cleared_above_1h: has price moved through a liq_density_above cluster in the last hour?
- liq_imbalance = (liq_density_above - liq_density_below) / total_liq (directional pressure)
- liq_total_usd_1h: total USDT liquidated in last 1h (volatility regime indicator)

**Integration with existing strategies:**
- cb_adx_breakout: after BB squeeze breakout, if liq_cleared_above (breakout cleared resistance)
  = stronger signal; if dense liq zone directly above = weaker signal (potential reversal)
- regime_adaptive: add liq_imbalance as an additional regime feature

**Note:** The liquidation orders endpoint provides a snapshot of recent liquidations, not a
continuous time series. Building a proper liq density model requires accumulating liquidation
data every ~5 minutes (simple polling script). Without historical liquidation time series,
this cannot be backtested on historical data.

**Data collection requirement:**
`scripts/collect_okx_liquidations.py` — poll every 5 minutes, store with timestamp, build
90-day history before backtesting. This is a 2-3 month data collection project before it
can be validated.

**Crypto fit:**
- 24/7 compatible: Yes
- Cannot backtest historically without collection: Blocker
- Works on ETH/SOL: Yes (OKX provides for all pairs)

**Evidence quality:** LOW (theoretical basis is sound, no published paper with OKX-specific
results found; requires data collection before any validation is possible).

**Next step:** `scripts/collect_okx_liquidations.py` to start data collection.
After 90 days, run `research/analyze_liquidation_clusters.py`.

---

## Strategy 6: BTC Canary Cross-Pair Feature

**Source:**
- arXiv:2506.08718 (2025): "Centralized markets typically lead ETH price discovery; futures
  markets lead overall" — BTC/centralized futures lead price formation
- Standard crypto market observation: BTC moves 1-3 bars before ETH/SOL at 15m resolution
- Current system: BTC is used only as a RULE-BASED filter (BTC ADX for regime_adaptive),
  not as ML features

**Applicability:** MEDIUM-HIGH
**Type:** cross-asset lead-lag

**Concept:**
BTC is the most liquid crypto pair. Its price movements tend to lead altcoins by 1-3 bars
at 15m resolution. Using BTC's real-time microstructure features as inputs to ETH/SOL ML
models captures the leading signal before it propagates.

**BTC canary feature set (for ETH/SOL ML models):**
- btc_return_1bar: BTC's last 15m return (direct lead-lag)
- btc_return_3bar: BTC's 45m cumulative return
- btc_taker_buy_ratio_1h: BTC's taker flow (aggressive buying = sector-wide risk-on)
- btc_oi_pct_change_1h: BTC OI expanding = new money entering = bullish for alts
- btc_dominance_change: BTC.D trending up = alts underperform; down = alt season
- eth_btc_ratio_deviation: ETH/BTC ratio vs its 20-bar mean (relative strength)

These features are INDEPENDENT of ETH's own OHLCV and do not suffer from the "no residual
signal" problem identified in OHLCV meta-labeling. They add cross-pair structural information.

**ML use:**
1. Add btc_* features to any future ML model as additional input features
2. Rule-based filter first (no ML needed for basic version):
   - If btc_return_1bar > +0.3% (BTC just pumped): suppress ETH SHORT entries for 1 bar
   - If btc_return_3bar < -0.5% (BTC in downtrend): suppress ETH/SOL LONG entries

**Crypto fit:**
- 24/7 compatible: Yes
- Survives costs: Only in combination (this is a feature set, not standalone strategy)
- Freqtrade implementable: Yes (BTC/USDT:USDT in informative_pairs)

**Evidence quality:** MEDIUM (price discovery paper confirms the direction, but the specific
1-3 bar lag at 15m is empirical observation, not academically validated for OKX).

**Next step:** Add `BTC/USDT:USDT` to informative_pairs in regime_adaptive and cb_adx_breakout.
Compute btc_taker_buy_ratio, btc_oi_change as features. Run 60d backtest with/without to
measure WR effect. This is a low-cost experiment using existing data + one API call change.

---

## Strategy 7: Regime-Only Classifier (No Per-Trade Labels Needed)

**Source:**
- oscartiz/trading (GitHub, 2025): 3-state Gaussian HMM for Hyperliquid perpetual futures
  - BTC 2024: 12 trades, +18.82, Sharpe 1.14 (but very few trades)
  - Multi-coin 2022-2025: 392 trades, -$254 (mixed results, BTC/AVAX only positive)
- arXiv:1902.10849 (Fons et al. 2019): Feature Saliency HMM achieves 60% excess return annually
- hmmlearn library (3.3k stars): GaussianHMM with n_components=3

**Applicability:** MEDIUM
**Type:** regime detection

**Key learning from oscartiz/trading failure on multi-coin:**
The HMM failed on most altcoins because it was tuned on BTC defaults. Per-coin calibration
is required (`tools/calibrate_gates.py`). ETH and SOL have different volatility dynamics.

**Concept:**
Train a 3-state GaussianHMM on 1h ETH features (NOT 15m — too noisy for HMM):
- Features: [1h_return, atr_ratio (current/20bar_avg), funding_rate, taker_buy_ratio_1h]
- States: TREND_UP / RANGE / TREND_DOWN (labeled post-hoc by mean emission values)
- Train on 150d of 1h ETH data (150 * 24 = 3600 bars = adequate for HMM fitting)
- At each 15m bar: look up the current 1h bar's regime label
- Gate strategies: regime_adaptive only activates in TREND_UP or TREND_DOWN
  (this is what it already tries to do with ADX, but HMM + funding adds structure)

**Why this avoids the sample size problem:**
HMM is UNSUPERVISED. No trade labels needed. It learns market state structure from the
raw features. The 3600 1h bars provide more than enough observations to fit a 3-component
Gaussian HMM reliably.

**Implementation:**
```python
# Minimum viable regime classifier
import numpy as np
from hmmlearn import hmm

# Features: [norm_return, atr_ratio, funding_rate, taker_buy_ratio]
obs = df_1h[['return_norm', 'atr_ratio', 'funding_rate', 'taker_buy_ratio']].values

# Fit 3-state model
model = hmm.GaussianHMM(n_components=3, covariance_type='full',
                         n_iter=1000, random_state=42)
model.fit(obs)
regimes = model.predict(obs)  # 0, 1, 2 — label post-hoc

# In Freqtrade: embed regime as a feature in informative_pairs 1h dataframe
# Then in 15m populate_indicators: forward-fill regime from 1h
```

**Validation:**
- After fitting, check that the 3 states have different mean returns (labeling is consistent)
- Walk-forward: fit on first 90d, predict next 30d; assess whether regimes align with
  periods where current strategies perform well vs poorly
- If regime 0 aligns with strategy win rates > 55% and regime 1 with WR < 45%, the
  classifier has demonstrated utility as a gating filter

**Crypto fit:**
- 24/7 compatible: Yes (1h timeframe, no session dependency)
- Does not need per-trade labels: Yes
- Regime shifts 5-15 times/month at 1h resolution: manageable
- Freqtrade implementable: Yes (hmmlearn on informative 1h data, regime as column)

**Evidence quality:** MEDIUM — oscartiz/trading shows it works for BTC single-pair.
The multi-coin failure is a warning: requires per-coin calibration.

**Next step:** `research/analyze_hmm_regime_eth.py`
Fit on ETH 1h data only. Label states. Check alignment with strategy performance periods.
This is a 1-day experiment using hmmlearn.

---

## Strategy 8: Online Learning Adaptation (River library)

**Source:**
- arXiv:2103.14079 (2021): Domain-specific concept drift detectors beat continuous retraining
  in computational efficiency while maintaining accuracy
- River library (riverml.xyz): production-ready online ML for Python 3.12
- Our failure: models trained on 60d regime fail when regime changes (documented)

**Applicability:** MEDIUM (for the specific problem of regime-dependent model degradation)
**Type:** adaptive ML

**Concept:**
Instead of batch-retraining a LightGBM every N days, use River's ADWIN + Hoeffding Adaptive
Tree (HAT) which:
1. Maintains a sliding window of recent examples using ADWIN drift detector
2. Updates the tree incrementally after every new bar (no retraining delay)
3. Detects concept drift (regime change) and automatically resets the degraded branches
4. CPU inference at <1ms (far faster than LightGBM batch inference)

**Usage in our context:**
Replace the scheduled retraining logic in ML strategies with River's online classifier.
At each 15m bar:
1. Receive new bar's features (OHLCV + taker volume + OI + funding)
2. `model.predict_proba_one(features)` → confidence for long/short
3. After trade closes: `model.learn_one(features, actual_label)` → update in real-time
4. ADWIN monitors accuracy; if drift detected, affected branches reset

**Honest limitation:**
River's models are simpler than LightGBM (decision trees, not gradient boosted ensembles).
They sacrifice some peak accuracy for adaptability. For our situation where regimes shift
every 2-4 weeks, the adaptability benefit likely outweighs the peak accuracy cost.

**Key prerequisite for using River in live trading:**
Requires actual trade outcomes as labels in near-real-time. This means the system needs
to record each bar's features when a signal fires and later update the model when the
trade closes. This is a moderate implementation effort in CryptoEngine.

**Crypto fit:**
- 24/7 compatible: Yes
- Real-time learning: Yes
- Freqtrade implementable: Yes (model.learn_one in on_exit hook, predict in on_tick)
- Not backtestable conventionally: PARTIAL (River has simulate_qa for offline evaluation)

**Evidence quality:** MEDIUM (concept drift literature supports the approach; no published
paper found specifically applying River's HAT to 15m crypto futures).

**Next step:** `research/analyze_online_learning_river.py`
Implement offline simulation: simulate River HAT on 6 months of backtest data using
simulate_qa function. Compare WR stability across regime transitions vs batch LightGBM.

---

## Ranked Action Priority

### Tier 1 — Data Available Now, No New Infrastructure

| Rank | Strategy | Blocker | Expected Lift | Time |
|------|----------|---------|---------------|------|
| 1 | Taker volume CVD as rule-based filter | None (API call, 1d) | +3-5pp WR on volume_spike_rev | 1-2d |
| 2 | Funding rate regime gate (rule-based first) | None | Suppresses bad trades in crowded regimes | 1d |
| 3 | BTC canary (btc_taker_buy_ratio as feature) | BTC in informative_pairs | Removes 10-15% bad altcoin entries | 1d |
| 4 | L/S divergence as rule-based filter | None (API call) | Moderate, pair-dependent | 1d |

### Tier 2 — Requires ML Training (available data)

| Rank | Strategy | Sample Requirement | Expected Lift | Time |
|------|----------|-------------------|---------------|------|
| 5 | Combined OI+funding+taker LightGBM regime classifier | 180d 4h data (1080 bars) | Stable regime gating | 3-5d |
| 6 | HMM regime classifier on ETH 1h | 150d 1h data (3600 bars) | Replaces ADX regime logic | 2-3d |
| 7 | River online learning on live signal stream | Real-time trade outcomes | Eliminates retraining lag | 1-2 weeks |

### Tier 3 — Requires Data Collection (not backtestable yet)

| Rank | Strategy | Collection Needed | Timeline |
|------|----------|------------------|----------|
| 8 | Liquidation cluster S/R | 90d liquidation polling | 3+ months |
| 9 | L2 orderbook ML (Tier A OFI) | 90d tick-level LOB | 3+ months |

### Do Not Pursue

| Approach | Reason |
|----------|--------|
| OHLCV meta-labeling on 15m signals | Confirmed failed 3 times; OHLCV has no residual signal |
| Direction prediction LSTM/TFT on OHLCV only | 51% accuracy ceiling; costs consume edge |
| RL (PPO/SAC) for 15m scalping | No published evidence; overfitting risk high (Gort 2022) |
| Meta-labeling with <500 IS trades | Sample size blocker is fundamental |
| Stacking ensemble on OHLCV features | Same feature space as failed meta-labeling |

---

## Implementation Notes for Freqtrade Integration

### Fetching OKX Microstructure Data in IStrategy

```python
# In populate_indicators or informative_pairs:
import requests

def get_okx_taker_volume(pair: str, period: str = "15m", limit: int = 300) -> pd.DataFrame:
    """Fetch taker buy/sell volume from OKX rubik endpoint."""
    inst_id = pair.replace("/", "-").replace(":USDT", "-SWAP")
    url = f"https://www.okx.com/api/v5/rubik/stat/taker-volume-contract"
    params = {"instId": inst_id, "period": period, "limit": limit}
    r = requests.get(url, params=params).json()
    df = pd.DataFrame(r['data'], columns=['ts', 'buy_vol', 'sell_vol'])
    df['ts'] = pd.to_datetime(df['ts'].astype(int), unit='ms', utc=True)
    df['taker_buy_ratio'] = df['buy_vol'].astype(float) / (
        df['buy_vol'].astype(float) + df['sell_vol'].astype(float))
    return df.set_index('ts').sort_index()

def get_okx_open_interest(pair: str, period: str = "15m", limit: int = 300) -> pd.DataFrame:
    """Fetch open interest history."""
    inst_id = pair.replace("/", "-").replace(":USDT", "-SWAP")
    url = f"https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history"
    params = {"instId": inst_id, "period": period, "limit": limit}
    r = requests.get(url, params=params).json()
    df = pd.DataFrame(r['data'], columns=['ts', 'oi_contracts', 'oi_ccy', 'oi_usd'])
    df['ts'] = pd.to_datetime(df['ts'].astype(int), unit='ms', utc=True)
    df['oi_usd'] = df['oi_usd'].astype(float)
    df['oi_pct_change'] = df['oi_usd'].pct_change()
    return df.set_index('ts').sort_index()

def get_okx_funding_rate(pair: str, limit: int = 100) -> pd.DataFrame:
    """Fetch funding rate history."""
    inst_id = pair.replace("/", "-").replace(":USDT", "-SWAP")
    url = f"https://www.okx.com/api/v5/public/funding-rate-history"
    params = {"instId": inst_id, "limit": limit}
    r = requests.get(url, params=params).json()
    df = pd.DataFrame(r['data'])
    df['fundingTime'] = pd.to_datetime(df['fundingTime'].astype(int), unit='ms', utc=True)
    df['fundingRate'] = df['fundingRate'].astype(float)
    df['fundingRate_3d_mean'] = df['fundingRate'].rolling(9).mean()  # 9 * 8h = 3d
    return df.set_index('fundingTime').sort_index()
```

### LightGBM Regime Classifier Template

```python
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Features: derivatives data, not OHLCV
feature_cols = [
    'funding_rate', 'funding_rate_3d_mean', 'oi_pct_change_4h',
    'oi_price_divergence', 'taker_buy_ratio_4h', 'taker_cvd_slope_4h',
    'ls_divergence', 'btc_return_1h', 'adx_1h', 'atr_ratio_1h'
]

# Labels: rule-based regime classification (no per-trade outcomes needed)
def label_regime(row):
    if row['adx_1h'] > 25 and row['atr_ratio_1h'] > 1.0:
        return 'TRENDING'
    elif row['atr_ratio_1h'] > 1.8:
        return 'VOLATILE'
    else:
        return 'RANGING'

df_4h['regime'] = df_4h.apply(label_regime, axis=1)
le = LabelEncoder()
y = le.fit_transform(df_4h['regime'])

# Train LightGBM regime classifier
params = {
    'objective': 'multiclass', 'num_class': 3,
    'n_estimators': 100, 'learning_rate': 0.05,
    'num_leaves': 15,  # small: low data volume (1080 4h bars)
    'min_child_samples': 20, 'verbose': -1
}
model = lgb.LGBMClassifier(**params)
model.fit(X_train[feature_cols], y_train)
# Output: probabilities for [RANGING, TRENDING, VOLATILE]
```

---

## Critical Warning: Evidence Quality Standards

All approaches above are ranked by how well the underlying mechanism is validated:

| Evidence Level | Description | Example |
|----------------|-------------|---------|
| CONFIRMED | Exchange-design structural mechanism, replicated over 5+ years | Funding carry (arXiv:2201.04699) |
| HIGH | Multiple academic papers confirm the mechanism | OFI as predictor (4 papers) |
| MEDIUM | One academic paper + theoretical basis | HMM regime detection |
| LOW | Theoretical only, no published validation on this specific data | Liquidation clusters |
| ANTI-PATTERN | Confirmed failures in our own research | OHLCV meta-labeling |

The rule-based versions (Tiers 1) of all approaches should be validated BEFORE building
any ML model on top of them. If taker_buy_ratio as a hard filter does not improve WR by
3+ percentage points, building a LightGBM on that signal will not help.

---

## References

- Borrageiro, G., Firoozye, N., Barucca, P. (2022). "The Recurrent Reinforcement Learning
  Crypto Agent." IEEE Access 10:38590-38599. arXiv:2201.04699.
  Key finding: 71% of 350% return from funding carry, not direction prediction.

- arXiv:2508.02356 (2025): Multi-timeframe CNN with on-chain + orderbook + GDELT.
  PF 1.15 with 0.05% costs; uses sub-second execution.

- arXiv:2509.10542 (2025): Adaptive TFT on ETH-USDT 10m, Binance.
  51.36% accuracy; 1-week test only; uses only close price volatility.

- arXiv:2606.04574 (2026): PPO+LSTM pair trading on Binance 1h.
  Statistical cointegration pairs, 3-5 features (Z-score + Hurst).

- arXiv:2506.08718 (2025): Price discovery in crypto markets.
  Confirms: centralized exchange futures lead ETH price discovery.

- arXiv:2103.14079 (2021): Concept drift detectors for financial time series.
  Domain-specific detectors beat continuous retraining.

- arXiv:2506.05764 (2025): Microstructural dynamics in crypto LOBs (Bybit, 100ms).
  XGBoost + Kalman filtering beats DeepLOB; simpler models with better preprocessing win.

- Lu, Reinert, Cucuringu (2022). "Trade Co-occurrence and Conditional Order Imbalance."
  arXiv:2209.10334. OFI correlates with future returns for isolated trades.

- Scaillet, Treccani, Trevisan (2017). "High-Frequency Jump Analysis of Bitcoin."
  arXiv:1704.08175. OFI + spread widening predicts BTC price jumps.

- oscartiz/trading (GitHub, 2025): HMM regime bot for Hyperliquid perpetual futures.
  BTC 2024 backtest: Sharpe 1.14; multi-coin failure: requires per-coin calibration.

- Gort et al. (2022). arXiv:2209.05559. DRL backtest overfitting in crypto.
  Warning: most published RL results are false positives; statistical test provided.

- Research files this supersedes/extends:
  research/ml_alternatives_5m.md, research/ml_comprehensive_catalogue.md,
  research/freqai_and_ml_frameworks.md

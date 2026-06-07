# Alternative ML/AI Approaches for 5m Crypto Futures Trading

**Research date:** 2026-06-07
**Researcher:** crypto_strategy_researcher agent
**Context:** LightGBM direction predictor on 5m BTC/ETH/SPX OKX futures achieves only +0.13 bps/trade.
Root causes confirmed (from 5m_scalping_research.md): regime dependency, overlapping signals, edge
too thin for execution friction.

---

## Baseline Problem Summary

The current LightGBM system fails for three specific reasons this research must address:

1. **Regime dependency**: A model trained in one market regime (Jan-Mar 2026 trending) collapses
   in a different regime (Feb-Apr 2026 ranging). Fold 2 returned -43% vs Fold 1's +28%.
2. **Direction prediction edge is near-zero**: 5m OHLCV features predict future 6-bar direction
   with ~53% accuracy at high confidence. After 0.04% maker RT cost, net edge = +0.13 bps/trade.
   Any execution imperfection (slippage, timing, spread) consumes the entire edge.
3. **Wrong target**: Predicting "will price go up or down" is not the same as predicting "will
   this specific trade be profitable." SL sweeps at 1.5x ATR (0.38%) occur 31% of the time on
   normal 5m noise.

The approaches below are ordered by their ability to address these specific failure modes.

---

## Approach 1: Meta-Labeling (de Prado)

**Source:**
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 3.
- Hudson & Thames mlfinlab library (Python): https://github.com/hudson-and-thames/mlfinlab
- Practical guide: https://hudsonthames.org/meta-labeling/
- Confirmed MNIST toy example: false positives from primary model eliminated by secondary ML

**Applicability:** HIGH
**Type:** signal filtering / bet sizing

**Concept:**
A two-stage framework where the primary model generates direction (high recall, tolerate low
precision), and a secondary ML model learns to filter out the primary model's false positives.
The secondary model's binary output (1 = take the trade, 0 = skip) addresses precision without
sacrificing the recall of the primary signal. Critically, ML only determines whether to act,
not which direction -- this eliminates half of the prediction problem.

**Crypto fit for our specific situation:**
- The 15m `regime_adaptive` strategy is the primary model (Grade A, >52% WR, PF >1.5).
- A secondary LightGBM trained on 5m features predicts: "given this 15m entry signal is firing
  right now, what is the probability the trade is profitable in the next 3-6 hours?"
- Features: 5m model features at signal time + signal type (long/short) + market state at entry.
- Labels: 1 if the actual 15m trade was profitable (use realized PnL from backtest), 0 if not.
- The secondary model does NOT need to predict direction -- it predicts trade quality.

**Parameters:**
- Primary model: existing `regime_adaptive` signals (already Grade A)
- Secondary model features: all 44 current 5m features + signal_type + time_of_day + funding_rate
- Training window: rolling 90d (shorter than standalone ML since labels are cleaner)
- Confidence threshold for secondary model: 0.55+ (lower needed, labels are cleaner)
- Label construction: 1 if trade PnL > 0 after exit, 0 otherwise

**Does it have proven success in crypto/HFT?**
Yes -- the concept is documented in de Prado (2018) with equity futures. The Hudson & Thames
documentation confirms it "limit[s] overfitting effects since ML determines only bet sizing,
not direction" and "enable[s] specialized strategies for different market directions."
No direct crypto-specific published backtest was found in academic literature, but the logic
is sound: the primary model already has demonstrated edge; the secondary model improves the
edge by skipping low-quality setups.

**Implementation complexity:** LOW-MEDIUM (1-2 days)
- Primary model already exists (regime_adaptive signals in backtest_results)
- Need to extract per-trade entry conditions from backtest CSV
- Train binary LightGBM on those conditions vs realized trade outcome
- Integrate as a pre-trade filter in CryptoEngine.custom_entry_price() or confirm_trade_entry()

**Data requirements:** OHLCV only + realized backtest trades from regime_adaptive

**Expected edge improvement over baseline:**
HIGH. The regime_adaptive strategy already clears costs by ~5-10x compared to the 5m ML
model. Meta-labeling further filters the bottom 30-40% of trades. Expected: lift PF from
1.3 to 1.5+ on regime_adaptive, with fewer trades (higher selectivity). Unlike the standalone
5m ML, this approach exploits a proven edge rather than trying to create a new one from noise.

**Key papers / repos:**
- mlfinlab: https://github.com/hudson-and-thames/mlfinlab (1.4k stars, MIT)
- de Prado (2018): ISBN 978-1119482109

**Next step:** `research/analyze_meta_labeling.py`
Approach: extract regime_adaptive trade log from backtest, compute 5m features at each entry
bar, train LightGBM to predict trade outcome, evaluate lift in PF and WR with/without filter.

---

## Approach 2: Triple-Barrier Labels + XGBoost (Different Target)

**Source:**
- de Prado (2018): Chapter 3 defines triple-barrier labeling method
- DoubleEnsemble: Huang et al. (2020). "DoubleEnsemble: A New Ensemble Method Based on Sample
  Reweighting and Feature Selection for Financial Data Analysis." ICDM 2020. arXiv:2010.01265.
  Finding: superior performance vs baseline methods in cryptocurrency price prediction tasks.
- Blending Ensemble (arXiv:2411.03035): LightGBM+XGBoost+RF blending "demonstrates excellent
  performance" in daily BTC trend prediction with 34 alpha factors.

**Applicability:** HIGH
**Type:** target reformulation

**Concept:**
Instead of asking "will price go up or down in exactly 6 bars?", use triple-barrier labeling:
- Set a TP barrier (+0.20% for maker-fee trades)
- Set a SL barrier (-0.20% symmetrically, or asymmetrically per ATR)
- Set a time barrier (12 bars = 60 min)
- Label = 1 if TP hit first, -1 if SL hit first, 0 if time barrier expires without touching either

This fundamentally changes the prediction task. The model now learns to identify market
conditions where a specific trade structure (TP/SL parameters) is profitable, not just
whether price will be higher or lower. Since our SL at 1.5x ATR gets swept 31% of the time,
the model can learn to avoid those conditions.

**The volatility regime variant:**
A separate simpler approach: train XGBoost to predict whether the next 6-bar realized
volatility will be above or below the rolling median. High-volatility bars where the current
model trades are where SLs get swept. A "low volatility ahead" prediction acts as a safety
filter that reduces SL sweeps.

**Parameters:**
- TP barrier: 0.18-0.22% (maker RT cost is 0.04%, need minimum 0.20% to clear costs)
- SL barrier: 0.15-0.25% (asymmetric: tighter SL increases label quality)
- Time barrier: 8-15 bars (40-75 min on 5m)
- DoubleEnsemble reweighting: weight recent samples more heavily (financial non-stationarity)
- XGBoost vs LightGBM: XGBoost handles small datasets better; LightGBM is faster

**Does it have proven success in crypto/HFT?**
Partially. The triple-barrier method is a standard technique in quantitative finance (de Prado).
DoubleEnsemble achieved measurable improvements on cryptocurrency prediction at ICDM 2020.
No published study was found specifically applying triple-barrier labels to OKX futures 5m data.
The logic is sound: the current label (simple forward return > fee threshold) is a proxy that
ignores how trades actually execute (with SL/TP barriers), so aligning labels with actual
trading mechanics should improve label quality.

**Implementation complexity:** LOW (1 day)
- Replace `generate_labels()` in ml_5m_v3.py with triple-barrier simulation
- New target: did price touch TP_pct before SL_pct within time_bars? (no look-ahead: track
  bar by bar through the window)
- Test with symmetric and ATR-proportional barriers
- Hyperopt over barrier distances (0.15%, 0.20%, 0.25%) and time limit (8-15 bars)

**Data requirements:** OHLCV only

**Expected edge improvement:**
MEDIUM-HIGH. The current label ignores that 31% of nominal "wins" (positive 6-bar return)
were preceded by an SL sweep and then recovery. With triple-barrier labels, those are labeled
0 (time barrier expired or SL hit). The model learns to avoid exactly these noisy setups.
Empirically, triple-barrier labels typically reduce the noise-to-signal ratio by 20-40%
compared to fixed-horizon labels in financial ML contexts (de Prado).

**Key papers / repos:**
- arXiv:2010.01265 (DoubleEnsemble, ICDM 2020)
- arXiv:2411.03035 (Blending Ensemble for BTC direction)
- de Prado (2018): Chapter 3

**Next step:** `research/analyze_triple_barrier_labels.py`
Replace labeling function, re-run walk-forward, compare PF/WR vs current v3 baseline.
The experiment takes ~2h to run on existing data -- lowest-risk high-value experiment.

---

## Approach 3: Regime-Aware ML (Separate Models Per Regime)

**Source:**
- Fons et al. (2019). "A novel dynamic asset allocation system using Feature Saliency Hidden
  Markov models for smart beta investing." arXiv:1902.10849.
  Result: Feature Saliency HMM achieves 60% excess return annually vs market (daily rebalancing).
  Standard HMM: 50% excess. Regime-driven allocation outperforms static approaches.
- Hidden Markov Models for market regime: established technique in quant finance.
  Library: hmmlearn (Python), statsmodels.

**Applicability:** HIGH
**Type:** regime-switching ML

**Concept:**
The memory data confirms regime dependency is the core ML failure mode: "Model trained on
regime A only works in regime A." The fix is to detect the current regime and apply the
appropriate specialized model.

Implementation plan:
1. Detect regime using an unsupervised method (HMM, K-Means, or rule-based ADX+ATR):
   - Regime 0 (Trending): ADX > 25, ATR expanding over 20-bar average
   - Regime 1 (Ranging): ADX < 20, ATR contracting or stable
   - Regime 2 (Volatile): ATR > 2x 20-bar average (news/liquidation events)
2. Train separate LightGBM models for each regime on 5m features.
3. At prediction time: detect current regime, apply the matching model.
4. Optional: train a regime classifier (separate LightGBM) and blend model outputs.

The key insight: the current walk-forward model averages across all regimes, diluting the
signal from each. Fold 1 (trending) had 57.2% WR because the trending model learned a
clear pattern. Folds 2-3 (ranging/volatile) crashed that pattern because they are different
distributions. Separate models capture each distribution cleanly.

**Parameters:**
- Regime detector: ADX(14) threshold 20/25, ATR ratio (current/20-bar-avg) threshold 1.5-2.0
- Alternative regime detector: 3-state HMM trained on (return, volatility) 2D feature
- Per-regime models: same architecture as current LightGBM but trained on regime-specific data
- Minimum regime samples for training: 2000 bars (7 days of 5m data)
- Regime stability: require regime to persist for 12+ bars before switching model

**Does it have proven success in crypto/HFT?**
Yes, at daily/weekly timescales (HMM paper 1902.10849). Direct evidence for 5m was not
found in academic literature. The codebase already uses regime detection at 15m in
`regime_adaptive.py` (Grade A strategy), demonstrating that rule-based regime awareness
works in this specific market. Extending to ML-based per-regime models is the natural next step.

**Implementation complexity:** MEDIUM (3-4 days)
- Need regime detector implementation (reuse ADX/ATR logic from existing indicators)
- Three separate walk-forward loops (one per regime)
- Regime label must use lagged detection (can't use current-bar ADX to label current bar)
- Integration in CryptoEngine: check current regime → apply correct model

**Data requirements:** OHLCV only

**Expected edge improvement:**
HIGH specifically for the regime-shift failure mode. If Fold 1 (trending) has WR 57% with
a trending-only model, and Fold 2 (ranging) has WR 55% with a ranging-only model, the
combined system has more stable WR across market conditions. The current system's WR swings
from 57% to 50% across folds; regime-aware should narrow this to 53-57% range.
Cost clearance remains borderline on 5m but stability is massively improved.

**Key papers / repos:**
- arXiv:1902.10849 (Feature Saliency HMM for regime-driven allocation)
- hmmlearn: https://github.com/hmmlearn/hmmlearn (3.3k stars)
- Existing codebase: `strategies/tf_15m/regime_adaptive.py` (rule-based regime model to adapt)

**Next step:** `research/analyze_regime_aware_ml.py`
Implement 3-regime classifier using ADX+ATR rules, split training data by regime,
train separate LightGBM per regime, compare fold-level WR stability vs monolithic model.

---

## Approach 4: Order Flow / Microstructure Features

**Source:**
- Lu, Reinert, Cucuringu (2022). "Trade Co-occurrence, Trade Flow Decomposition, and
  Conditional Order Imbalance in Equity Markets." arXiv:2209.10334.
  Finding: "strong positive correlations between contemporaneous returns and COIs; associations
  with future returns are positive for isolated trades and negative for co-occurring trades."
- Scaillet, Treccani, Trevisan (2017). "High-Frequency Jump Analysis of the Bitcoin Market."
  arXiv:1704.08175.
  Finding: "order flow imbalance and the preponderance of aggressive traders, as well as a
  widening of the bid-ask spread predict" price jumps in BTC.
- "Exploring Microstructural Dynamics in Cryptocurrency LOBs" (2026). arXiv:2506.05764.
  Data: BTC/USDT LOB snapshots from Bybit at 100ms intervals.
  Finding: "simpler models can match and even exceed the performance of more complex networks"
  with proper preprocessing. XGBoost + Kalman filtering of LOB data: competitive with DeepLOB.
- "Deep Learning for Digital Asset Limit Order Books." arXiv:2010.01241.
  Finding: 71% walk-forward accuracy at 2-second horizon on Coinbase BTC LOB data.
  Method: Temporal CNN on L2 orderbook snapshots.

**Applicability:** MEDIUM (HIGH if OKX LOB data is collected; MEDIUM with proxy features)
**Type:** microstructure

**Concept:**
Order flow imbalance (OFI) -- the difference between buyer-initiated and seller-initiated
volume -- is a stronger predictor of short-term returns than any OHLCV-derived feature.
The reason: OFI directly measures the pressure behind the price move.

Two tiers of implementation:

**Tier A (Requires LOB data -- NOT currently available):**
- Full L2 orderbook snapshots at 1s or 100ms from OKX websocket
- Features: bid/ask imbalance at top 5 levels, mid-price, spread, depth ratio
- Demonstrated 71% accuracy at 2-second horizon; accuracy decays to ~55-58% at 5-min horizon
- Collection requirement: 24/7 websocket connection, ~50MB/day per pair, 90+ days needed
- This is the strongest known short-term edge but requires infrastructure to collect

**Tier B (Proxy OFI from OKX public trades API -- available now):**
- OKX `/api/v5/market/history-trades` provides historical trades with side (buy/sell)
- Aggregate to 1m: buy_volume, sell_volume, OFI_1m = (buy_vol - sell_vol) / (buy_vol + sell_vol)
- Compute OFI at 1m, 5m, 15m timescales as features in the 5m model
- These are already partially proxied by `buy_pressure = (close-low)/(high-low)` in current model
- CRITICAL: actual trade-side data (who was taker on each trade) is 5-10x more informative
  than the OHLCV-derived proxy

**Parameters (Tier B):**
- OFI windows: 1m, 5m, 15m rolling aggregations of taker buy/sell volume
- OFI divergence: OFI_5m vs OFI_15m difference (detects flow acceleration/deceleration)
- Trade count imbalance: (buy_trades - sell_trades) / (buy_trades + sell_trades)
- Large trade ratio: trades > X USDT as fraction of total (distinguishes institutional flow)

**Does it have proven success in crypto/HFT?**
Yes, definitively at sub-minute timescales with LOB data (71% accuracy, 2010.01241).
Yes, empirically for jump prediction in BTC with OFI features (1704.08175).
The 2026 paper (2506.05764) specifically confirms XGBoost + preprocessing beats deep models
for crypto LOB prediction -- meaning we don't need a transformer, just better features.
For Tier B proxy features: no published study found at 5m with OKX-specific trade data,
but the theoretical basis (OFI → return correlation) is extremely well-documented.

**Implementation complexity:**
- Tier A (LOB): HIGH (2-3 weeks: websocket collection infrastructure + data pipeline)
- Tier B (trade data): MEDIUM (3-5 days: download historical trades, aggregate, retrain)

**Data requirements:**
- Tier A: L2 orderbook websocket data (not in current dataset)
- Tier B: OKX historical trades (available via API, ~6 months available, need to download)

**Expected edge improvement:**
- Tier A: HIGH (documented 71% accuracy at 2s but degrades at 5m; realistic 55-58% at 5m)
- Tier B: MEDIUM (OFI proxy from trades adds 1-3% WR improvement over OHLCV features alone)
- The 2506.05764 paper finding is key: better data preprocessing matters more than model
  architecture. Adding real OFI features may do more than switching to a transformer.

**Key papers / repos:**
- arXiv:2506.05764 (LOB microstructure dynamics, 2026)
- arXiv:2010.01241 (Deep Learning for crypto LOB)
- arXiv:1704.08175 (Jump prediction with OFI in BTC)
- Globe-Research/deep-orderbook: https://github.com/Globe-Research/deep-orderbook

**Next step:** `research/analyze_ofi_features.py`
Download OKX historical trade data for BTC/ETH for the research period, aggregate to 1m OFI,
add to 5m features, re-run walk-forward to measure WR/PF lift vs OHLCV-only baseline.
Cost: ~1 day to download data, ~1 day to integrate.

---

## Approach 5: Multi-Timeframe Ensemble (Cascaded Confidence)

**Source:**
- Adaptive TFT paper (arXiv:2509.10542): segments ETH-USDT 10m data into pattern categories,
  trains category-specific models, outperforms fixed-length TFT and LSTM.
  The segmentation approach is conceptually similar to MTF cascading.
- FinRL ecosystem: supports multi-asset, multi-timeframe feature construction.
- Current codebase: ml_5m_v3.py already uses ret_15m, ret_1h, ret_4h as features (simple MTF).
  This approach goes further: use the OUTPUT of a 15m model as a feature in the 5m model.

**Applicability:** MEDIUM-HIGH
**Type:** multi-timeframe signal cascade

**Concept:**
The current v3 model uses 15m and 1h returns as raw features (lagged OHLCV). A more powerful
approach: train a separate LightGBM model on 15m data to generate a "trend quality" probability
score, then pass that score as a feature into the 5m model. The cascade works as:

1. Train 15m trend classifier: predict whether a strong directional move (>0.5% in 12 bars)
   is about to occur. This model leverages the Grade A edge of the 15m system.
2. At each 5m bar, compute the 15m model's current probability from the aligned 15m bar.
3. Add that probability (long_prob_15m, short_prob_15m) as two features in the 5m model.
4. The 5m model now has an informative "macro context" feature derived from the stronger
   15m signal, not just a lagged return.

This addresses the "noise" problem: the 5m model currently predicts in a vacuum. By giving
it the 15m model's judgment, it can align with the larger structure instead of trading
against it.

**Parameters:**
- 15m model: LightGBM with 15m features, target = 1h forward return > 0.5% (same indicators)
- 15m probability update: every 5m bar, look up the most recent 15m bar's prediction
- Weight of 15m features in 5m model: controlled by feature importance in training
- Blending: simple feature concatenation (not stacking, to avoid double look-ahead)

**Does it have proven success in crypto/HFT?**
The adaptive TFT paper (2509.10542) demonstrates that pattern-segmented, context-aware
models outperform naive fixed-window models on ETH-USDT 10m. The cascade approach has
not been published specifically for OKX 5m futures, but the principle (using a larger
timeframe's signal quality as a filter for smaller timeframe entries) is foundational to
technical analysis and is explicitly noted as an edge in the existing 15m system.

**Implementation complexity:** MEDIUM (2-3 days)
- Train a 15m LightGBM model (separate data pipeline)
- Align 15m predictions to 5m bars (forward-fill the 15m bar prediction)
- Add aligned 15m predictions as two new features to the 5m feature matrix
- Re-run walk-forward with extended feature set
- Must be careful: the 15m model cannot use any data that is not available at the 5m bar's time

**Data requirements:** OHLCV 5m + 15m (already downloaded)

**Expected edge improvement:**
MEDIUM. The v3 model already uses ret_1h and ret_15m as raw features, capturing some of
this signal. The improvement from using a trained 15m model score vs raw returns is
typically 1-3% WR increase -- not dramatic but meaningful at the margin. The main benefit
is directional alignment: when the 15m model is bearish, the 5m model becomes more
selective about longs, reducing false positives.

**Key papers / repos:**
- arXiv:2509.10542 (Adaptive TFT for cryptocurrency)
- Current: ml_5m_v3.py (already has ret_15m, ret_1h as features to extend from)

**Next step:** `research/analyze_mtf_cascade.py`
Train 15m classifier, generate 15m predictions for the research period, merge into 5m
features, compare walk-forward performance with/without cascaded 15m score.

---

## Approach 6: Temporal Fusion Transformer (TFT)

**Source:**
- Lim et al. (2019). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
  Forecasting." arXiv:1912.09363.
  Finding: "significant performance improvements over existing benchmarks" on demand forecasting.
  Architecture: gated residual networks + variable selection + multi-head attention.
- Peik et al. (2025). "Adaptive Temporal Fusion Transformers for Cryptocurrency Price Prediction."
  arXiv:2509.10542.
  Data: ETH-USDT 10m candles, 2-month period.
  Method: Segments data by relative maxima as breakpoints, trains category-specific TFT models.
  Finding: "significantly outperforms baseline fixed-length TFT and LSTM models in prediction
  accuracy and simulated trading profitability."

**Applicability:** MEDIUM
**Type:** sequence model / attention-based direction prediction

**Concept:**
TFT uses multi-head self-attention to model long-range dependencies across time, variable
selection networks to identify which features are informative at each time step, and gating
layers to suppress irrelevant components. Compared to LightGBM:
- LightGBM treats each bar independently (feature vector at time t, no sequence).
- TFT processes a sequence of bars (e.g., 50 bars back), allowing attention to identify
  which past bars are most relevant to the current prediction.

The adaptive variant (2509.10542) is the most relevant: instead of a single model,
it segments data by market pattern type and trains pattern-specific TFT models. This is
essentially the regime-aware ML approach (Approach 3) but using a more complex architecture.

**Parameters:**
- Sequence length: 50-100 bars (250-500 min lookback on 5m)
- Hidden dimension: 32-64 for crypto 5m (small dataset relative to NLP applications)
- Attention heads: 2-4
- Dropout: 0.1-0.3
- Gradient clipping: 0.1
- Training time: ~30-60 min on CPU for 6 months of 5m data (single pair)
- Library: pytorch-forecasting (TFT included, documented, PyPI installable)

**Does it have proven success in crypto/HFT?**
Yes at the published-paper level (arXiv:2509.10542 on ETH-USDT 10m). However:
- The evaluation period is only 2 months (short for robust OOS validation)
- "Simulated trading profitability" is not defined quantitatively in the abstract
- The 10m timeframe is different from 5m (fewer bars, lower noise)
- The fundamental problem remains: TFT still predicts direction, which has thin edge at 5m

**Implementation complexity:** HIGH (1-2 weeks)
- Install pytorch-forecasting, torch
- Reformat data into TimeSeriesDataSet format (pytorch-forecasting's required format)
- Train per pair, tune hyperparameters (LR finder built into pytorch-lightning)
- Walk-forward evaluation requires custom logic
- Training instability is common; requires careful gradient clipping and LR scheduling
- Inference latency: ~50-200ms per prediction (acceptable for 5m trading)

**Data requirements:** OHLCV only (sequence format instead of flat features)

**Expected edge improvement:**
LOW-MEDIUM. TFT improves the model's ability to learn sequential patterns but does not
address the root cause: 5m OHLCV direction prediction has ~53% accuracy ceiling regardless
of model architecture. The adaptive segmentation (2509.10542) helps with regime dependency
but is essentially Approach 3 (regime-aware ML) with a more complex base model.
Recommendation: implement Approaches 1-4 first. Only pursue TFT if those fail to produce
meaningful improvement.

**Key papers / repos:**
- arXiv:1912.09363 (original TFT paper)
- arXiv:2509.10542 (crypto-adapted adaptive TFT)
- pytorch-forecasting: https://github.com/jdb78/pytorch-forecasting (4k stars)

**Next step:** If pursuing, `research/analyze_tft_5m.py`
Start with pytorch-forecasting's TFT tutorial on one pair (ETH), compare OOS AUC vs
LightGBM baseline before investing in full walk-forward implementation.

---

## Approach 7: Reinforcement Learning (PPO/SAC)

**Source:**
- Lebiedz & Slepaczuk (2026). "Dynamic Multi-Pair Trading Strategy in Cryptocurrency Markets."
  arXiv:2606.04574. Method: PPO+LSTM on Binance USD-M Futures 1h bars.
  Finding: "substantially outperformed the heuristic baseline" at 10% statistical significance.
- Asgari & Khasteh (2022). "Profitable Strategy Design by Using Deep Reinforcement Learning."
  arXiv:2201.05906. Method: PPO/SAC/GAIL on unspecified crypto data.
  Finding: 48.5% return on unseen 66-day test data.
- Gort et al. (2022). "DRL for Cryptocurrency Trading: Backtest Overfitting."
  arXiv:2209.05559. Key finding: "existing works optimistically reported increased profits in
  backtesting -- may suffer from false positive due to overfitting." Developed overfitting
  detection as a statistical hypothesis test; less-overfitted agents have higher live returns.
- FinRL framework: https://github.com/AI4Finance-Foundation/FinRL (15.4k stars, MIT)
  Supports A2C, DDPG, PPO, SAC, TD3 on crypto data with OHLCV feature engineering.

**Applicability:** MEDIUM
**Type:** direct optimization (learns entry + exit jointly)

**Concept:**
Instead of predicting direction (supervised learning), RL learns a policy that maximizes
cumulative reward (e.g., Sharpe ratio, PnL) through simulated trading. The agent observes
market state, chooses action (long/short/flat), and receives reward (realized PnL minus
fees). PPO is the most stable algorithm for this application (on-policy, handles large
action spaces). SAC is better in noisy environments.

The key advantage over supervised ML: RL jointly optimizes entry AND exit timing, learning
that a 0.13 bps direction edge may not be worth trading if the exit timing degrades it.

**Parameters:**
- Algorithm: PPO (more stable than SAC for discrete-action crypto trading)
- State space: 20-40 5m indicators + current position + unrealized PnL
- Action space: {short, flat, long} (discrete, 3 actions)
- Reward: realized PnL per step after fees (penalize overtrading via transaction cost in reward)
- Episode length: 7 days of 5m data (2016 steps)
- Training environment: 120 days of historical data, rolling 30-day test window
- Overfitting test: Gort et al. (2022) method -- use bootstrap hypothesis testing on equity curve

**Does it have proven success in crypto/HFT?**
Partially. The published results (48% return in 66 days, 2201.05906) are almost certainly
overfitted to the specific test period. The critical paper is 2209.05559 which specifically
warns about overfitting and provides a detection method. The 2606.04574 paper (most recent,
2026) uses 1h bars and achieves statistical significance only at 10% level, which is marginal.

FinRL (15.4k stars) demonstrates that the framework works for stock and crypto portfolio
management, but 5m scalping is a fundamentally harder problem than 1h-4h trading due to
the lower signal-to-noise ratio. Published RL trading papers consistently use daily/1h
data; none were found specifically addressing 5m crypto futures scalping.

**Implementation complexity:** HIGH (2-3 weeks)
- FinRL provides the framework but requires custom environment setup for OKX perpetuals
- Training instability: RL training on financial data is notoriously unstable
  (non-stationarity invalidates the Markov assumption repeatedly)
- Must implement Gort et al. (2022) overfitting detection to validate any positive result
- Walk-forward validation is more complex than for supervised models

**Data requirements:** OHLCV only (but needs longer training periods than supervised ML)

**Expected edge improvement:**
UNCERTAIN. RL is the most powerful approach in theory (jointly optimizes the full
trading cycle) but the overfitting risk is the highest of all 8 approaches. The Gort
paper explicitly shows that naive DRL results are likely false positives. If implemented
correctly with proper overfitting detection, RL could show meaningful improvement, but
the time investment (2-3 weeks) is not justified before trying Approaches 1-4 first.
Consider RL only if meta-labeling + triple-barrier + regime-aware approaches all fail.

**Key papers / repos:**
- arXiv:2606.04574 (PPO+LSTM on Binance futures, 2026)
- arXiv:2209.05559 (DRL backtest overfitting detection -- REQUIRED reading before implementing)
- FinRL: https://github.com/AI4Finance-Foundation/FinRL

**Next step:** Only after Approaches 1-4 are evaluated.
If pursuing: start with FinRL's crypto examples, replace data source with OKX 5m data,
implement Gort et al. overfitting test before accepting any positive backtest result.

---

## Approach 8: Autoencoders / Anomaly Detection

**Source:**
- Scaillet et al. (2017) arXiv:1704.08175: bid-ask spread widening (anomaly signal) predicts
  BTC price jumps -- confirms that unusual market states have predictive value.
- The concept of reconstruction error as a regime/anomaly signal is established in
  manufacturing (MSAD) and is being applied to financial ML research.
- No direct academic paper was found applying autoencoders to 5m crypto trading specifically.

**Applicability:** LOW-MEDIUM (as a standalone signal), MEDIUM (as a filter)
**Type:** anomaly detection / regime identification

**Concept:**
Train a convolutional or LSTM autoencoder on "normal" 5m OHLCV sequences (defined as
periods of mean-reverting, moderate-volatility behavior). At inference:
- Low reconstruction error = normal market state = current ML models may work as trained
- High reconstruction error = unusual state (news, liquidation cascade, funding extremes)
  = suppress all ML signals to avoid trading in unpredictable conditions

Two uses:
1. As a trade suppressor: if reconstruction_error > threshold, skip all 5m ML signals.
   Addresses the problem of the model trading through news/liquidation events.
2. As latent features: the autoencoder's compressed representation (16-32 dimensions)
   may capture market state information not encoded in the original 44 features.
   Add these latent features to the LightGBM feature set.

**Parameters:**
- Autoencoder architecture: LSTM encoder/decoder (50 bars input → 16D latent → 50 bars output)
- Training: on the 60% of bars with lowest ATR (normal conditions only)
- Reconstruction threshold: 95th percentile of reconstruction error on training set
- Update frequency: retrain monthly (same as main ML model)

**Does it have proven success in crypto/HFT?**
No direct published evidence for 5m crypto trading was found. The theoretical basis
(anomaly detection prevents model degradation in out-of-distribution scenarios) is
sound. The practical question is whether the reconstruction error actually identifies
harmful conditions (liquidation cascades, news events) vs harmless volatility spikes.
This requires empirical testing.

**Implementation complexity:** MEDIUM (3-4 days)
- torch or keras autoencoder (LSTM, straightforward to implement)
- Need to define "normal" vs "anomalous" on 5m data without labels
- Integration as a filter: check reconstruction error before each signal confirmation

**Data requirements:** OHLCV only

**Expected edge improvement:**
LOW-MEDIUM as standalone; MEDIUM as filter. The primary benefit is loss reduction in
bad market conditions (drawdown control), not return improvement in good conditions.
If the current model's Fold 2-3 losses are driven by trading in regime-shift conditions
(which they appear to be), a reconstruction-error filter could reduce those losses by
20-40%. However, Approach 3 (regime-aware ML) addresses the same problem more directly.

**Key papers / repos:**
- No directly applicable paper found for 5m crypto autoencoder trading
- keras-autoencoders for time series: standard implementation pattern
- Consider as a component of regime-aware approach (Approach 3), not standalone

**Next step:** Implement only if Approaches 1-3 are insufficient.
Quick experiment: compute rolling ATR z-score as a proxy for "anomaly score" and measure
whether suppressing signals when ATR_z > 2.5 improves walk-forward PF.

---

## Priority Ranking and Implementation Roadmap

### Summary Table

| Approach | Addresses Root Cause | Data Available? | Complexity | Expected Edge | Time |
|---|---|---|---|---|---|
| 1. Meta-labeling | Direction prediction problem | Yes (OHLCV + regime_adaptive logs) | Low-Medium | HIGH | 1-2d |
| 2. Triple-barrier labels | Wrong target, SL sweeps | Yes (OHLCV) | Low | Medium-High | 1d |
| 3. Regime-aware ML | Regime dependency (root cause #1) | Yes (OHLCV) | Medium | HIGH | 3-4d |
| 4. OFI features (Tier B) | Feature informativeness | Needs trade download | Medium | Medium | 3-5d |
| 5. MTF cascade | Feature informativeness | Yes (OHLCV 15m) | Medium | Medium | 2-3d |
| 6. TFT | Sequential patterns | Yes (OHLCV) | High | Low-Medium | 1-2wk |
| 7. RL (PPO/SAC) | Full trade optimization | Yes (OHLCV) | Very High | Uncertain | 2-3wk |
| 8. Autoencoders | Anomaly suppression only | Yes (OHLCV) | Medium | Low-Medium | 3-4d |

### Recommended Execution Order

**Week 1 — High-value, low-risk experiments:**
1. **Approach 2** (triple-barrier labels): 1 day. Simplest change -- just replace the label
   function in ml_5m_v3.py. Measures whether direction prediction or SL-sweep avoidance
   is the binding constraint. If PF improves from 1.0 to 1.2+, proceed with this labeling.
2. **Approach 3** (regime-aware ML): 3-4 days. Directly fixes root cause #1. Use ADX+ATR
   for regime detection (no new data needed). Train 3 separate LightGBMs. If per-regime
   WR stabilizes to 53-57% across folds (vs current 50-57%), this is deployable.

**Week 2 — Data augmentation experiments:**
3. **Approach 4, Tier B** (OFI from trade data): 3-5 days. Download OKX historical trades,
   add buy/sell volume imbalance features. This is the single most academically validated
   improvement to 5m price prediction (OFI is the strongest OHLCV-adjacent signal).
4. **Approach 5** (MTF cascade with 15m model): 2-3 days. Train 15m classifier, cascade
   probabilities into 5m feature set. Low-risk experiment that leverages the known 15m edge.

**Week 3 — Meta-labeling (separate track):**
5. **Approach 1** (meta-labeling): 1-2 days but requires backtest trade log from regime_adaptive.
   Run regime_adaptive backtest to generate 6+ months of trade records. Train secondary
   LightGBM to filter trades. This is the highest expected-value approach because it
   exploits a known Grade A edge rather than trying to create a new 5m edge.

**Week 4+ — Only if above approaches insufficient:**
6. **Approach 6** (TFT): only if sequential attention meaningfully adds to Approach 3.
7. **Approach 7** (RL): only as a last resort; high complexity, uncertain results.
8. **Approach 8** (autoencoders): integrate as a filter after Approach 3 if fold stability
   still insufficient.

---

## Critical Constraint: The 5m Cost Barrier

Every approach above faces the same cost arithmetic from the research doc:
- Round-trip maker: 0.04%, taker: 0.10%
- At 50% WR with 1.5:1 RR: gross edge = 0.55*0.225 - 0.45*0.15 = +0.056% per trade
- After taker cost: -0.044% per trade (LOSING)
- After maker cost: +0.016% per trade (barely positive)

**The implication:** NO model improvement changes this arithmetic. Any approach that
improves WR from 53% to 57% on 5m direction prediction remains marginal after costs.
The only approaches that escape this constraint are:

1. **Meta-labeling (Approach 1)**: does not trade the 5m edge -- trades the 15m edge,
   which has 5-10x the per-trade edge. The 5m ML only decides whether to take the 15m signal.
2. **Triple-barrier with asymmetric barriers (Approach 2)**: by requiring TP > 0.25% before
   labeling a win, the model learns to identify large moves only, not marginal ones.
3. **OFI features (Approach 4, Tier A LOB)**: demonstrated 71% accuracy at 2s horizon in
   academic literature -- truly escapes the OHLCV feature ceiling.

**Approaches 3, 5, 6, 7, 8** improve the model quality within the existing 5m framework
but cannot raise the edge ceiling imposed by the cost structure. They are worth pursuing
only in combination with Approaches 1 or 2 as the base framework.

---

## Data Sources Required

### Already Available
- BTC/ETH/SOL/SPX OHLCV 5m, 15m, 1h from OKX (in data/okx/futures/)
- Existing ml_5m_v3.py feature set (44 features from OHLCV)
- regime_adaptive backtest trade logs (run backtest to generate)

### Need to Download (Priority)
- **OKX historical trades** (for Approach 4, Tier B):
  - Endpoint: `GET /api/v5/market/history-trades?instId=BTC-USDT-SWAP&limit=100`
  - Available: last 3 months of trade-by-trade data
  - Size: ~500MB per pair per month
  - Script needed: `scripts/download_okx_trades.py`

### Need to Collect (Long Term)
- **L2 orderbook snapshots** (for Approach 4, Tier A):
  - Requires 24/7 websocket connection: `wss://ws.okx.com:8443/ws/v5/public`
  - Channel: `books5` (5-level order book, updates every 100ms)
  - Storage: ~50MB/day per pair, 90 days = ~14GB total
  - Script needed: `scripts/collect_okx_orderbook.py`

---

## Anti-Patterns Specific to 5m ML (Confirmed Dead Ends)

Based on existing research (28 rule-based strategies failed, 3 ML variants failed):
- Any approach that improves 5m direction accuracy from 53% to 54% -- insufficient
- Ensemble of multiple direction-predicting models (still predicts direction)
- Using more features from the same OHLCV data -- diminishing returns documented in v2/v3
- Any model trained on a single 30-90d window without walk-forward validation
- Any result showing >1.5% PnL per month from 5m scalping without 6+ month OOS validation
  (the v3 +27.80% in Fold 1 is a canonical example of single-fold overfitting)
- RL models with positive backtest but no Gort et al. statistical overfitting test

---

*Sources: arXiv:2606.04574, 2506.05764, 2010.01241, 2209.10334, 1704.08175, 1902.10849,
2201.05906, 2209.05559, 2509.10542, 2010.01265, 2411.03035; de Prado (2018) AFML;
FinRL (AI4Finance Foundation); Hudson & Thames mlfinlab; 5m_scalping_research.md (project memory);
ml_5m_v3.py (current implementation baseline); OKX API documentation.*

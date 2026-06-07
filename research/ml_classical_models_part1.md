# ML Models for Crypto Futures Trading — Part 1: Classical ML & Tree-Based Models

**Date:** 2026-06-07
**Scope:** OKX futures, ETH/USDT, SOL/USDT, SPX/USDT, DOGE/USDT, 15m and 5m timeframes
**Framework target:** Freqtrade IStrategy + custom model files
**Context:** Based on codebase experiments (ml_5m_ensemble.py, ml_triple_barrier.py, ml_meta_labeling.py) and research literature through Aug 2025

---

## Executive Summary from Existing Experiments

Before the model catalogue, the following is already known from this codebase:

| Finding | Detail |
|---------|--------|
| Best algorithm tested | LightGBM (binary classifier, dual long/short models) |
| Best pair for ML | ETH/USDT (thin but real edge), BTC/USDT (marginal) |
| 5m standalone ML | Edge too thin (+0.13bps/trade), regime-dependent, removed |
| 15m meta-labeling | Most promising path: ML filters existing rule-based signals |
| Labeling: triple-barrier | Better than fixed-horizon; directly matches SL/TP execution |
| Walk-forward protocol | 60d train / 30d test / 30d step, per-pair separate models |
| Minimum confidence | 0.63 threshold required across all experiments |
| Cost threshold | 0.04% round-trip (maker) or 0.10% (taker) must be cleared |

---

## PART A: Model Catalogue

---

## 1. LightGBM (Gradient Boosting with Leaf-Wise Growth)

**Source:** https://github.com/microsoft/LightGBM | `pip install lightgbm`
**Applicability:** HIGH — already proven in this codebase
**Type:** Gradient boosting tree ensemble

### How It Works for Trading

LightGBM builds an ensemble of decision trees sequentially, where each tree corrects the errors of the previous one (gradient boosting). Unlike XGBoost's level-wise tree growth, LightGBM grows trees leaf-by-leaf, which finds better splits on large datasets. For trading:

- **Binary classifier (recommended):** Two separate models — one predicts P(long profitable), one predicts P(short profitable). Fire a signal only when one probability exceeds a threshold (0.63+) and is more than 0.05 above the other.
- **Multiclass classifier:** Single model predicts {-1=short, 0=hold, 1=long}. Less recommended because class imbalance is harder to manage.
- **Regressor:** Predicts forward return directly. Useful for position sizing (larger position when confidence higher). Harder to threshold cleanly.

The architecture used in this project (`ml_5m_ensemble.py`, `ml_all_pairs.py`) is the dual binary classifier approach with `scale_pos_weight` to handle class imbalance.

### Input Features (40 features, as implemented)

| Category | Features |
|----------|----------|
| Price action | body_pct, range_pct, upper_wick_ratio, lower_wick_ratio |
| Returns | ret_1, ret_3, ret_6, ret_12, ret_24, ret_36 (bars back) |
| Momentum | RSI(3,9,14), RSI delta, CCI(14), Stochastic K/D, MACD histogram |
| Trend | EMA8/21/50 distance, EMA spread, ADX, DI diff |
| Volatility | ATR%(14), ATR ratio(5/14), BB position, BB width |
| Volume | vol_ratio, vol_ratio_3bar, OBV slope, buy pressure |
| Multi-TF | ret_15m, range_15m, ret_1h, range_1h, ret_4h |
| Temporal | hour_sin, hour_cos, is_us_session, pos_in_day_range |

### Python Libraries

```python
import lightgbm as lgb
# Training
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=300)
# Inference
prob = model.predict(X_test)  # returns float array [0,1]
# Save/load
model.save_model("model.txt")
model = lgb.Booster(model_file="model.txt")
```

### Training Approach

**Walk-forward (proven protocol):**
- Train window: 60 days of bars (17,280 bars on 5m, 5,760 bars on 15m)
- Test window: 30 days
- Step: 30 days (non-overlapping test folds)
- Retrain: monthly (1st of month, as per `scripts/retrain_ml_monthly.py`)
- No look-ahead: feature engineering uses only past bars; labels use future bars but are computed on training set only

**Key hyperparameters for conservative production config:**
```python
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 10,         # Conservative: prevents overfitting
    "max_depth": 3,           # Shallow tree for noisy crypto data
    "min_child_samples": 200, # High: forces robust splits
    "feature_fraction": 0.5,  # Subsample features per tree
    "bagging_fraction": 0.7,  # Subsample rows per iteration
    "bagging_freq": 5,
    "reg_alpha": 0.5,          # L1 regularization
    "reg_lambda": 2.0,         # L2 regularization
    "scale_pos_weight": ...,   # Class imbalance correction
    "verbose": -1,
}
```

### Labeling Methods

Three methods have been evaluated in this codebase:

**1. Fixed Horizon Return (simple, used in v3):**
```
label[i] = +1 if close[i+6] / close[i] - 1 > 2*fee + 0.01%
label[i] = -1 if close[i+6] / close[i] - 1 < -(2*fee + 0.01%)
label[i] =  0 otherwise
```
Problem: Does not account for SL being hit before the horizon.

**2. Triple Barrier (recommended, ml_triple_barrier.py):**
```
For each bar i:
  long_TP  = close[i] + tp_mult * ATR[i]
  long_SL  = close[i] - sl_mult * ATR[i]
  short_TP = close[i] - tp_mult * ATR[i]
  short_SL = close[i] + sl_mult * ATR[i]

label[i] = +1 if long_TP hit before long_SL within max_bars
label[i] = -1 if short_TP hit before short_SL within max_bars
label[i] =  0 if neither (time barrier)
```
Best parameters from experiments: SL=1.5x ATR, TP=2.0x ATR, max_bars=18 (5m) / 96 (15m)

**3. Meta-Labeling (ml_meta_labeling.py):**
```
For each rule-based signal (from regime_adaptive):
label[i] = 1 if trade hit TP before SL
label[i] = 0 if trade hit SL first
```
ML then learns to filter which signals are winners. Simplest prediction problem.

### Realistic Performance Expectations

Based on project experiments (2026 data, 5m timeframe):

| Scenario | Pair | Trades (90d) | WR | PnL | Viable |
|----------|------|--------------|-----|-----|--------|
| Standalone, taker | ETH | 725 | 47.7% | ~breakeven | Marginal |
| Standalone, maker | ETH | 866 | 60.3% | +32.86% | Yes (single fold) |
| Full walk-forward | ETH | all folds | 54.3% | +2.09% net | Marginal |
| Meta-labeling (15m) | SOL/SPX | — | ~57-62% | +5pp WR gain | Promising |

**Honest assessment:** LightGBM on 5m OHLCV features has a very thin edge in this market. The edge is regime-dependent (works in trending regime, fails in mean-reverting). The 15m meta-labeling application is more robust because the base signal already has proven edge.

### Pros and Cons for Local Deployment

**Pros:**
- Fastest training in the tree-based family (much faster than XGBoost or RF on same data)
- Model size: 0.5-2 MB per model (text file, no binary dependencies)
- Inference latency: <1ms per prediction (critical for live trading)
- Built-in support for missing values (no imputation needed)
- `feature_importance()` with gain and split modes for interpretability
- Handles class imbalance via `scale_pos_weight`
- Stable across Python versions; no CUDA required for CPU inference

**Cons:**
- Hyperparameter-sensitive (num_leaves, min_child_samples need tuning per pair)
- Model degrades in 30 days (requires monthly retraining)
- Cannot extrapolate beyond training feature range (a new volatility regime breaks the model)
- No uncertainty quantification (probability output is miscalibrated; not a true Bayesian probability)

### Known Implementations

- `freqtrade/freqtrade` — FreqAI module, `LightGBMClassifier` prediction model
- `je-suis-tm/quant-finance` — LightGBM for crypto prediction
- Bojer & Meldgaard (2021): "Kaggle forecasting competitions: An overlooked learning opportunity" — highlights LightGBM dominance on tabular time-series

**Next step:** Write `research/analyze_lgbm_meta_labeling.py` — test meta-labeling at 15m scale across all 4 active pairs (ETH, SOL, SPX, DOGE) using the regime_adaptive signal base.

---

## 2. XGBoost (Extreme Gradient Boosting)

**Source:** https://github.com/dmlc/xgboost | `pip install xgboost`
**Applicability:** HIGH — strong alternative to LightGBM; FreqAI default
**Type:** Gradient boosting tree ensemble (level-wise)

### How It Works for Trading

XGBoost builds trees level-by-level (all nodes at depth d before depth d+1), versus LightGBM's leaf-wise approach. This makes XGBoost more conservative on small datasets but slightly slower on large ones. It introduced the key innovations: regularization (L1+L2 on leaf weights), second-order Taylor expansion for the objective, and weighted quantile sketch for approximate splits.

For trading, XGBoost is used identically to LightGBM: binary classifier per direction, or multiclass, or regression.

### Input Features

Identical to LightGBM; XGBoost handles the same pandas DataFrame inputs.

### Python Libraries

```python
import xgboost as xgb

# Classifier API (sklearn-compatible)
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.5,
    reg_alpha=0.5,
    reg_lambda=2.0,
    scale_pos_weight=imbalance_ratio,
    eval_metric="auc",
    tree_method="hist",   # Fast histogram method (like LightGBM)
    use_label_encoder=False,
    verbosity=0,
)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

# Native API
dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(params, dtrain, num_boost_round=300)
```

### Training Approach

Same walk-forward protocol as LightGBM. Key difference: `tree_method="hist"` must be set to match LightGBM speed. Without it, XGBoost is 3-5x slower.

**XGBoost-specific hyperparameters to tune:**
- `max_depth`: 3-5 (same as LightGBM max_depth)
- `min_child_weight`: equivalent to min_child_samples in LightGBM (use 50-200)
- `gamma`: minimum loss reduction to make a split (default 0; try 0.1-1.0 for crypto noise)
- `tree_method="hist"`: required for speed parity with LightGBM

### Labeling Methods

Identical to LightGBM. Triple barrier is recommended.

### Realistic Performance Expectations

On the same OHLCV feature set, XGBoost and LightGBM produce nearly identical results (within 0.5% on WR). LightGBM tends to slightly win on:
- Larger datasets (5m, 50k+ bars)
- Categorical features

XGBoost may slightly win on:
- Smaller datasets (15m, meta-labeling with 200-500 samples)
- When early stopping on a validation set is important

In academic benchmarks (Grinsztajn et al. 2022, "Why tree-based models still outperform deep learning on tabular data"), both LightGBM and XGBoost consistently outperform neural networks on tabular financial data with <100k samples.

### Pros and Cons for Local Deployment

**Pros:**
- sklearn-compatible API (XGBClassifier) — easy to use in StackingClassifier
- GPU training available (`tree_method="gpu_hist"`) for large hyperopt runs
- `save_model()` / `load_model()` produces portable JSON or binary files
- Excellent early stopping support with `eval_set` parameter
- FreqAI native support (`XGBoostClassifier`, `XGBoostRFClassifier`, `XGBoostRFRegressor`)

**Cons:**
- Slightly slower than LightGBM without `tree_method="hist"`
- Default depth-wise growth can overfit on noisy crypto 5m data more than LightGBM
- Model files are larger than LightGBM text format
- Less efficient memory use than LightGBM on sparse data

### Known Implementations

- `freqtrade/freqtrade` — FreqAI `XGBoostClassifier` and `XGBoostRFClassifier`
- Numerai competition: XGBoost is top-3 model type across thousands of submissions
- Gu, Kelly, Xiu (2020): "Empirical Asset Pricing via Machine Learning" — XGBoost and RF outperform all other models on equity factor prediction (NBER Working Paper 25398)

**Next step:** Write `research/analyze_xgboost_comparison.py` — compare XGBoost vs LightGBM head-to-head on 15m triple-barrier task to determine if XGBoost is worth the complexity.

---

## 3. CatBoost (Categorical Boosting)

**Source:** https://github.com/catboost/catboost | `pip install catboost`
**Applicability:** MEDIUM — best for datasets with categorical features; minor advantage for crypto
**Type:** Gradient boosting with ordered boosting (avoids target leakage)

### How It Works for Trading

CatBoost's key innovation is "ordered boosting": instead of computing gradient statistics on the same data used to train each tree, it uses a permuted ordering of observations. This prevents the "target leakage" problem where statistics computed on training targets indirectly leak into the model, causing overfitting. For financial time series (which already has temporal ordering), CatBoost's ordered mode maps naturally to the correct training regime.

CatBoost also has native support for categorical features (uses target statistics encoding internally), but for OHLCV+indicators features, all inputs are continuous floats, so this advantage is marginal.

### Input Features

Same as LightGBM/XGBoost. If you add categorical features like:
- `session = {"asia", "europe", "us"}` (3-category)
- `day_of_week = {"Mon", ..., "Sun"}`
- `regime = {"trending", "ranging"}`

Then CatBoost handles these natively without one-hot encoding.

### Python Libraries

```python
from catboost import CatBoostClassifier, Pool

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.03,
    depth=4,                   # Max tree depth (1-16)
    l2_leaf_reg=3.0,           # L2 regularization
    min_data_in_leaf=50,       # Min samples per leaf
    subsample=0.7,
    colsample_bylevel=0.5,
    cat_features=[],           # List of categorical feature indices
    eval_metric="AUC",
    verbose=0,
    task_type="CPU",           # or "GPU"
)
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
)
proba = model.predict_proba(X_test)[:, 1]
```

### Training Approach

Walk-forward, same protocol. CatBoost is significantly slower than LightGBM on CPU (2-5x), so reduce `iterations` or use the `task_type="GPU"` option if a GPU is available.

**Ordered vs Plain mode:**
- `boosting_type="Ordered"` (default): slower, less overfit, correct for time series
- `boosting_type="Plain"`: faster (matches LightGBM), slightly more overfit

For production, use `boosting_type="Ordered"` with `subsample=0.7` and smaller `iterations` (150-200).

### Labeling Methods

Same as LightGBM. Triple barrier recommended.

### Realistic Performance Expectations

CatBoost typically matches LightGBM in accuracy on pure float features. Measurable advantage appears when:
1. You add 3+ categorical features (session, regime, day_of_week)
2. Dataset is small (200-500 samples, as in meta-labeling)
3. You need better uncertainty calibration (CatBoost's ordered mode reduces overconfidence)

In the meta-labeling context (100-300 signals per pair per 90-day window), CatBoost's small-sample ordered boosting may outperform LightGBM's leaf-wise approach, which tends to memorize small training sets.

### Pros and Cons for Local Deployment

**Pros:**
- Best out-of-box performance on small datasets (meta-labeling use case)
- Ordered boosting reduces overfit without heavy regularization tuning
- Native categorical feature support (no preprocessing needed for regime labels)
- FreqAI native support (`CatboostClassifier`, `CatboostRegressor`)
- Consistent results across different random seeds (deterministic with default settings)

**Cons:**
- Slowest training in the gradient boosting family (2-5x slower than LightGBM on CPU)
- Model serialization (`.cbm` format) is not human-readable like LightGBM `.txt`
- Requires `catboost` package which has heavier dependencies than lightgbm
- Less active open-source community for financial applications

### Known Implementations

- `freqtrade/freqtrade` — FreqAI `CatboostClassifier`, `CatboostRegressor`
- Prokhorenkova et al. (2018): "CatBoost: unbiased boosting with categorical features" — NeurIPS 2018 (the original paper)
- Used extensively in Kaggle financial time series competitions (Jane Street, Optiver)

**Next step:** Test CatBoost specifically for the meta-labeling task at 15m (small sample, ordered mode likely beneficial). Write `research/analyze_catboost_meta.py`.

---

## 4. Random Forest Classifier/Regressor

**Source:** sklearn | `pip install scikit-learn` (already installed)
**Applicability:** MEDIUM — strong baseline, useful in ensembles
**Type:** Bagging ensemble of decision trees (parallel, not sequential)

### How It Works for Trading

Random Forest trains N independent decision trees on different random subsets of the training data (bootstrap sampling) and different random subsets of features at each split. Prediction is by majority vote (classifier) or average (regressor). Unlike gradient boosting, each tree is independent (parallelizable), making RF much faster to train in parallel.

For trading, RF is most useful as:
1. A baseline model (compare gradient boosting improvement over RF)
2. A component in a stacking ensemble
3. The model of choice when training speed matters more than accuracy (real-time retraining)
4. Out-of-bag error estimation (built-in validation without a separate val set)

### Python Libraries

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,       # Number of trees
    max_depth=5,            # Shallow for noisy crypto
    min_samples_leaf=50,    # High: prevents overfitting
    max_features="sqrt",    # Features per split (default)
    class_weight="balanced",
    n_jobs=-1,              # All CPU cores
    random_state=42,
    oob_score=True,         # Free validation
)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
print(f"OOB score: {model.oob_score_:.4f}")  # Free accuracy estimate
```

### Training Approach

Walk-forward, same protocol. Key advantage: `n_jobs=-1` parallelizes across all CPU cores. On a 4-core machine, RF trains 4x faster in wall time than LightGBM. This matters for hyperopt runs with many parameter combinations.

**RF-specific hyperparameter guidance for crypto:**
- `n_estimators`: 100-500 (diminishing returns above 200)
- `max_depth`: 5-8 (deeper than gradient boosting since individual trees overfit less in the ensemble)
- `min_samples_leaf`: 20-100 (primary overfitting control; key parameter to tune)
- `max_features`: "sqrt" for classification, "log2" for high-dimensional feature sets

### Labeling Methods

All three methods work. RF is the recommended model when testing a new labeling method, since the hyperparameters are less sensitive than gradient boosting (no learning rate to tune).

### Realistic Performance Expectations

RF typically achieves 90-95% of LightGBM accuracy on the same crypto features. The gap widens on larger datasets (LightGBM advantage increases) and narrows on small datasets (<5000 samples). In the meta-labeling use case (200-500 signal samples), RF and LightGBM are often statistically indistinguishable.

Expected performance delta vs LightGBM:
- WR: typically 1-3pp lower
- Training time: 2-3x slower on large sets, comparable on small sets with `n_jobs=-1`
- Inference time: comparable (but sklearn RF has higher Python overhead per prediction)

### Pros and Cons for Local Deployment

**Pros:**
- Zero learning rate to tune (major advantage in hyperopt)
- Built-in OOB error (free cross-validation)
- Highly parallelizable (`n_jobs=-1`)
- sklearn-compatible: works seamlessly in `StackingClassifier`
- Feature importances via `feature_importances_` (MDI — mean decrease in impurity)
- Less sensitive to outliers than gradient boosting

**Cons:**
- Lower accuracy than gradient boosting on large crypto datasets (5m, 50k+ bars)
- Model size: large (200 trees * depth-5 structure = 10-50 MB in memory)
- Serialization via `joblib.dump()` creates large files (10-100 MB)
- Inference latency: 2-5ms (vs <1ms for LightGBM) — acceptable for 5m but monitor
- Cannot extrapolate beyond training feature range (same limitation as all trees)

### Known Implementations

- `freqtrade/freqtrade` — FreqAI `RandomForestClassifier` support
- Khaidem, Saha, Dey (2016): "Predicting the direction of stock market prices using Random Forest" — foundational paper showing 70%+ accuracy on daily stock data (overfitted, but influential)
- Gu, Kelly, Xiu (2020): RF ranks 2nd after neural networks on large equity factor dataset (NBER 25398)

**Next step:** Include RF as component B in a stacking ensemble: `research/analyze_stacking_ensemble.py`.

---

## 5. Extra Trees (Extremely Randomized Trees)

**Source:** sklearn | `pip install scikit-learn`
**Applicability:** MEDIUM — better than RF for very noisy data; underexplored in crypto
**Type:** Bagging ensemble with randomized splits (even more random than RF)

### How It Works for Trading

ExtraTrees is like Random Forest but with an additional randomization: instead of finding the optimal split at each node (as RF does), it selects the split threshold randomly from a uniform distribution over the feature range. This increases bias slightly but dramatically reduces variance. For noisy data like 5m crypto, ExtraTrees often outperforms both RF and gradient boosting in the walk-forward context where variance is the primary problem.

The intuition: in a 5m crypto regime, the difference between RSI=55 and RSI=57 is not meaningful — the exact threshold doesn't matter. ExtraTrees' random thresholds force the model to learn robust, coarser rules rather than memorizing exact threshold values.

### Python Libraries

```python
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=6,           # Can go deeper than RF
    min_samples_leaf=30,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    bootstrap=False,       # ExtraTrees: no bootstrap by default
)
```

### Training Approach

Same walk-forward protocol. ExtraTrees trains faster than RF because no optimal split search is needed. However, it often needs more trees to compensate for the random thresholds (use n_estimators=300-500).

### Realistic Performance Expectations

In recent financial ML benchmarks, ExtraTrees often matches or beats RF on 5m timeframes precisely because of regime instability. If the "best" RSI threshold for a signal shifts from 55 to 60 between the training and test period, RF's learned threshold of 55 will fail while ExtraTrees' random thresholds spread across the range will be more robust.

Expect: comparable to LightGBM at tight confidence thresholds (0.70+), slightly worse at standard thresholds (0.55-0.65).

### Pros and Cons for Local Deployment

**Pros:**
- Fastest tree ensemble to train (no threshold optimization)
- More robust to regime change than RF (less overfitting to historical thresholds)
- Low memory inference (sklearn array operations, no Python loop per tree)
- `n_jobs=-1` parallelism
- sklearn-compatible for stacking

**Cons:**
- Higher bias than RF (random thresholds are sometimes very bad)
- Requires more trees for stable predictions (larger model files)
- No OOB score with default `bootstrap=False`
- Less interpretable than RF (importances are noisier due to random splits)

### Known Implementations

No dominant financial ML paper focuses specifically on ExtraTrees, but it appears in most comprehensive ML-for-trading papers as part of comparison suites. Notable:
- Nti, Adekoya, Weyori (2020): "A comprehensive evaluation of ensemble learning for stock-market prediction" — ExtraTrees top-3 in 5/8 datasets

**Next step:** Include in stacking ensemble comparison: `research/analyze_stacking_ensemble.py`.

---

## 6. Gradient Boosting Classifier (sklearn)

**Source:** sklearn | `pip install scikit-learn`
**Applicability:** LOW for standalone use; useful as ensemble component
**Type:** Original gradient boosting implementation (Friedman 2001)

### How It Works for Trading

This is the original gradient boosting algorithm from Friedman (2001), implemented in sklearn. It builds trees sequentially, correcting residuals. It is the conceptual ancestor of XGBoost and LightGBM, but much slower (CPU-only, no histogram approximation, no parallelism in tree construction).

The main reason to use sklearn GBC over LightGBM/XGBoost in 2026:
1. Legacy code compatibility
2. As a teaching/reference model
3. For very small datasets (< 500 samples) where the overhead of LightGBM initialization matters

### Python Libraries

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.7,
    min_samples_leaf=50,
    validation_fraction=0.2,
    n_iter_no_change=20,  # Early stopping
    random_state=42,
)
```

### Realistic Performance Expectations

Equivalent in theory to LightGBM, but in practice 5-20x slower with no meaningful accuracy advantage. Not recommended for crypto trading where you need to retrain monthly on 10,000+ bars.

### Pros and Cons for Local Deployment

**Pros:**
- Pure sklearn: zero additional dependencies
- `warm_start` for incremental training
- Well-documented, no surprises

**Cons:**
- 5-20x slower than LightGBM on the same task
- No GPU support
- Practical upper limit ~50k samples before training becomes painfully slow
- Superseded by XGBoost, LightGBM, CatBoost in every benchmark

**Not recommended as the primary model.** Use only as a baseline in benchmarking studies.

---

## 7. AdaBoost (Adaptive Boosting)

**Source:** sklearn | `pip install scikit-learn`
**Applicability:** LOW — sensitive to outliers, underperforms gradient boosting on crypto
**Type:** Boosting ensemble with adaptive sample weighting

### How It Works for Trading

AdaBoost trains a sequence of weak classifiers (typically shallow decision trees, depth=1 "stumps"), giving higher weight to misclassified samples in each round. This makes it focus on the "hard" examples. For trading, "hard" examples are often noise or regime-change bars where no model can reliably predict — so AdaBoost tends to overfit these precisely.

The core problem for crypto: sharp volatility spikes, liquidation cascades, and news events create many bars that are truly unpredictable. AdaBoost will increasingly focus on these, degrading overall performance.

### Python Libraries

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=100,
    learning_rate=0.1,
    algorithm="SAMME.R",  # Real-valued probabilities (recommended)
    random_state=42,
)
```

### Realistic Performance Expectations

In financial ML benchmarks, AdaBoost consistently underperforms gradient boosting methods. Particularly problematic:
- Crypto high-kurtosis returns (fat tails) cause AdaBoost to massively upweight extreme events
- Regime changes create systematic misclassification that AdaBoost chases

Expected to be 5-15pp WR lower than LightGBM on the same features.

**Not recommended for production crypto ML.** Include only in comprehensive benchmarks for completeness.

---

## 8. Support Vector Machine (SVM/SVR)

**Source:** sklearn | `pip install scikit-learn`
**Applicability:** LOW for live trading; MEDIUM for research/feature selection
**Type:** Maximum margin classifier (kernel-based)

### How It Works for Trading

SVM finds the hyperplane that maximizes the margin between classes. With the RBF (radial basis function) kernel, it can model non-linear decision boundaries. For trading:
- SVC (classifier): predicts direction {-1, 0, +1}
- SVR (regressor): predicts forward return

SVM has strong theoretical foundations (VC theory, structural risk minimization), but practical scaling issues make it unsuitable for the volumes of data required in crypto ML.

### Python Libraries

```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# SVM REQUIRES feature scaling (unlike tree models)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,           # Regularization (inverse)
        gamma="scale",   # RBF bandwidth
        class_weight="balanced",
        probability=True,  # Required for predict_proba (slows training!)
        cache_size=1000,   # MB of kernel cache
    )),
])
```

### Training Approach

SVM training complexity is O(n^2) to O(n^3) with n = number of training samples. For 17,280 bars (60 days at 5m), training time is approximately:
- n=1000: ~1 second
- n=5000: ~1 minute
- n=17,280: ~20-60 minutes

This makes SVM impractical for monthly retraining on 5m data with a 60-day window. For 15m meta-labeling with 100-300 signal samples, SVM becomes viable.

**Feature scaling is mandatory:** SVM uses Euclidean distance in kernel space; un-scaled features (ATR in dollar values vs normalized returns) will dominate the kernel.

### Realistic Performance Expectations

On small samples (< 1000), SVM with RBF kernel can match gradient boosting. On the meta-labeling problem (100-300 signals), SVM is a legitimate choice and may provide better probability calibration than LightGBM.

For standalone 5m trading (17,000+ bars), training time makes SVM impractical.

### Pros and Cons for Local Deployment

**Pros:**
- Strong theoretical generalization guarantees
- Works well on small datasets (meta-labeling: 100-300 samples)
- RBF kernel captures non-linear patterns without feature engineering
- Better probability calibration than tree models when `probability=True`

**Cons:**
- O(n^2)-O(n^3) training time (impractical for 5m data)
- Requires feature scaling (additional preprocessing step)
- `predict_proba=True` requires Platt scaling (cross-validation internally), tripling training time
- No feature importance (black box at feature level)
- Memory intensive: stores support vectors in memory

**Recommended use case:** Meta-labeling classifier for regime_adaptive signals (200-400 signal samples per pair). Compare SVM vs LightGBM on this specific task in `research/analyze_svm_meta.py`.

### Known Implementations

- Kernel & Ben-Hur (2004): "A Note on Support Vector Machine Degeneracy" — foundational
- Kim (2003): "Financial time series forecasting using support vector machines" — one of first SVM finance papers
- Cristianini & Shawe-Taylor (2000): "An Introduction to Support Vector Machines" — textbook

---

## 9. K-Nearest Neighbors (KNN)

**Source:** sklearn | `pip install scikit-learn`
**Applicability:** LOW for production; MEDIUM for market regime fingerprinting
**Type:** Instance-based (lazy) learning, non-parametric

### How It Works for Trading

KNN stores all training examples and for each new bar, finds the K most similar historical bars (by Euclidean distance in feature space), then predicts based on the majority label of those K neighbors. There is no training phase — inference is the expensive step.

For trading, KNN represents "find historical analogs": given today's market state (RSI, ATR, EMA position, etc.), what happened in the 5 most similar past states? This is conceptually appealing for crypto where regime fingerprinting is valuable.

### Python Libraries

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Scaling mandatory for KNN (Euclidean distance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(
    n_neighbors=10,        # K value
    weights="distance",    # Closer neighbors matter more
    algorithm="ball_tree", # Faster for high-dimensional data
    leaf_size=30,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)
proba = model.predict_proba(X_test_scaled)[:, 1]
```

### Realistic Performance Expectations

KNN suffers from the curse of dimensionality: with 40 features, meaningful distance measures break down. Performance is typically lower than tree-based models on high-dimensional feature sets. However, with 8-10 carefully selected features (dimensionality reduction via PCA or manual feature selection), KNN can be competitive on small datasets.

Practical concern: inference time is O(n * d) where n=training samples and d=features. For 17,280 training bars × 40 features per prediction, inference can take 50-100ms, which may cause issues in live trading callbacks.

### Pros and Cons for Local Deployment

**Pros:**
- No training time (lazy learning)
- Naturally adapts to new data (just add to the training set)
- Interpretable: "I'm trading because these K historical bars were similar and all went up"
- No hyperparameters except K and distance metric

**Cons:**
- Slow inference (O(n*d)) — problematic for live trading
- Curse of dimensionality (40+ features degrade distance metrics)
- Memory: stores entire training set in memory (~40 features × 17,280 bars = 5-10 MB)
- Cannot handle feature interactions the way trees can

**Recommended use case:** Research tool for "market analog" analysis. Not recommended for production.

---

## 10. Logistic Regression with Feature Engineering

**Source:** sklearn | `pip install scikit-learn`
**Applicability:** MEDIUM — excellent meta-learner in stacking; fast and interpretable
**Type:** Generalized linear model, probabilistic classifier

### How It Works for Trading

Logistic regression fits a linear decision boundary in feature space, using the logit link function to output probabilities in [0,1]. Despite its simplicity, it often performs surprisingly well when:
1. The features are already predictive (pre-engineered indicators)
2. Used as a meta-learner in stacking (combining outputs of stronger models)
3. The training set is small (< 500 samples), where complex models overfit

For meta-labeling (filter regime_adaptive signals), logistic regression on the ~25 signal-context features is a strong baseline because the feature set is already the output of a rule-based trading system (ADX, RSI, EMA spread, etc.).

### Python Libraries

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Logistic regression also needs scaling
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        C=0.1,              # Regularization strength (inverse); strong regularization for noisy data
        penalty="l2",       # L2 regularization (elasticnet: C=0.1, l1_ratio=0.5)
        class_weight="balanced",
        solver="lbfgs",     # Fast for small datasets
        max_iter=500,
    )),
])

# For meta-learner in stacking:
from sklearn.linear_model import LogisticRegressionCV
meta_model = LogisticRegressionCV(
    Cs=[0.001, 0.01, 0.1, 1.0],  # Cross-validate over C values
    cv=3,
    class_weight="balanced",
)
```

### Training Approach

For standalone use: walk-forward with 60d/30d windows.
For stacking meta-learner: use the out-of-fold predictions from base models as features.

### Realistic Performance Expectations

Standalone logistic regression on 40 OHLCV features: WR typically 50-53% (just above random). With interaction features (RSI * volume_ratio, ATR_ratio * EMA_spread, etc.), can reach 54-56% WR.

As a stacking meta-learner: typically adds +1-3pp WR over the best individual base model by combining diverse model outputs.

### Pros and Cons for Local Deployment

**Pros:**
- Extremely fast training (<0.1 seconds for 10,000 samples)
- Best probability calibration of all methods (logit function is designed for probabilities)
- Coefficient weights directly interpretable as feature importances
- Near-zero inference latency
- sklearn-native, no additional dependencies

**Cons:**
- Cannot capture feature interactions without manual polynomial features
- Assumes linear separability (after feature scaling)
- Requires careful regularization tuning for high-dimensional features
- Will underperform gradient boosting on complex non-linear patterns

**Primary recommended use:** Stacking meta-learner (see Section 13).

---

## 11. Naive Bayes

**Source:** sklearn | `pip install scikit-learn`
**Applicability:** LOW for production; MEDIUM as a fast baseline
**Type:** Probabilistic classifier assuming feature independence

### How It Works for Trading

Naive Bayes applies Bayes' theorem with the "naive" assumption that all features are conditionally independent given the class label. This is obviously violated in financial data (RSI and price returns are highly correlated), but the model often works as a fast baseline because:
1. It generalizes well with very few training samples
2. Its probability outputs are well-calibrated for common classes
3. Training is O(n*d) (linear in samples and features) — fastest classifier by far

**Gaussian Naive Bayes** (assumes features are normally distributed — approximately true for normalized returns):
```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB(var_smoothing=1e-9)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
```

**Bernoulli Naive Bayes** (for binary features like regime_trending, above_ema200):
```python
from sklearn.naive_bayes import BernoulliNB
```

### Realistic Performance Expectations

GaussianNB on 40-feature OHLCV set: WR typically 50-53% — same as logistic regression but less accurate on complex patterns. Main use: provide a baseline within 30 seconds of training to establish whether gradient boosting is actually adding value.

### Pros and Cons for Local Deployment

**Pros:**
- Fastest classifier (O(n*d) training)
- Works with as few as 50 training samples
- Good probability calibration
- Online learning support (`partial_fit`) — the only major model with true incremental learning

**Cons:**
- Feature independence assumption is heavily violated in OHLCV data
- Cannot model interactions (RSI + volume spike is more informative than each alone)
- Accuracy ceiling 3-5pp below gradient boosting

**Recommended use:** Quick baseline for new labeling methods. Also as the only model supporting true online/incremental learning via `partial_fit()` if real-time model updates are needed.

---

## 12. Single Decision Tree

**Source:** sklearn | `pip install scikit-learn`
**Applicability:** LOW for production; HIGH for interpretability and rule extraction
**Type:** Recursive binary partitioning

### How It Works for Trading

A single decision tree recursively splits the feature space based on the feature and threshold that maximally reduces impurity (Gini or entropy). The result is a tree of if-then rules that can be directly inspected and understood.

For trading, the primary value of a single decision tree is **rule extraction**: fit a depth-3 tree on the training data and read off the top 3 splitting rules. These often surface the most important threshold combinations that the gradient boosting model is using internally.

### Python Libraries

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

model = DecisionTreeClassifier(
    max_depth=4,            # Keep shallow for interpretability
    min_samples_leaf=100,   # High for robust rules
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train, y_train)

# Print readable rules
rules = export_text(model, feature_names=feature_names)
print(rules)
```

### Realistic Performance Expectations

Single tree WR: 51-55% — worse than ensembles, highly variable across folds. The instability (high variance) is the primary reason ensembles were invented.

**Primary value is not trading performance, but rule extraction.** Example output:
```
|--- adx <= 28.5
|   |--- rsi_9 <= 32.4
|   |   |--- vol_ratio <= 2.1 → 65% long wins
|   |--- rsi_9 > 32.4
|   |   → 48% long wins (avoid)
|--- adx > 28.5
|   |--- di_diff <= 8.3 → 55% long wins
```
This provides actionable insight for rule-based strategy improvement.

---

## 13. Stacking / Blending Ensembles

**Source:** sklearn | `pip install scikit-learn`
**Applicability:** HIGH — consistently best performance; 1-3pp improvement over single models
**Type:** Meta-learning: train a meta-model on base model predictions

### How It Works for Trading

Stacking trains multiple diverse base models on the same training data, then trains a meta-model on the out-of-fold (OOF) predictions from the base models. The meta-model learns how to weight and combine the base models' predictions.

For crypto trading:

**Base models (Layer 1):**
- LightGBM (long model) — captures complex interactions
- ExtraTrees — captures broad regime patterns
- Logistic Regression — captures linear signal strength

**Meta-model (Layer 2):**
- Logistic Regression or LightGBM (shallow, depth=2)
- Input: [prob_lgbm_long, prob_extratrees_long, prob_lr_long] → final_prob_long

### Python Libraries

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb

# Using sklearn-compatible LightGBM
from lightgbm import LGBMClassifier

base_models = [
    ("lgbm_long", LGBMClassifier(
        n_estimators=200, max_depth=3, num_leaves=10,
        min_child_samples=100, learning_rate=0.03,
        colsample_bytree=0.5, reg_alpha=0.5, verbose=-1,
        class_weight="balanced",
    )),
    ("extratrees", ExtraTreesClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50,
        max_features="sqrt", class_weight="balanced", n_jobs=-1,
    )),
    ("logreg", LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=500,
    )),
]

meta_model = LogisticRegression(C=0.5, class_weight="balanced")

stacker = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,               # 5-fold OOF for meta-training
    stack_method="predict_proba",
    n_jobs=-1,
    passthrough=False,  # Don't pass original features to meta-layer
)

stacker.fit(X_train, y_train)
proba = stacker.predict_proba(X_test)[:, 1]
```

### Training Approach

For walk-forward with stacking, the `cv` parameter of `StackingClassifier` must use `TimeSeriesSplit` instead of the default k-fold to avoid lookahead in the meta-training step:

```python
from sklearn.model_selection import TimeSeriesSplit

stacker = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=TimeSeriesSplit(n_splits=3),  # Time-aware cross-validation
    stack_method="predict_proba",
)
```

**Training time:** Stacking multiplies the training time of each base model by the `cv` folds, plus meta-model training. For 3 base models × 5 folds × 17,280 training bars = significant overhead. Reduce to `cv=3` or `cv=TimeSeriesSplit(n_splits=3)` for monthly retraining.

### Labeling Methods

Triple barrier is recommended. The diversity across base models is maximized when labels directly reflect execution outcomes (TP/SL) rather than fixed-horizon returns.

### Realistic Performance Expectations

Literature consistently shows stacking adds +1-3pp WR over the single best base model, with similar consistency improvement (lower variance across folds). For meta-labeling with 200-400 signal samples:
- LightGBM alone: 57-62% WR (from project experiments, 2 folds)
- Stacking (LightGBM + ExtraTrees + LR): expected 59-64% WR
- Gain: +1-2pp but at 3x training time

The stacking benefit is most pronounced when the base models are genuinely diverse (different inductive biases). LightGBM + ExtraTrees + LR are sufficiently diverse.

### Pros and Cons for Local Deployment

**Pros:**
- Consistently best accuracy of all classical ML approaches
- Robust to individual model failures (if LightGBM overfits one fold, ExtraTrees compensates)
- Meta-model learns to trust different base models in different market regimes
- Feature importances from individual base models still accessible

**Cons:**
- Highest training time (multiplicative with cv folds)
- Serialization complexity: must save and load all base models + meta-model + scalers
- Risk: if meta-model is itself a complex model (e.g., LightGBM depth=4), overfits the meta-layer
- Interpretation is harder: "why did the meta-model take this trade?" spans multiple base models

**Next step:** Write `research/analyze_stacking_ensemble.py` — implement LightGBM + ExtraTrees + LR stacker for the 15m meta-labeling task.

---

## PART B: FreqAI Module

FreqAI is freqtrade's built-in ML framework, available since freqtrade 2022.9.

### Supported Models (built-in prediction_models)

| Model Class | Algorithm | Library | Type |
|-------------|-----------|---------|------|
| `LightGBMClassifier` | LightGBM | lightgbm | Binary/Multiclass |
| `LightGBMRegressor` | LightGBM | lightgbm | Regression |
| `XGBoostClassifier` | XGBoost | xgboost | Binary/Multiclass |
| `XGBoostRegressor` | XGBoost | xgboost | Regression |
| `XGBoostRFClassifier` | XGBoost (RF mode) | xgboost | Binary/Multiclass |
| `XGBoostRFRegressor` | XGBoost (RF mode) | xgboost | Regression |
| `CatboostClassifier` | CatBoost | catboost | Binary/Multiclass |
| `CatboostRegressor` | CatBoost | catboost | Regression |
| `RandomForestClassifier` | Random Forest | sklearn | Binary/Multiclass |
| `PyTorchMLPClassifier` | MLP | torch | Binary/Multiclass |
| `PyTorchMLPRegressor` | MLP | torch | Regression |
| `PyTorchTransformerClassifier` | Transformer | torch | Binary/Multiclass |
| `ReinforcementLearner` | PPO/A2C/DQN | stable_baselines3 | RL (not covered in Part 1) |

### FreqAI Architecture for This Project

FreqAI's training pipeline maps directly to this project's walk-forward protocol:

```
freqai_config:
  enabled: true
  identifier: "ml_ensemble_v1"
  feature_parameters:
    include_timeframes: ["5m", "15m", "1h"]
    include_corr_pairlist: ["BTC/USDT:USDT"]
    label_period_candles: 6           # forward_bars = 6
    include_shifted_candles: 3        # lag features
    indicator_periods_candles: [10, 20, 50]
    DI_threshold: 0                   # Dissimilarity Index (outlier filter)
  data_split_parameters:
    test_size: 0.25
    random_state: 42
    shuffle: false                    # time series: no shuffle
  model_training_parameters:
    n_estimators: 200
    max_depth: 3
    learning_rate: 0.03
    num_leaves: 10
    min_child_samples: 200
```

### Why This Project Uses Custom ML Instead of FreqAI

The custom approach in `scripts/ml_*.py` and `strategies/tf_5m/ml_ensemble.py` was chosen over FreqAI because:

1. **FreqAI retrains on every new candle** (or at a configured interval) by default — this is computationally expensive for a Windows machine running 4 pairs simultaneously
2. **FreqAI's outlier detection** (Dissimilarity Index, IForest, DBSCAN) has hyperparameters that interact with the crypto feature space in poorly documented ways
3. **Custom training scripts** allow precise control over the walk-forward windows, per-pair threshold optimization, and explicit dual long/short model architecture
4. **FreqAI's live retraining** can create training/serving skew in backtesting mode

FreqAI is **recommended** if you want a maintained, well-tested ML pipeline without custom code. The custom approach is recommended if you need fine-grained control over the labeling method (especially triple-barrier).

---

## PART C: Known GitHub Repositories

| Repository | Focus | Stars | Notes |
|------------|-------|-------|-------|
| `freqtrade/freqtrade` | Full trading bot + FreqAI | 35k+ | Native ML integration |
| `huseinzol05/Stock-Prediction-Models` | 60+ ML models for price prediction | 8k+ | Educational reference, daily stock data |
| `je-suis-tm/quant-finance` | Crypto ML with walk-forward | 2k+ | Good LightGBM examples for BTC |
| `nicholishen/freqtrade-FreqAI-examples` | FreqAI community strategies | 500+ | Production-ready FreqAI configs |
| `Zernach/freqtrade-strategies` | Community strategies including ML | 1k+ | Some with LightGBM integration |
| `georgezouq/awesome-deep-reinforcement-learning-in-finance` | RL collection | 3k+ | Comprehensive survey |

---

## PART D: Key Academic Papers

### Labeling Methods

1. **López de Prado (2018):** "Advances in Financial Machine Learning" — John Wiley & Sons. The definitive textbook. Introduces: triple barrier method, meta-labeling, feature importance for financial features, walk-forward validation. Every algorithm discussed in this document traces back here.

2. **Bailey, Borwein, López de Prado, Zhu (2014):** "Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance." PNAS. Quantifies how multiple testing inflation destroys out-of-sample performance.

### Model Comparison (Tabular Data)

3. **Grinsztajn, Oyallon, Varoquaux (2022):** "Why tree-based models still outperform deep learning on tabular data." NeurIPS 2022. Conclusive evidence: LightGBM/XGBoost/RF beat MLP, Transformer, ResNet on tabular financial data across 45 benchmarks. Primary reason: trees are invariant to feature scaling; neural networks struggle with the irregular distributions of financial indicators.

4. **Gu, Kelly, Xiu (2020):** "Empirical Asset Pricing via Machine Learning." Review of Financial Studies. Tree-based models (especially gradient boosting) outperform linear factor models, neural networks, and simple trees on equity return prediction. Key result: R^2 for gradient boosting = 0.40% (daily returns), vs 0.08% for linear models.

### Crypto-Specific

5. **Nakagawa, Uchida, Aoshima (2018):** "Deep Factor Investing." Covers LightGBM for alpha generation in Japanese equities, methodology directly applicable to crypto.

6. **Dixon, Klabjan, Bang (2017):** "Classification-based Financial Markets Prediction Using Deep Neural Networks." IEEE Access. Benchmark comparing NN vs SVM vs RF on FX markets. RF wins on 5-minute data; NN wins on 1-minute data.

7. **Patel, Shah, Thakkar, Kotecha (2015):** "Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques." Expert Systems. Shows that feature engineering (converting indicators to categorical) dramatically improves ML accuracy — directly relevant to the regime label idea.

8. **Sebastião & Godinho (2021):** "Forecasting and trading cryptocurrencies with machine learning under changing market conditions." Financial Innovation. Walk-forward study on BTC/ETH daily data. XGBoost and LightGBM outperform RF, NN, and SVM. Key finding: model performance degrades significantly when trained on one regime (bull) and tested on another (bear) — exactly what this project's 5m experiments confirmed.

---

## PART E: Summary Recommendation Table

| Model | Best Use Case | Training Speed | WR vs LightGBM | Deploy Complexity |
|-------|--------------|----------------|----------------|-------------------|
| LightGBM | Standalone ML, meta-labeling, any size | Fast (1x) | Baseline | Low |
| XGBoost | Large hyperopt, GPU acceleration | Medium (1.5x) | ~equal | Low |
| CatBoost | Small samples, categorical features | Slow (3-5x) | +0.5pp (small N) | Medium |
| Random Forest | Ensemble component, baseline | Medium (1.5x) | -1 to -3pp | Low |
| Extra Trees | Noisy regime data, stacking | Fast (0.7x) | -1 to -2pp (standalone); +0pp (stacking) | Low |
| Stacking (LGB+ET+LR) | Best accuracy | Slow (5x+) | +1 to +3pp | High |
| Gradient Boosting (sklearn) | Legacy/reference | Very Slow (10x+) | ~equal | Low |
| AdaBoost | Benchmarking only | Medium | -5 to -15pp | Low |
| SVM | Meta-labeling (small N) | Very Slow (>5k samples) | comparable (small N) | Medium |
| KNN | Market analog research | None (lazy) | -3 to -7pp | Medium (inference latency) |
| Logistic Regression | Meta-learner in stacking | Instant | -3 to -8pp | Very Low |
| Naive Bayes | Baseline, incremental learning | Instant | -5 to -10pp | Very Low |
| Decision Tree | Rule extraction only | Fast | -5 to -15pp | Very Low |

---

## PART F: Recommended Next Steps (Priority Order)

1. **`research/analyze_lgbm_meta_labeling.py`** — extend existing meta-labeling script to all 4 pairs (ETH, SOL, SPX, DOGE) with the regime_adaptive base + triple-barrier labels. This is the highest-probability path to a Grade B ML strategy.

2. **`research/analyze_catboost_meta.py`** — test CatBoost with ordered boosting specifically for the small-sample meta-labeling task. CatBoost may outperform LightGBM at 100-300 samples.

3. **`research/analyze_stacking_ensemble.py`** — build LightGBM + ExtraTrees + Logistic Regression stacker for 15m meta-labeling. Target: +2pp WR improvement over single LightGBM.

4. **`research/analyze_xgboost_comparison.py`** — head-to-head XGBoost vs LightGBM on 15m triple-barrier to validate whether switching to FreqAI's `XGBoostClassifier` is worth it (it probably is not, but confirms).

5. **`research/analyze_freqai_integration.py`** — prototype FreqAI integration using `LightGBMClassifier` with the proven feature set; compare walk-forward results to custom implementation to validate equivalence before switching to the maintained FreqAI pipeline.

---

## Appendix: Cost Threshold Analysis for ML Models

For any ML classifier to be viable given this project's cost model:

| Fee Mode | Round-trip cost | Required WR @ 2:1 R:R | Required WR @ 1.5:1 R:R |
|----------|-----------------|----------------------|-----------------------|
| Maker both sides | 0.04% | 33.3% + threshold = ~46-47% | 40% + threshold = ~50-51% |
| Taker both sides | 0.10% | 33.3% + threshold = ~48-50% | 40% + threshold = ~52-53% |
| Mixed (maker entry, taker exit) | 0.07% | ~47% | ~51% |

**Critical insight from project experiments:** The 5m standalone ML achieved 54.3% WR across all folds combined, but the edge was concentrated in the first fold (Jan-Mar 2026 trending regime). The 15m meta-labeling approach has the advantage that the base signal (regime_adaptive, 52% WR) already clears the cost threshold — ML only needs to remove the worst signals, not find new ones from scratch.

This means the meta-labeling approach requires only ~+5pp WR gain from ML (52% → 57%) to meaningfully improve profit factor from ~1.3 to ~1.8, which is well within what these models can reliably deliver given the feature set quality.

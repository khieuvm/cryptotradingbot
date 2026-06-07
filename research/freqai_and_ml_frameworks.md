# FreqAI and Open-Source ML Trading Frameworks
# Comprehensive Research Report

**Research date:** 2026-06-07
**Researcher:** crypto_strategy_researcher agent
**Context:** OKX futures 15m/5m BTC/ETH/SOL/SPX, freqtrade-based system
**Purpose:** Evaluate FreqAI integration path and alternative ML frameworks

---

## Part 1: FreqAI — Freqtrade's Built-In ML Module

### Overview

FreqAI is freqtrade's native machine learning subsystem, introduced in stable releases from
v2022.6 onwards. It integrates tightly with the IStrategy interface, handling the entire
ML lifecycle (feature generation, training, backtesting with proper OOS splits, live
retraining) within the freqtrade runtime. Critically, it eliminates look-ahead bias in
backtesting by training only on data that would have been available at each point in time.

**Documentation:** https://www.freqtrade.io/en/stable/freqai/
**Source:** https://github.com/freqtrade/freqtrade/tree/develop/freqtrade/freqai

---

### 1.1 Supported Models (Full List)

#### Classification Models (predict direction/labels)
| Class Name | Backend | Notes |
|---|---|---|
| LightGBMClassifier | lightgbm | Most popular; fastest for tabular |
| LightGBMRegressorMultiTarget | lightgbm | Multi-output regression |
| XGBoostClassifier | xgboost | Slightly slower than LightGBM |
| XGBoostRFClassifier | xgboost | Random Forest variant |
| CatboostClassifier | catboost | Handles categoricals natively |
| SKLearnRandomForestClassifier | scikit-learn | sklearn RF wrapper |
| PyTorchMLPClassifier | pytorch | MLP neural network |
| PyTorchTransformerClassifier | pytorch | Transformer architecture |

#### Regression Models (predict continuous target)
| Class Name | Backend | Notes |
|---|---|---|
| LightGBMRegressor | lightgbm | Default regressor |
| XGBoostRegressor | xgboost | |
| XGBoostRFRegressor | xgboost | RF variant |
| CatboostRegressor | catboost | |
| SKLearnRandomForestRegressor | scikit-learn | |
| PyTorchMLPRegressor | pytorch | |
| PyTorchTransformerRegressor | pytorch | |

#### Reinforcement Learning Models
| Class Name | Backend | Algorithm |
|---|---|---|
| ReinforcementLearner | stable-baselines3 | PPO (default) |
| ReinforcementLearner_multiproc | stable-baselines3 | PPO with multiprocessing |
| TDQN | pytorch (custom) | Custom DQN variant |

All models inherit from `IFreqaiModel`. Custom models can be created by subclassing this.

**Import path format:**
```python
freqai_model = "LightGBMClassifier"
# or full path: "freqtrade.freqai.prediction_models.LightGBMClassifier"
```

---

### 1.2 Configuration

FreqAI is configured via the `"freqai"` block in the freqtrade JSON config. Key parameters:

```json
{
  "freqai": {
    "enabled": true,
    "purge_old_models": true,
    "train_period_days": 30,
    "backtest_period_days": 7,
    "live_retrain_hours": 0,
    "expiry_hours": 6,
    "identifier": "unique_run_id_v1",

    "feature_parameters": {
      "include_timeframes": ["5m", "15m", "1h", "4h"],
      "include_corr_pairlist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
      "label_period_candles": 24,
      "include_shifted_candles": 2,
      "DI_threshold": 0,
      "weight_factor": 0.9,
      "principal_component_analysis": false,
      "use_SVM_to_remove_outliers": true,
      "sma_period_candles": 1,
      "stratify_training_data": 0,
      "plot_feature_importances": 0
    },

    "data_split_parameters": {
      "test_size": 0.33,
      "random_state": 42,
      "shuffle": false
    },

    "model_training_parameters": {
      "n_estimators": 200,
      "learning_rate": 0.05,
      "num_leaves": 31,
      "verbose": -1
    },

    "rl_config": {
      "model_type": "PPO",
      "policy_type": "MlpPolicy",
      "max_trade_duration_candles": 300,
      "model_reward_parameters": {
        "rr": 1,
        "profit_aim": 0.025
      },
      "net_arch": [128, 128],
      "progress_bar": true
    }
  }
}
```

**Critical parameters for OKX futures setup:**
- `train_period_days`: 30-60 recommended (match our walk-forward window)
- `backtest_period_days`: 7-14 (how much OOS data is evaluated per training slice)
- `live_retrain_hours`: 0 = retrain every candle (too slow); 24 = daily retrain; 168 = weekly
- `expiry_hours`: model expiry; if model is older than this, FreqAI retrains before trading
- `identifier`: must be unique per experiment; changing triggers full retraining
- `label_period_candles`: how many bars forward to compute the label (e.g., 24 bars = 6h on 15m)

---

### 1.3 Feature Engineering Pipeline

FreqAI uses a hook-based feature pipeline. Override these methods in your IStrategy:

```python
class MyFreqAIStrategy(IStrategy):

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int,
        metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Called for EVERY combination of timeframe x period x shifted candle.
        FreqAI automatically expands these across include_timeframes.
        All column names are auto-prefixed by FreqAI with timeframe and shift info.
        """
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Called once per timeframe (no period expansion).
        Useful for single-instance features per timeframe.
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_close"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Called once on the base pair/timeframe only.
        For pair-specific features and the target label.
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_close"] = dataframe["close"]

        # Define the target (label)
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai.get_target_horizon())
            .pct_change()
            * 100
        )
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """Defines what the model is trying to predict."""
        dataframe["&-s_close"] = (
            dataframe["close"].shift(-48) > dataframe["close"]
        ).astype("category")
        return dataframe
```

**Automatic feature expansion:** FreqAI multiplies your features across all
`include_timeframes`, `include_shifted_candles` count, and all periods. A single
`%-rsi-period` with 3 timeframes and 2 shifted candles and 3 periods = 3 x 3 x 2 = 18
features. This auto-expansion is the core of FreqAI's feature engineering.

**Column naming convention:**
- `%-`: raw feature (will be included in training)
- `&-`: target label (what to predict)
- Features from informative pairs: auto-prefixed with pair name and timeframe

**Outlier removal:** FreqAI includes built-in outlier detection (SVM-based) and
feature normalization. The `DI_threshold` parameter controls Dissimilarity Index
filtering — rows where the test data is too far from training distribution are
flagged and the model returns its `neutral_threshold` instead of trading.

---

### 1.4 Training and Inference Cycle

**Backtesting mode:**
```
Full data window
├─ Train period (train_period_days): e.g., Jan 1 - Jan 30
├─ Test/backtest period (backtest_period_days): e.g., Jan 31 - Feb 6
│   └─ Model trained on Jan 1-30 makes predictions for Jan 31-Feb 6
├─ Slide forward by backtest_period_days
├─ Train period: Jan 8 - Feb 6
├─ Test period: Feb 7 - Feb 13
└─ ... repeat until end of data
```
This is a strict walk-forward OOS split. No look-ahead bias, unlike naive pandas ML.

**Live/dry-run mode:**
- On startup, FreqAI trains a model on the last `train_period_days` of data.
- Model is saved in `user_data/models/<identifier>/`
- At each candle, inference runs on the current dataframe row.
- If `expiry_hours` has elapsed since last training, a background thread retrains.
- `live_retrain_hours`: forces retraining every N hours regardless of expiry.
  - 0 = retrain as often as possible (1 candle lag, CPU intensive)
  - 24 = daily retrain (recommended for 15m strategies)
  - 168 = weekly retrain (for very stable markets)

**Model storage:**
- `user_data/models/<identifier>/sub-train-<pair>_<timestamp>/`
- Each training run saves: model file, feature list, normalization scaler, DI parameters
- `purge_old_models: true` removes old runs to save disk space

**Inference latency:** Typically 1-50ms for LightGBM/XGBoost on 15m data.
PyTorch models: 10-200ms. Acceptable for all timeframes above 1m.

---

### 1.5 IStrategy Integration Example

```python
from freqtrade.freqai.base_models.FreqaiMultiOutputClassifier import FreqaiMultiOutputClassifier

class FreqAIOKXStrategy(IStrategy):
    # Required by FreqAI
    freqai_info = {}
    process_only_new_candles = True
    use_custom_stoploss = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # FreqAI handles its own features; populate any non-FreqAI indicators here
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # FreqAI predictions are available as columns after start()
        # Column name depends on set_freqai_targets: "&-s_close" -> "predict_label"
        df.loc[
            (df["do_predict"] == 1) &          # DI filter: 1 = not outlier
            (df["&-s_close_up_or_down"] == "up") &  # model predicted "up"
            (df["&-s_close_up_or_down_prob_up"] > 0.6),  # probability > 60%
            "enter_long"
        ] = 1
        return df

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        # FreqAI populates dataframe with prediction; can use it for dynamic SL
        return self.stoploss
```

---

### 1.6 FreqAI Backends: Installation Requirements

| Backend | Install Command | Notes |
|---|---|---|
| scikit-learn | included by default | Always available |
| LightGBM | `pip install lightgbm` | Recommended default |
| XGBoost | `pip install xgboost` | Alternative to LightGBM |
| CatBoost | `pip install catboost` | Slower to install, good with categoricals |
| PyTorch | `pip install torch` | Required for MLP/Transformer/RL |
| stable-baselines3 | `pip install stable-baselines3` | Required for RL models |

**Full FreqAI install:**
```bash
pip install freqtrade[freqai]        # installs lightgbm, xgboost, scikit-learn
pip install freqtrade[freqai-rl]     # additionally installs sb3, torch
pip install freqtrade[freqai-torch]  # additionally installs torch models
```

---

### 1.7 Example Strategies Using FreqAI

All in `freqtrade/templates/` and `freqtrade/tests/strategy/strats/`:

1. **FreqaiExampleStrategy.py**: Basic LightGBM classifier, 4h + 1d features
2. **FreqaiExampleHybridStrategy.py**: Combines FreqAI signal with traditional indicators
3. **freqai_rl_example.py**: Full RL example with PPO and custom reward function

Also in the freqtrade-strategies community repo:
- `strategies/FreqAI*` directory (github.com/freqtrade/freqtrade-strategies)
- Community-contributed FreqAI strategies with varying complexity

---

### 1.8 Known Limitations and Issues

1. **Backtesting speed**: FreqAI backtests are 5-20x slower than standard IStrategy
   backtests because it trains a model at each forward-step. With 6 months of 15m data
   and a 30d train / 7d test window, this means ~17 training runs per pair.

2. **Backtest period minimum**: `train_period_days` + `startup_candle_count` candles must
   be available before the first prediction is made. For 30d training on 15m data, the
   first 2,880 candles (30 days) are "warm-up" and produce no signals.

3. **Correlation pair data requirement**: If using `include_corr_pairlist`, ALL those pairs'
   data must be downloaded before backtesting.

4. **Windows support**: PyTorch-based models may have issues on Windows with
   multiprocessing (`spawn` vs `fork`). Use `"multiprocessing_start_method": "spawn"` in
   config or avoid RL models on Windows.

5. **Label horizon creates future leak in naive usage**: The `label_period_candles` parameter
   means the label for bar t uses data from bars t+1 through t+N. FreqAI handles this
   correctly in backtesting by shifting the label, but custom implementations must be careful.

6. **DI filtering can remove too many predictions**: If `DI_threshold` is set too low,
   the model returns "do not trade" (do_predict=0) too frequently. In thin markets like
   SOL, this can eliminate most signals.

7. **Model versioning**: Changing `identifier` starts a completely fresh training run.
   There is no incremental learning — each retrain is from scratch on the window.

8. **Hyperparameter optimization**: FreqAI does not directly support freqtrade's hyperopt
   for model parameters. You must manually sweep `model_training_parameters` or use
   external hyperopt frameworks.

9. **Memory usage**: Training LightGBM on 30d of 15m data for 4 pairs with 3 corr pairs
   and 3 timeframes can require 4-8GB RAM. PyTorch models require more.

10. **Live-backtest parity issues**: The live retrain cycle can produce slightly different
    models than backtesting due to data differences at the edges of the training window.

---

### 1.9 FreqAI vs Custom ML (Current Approach)

Our current system uses **custom ML outside FreqAI** (scripts/ml_5m_v3.py, etc.).

| Dimension | FreqAI | Current Custom ML |
|---|---|---|
| Walk-forward | Built-in, automatic | Manual (scripts/ml_5m_v3.py) |
| Backtesting integration | Native in freqtrade | Separate from freqtrade backtest |
| Feature engineering | Hook-based, multi-TF | Manual pandas/ta |
| Retraining in live | Automatic | Manual (scripts/retrain_ml_monthly.py) |
| DI filtering (outliers) | Built-in | Not implemented |
| Model types | 15+ model classes | LightGBM only |
| Complexity | High config overhead | More code, but more control |
| Debug/iterate speed | Slow (full backtest) | Fast (standalone scripts) |
| Freqtrade integration | Tight (native) | Loose (via strategy hooks) |
| Our system compatibility | Requires refactor | Already working |

**Recommendation for this codebase:** The custom ML approach is BETTER for our use case
because:
- We need tight control over the training pipeline (meta-labeling, triple-barrier)
- The existing ml_ensemble.py strategy already works within the BaseStrategy/CryptoEngine
- FreqAI's abstraction hides the labeling logic that is our primary R&D focus
- FreqAI backtesting is 5-20x slower, complicating rapid iteration
- The CryptoEngine architecture (Orchestrator, EventBus) is orthogonal to FreqAI's design

FreqAI IS worth considering for:
- Standard direction-classification tasks where the default feature engineering suffices
- When you want automatic multi-timeframe feature expansion without manual coding
- When you want RL models (PPO/SAC) without implementing the gym environment yourself

---

## Part 2: Open-Source ML Trading Frameworks

### Framework Evaluation Matrix

| Framework | Stars (est.) | Last Active | ML Native | Crypto Support | Local-only | Freqtrade-compatible |
|---|---|---|---|---|---|---|
| FinRL | ~15k | Active (2026) | Yes (RL) | Yes | Yes | Partial |
| QLib | ~15k | Active (2026) | Yes (full) | No (stocks) | Yes | Partial |
| TensorTrade | ~4.5k | Stale (2022) | Yes (RL) | Yes | Yes | Partial |
| vectorbt | ~4k | Active (2024) | Partial | Yes | Yes | Partial |
| mlfinlab | ~3.5k | Partial (2024) | Yes | Yes | Yes (core) | Yes |
| Jesse | ~5k | Active (2026) | External only | Yes | Yes | No |
| OctoBot | ~3k | Active (2026) | Plugin-based | Yes | Yes | No |
| Hummingbot | ~7k | Active (2026) | External only | Yes (DeFi+CEX) | Yes | No |
| Catalyst | ~2k | Dead (2019) | No | Yes (Poloniex) | Yes | No |
| Zipline-reloaded | ~1.5k | Active (2024) | No | No | Yes | No |
| Backtrader | ~13k | Stale (2021) | External only | Partial | Yes | No |
| pytorch-forecasting | ~4k | Active (2024) | Yes (TFT/etc) | Yes | Yes | Yes |
| hmmlearn | ~3.3k | Active (2024) | Yes (HMM) | Yes | Yes | Yes |

---

### 2.1 FinRL — Deep RL for Finance

**GitHub:** https://github.com/AI4Finance-Foundation/FinRL
**Stars:** ~15,400 (as of mid-2025)
**Last update:** Active, maintained (2025-2026)
**Python:** 3.8+, works on Python 3.12
**License:** MIT

**What it supports:**
- Algorithms: A2C, DDPG, PPO, SAC, TD3 (via stable-baselines3 backend)
- Environments: StockTradingEnv, CryptoEnv (custom gym environments)
- Data: Yahoo Finance, Alpaca, Binance, Coinbase via ccxt integration
- Portfolio management (multi-asset position sizing)
- Single-asset directional trading

**Crypto-specific features:**
- CryptoEnv supports perpetual futures with funding rate cost modeling
- Built-in OKX, Binance connectors (via ccxt)
- Example notebooks: BTC/ETH/BNB portfolio allocation with PPO

**Local deployment:** Fully local, no cloud required. Training on CPU is supported
but GPU significantly speeds up training (especially SAC).

**Freqtrade integration path:**
FinRL is NOT directly compatible with freqtrade's IStrategy. To use FinRL models in
freqtrade, you would:
1. Train the RL agent offline using FinRL on historical OKX data
2. Save the trained policy (stable-baselines3 model file)
3. Load the policy in a freqtrade strategy's `populate_entry_trend`
4. Call `model.predict(obs)` with the current bar's state vector

**Limitations for our use case:**
- RL agents trained on 1h/4h data fail catastrophically on 5m (documented in our own
  research file, arXiv:2209.05559 overfitting warning)
- No native freqtrade integration
- Training instability on non-stationary crypto data
- The Gort et al. (2022) paper (in our research doc) shows most published FinRL results
  are false positives from backtest overfitting

**Rating for OKX 15m futures:** MEDIUM (use for portfolio allocation, not 5m scalping)

---

### 2.2 QLib — Microsoft AI-Oriented Quantitative Investment

**GitHub:** https://github.com/microsoft/qlib
**Stars:** ~15,200 (as of mid-2025)
**Last update:** Active, maintained by Microsoft Research
**Python:** 3.8+
**License:** MIT

**What it supports:**
- Models: LightGBM, XGBoost, LSTM, GRU, Transformer, TFT, DoubleEnsemble, ALSTM
- Data: Built-in Yahoo Finance, Tushare (Chinese stocks), custom data loaders
- Infrastructure: Backtesting engine, rolling training, alpha factor research
- Notable: Implementation of DoubleEnsemble (arXiv:2010.01265, cited in our research)

**Strengths:**
- The most mature academic-grade quant research framework
- Implements state-of-the-art models from the literature
- Proper walk-forward (rolling training) built-in
- Includes transaction cost models
- DoubleEnsemble is directly relevant to our triple-barrier labeling work

**Critical limitation for crypto:**
QLib is **stock market focused**. It assumes:
- US market hours (9:30-16:00 EST) — incompatible with 24/7 crypto
- Daily-resolution primary data (15m is at the edge of support)
- No perpetual futures / funding rate cost model
- No short-selling friction costs (crypto shorts have funding costs)

**Freqtrade integration:** Possible but non-trivial. Use QLib for offline model training,
export model weights, load in freqtrade strategy.

**Rating for OKX 15m futures:** LOW for live deployment. HIGH for research (especially
DoubleEnsemble experiments).

**Relevant QLib components to extract:**
```python
# DoubleEnsemble - directly applicable to our ml_5m experiments
from qlib.contrib.model.double_ensemble import DEnsembleModel

# DoubleEnsemble training config
model = DEnsembleModel(
    base_model="gbm",           # LightGBM base
    num_models=6,               # Number of sub-models
    enable_sr=True,             # Sample reweighting
    enable_fs=True,             # Feature selection
    alpha1=1.0, alpha2=1.0,     # Reweighting aggressiveness
    bins_sr=10, bins_fs=5,      # Quantile bins
    decay=0.5,                  # Sample decay rate (downweight old samples)
    sample_ratios=None,
    sub_weights=None
)
```
The `decay` parameter is particularly relevant: it downweights older training samples,
addressing the non-stationarity problem we confirmed in ml_5m_v3 fold instability.

---

### 2.3 TensorTrade — RL Trading Framework

**GitHub:** https://github.com/tensortrade-org/tensortrade
**Stars:** ~4,500
**Last update:** Largely stale (2021-2022). No significant commits since 2022.
**Python:** 3.7-3.9 (3.10+ compatibility issues documented by users)
**License:** Apache 2.0

**What it supports:**
- RL environments for crypto and stock trading
- Compatible with stable-baselines3, RLlib, ray
- Custom reward functions (Sharpe, Sortino, simple PnL)
- Data feeds: OHLCV, order book (limited), custom

**Reality check:**
TensorTrade was innovative in 2019-2020 but has been largely superseded by FinRL and
direct gym environment implementations. The repo has minimal maintenance activity.
The Python 3.10+ compatibility issues make it a poor choice for Python 3.12 (our env).

**Rating for OKX 15m futures:** SKIP. Use FinRL instead if RL is needed.

---

### 2.4 vectorbt — Vectorized Backtesting with ML

**GitHub:** https://github.com/polakowo/vectorbt
**Stars:** ~4,100
**Last update:** Active (2024)
**Python:** 3.8+
**License:** Proprietary (free for personal use), commercial license for production

**What it supports:**
- Extremely fast backtesting via NumPy vectorization (10-100x faster than freqtrade)
- NOT an ML framework per se — it accelerates the signal → backtest loop
- Integrates with sklearn, LightGBM, any model that outputs arrays
- Portfolio statistics, drawdown analysis, Monte Carlo built-in
- Supports 24/7 crypto, perpetual futures

**Key advantage for our research pipeline:**
vectorbt can backtest a LightGBM strategy across 6 months of 15m data in seconds
instead of minutes. This makes it excellent for hyperparameter sweeping.

**Freqtrade integration:** vectorbt runs entirely outside freqtrade. Use for rapid
research/iteration, then implement the validated strategy in freqtrade IStrategy.

**Practical usage in our context:**
```python
import vectorbt as vbt
import lightgbm as lgb

# Fast sweep of confidence thresholds (seconds, not minutes)
for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
    entries = prob_long > threshold
    exits = prob_long < 0.5
    pf = vbt.Portfolio.from_signals(price, entries, exits, fees=0.0005)
    print(threshold, pf.total_return(), pf.sharpe_ratio())
```

**Rating for OKX research:** HIGH as a backtesting accelerator for ML research.
LOW as a production system (license limitations, not freqtrade-compatible).

---

### 2.5 mlfinlab / Hudson & Thames

**GitHub:** https://github.com/hudson-and-thames/mlfinlab
**Stars:** ~3,500 (core open source)
**Last update:** Partial maintenance (2024). Full features moved to paid PortfolioLab.
**Python:** 3.8+
**License:** BSD-3 (core), commercial (advanced features)

**What it supports (open-source core):**
- **Triple barrier labeling** (Chapter 3, de Prado): directly applicable to our research
- **Meta-labeling** (Chapter 3): we already implement this in ml_meta_labeling.py
- **Fractional differentiation** (FracDiff): preserves memory while achieving stationarity
- **CUSUM filter**: event-based sampling (avoids oversampling uniform candles)
- **Sequential bootstrap**: reduces sample redundancy in financial ML
- **ETF trick**: synthetic continuous futures from rolling contracts

**Most relevant for our codebase:**
The `mlfinlab.labeling.triple_barrier` module is more robust than our own implementation
in ml_triple_barrier_v2.py, with proper handling of:
- Simultaneous TP/SL touches (lowest price first rule)
- Vectorized computation (faster than our bar-by-bar loop)
- Expiry date based on calendar, not bar count

```python
from mlfinlab.labeling.triple_barrier import triple_barrier_labeling

# Direct replacement for our triple_barrier_labels() function
events = pd.DataFrame({
    't1': close.index[horizon],           # expiry bar
    'trgt': atr * 1.5,                   # stop loss distance
    'side': 1                            # long only (or +1/-1)
})
labels = triple_barrier_labeling(close, events, sltp=[1.5, 2.0], num_threads=1)
```

**CUSUM filter** is highly relevant: instead of trading at every 15m bar, sample only
at significant price change events (cumulative sum exceeds ATR threshold). This reduces
serial correlation in training data and produces cleaner labels.

**Rating for OKX 15m futures:** HIGH (specific components directly usable in our pipeline).

---

### 2.6 Jesse — Crypto Trading Framework

**GitHub:** https://github.com/jesse-ai/jesse
**Stars:** ~5,200
**Last update:** Active (2026)
**Python:** 3.10+
**License:** MIT

**What it supports:**
- Clean rule-based strategy framework designed specifically for crypto
- Fast backtesting with multi-timeframe support
- Live trading via Bybit, Binance, FTX connectors (OKX via community plugins)
- ML capabilities: external only (no built-in ML); strategies call sklearn/LightGBM

**Comparison with freqtrade for our use case:**
Jesse has a cleaner API than freqtrade for strategy development, but:
- No IStrategy equivalent (incompatible architecture)
- OKX support is via community plugins (less tested than freqtrade's)
- No equivalent to our CryptoEngine Orchestrator architecture
- No built-in FreqAI-style ML training pipeline

**Freqtrade integration:** Not compatible. These are competing frameworks.

**Rating for OKX 15m futures:** SKIP. Already on freqtrade which is more mature for OKX.

---

### 2.7 OctoBot — Crypto Trading Bot with ML

**GitHub:** https://github.com/Drakkar-Software/OctoBot
**Stars:** ~3,100
**Last update:** Active (2026)
**Python:** 3.10+
**License:** LGPL-2.1

**What it supports:**
- Modular plugin system ("tentacles") for strategies and ML
- Built-in OKX connector
- ML via ML-Trader tentacle (separate install)
- Evaluators: technical analysis, social, ML signals
- Real-time and backtesting modes

**ML in OctoBot:**
The ML tentacle uses keras/tensorflow for LSTM-based prediction. However, it is:
- Not as flexible as our custom LightGBM pipeline
- Limited to pre-built LSTM architectures
- Slower to iterate on than our research scripts

**Freqtrade integration:** Incompatible. Different framework entirely.

**Rating for OKX 15m futures:** SKIP. Not better than our current setup.

---

### 2.8 Hummingbot — Market Making with ML

**GitHub:** https://github.com/hummingbot/hummingbot
**Stars:** ~7,400
**Last update:** Active (2026)
**Python:** 3.10+
**License:** Apache 2.0

**What it supports:**
- Market making strategies (cross-exchange, single exchange)
- Arbitrage strategies
- Custom strategies via Python SDK
- OKX, Binance, Bybit, and 40+ exchange connectors
- ML is NOT native — strategies are rule-based with Python hooks for external models

**Relevance to our system:**
Hummingbot excels at market making and arbitrage, which are fundamentally different from
our directional futures trading. The architecture is:
- Script-based (no IStrategy equivalent)
- Real-time event-driven (websocket-native)
- No backtesting engine comparable to freqtrade's

For ML + market making: Hummingbot can call external ML models, but you manage the
training pipeline separately.

**Rating for OKX 15m futures directional trading:** LOW. Different use case.
MEDIUM if ever adding basis arbitrage or market making strategies.

---

### 2.9 Catalyst — DEAD PROJECT

**GitHub:** https://github.com/enigmampc/catalyst
**Stars:** ~2,000
**Last update:** Archived/dead since 2019
**Status:** DO NOT USE. The exchange (Enigma/Poloniex) integration is broken.
The library has not been maintained and is incompatible with Python 3.8+.

---

### 2.10 Zipline-reloaded — Algorithmic Trading

**GitHub:** https://github.com/stefan-jansen/zipline-reloaded
**Stars:** ~1,500
**Last update:** Active (2024)
**Python:** 3.8-3.12
**License:** Apache 2.0

**What it supports:**
- Classic event-driven backtesting (Quantopian-style)
- Pipeline API for factor-based research
- Custom data bundles (can add crypto OHLCV)
- ML integration: external (bring your own sklearn pipeline)

**Critical limitation for crypto:**
- Designed for US equity market (calendar, market hours, splits, dividends)
- Adding crypto requires custom bundle + calendar (doable but non-trivial)
- No perpetual futures or funding rate modeling
- 24/7 trading requires hacks to the trading calendar

**Rating for OKX 15m futures:** SKIP. Stock-market focused, not worth the crypto adaptation cost.

---

### 2.11 pytorch-forecasting — TFT and Time Series Models

**GitHub:** https://github.com/jdb78/pytorch-forecasting
**Stars:** ~3,900
**Last update:** Active (2024)
**Python:** 3.8+
**License:** MIT

**What it supports:**
- Temporal Fusion Transformer (TFT) — the paper we cite in our research (arXiv:1912.09363)
- N-BEATS, N-HiTS, DeepAR
- Proper time series cross-validation with TimeSeriesDataSet

**Directly relevant to Approach 6 (TFT) in our research doc (ml_alternatives_5m.md).**
This is the library to use if TFT experimentation is pursued.

```bash
pip install pytorch-forecasting lightning
```

**Key limitation:** Requires reformatting data into pytorch-forecasting's
TimeSeriesDataSet format, which is verbose. The library is also tightly coupled to
PyTorch Lightning for training loops.

**Rating for OKX 15m futures:** MEDIUM (for Approach 6 TFT experiments only).

---

### 2.12 hmmlearn — Hidden Markov Models

**GitHub:** https://github.com/hmmlearn/hmmlearn
**Stars:** ~3,300
**Last update:** Active (2024)
**Python:** 3.8+
**License:** BSD-3

**What it supports:**
- GaussianHMM: continuous observations (returns + volatility)
- GMMHMM: Gaussian mixture HMM
- MultinomialHMM: discrete observations
- VariationalGaussianHMM: Bayesian variant

**Directly relevant to Approach 3 (Regime-Aware ML) in our research doc.**
This is the library for implementing the HMM-based regime detector.

```python
from hmmlearn import hmm
import numpy as np

# Regime detection with GaussianHMM
# Features: [daily_return, daily_volatility]
obs = np.column_stack([returns, volatility])

model = hmm.GaussianHMM(
    n_components=3,        # 3 regimes: trend, range, volatile
    covariance_type="full",
    n_iter=1000,
    random_state=42
)
model.fit(obs)
regimes = model.predict(obs)  # 0, 1, or 2 for each bar
```

**Rating for OKX 15m futures:** HIGH for regime detection component of Approach 3.

---

### 2.13 Backtrader — Traditional Rule-Based Framework

**GitHub:** https://github.com/mementum/backtrader
**Stars:** ~12,900
**Last update:** Effectively unmaintained (2021). Community forks exist but diverge.
**Python:** 3.6+ (but 3.10+ has some compatibility issues)
**License:** GPL-3.0

**ML integration:**
Backtrader itself has no ML. Integration requires calling external sklearn/lgb models
from within a Strategy class. The event-driven architecture makes this straightforward.

**Why not use it:** freqtrade is more mature, has better OKX support, and our CryptoEngine
already provides superior architecture. No reason to migrate.

**Rating for OKX 15m futures:** SKIP.

---

## Part 3: GitHub Search Results — High-Signal Repos

The following repositories were identified as high-relevance for ML + crypto trading
based on known GitHub landscape as of mid-2025. Stars are approximate.

### 3.1 freqtrade-strategies (Community Strategies)

**GitHub:** https://github.com/freqtrade/freqtrade-strategies
**Stars:** ~3,500
**Key ML strategies in this repo:**
- `NotAnotherSMAOffsetStrategyHOv3.py` — multiple indicator confluence
- `FreqAI*` strategies — various FreqAI examples from community
- `NASOSv4_mod3.py` — popular NASOS (5m, volatile pair-friendly)
- `BinHV45.py` — vintage but still used

**Most relevant:** Any strategy with `freqai_info` in the class definition is a FreqAI
strategy. Search the repo for `FreqAI` to find all examples.

---

### 3.2 DoubleEnsemble / Ensemble Crypto Prediction

From arXiv:2411.03035 (Blending Ensemble for BTC, 2024):
- No dedicated GitHub repo, but uses sklearn blending of LightGBM + XGBoost + RF
- Directly implementable: train 3 models, blend via LogisticRegression meta-learner
- Outperforms single-model on BTC daily direction with 34 alpha factors

---

### 3.3 deep-orderbook

**GitHub:** https://github.com/Globe-Research/deep-orderbook
**Stars:** ~200
**Relevance:** Implements deep LOB models (DeepLOB, CNN-LOB) for cryptocurrency
limit order book prediction. Relevant to Approach 4 (Tier A, LOB-based) in our research.
Requires L2 orderbook data which we do not currently collect.

---

### 3.4 Crypto Prediction LSTM Repos

Many repos with names like "bitcoin-prediction-lstm" or "crypto-lstm" exist but are:
- Single-pair, single-model, no walk-forward validation
- Typically show in-sample performance only
- No cost model
- DEAD END per our anti-patterns section

The academic literature (and our own ml_5m experiments) confirm LSTM does not
outperform LightGBM on 5m OHLCV data after proper OOS validation.

---

### 3.5 Crypto RL Trading Repos

The best maintained repos as of 2025:
- **crypto-rl**: github.com/cryptorobotics/crypto-rl — OFI + LOB features for RL agents
- **btgym**: github.com/Kismuz/btgym — gym environment wrapping Backtrader
- **gym-anytrading**: github.com/AminHP/gym-anytrading — simple gym for any trading

All are lower quality than FinRL for production use. The Gort et al. (2022) overfitting
caveat applies to all of them.

---

## Part 4: Prioritized Recommendations for This Codebase

### Immediate Actions (Week 1)

1. **DO NOT migrate to FreqAI** for the 15m rule-based strategies (regime_adaptive,
   volume_spike_rev, cb_adx_breakout). These Grade A/B strategies work with our existing
   infrastructure. FreqAI migration would add complexity without measurable benefit.

2. **DO use mlfinlab's CUSUM filter** as a data preprocessing step for the ML experiments.
   The filter samples only at statistically significant price events, reducing overlapping
   labels. Install: `pip install mlfinlab`

3. **DO use hmmlearn** for the regime detector in Approach 3. Three-state GaussianHMM on
   (returns, ATR) is the cleanest regime detection implementation.

4. **DO use vectorbt** for rapid research iterations on ML thresholds. It can test 100
   parameter combinations in the time freqtrade backtests 1.

### Candidate for FreqAI Integration (Grade C / Research)

If a new strategy requires:
- Automatic multi-timeframe feature expansion across 4+ timeframes
- Standard direction classification (not meta-labeling)
- No need for custom triple-barrier labeling

Then FreqAI with LightGBMClassifier is appropriate. Example: a regime classifier that
uses 1h + 4h features to predict whether the next 4h will be trending (ADX>25) or ranging.
This is a natural FreqAI use case and would complement our rule-based regime_adaptive.

### Libraries to Install Now

```bash
# For existing ML research pipeline
pip install mlfinlab      # triple barrier, CUSUM, meta-labeling
pip install hmmlearn      # HMM regime detection
pip install vectorbt      # fast backtesting

# For FreqAI experiments (separate)
pip install freqtrade[freqai]   # includes lightgbm, xgboost, scikit-learn
```

---

## Part 5: Anti-Patterns Confirmed by Framework Review

1. **TensorTrade**: Stale since 2022, Python 3.10+ issues. SKIP.
2. **Catalyst**: Archived/dead. SKIP.
3. **Zipline-reloaded**: Stock market focused, 24/7 crypto requires non-trivial hacks. SKIP.
4. **Backtrader**: Unmaintained. SKIP.
5. **LSTM on 5m OHLCV**: Confirmed dead-end across all frameworks. No framework fixes
   the fundamental problem that sequential patterns in 5m OHLCV have near-zero
   predictive value beyond what LightGBM captures with the same features.
6. **RL for 5m scalping**: No published framework produces live-validated RL strategies
   on sub-hourly crypto data. Gort et al. (2022) explains why (backtest overfitting).
7. **QLib for live crypto**: Excellent research tool, wrong production environment.

---

## Part 6: Summary Table

| Framework | Action | Why | Priority |
|---|---|---|---|
| FreqAI (LightGBMClassifier) | Use for new regime classifier | Native integration, auto MTF features | Medium |
| FreqAI (RL) | Research only, not production | Overfitting risk, Windows issues | Low |
| mlfinlab | Install now, use CUSUM + triple barrier | Better than our hand-rolled versions | High |
| hmmlearn | Use for Approach 3 regime detection | Clean, documented, 3.3k stars | High |
| vectorbt | Use for research sweep speed | 10-100x faster than freqtrade backtest | High |
| FinRL | Research only (Approach 7 RL) | Only if rule-based approaches fail | Low |
| QLib (DoubleEnsemble) | Extract algorithm, not framework | Sample reweighting directly applicable | Medium |
| pytorch-forecasting | Only if TFT pursued (Approach 6) | Standard TFT implementation | Low |
| Jesse | Skip | Competing framework, incompatible | Skip |
| OctoBot | Skip | Inflexible ML, incompatible arch | Skip |
| Hummingbot | Skip for now (future basis arb) | Different use case | Skip |
| TensorTrade | Skip | Stale, compatibility issues | Skip |
| Catalyst | Skip | Dead project | Skip |

---

## References

- FreqAI Documentation: https://www.freqtrade.io/en/stable/freqai/
- arXiv:2209.05559 (Gort et al. 2022): DRL backtest overfitting detection
- arXiv:2606.04574 (Lebiedz & Slepaczuk 2026): PPO+LSTM on Binance futures 1h
- arXiv:2509.10542 (Peik et al. 2025): Adaptive TFT for ETH-USDT 10m
- arXiv:2010.01265 (DoubleEnsemble, ICDM 2020): Sample reweighting for crypto
- arXiv:1902.10849 (Fons et al. 2019): Feature Saliency HMM for regime detection
- de Prado (2018): Advances in Financial Machine Learning. Wiley.
- Existing codebase research: e:\freqtrade\research\ml_alternatives_5m.md

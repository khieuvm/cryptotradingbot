# Comprehensive ML Model Catalogue for Crypto Futures Trading

**Date:** 2026-06-07
**Target:** OKX futures (ETH/USDT, SOL/USDT, SPX/USDT, DOGE/USDT), 15m + 5m
**Environment:** Windows 11, Python 3.12, local training (CPU + optional GPU)

---

## OVERVIEW: 60+ Models Across 5 Categories

| Category | Models | Best For |
|----------|--------|----------|
| A. Classical ML / Tree-Based | 13 models | Tabular OHLCV features, proven best for crypto |
| B. Deep Learning | 20 models | Sequence patterns, multi-timeframe fusion |
| C. Reinforcement Learning | 10 models | Joint entry/exit/sizing optimization |
| D. Probabilistic / Bayesian | 8 models | Uncertainty quantification, regime detection |
| E. Advanced / Hybrid | 12 models | Meta-learning, online learning, anomaly detection |

---

## CATEGORY A: Classical ML & Tree-Based (13 models)

*Detailed analysis in `research/ml_classical_models_part1.md`*

### Tier 1 — HIGH Priority

| # | Model | Library | Speed | WR vs LGBM | Best Use Case |
|---|-------|---------|-------|------------|---------------|
| 1 | **LightGBM** | `lightgbm` | 1x (baseline) | baseline | Standalone ML, meta-labeling |
| 2 | **XGBoost** | `xgboost` | 1.5x | ~equal | GPU hyperopt, FreqAI native |
| 3 | **Stacking Ensemble** | `sklearn` | 5x | +1-3pp | Best accuracy (LGB+ET+LR) |

### Tier 2 — MEDIUM Priority

| # | Model | Library | Speed | WR vs LGBM | Best Use Case |
|---|-------|---------|-------|------------|---------------|
| 4 | **CatBoost** | `catboost` | 3-5x | +0.5pp (small N) | Small samples, categoricals |
| 5 | **Random Forest** | `sklearn` | 1.5x | -1 to -3pp | Ensemble component, OOB score |
| 6 | **Extra Trees** | `sklearn` | 0.7x | -1pp | Regime-unstable data |
| 7 | **SVM (RBF)** | `sklearn` | O(n²) | comparable (small N) | Meta-labeling <400 samples |
| 8 | **Logistic Regression** | `sklearn` | instant | -3 to -8pp | Meta-learner in stacking |

### Tier 3 — LOW Priority

| # | Model | Library | Notes |
|---|-------|---------|-------|
| 9 | Gradient Boosting (sklearn) | `sklearn` | 10-20x slower than LGB, same accuracy |
| 10 | AdaBoost | `sklearn` | Fat-tailed returns break it |
| 11 | KNN | `sklearn` | Slow inference, curse of dimensionality |
| 12 | Naive Bayes | `sklearn` | Only value: `partial_fit()` online learning |
| 13 | Decision Tree | `sklearn` | Rule extraction only |

**Key Finding:** Grinsztajn et al. (2022, NeurIPS) proved tree-based models beat neural networks on tabular financial data across 45 benchmarks. LightGBM/XGBoost are the correct starting point.

---

## CATEGORY B: Deep Learning (20 models)

### B1. Recurrent Neural Networks (RNN family)

| # | Model | Architecture | Library | GPU Required | Params | Training Time (180d 15m) |
|---|-------|-------------|---------|-------------|--------|--------------------------|
| 14 | **LSTM** | 2-layer LSTM, hidden=64-128 | PyTorch/Keras | Optional | 50-200K | 5-15 min (GPU), 30-60 min (CPU) |
| 15 | **GRU** | 2-layer GRU, hidden=64-128 | PyTorch/Keras | Optional | 40-150K | 4-12 min (GPU) |
| 16 | **Bi-LSTM** | Bidirectional LSTM | PyTorch | Optional | 100-400K | 10-30 min (GPU) |
| 17 | **Bi-GRU** | Bidirectional GRU | PyTorch | Optional | 80-300K | 8-25 min (GPU) |

**How LSTM/GRU work for trading:**
- Input: sliding window of N bars × M features (e.g., 24 bars × 40 features)
- Output: P(long win), P(short win) or direction class
- Captures temporal dependencies that trees miss (sequential patterns)
- Training: BackPropagation Through Time (BPTT)

```python
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 2)  # [P(short), P(long)]

    def forward(self, x):  # x: (batch, seq_len, features)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])
```

**Realistic performance:** On 5m crypto OHLCV, LSTM matches LightGBM (~53% WR) but with 10-50x more training time. The sequential patterns in 5m data have near-zero additional predictive value beyond what tabular features capture. On 1h+ data with order flow features, LSTM may add 1-2pp.

### B2. Convolutional Networks

| # | Model | Architecture | Library | GPU | Params | Notes |
|---|-------|-------------|---------|-----|--------|-------|
| 18 | **1D CNN** | Conv1D + MaxPool stacks | PyTorch/Keras | Optional | 30-100K | Fast inference, good for pattern detection |
| 19 | **TCN** | Dilated causal convolutions | `pytorch-tcn` | Optional | 50-200K | Captures long-range dependencies without RNN |
| 20 | **WaveNet** | Dilated causal + skip connections | PyTorch | Recommended | 200K-1M | Audio-style, rarely used for trading |
| 21 | **CNN-LSTM** | CNN feature extractor → LSTM | PyTorch | Recommended | 100-500K | CNN captures local patterns, LSTM sequences |

```python
# TCN example (Temporal Convolutional Network)
# pip install pytorch-tcn
from tcn import TemporalConvNet

class TCNClassifier(nn.Module):
    def __init__(self, input_size=40, num_channels=[32, 32, 32], kernel_size=3):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=0.2)
        self.fc = nn.Linear(num_channels[-1], 2)

    def forward(self, x):  # x: (batch, features, seq_len) - note: features first!
        y = self.tcn(x)
        return self.fc(y[:, :, -1])  # last timestep
```

**TCN advantage:** Parallelizable (unlike LSTM), handles long sequences efficiently with dilated convolutions. Each layer doubles the receptive field: kernel_size=3, 3 layers → 27-bar receptive field.

### B3. Transformer-Based Models

| # | Model | Architecture | Library | GPU | Params | Training Time |
|---|-------|-------------|---------|-----|--------|---------------|
| 22 | **Vanilla Transformer** | Multi-head self-attention | PyTorch | Recommended | 200K-2M | 15-60 min |
| 23 | **Temporal Fusion Transformer (TFT)** | Attention + variable selection | `pytorch-forecasting` | Recommended | 500K-5M | 30-120 min |
| 24 | **Informer** | ProbSparse attention | `tsai` / custom | Required | 1-10M | 30-90 min |
| 25 | **PatchTST** | Patching + channel-independent | `neuralforecast` | Recommended | 200K-2M | 15-45 min |
| 26 | **iTransformer** | Inverted: attention over features | Custom PyTorch | Recommended | 200K-2M | 15-45 min |
| 27 | **FT-Transformer** | Feature Tokenizer + Transformer | `tab-transformer-pytorch` | Optional | 100K-1M | 10-30 min |
| 28 | **TabNet** | Attention-based tabular | `pytorch-tabnet` | Optional | 100K-500K | 10-30 min |

**TFT (Temporal Fusion Transformer)** — Most promising DL model for trading:
- Variable selection network: automatically learns which features matter per timestep
- Interpretable multi-horizon attention
- Handles static (pair, strategy) + temporal (OHLCV, indicators) features
- Published by Google (Lim et al., 2019)

```python
# pip install pytorch-forecasting lightning
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

training = TimeSeriesDataSet(
    data, time_idx="bar_idx", target="forward_return",
    group_ids=["pair"],
    max_encoder_length=48,  # 48 bars lookback (12h on 15m)
    max_prediction_length=6,  # predict 6 bars ahead
    time_varying_known_reals=["hour_sin", "hour_cos"],
    time_varying_unknown_reals=["close", "volume", "rsi", "atr", "adx"],
    static_categoricals=["pair"],
)

model = TemporalFusionTransformer.from_dataset(
    training, hidden_size=32, attention_head_size=4,
    dropout=0.3, learning_rate=1e-3,
)
```

**PatchTST** — Efficient alternative to TFT:
- Divides time series into patches (like ViT for images)
- Channel-independent: each feature processed separately
- Fewer parameters, faster training than TFT
- State-of-the-art on many forecasting benchmarks (2023)

```python
# pip install neuralforecast
from neuralforecast.models import PatchTST

model = PatchTST(
    h=6,              # forecast horizon
    input_size=48,    # lookback
    patch_len=8,      # patch length
    stride=4,         # patch stride
    hidden_size=32,
    n_heads=4,
    learning_rate=1e-3,
)
```

**TabNet** — Best DL model for tabular data (when you don't want trees):
```python
# pip install pytorch-tabnet
from pytorch_tabnet.tab_model import TabNetClassifier

model = TabNetClassifier(
    n_d=8, n_a=8,           # decision/attention dimensions
    n_steps=3,               # number of attention steps
    gamma=1.5,               # attention coefficient
    lambda_sparse=1e-3,      # sparsity regularization
    optimizer_fn=torch.optim.Adam,
    scheduler_params={"step_size": 10, "gamma": 0.9},
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          max_epochs=100, patience=20, batch_size=256)
```

### B4. Specialized Time Series Models

| # | Model | Architecture | Library | GPU | Params | Notes |
|---|-------|-------------|---------|-----|--------|-------|
| 29 | **N-BEATS** | Fully connected residual stacks | `neuralforecast` / `darts` | Optional | 100K-1M | Interpretable decomposition |
| 30 | **N-HiTS** | Hierarchical interpolation | `neuralforecast` | Optional | 50-500K | Multi-scale temporal patterns |
| 31 | **TimesNet** | Multi-period 2D convolution | Custom PyTorch | Recommended | 500K-5M | Converts 1D to 2D via FFT periods |
| 32 | **DeepAR** | Autoregressive + probabilistic | `pytorch-forecasting` / `gluonts` | Optional | 200K-2M | Returns distribution, not point |

**N-BEATS** — Interpretable neural basis expansion:
```python
from neuralforecast.models import NBEATS

model = NBEATS(
    h=6, input_size=48,
    stack_types=["trend", "seasonality"],  # interpretable mode
    n_blocks=[3, 3],
    mlp_units=[[256, 256], [256, 256]],
)
```

### B5. Generative / Unsupervised DL

| # | Model | Architecture | Library | Use Case |
|---|-------|-------------|---------|----------|
| 33 | **VAE** | Variational Autoencoder | PyTorch | Regime detection, anomaly |
| 34 | **GAN (TimeGAN)** | Generative Adversarial | PyTorch | Synthetic data augmentation |

**VAE for regime detection:**
```python
class VAERegimeDetector(nn.Module):
    def __init__(self, input_dim=40, latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )
    # Latent space clusters → regime labels via KMeans
```

### B6. Time Series Libraries (all-in-one)

| Library | Models Included | Install | Stars |
|---------|----------------|---------|-------|
| **Darts** | LSTM, TCN, TFT, N-BEATS, N-HiTS, Transformer, DeepAR + classical | `pip install darts` | ~8k |
| **NeuralForecast** | PatchTST, N-BEATS, N-HiTS, TFT, LSTM, GRU, MLP | `pip install neuralforecast` | ~3k |
| **tsai** | ROCKET, InceptionTime, ResNet, LSTM, GRU, TCN, Transformer | `pip install tsai` | ~4k |
| **PyTorch Forecasting** | TFT, N-BEATS, DeepAR, NHiTS | `pip install pytorch-forecasting` | ~4k |
| **GluonTS** | DeepAR, WaveNet, Transformer, MQ-CNN | `pip install gluonts` | ~4k |

**tsai** — Particularly interesting for classification (buy/sell/hold):
```python
# pip install tsai
from tsai.all import *

X, y, splits = get_classification_data(df, target_col='label', window=24)
learn = TSClassifier(X, y, splits, arch=InceptionTimePlus, metrics=accuracy)
learn.fit_one_cycle(25, 1e-3)
```

**Key models from tsai:**
- **ROCKET** (Random Convolutional Kernel Transform): ~10K random kernels, linear classifier on top. Fastest training of all DL models. Often matches LSTM/CNN on time series classification.
- **InceptionTime**: Multi-scale CNN with bottleneck layers. State-of-art on UCR time series benchmarks.

---

## CATEGORY C: Reinforcement Learning (10 models)

### C1. Value-Based RL

| # | Model | Algorithm | Library | Action Space | Notes |
|---|-------|-----------|---------|-------------|-------|
| 35 | **DQN** | Deep Q-Network | `stable-baselines3` | Discrete: {buy, sell, hold} | Simple but effective |
| 36 | **Double DQN** | Reduces overestimation | `sb3` | Discrete | Better than vanilla DQN |
| 37 | **Dueling DQN** | Value + advantage streams | `sb3` | Discrete | Better state evaluation |
| 38 | **TDQN** | Custom DQN in FreqAI | `freqtrade` | Discrete | Native freqtrade support |

### C2. Policy-Based RL

| # | Model | Algorithm | Library | Action Space | Notes |
|---|-------|-----------|---------|-------------|-------|
| 39 | **PPO** | Proximal Policy Optimization | `sb3` | Both | Most stable, recommended start |
| 40 | **A2C** | Advantage Actor-Critic | `sb3` | Both | Simpler than PPO |
| 41 | **SAC** | Soft Actor-Critic | `sb3` | Continuous | Best for continuous sizing |
| 42 | **TD3** | Twin Delayed DDPG | `sb3` | Continuous | Position sizing + direction |
| 43 | **DDPG** | Deep Deterministic PG | `sb3` | Continuous | Predecessor to TD3 |
| 44 | **Multi-Agent RL** | Independent learners | `RLlib` | Per-strategy | One agent per strategy |

**PPO for trading (recommended RL starting point):**
```python
# pip install stable-baselines3[extra]
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class CryptoTradingEnv(gym.Env):
    """Custom gym environment for OKX futures."""
    def __init__(self, df, initial_balance=10000):
        self.action_space = gym.spaces.Discrete(3)  # 0=hold, 1=long, 2=short
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32
        )
        self.df = df
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # -1, 0, 1

    def step(self, action):
        # Execute action, calculate reward
        reward = self._calculate_reward(action)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()
        return obs, reward, done, False, {}

    def _calculate_reward(self, action):
        # Risk-adjusted reward (Sharpe-like)
        pnl = self._execute_trade(action)
        return pnl - 0.0005  # subtract transaction cost

env = DummyVecEnv([lambda: CryptoTradingEnv(df_train)])
model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, clip_range=0.2)
model.learn(total_timesteps=100_000)
```

**RL Reward Design Options:**
1. **Simple PnL:** `reward = profit_pct - fees` (tends to overtrade)
2. **Sharpe-like:** `reward = (return - risk_free) / volatility` (better risk-adjusted)
3. **Differential Sharpe:** Incremental Sharpe ratio update (Moody & Saffell, 2001)
4. **Max Drawdown penalty:** `reward = pnl - lambda * max_dd` (controls DD)

**RL Frameworks:**

| Framework | Install | Models | Notes |
|-----------|---------|--------|-------|
| **Stable-Baselines3** | `pip install stable-baselines3` | PPO, A2C, SAC, TD3, DQN | Best maintained, easiest |
| **FinRL** | `pip install finrl` | All SB3 + custom | Crypto-specific environments |
| **RLlib** (Ray) | `pip install ray[rllib]` | All + multi-agent | Heavy dependency, scalable |
| **FreqAI RL** | built-in freqtrade | PPO, TDQN | Native integration |

**Critical warning:** Gort et al. (2022, arXiv:2209.05559) showed that most published RL trading results are overfitting artifacts. Always validate with walk-forward OOS testing.

---

## CATEGORY D: Probabilistic & Bayesian (8 models)

| # | Model | Library | Use Case | Notes |
|---|-------|---------|----------|-------|
| 45 | **Hidden Markov Model (HMM)** | `hmmlearn` | Regime detection | 3-state: trend/range/volatile |
| 46 | **Gaussian Process Regression** | `sklearn.gaussian_process` | Uncertainty in predictions | O(n³), only for small N |
| 47 | **Bayesian Neural Network** | `blitz-bayesian-pytorch` | Uncertainty quantification | Weight distributions, not points |
| 48 | **MC Dropout** | PyTorch (native) | Cheap uncertainty | Run inference N times with dropout |
| 49 | **Mixture of Experts (MoE)** | Custom PyTorch | Regime-conditional prediction | Different expert per regime |
| 50 | **Bayesian Optimization** | `optuna` / `hyperopt` | Hyperparameter tuning | Not a trading model per se |
| 51 | **Quantile Regression** | `lightgbm` (built-in) | Risk estimation | Predict 5th/95th percentile |
| 52 | **Conformal Prediction** | `mapie` | Prediction intervals | Distribution-free intervals |

**HMM for regime detection (highest priority in this category):**
```python
# pip install hmmlearn
from hmmlearn import hmm
import numpy as np

# Features: daily returns + ATR + volume_ratio
obs = np.column_stack([
    df['close'].pct_change().rolling(4).mean(),  # smoothed return
    df['atr'] / df['close'],                     # normalized ATR
    df['volume'] / df['volume'].rolling(20).mean(),  # volume ratio
]).dropna()

model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(obs)
regimes = model.predict(obs)
# regime 0: low vol ranging, regime 1: trending, regime 2: high vol
```

**MC Dropout for uncertainty:**
```python
# During inference, keep dropout ON and run N forward passes
model.train()  # keeps dropout active
predictions = []
for _ in range(50):
    with torch.no_grad():
        pred = model(x_test)
        predictions.append(pred)

mean_pred = torch.stack(predictions).mean(0)
std_pred = torch.stack(predictions).std(0)
# Only trade when std_pred < threshold (high confidence)
```

**LightGBM Quantile Regression:**
```python
# Predict 5th and 95th percentile of returns
model_q05 = lgb.train({"objective": "quantile", "alpha": 0.05, ...}, train_data)
model_q95 = lgb.train({"objective": "quantile", "alpha": 0.95, ...}, train_data)

lower = model_q05.predict(X_test)
upper = model_q95.predict(X_test)
# Only trade when lower > 0 (even worst case is profitable)
```

---

## CATEGORY E: Advanced & Hybrid (12 models/approaches)

### E1. Meta-Learning

| # | Model | Library | Use Case |
|---|-------|---------|----------|
| 53 | **MAML** | `learn2learn` | Adapt to new regime with few samples |
| 54 | **Reptile** | Custom PyTorch | Simpler MAML alternative |
| 55 | **Prototypical Networks** | Custom PyTorch | Regime classification with few examples |

**MAML for regime adaptation:**
```python
# pip install learn2learn
import learn2learn as l2l

# Train on multiple regime "tasks", adapt to new regime with 5-10 bars
maml = l2l.algorithms.MAML(model, lr=0.01, first_order=True)
for task in regime_tasks:
    learner = maml.clone()
    train_loss = compute_loss(learner, task.support)
    learner.adapt(train_loss)
    eval_loss = compute_loss(learner, task.query)
    eval_loss.backward()
```

### E2. Online Learning

| # | Model | Library | Use Case |
|---|-------|---------|----------|
| 56 | **Online Gradient Descent** | `river` | Real-time model updates |
| 57 | **Adaptive Random Forest** | `river` | Concept drift handling |
| 58 | **ADWIN** | `river` | Drift detection |

```python
# pip install river
from river import tree, drift

# Incremental learning - update model bar by bar
model = tree.HoeffdingAdaptiveTreeClassifier()
drift_detector = drift.ADWIN()

for x, y in stream:
    pred = model.predict_proba_one(x)
    model.learn_one(x, y)
    drift_detector.update(int(pred != y))
    if drift_detector.drift_detected:
        model = tree.HoeffdingAdaptiveTreeClassifier()  # reset
```

### E3. Anomaly Detection (for regime changes)

| # | Model | Library | Use Case |
|---|-------|---------|----------|
| 59 | **Isolation Forest** | `sklearn` | Detect unusual market conditions |
| 60 | **DBSCAN** | `sklearn` | Cluster market states |
| 61 | **Autoencoder** | PyTorch | Reconstruct normal → high error = anomaly |

### E4. De Prado Methods (Financial ML)

| # | Method | Library | Use Case |
|---|--------|---------|----------|
| 62 | **Triple Barrier Labeling** | `mlfinlab` | Better labels than fixed-horizon |
| 63 | **Meta-Labeling** | `mlfinlab` | Filter existing signals with ML |
| 64 | **CUSUM Filter** | `mlfinlab` | Event-driven sampling |
| 65 | **Fractional Differentiation** | `mlfinlab` | Stationarity while preserving memory |
| 66 | **Sequential Bootstrap** | `mlfinlab` | Reduce sample redundancy |

### E5. AutoML

| # | Framework | Library | Notes |
|---|-----------|---------|-------|
| 67 | **AutoGluon** | `autogluon` | AutoML by Amazon, auto-stacking |
| 68 | **FLAML** | `flaml` | Fast AutoML by Microsoft |
| 69 | **auto-sklearn** | `auto-sklearn` | Auto model/feature selection |

```python
# pip install autogluon
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='target', eval_metric='roc_auc')
predictor.fit(train_data, time_limit=600, presets='best_quality')
# Auto-selects and stacks: LightGBM + CatBoost + XGBoost + NN + KNN
```

### E6. Survival Analysis & Duration

| # | Model | Library | Use Case |
|---|-------|---------|----------|
| 70 | **Cox Proportional Hazards** | `lifelines` | Predict trade duration to TP/SL |
| 71 | **Random Survival Forest** | `scikit-survival` | Non-linear survival prediction |

```python
# pip install lifelines
from lifelines import CoxPHFitter

# Predict time-to-exit for each trade
cph = CoxPHFitter(penalizer=0.1)
cph.fit(trade_data, duration_col='bars_to_exit', event_col='hit_tp')
# Hazard ratio tells which features accelerate TP hit
```

---

## MASTER COMPARISON TABLE

### By Practical Viability for OKX 15m Futures

| Rank | Model | Category | CPU-OK | Train Time | WR Gain | Complexity | Recommendation |
|------|-------|----------|--------|------------|---------|------------|----------------|
| 1 | LightGBM meta-labeling | A | Yes | 1 min | +5pp | Low | **DO NOW** |
| 2 | Stacking (LGB+ET+LR) | A | Yes | 5 min | +7pp | Medium | **DO NOW** |
| 3 | HMM regime detection | D | Yes | 10 sec | indirect | Low | **DO NOW** |
| 4 | CatBoost meta-labeling | A | Yes | 3 min | +5.5pp | Low | Test next |
| 5 | XGBoost (FreqAI) | A | Yes | 2 min | +5pp | Low | Test next |
| 6 | LGB Quantile Regression | D | Yes | 1 min | risk mgmt | Low | Test next |
| 7 | CUSUM + triple barrier | E | Yes | instant | +2pp labels | Low | **DO NOW** |
| 8 | ROCKET (tsai) | B | Yes | 30 sec | ~equal | Medium | Research |
| 9 | TCN | B | GPU pref | 15 min | +0-1pp | Medium | Research |
| 10 | TabNet | B | Yes | 10 min | +0-1pp | Medium | Research |
| 11 | TFT | B | GPU req | 60 min | +0-2pp | High | Research |
| 12 | PatchTST | B | GPU pref | 30 min | +0-1pp | Medium | Research |
| 13 | PPO (RL) | C | GPU pref | hours | unknown | Very High | Low priority |
| 14 | LSTM | B | Optional | 30 min | ~equal | Medium | Low priority |
| 15 | VAE regime | B | Optional | 15 min | indirect | High | Low priority |

### By Hardware Requirements

**CPU-only (no GPU needed):**
- All tree-based (LightGBM, XGBoost, CatBoost, RF, ET)
- Logistic Regression, SVM, KNN, Naive Bayes
- HMM, Gaussian Process (small N)
- CUSUM, triple barrier, sequential bootstrap
- River online learning models
- AutoGluon, FLAML

**GPU recommended (RTX 3060+ / 6GB VRAM):**
- LSTM, GRU, TCN (small models fit CPU, but slow)
- TabNet, FT-Transformer
- N-BEATS, N-HiTS
- PatchTST
- PPO, SAC (RL)

**GPU required (RTX 4060+ / 8GB VRAM):**
- TFT with multi-timeframe features
- Informer, TimesNet (large models)
- MAML, Prototypical Networks
- GAN for synthetic data
- Large transformers (>2M params)

---

## RECOMMENDED IMPLEMENTATION ORDER

### Phase 1: Immediate (use existing infrastructure)
1. **LightGBM meta-labeling on 15m** — filter regime_adaptive signals
2. **CUSUM filter** via `mlfinlab` — better event sampling
3. **HMM regime detection** via `hmmlearn` — 3-state market classifier
4. **Stacking ensemble** — LightGBM + ExtraTrees + LR for meta-labeling

### Phase 2: Short-term (1-2 weeks)
5. **CatBoost comparison** — ordered boosting for small-sample meta-labeling
6. **LGB quantile regression** — uncertainty-based position sizing
7. **ROCKET via tsai** — fastest DL baseline for classification

### Phase 3: Medium-term (research)
8. **TabNet** — attention-based tabular (potential tree-beater)
9. **TCN** — temporal convolutions for sequential patterns
10. **PatchTST** — efficient transformer for time series
11. **MC Dropout** — uncertainty quantification on any model

### Phase 4: Long-term (if Phase 1-3 succeed)
12. **TFT** — full temporal fusion transformer
13. **PPO** — reinforcement learning for joint optimization
14. **Online learning (River)** — real-time model adaptation
15. **AutoGluon** — automated model selection + stacking

---

## LIBRARIES TO INSTALL

```bash
# Core ML (likely already installed)
pip install lightgbm xgboost catboost scikit-learn

# Financial ML
pip install mlfinlab        # triple barrier, CUSUM, meta-labeling
pip install hmmlearn        # HMM regime detection
pip install lifelines       # survival analysis

# Deep Learning
pip install torch           # PyTorch (if not installed)
pip install pytorch-tabnet  # TabNet
pip install tsai            # ROCKET, InceptionTime, etc.
pip install neuralforecast  # PatchTST, N-BEATS, N-HiTS
pip install pytorch-forecasting  # TFT, DeepAR

# Time Series
pip install darts           # all-in-one time series (LSTM, TCN, TFT, etc.)

# Reinforcement Learning
pip install stable-baselines3[extra]  # PPO, SAC, DQN
pip install gymnasium       # gym environments

# Online Learning
pip install river           # incremental ML

# AutoML
pip install autogluon       # auto model selection
pip install flaml           # fast AutoML

# Research acceleration
pip install vectorbt        # fast backtesting
pip install optuna          # hyperparameter optimization
pip install mapie           # conformal prediction

# Uncertainty
pip install blitz-bayesian-pytorch  # Bayesian neural networks
```

---

## KEY REFERENCES

### Textbooks
- López de Prado (2018): "Advances in Financial Machine Learning" — Wiley
- López de Prado (2020): "Machine Learning for Asset Managers" — Cambridge

### Benchmark Papers
- Grinsztajn et al. (2022): "Why tree-based models still outperform deep learning on tabular data" — NeurIPS
- Gu, Kelly, Xiu (2020): "Empirical Asset Pricing via Machine Learning" — RFS
- Sebastião & Godinho (2021): "Forecasting and trading cryptocurrencies with ML" — Financial Innovation

### Model-Specific
- Lim et al. (2019): "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- Nie et al. (2023): "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (PatchTST)
- Wu et al. (2023): "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"
- Arik & Pfister (2021): "TabNet: Attentive Interpretable Tabular Learning"

### RL for Trading
- Gort et al. (2022): "Detecting Backtest Overfitting in DRL" — arXiv:2209.05559
- Moody & Saffell (2001): "Learning to Trade via Direct Reinforcement" — IEEE

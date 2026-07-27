# StockM — Full Project Documentary

> Status snapshot: **Phase 7 (Deep Learning), Session 12, Lesson 7 of 14 complete.**
> Last updated: 2026-07-27. All facts below verified against the live codebase.

---

## 0. The 30-second orientation

StockM is a self-built, end-to-end system that tries to predict **next-day stock
movement** from historical OHLCV (Open/High/Low/Close/Volume) data for NIFTY-50
tickers. It goes all the way from raw data download → feature engineering →
machine-learning baselines → hyperparameter optimization → deep-learning models
→ (planned) prediction, evaluation, backtesting, paper trading.

**Where "predicting stocks" actually happens in code** (the short answer):

| Layer | File | What it does |
|---|---|---|
| **ML prediction (works TODAY)** | [src/models/prediction.py](../src/models/prediction.py) | `predict_single_stock()` / `predict_batch()` — loads a saved model, runs it on the latest unseen row, emits BUY/HOLD/SELL |
| **DL prediction (being built, Lesson 12)** | [src/deep_learning/](../src/deep_learning/) | the four neural nets (LSTM/GRU/CNN/Transformer) + sequence builder; not yet wired into `prediction.py` |

So today you can already predict with the classical ML models. The deep-learning
models are built and trained but not yet plugged into the live prediction path —
that's Lesson 12.

---

## 1. Which phase are we in?

The project is structured in phases. Here's the honest map:

| Phase | Status | What it is |
|---|---|---|
| Phase 1 — Historical Data Collection | ✅ Done | yfinance download → `data/raw/<SYM>.csv` |
| Phase 2 — Data Validation | ✅ Done | quality checks on raw CSVs |
| Phase 3 — Feature Engineering | ✅ Done | OHLCV → ~114 features (returns, SMA/EMA, RSI, MACD, Bollinger, ATR, lags, rolling stats) |
| Phase 4 — EDA + Dataset Prep | ✅ Done | feature selection (114→40), scaling (StandardScaler, train-only), chronological train/val/test split with 1-day no-leakage gap |
| Phase 5 — ML Baseline Models | ✅ Done | LinearRegression, Ridge, DecisionTree, RandomForest, GradientBoosting, XGBoost, LightGBM per ticker |
| Phase 6 — Hyperparameter Optimization | ✅ Done | Optuna/grid search with TimeSeriesSplit; tuned models beat baselines where they could |
| **Phase 7 — Deep Learning** | 🔄 **In progress (you are here)** | LSTM, GRU, 1D CNN, Transformer; 7/14 lessons done |
| Phase 8 — Evaluation & Comparison | ⏳ Partial | baselines evaluated; full DL eval = Lessons 9–10 |
| Phase 9 — Prediction pipeline (DL) | ⏳ Lesson 12 | wire DL into `prediction.py` |
| Phase 10 — Backtesting / Paper trading | ⏳ Future | `configs/backtest_config.yaml` exists; engine not built |

**Within Phase 7**, the 14-lesson roadmap:

```
Lesson 1  Deep Learning Fundamentals          ✅ done
Lesson 2  Sequential Data Preparation          ✅ done
Lesson 3  PyTorch framework setup              ✅ done
Lesson 4  LSTM                                 ✅ done
Lesson 5  GRU                                  ✅ done
Lesson 6  1D CNN                               ✅ done
Lesson 7  Transformer                          ✅ done  ← we are here
Lesson 8  Training pipeline (early stop, LR)   ⏳ next
Lesson 9  DL evaluation (full metric suite)    ⏳
Lesson 10 Model comparison report              ⏳
Lesson 11 Save DL models                       ⏳
Lesson 12 Production prediction pipeline       ⏳
Lesson 13 Error analysis                       ⏳
Lesson 14 Final architecture review            ⏳
```

---

## 2. Where the main algorithms live (the full map)

### 2a. The data → ML chain (already built, prior phases)

```
config/tickers.csv  (51 NIFTY symbols)
   │
   ▼  src/main.py  /  src/collectors/historical_collector.py
data/raw/<SYM>.csv                        (yfinance OHLCV)
   │
   ▼  src/run_feature_pipeline.py  /  src/feature_engineering/
data/processed/features/<SYM>_features.csv   (~114 features)
   │
   ▼  src/run_eda_pipeline.py  /  src/eda/
data/prepared/<SYM>/
    ├─ train.csv  validation.csv  test.csv   (40 selected, SCALED features + 5 targets)
    ├─ feature_metadata.json   (which 40 features, split dates, scaler)
    └─ scaler_params.json      (mean/scale, fit on TRAIN only)
   │
   ▼  src/models/data_loader.py  → load_dataset()   (leakage-safe X/y split)
   │
   ▼  src/run_baseline_training.py  /  src/models/
models/saved_models/<SYM>/   (per-ticker baseline models + metadata)
   │
   ▼  src/run_optimization.py  /  src/optimization/
models/optimized/<SYM>/      (tuned models; deployed if they beat baseline)
   │
   ▼  src/models/prediction.py   ← THE PREDICTION ENTRY POINT (works today)
predict_single_stock(symbol) → {date, predicted_return, signal, realised_return, correct_direction}
```

### 2b. The deep-learning chain (built this phase, Phase 7)

```
data/prepared/<SYM>/{train,validation,test}.csv
   │
   ▼  src/models/data_loader.py  → load_dataset()           (reused — identical data)
   │
   ▼  src/deep_learning/sequence_builder.py  → build_sequences()
X_train_seq / y_train_seq / X_validation_seq / ...           (3540×30×40 windows)
   │
   ▼  src/deep_learning/framework.py   (seed, device, dataloaders, get_loss, checkpoints)
   │
   ▼  src/deep_learning/trainer.py  → train_one()           (minimal training loop, shared)
   │
   ▼  the four architectures:
      src/deep_learning/lstm.py        build_lstm()
      src/deep_learning/gru.py         build_gru()
      src/deep_learning/cnn.py         build_cnn()
      src/deep_learning/transformer.py build_transformer()
   │
   ▼  (Lesson 12 will wire these into src/models/prediction.py)
```

### 2c. Key files, one line each

| File | Role |
|---|---|
| [src/deep_learning/sequence_builder.py](../src/deep_learning/sequence_builder.py) | Converts daily rows → `(n_windows, lookback=30, n_features=40)` tensors. Leakage-safe (per-split windowing, causal target pairing). |
| [src/deep_learning/framework.py](../src/deep_learning/framework.py) | PyTorch utilities: `set_global_seed`, `get_device`, `make_dataloader`, `make_tensor_dataset`, `get_loss`, `save_checkpoint`, `load_checkpoint`, `count_parameters`. |
| [src/deep_learning/trainer.py](../src/deep_learning/trainer.py) | The shared `train_one()` training loop (minimal — Lesson 8 adds early stopping / LR scheduling / checkpointing / curves). |
| [src/deep_learning/lstm.py](../src/deep_learning/lstm.py) | `LSTMModel` + `build_lstm()`. 60k params. |
| [src/deep_learning/gru.py](../src/deep_learning/gru.py) | `GRUModel` + `build_gru()` + `compare_rnn()`. 45k params. |
| [src/deep_learning/cnn.py](../src/deep_learning/cnn.py) | `CNNModel` + `build_cnn()` + `compare_cnn()`. 20k params. |
| [src/deep_learning/transformer.py](../src/deep_learning/transformer.py) | `TransformerModel` + `build_transformer()` + `compare_all()`. 103k params. |
| [src/models/prediction.py](../src/models/prediction.py) | **The production prediction entry point** (works today for ML). `load_model`, `predict`, `predict_single_stock`, `predict_batch`. |
| [src/models/data_loader.py](../src/models/data_loader.py) | `load_dataset()` — the leakage-safe X/y split shared by ML and DL. |
| [src/models/baseline_models.py](../src/models/baseline_models.py) | Factory for the 7 classical models. |
| [configs/model_config.yaml](../configs/model_config.yaml) | Architecture params: `lstm_baseline_v1`, `transformer_v1`, `cnn_v1`. |
| [configs/feature_config.yaml](../configs/feature_config.yaml) | `windowing.lookback_window=30`, `horizon=1`; feature groups. |
| [configs/training_config.yaml](../configs/training_config.yaml) | optimizer, lr, epochs, batch_size, early stopping, checkpoints. |

---

## 3. How the logic is handled (the design principles)

1. **One leakage firewall, reused everywhere.** `data_loader.load_dataset()` is the single place that separates features (X) from targets (y) and asserts no `target_*` column leaks into X. Both the ML baselines and the DL models consume it, so they train on byte-identical data. Apples-to-apples by construction.

2. **Config-driven, not hard-coded.** Lookback (30), horizon (1), hidden_size (64), learning_rate (0.001), batch_size (64) — all read from YAML. Change a number in a config file, not in code.

3. **Factory pattern for models.** Each architecture has a `build_<name>(input_size, **overrides)` factory that reads its config block. Adding a model = one file + one config block. The training pipeline never changes.

4. **Shared trainer for fair comparison.** `trainer.train_one()` trains any `nn.Module` with identical optimizer/loss/batching. When we compare LSTM vs GRU vs CNN vs Transformer, the only difference is the model — so any gap is the architecture, not the setup.

5. **Honesty by construction.** Every evaluation reports `beats_naive` (does it beat "always predict zero"?) and `directional_accuracy` (vs the 50% coin-flip floor). The pipeline is designed to surface "no edge" rather than hide it behind a gamed metric.

---

## 4. How to implement your own logic

### 4a. Add a new deep-learning architecture (e.g. a BiLSTM-Attention hybrid)

1. **Create** `src/deep_learning/my_model.py`, mirroring [lstm.py](../src/deep_learning/lstm.py):
   ```python
   class MyModel(nn.Module):
       def __init__(self, input_size, hidden_size=64, dropout=0.2, task="regression", **kw):
           super().__init__()
           ...  # your layers
       def forward(self, x):        # x: (batch, lookback, input_size)
           ...                       # return (batch, 1)
   def build_my_model(input_size, config_path=None, **overrides):
       cfg = _load_my_config(config_path); cfg.update(overrides)
       return MyModel(input_size=input_size, **cfg)
   ```
   The only contract: `forward` takes `(batch, lookback, n_features)` and returns `(batch, 1)`.

2. **Add a config block** to [configs/model_config.yaml](../configs/model_config.yaml):
   ```yaml
   my_model_v1:
     family: deep_learning
     type: my_model
     hidden_size: 64
     dropout: 0.2
   ```

3. **Train it** with the shared trainer — no new training code needed:
   ```python
   from deep_learning.framework import set_global_seed, make_dataloader, make_tensor_dataset, load_training_config, get_device
   from deep_learning.sequence_builder import build_sequences
   from deep_learning.trainer import train_one
   from deep_learning.my_model import build_my_model

   seq = build_sequences("RELIANCE.NS")
   model = build_my_model(seq["n_features"], task="regression")
   # ... make dataloaders from seq ... then:
   result = train_one(model, train_dl, val_dl, test_dl, task="regression",
                      epochs=20, lr=0.001, weight_decay=1e-4, grad_clip=1.0, device="cpu")
   ```

### 4b. Change the prediction target (regression ↔ classification)

- **Regression** (predict next-day return value): `target_col="target_next_return"`, `task="regression"`.
- **Classification** (predict UP/DOWN): `target_col="target_direction"`, `task="classification"`.

One parameter in `build_sequences(..., target_col=...)` and `build_<model>(..., task=...)`. The sequence builder is target-agnostic.

### 4c. Change the inputs (features / lookback / horizon)

- **Lookback window**: edit `windowing.lookback_window` in [feature_config.yaml](../configs/feature_config.yaml) (default 30).
- **More feature groups**: enable `technical`/`fundamental`/`macro` under `active_groups` in feature_config (only `ohlcv` is active now — the others are reserved placeholders).
- **Longer horizon**: use `target_return_5d` as the target, or set `windowing.horizon`.

### 4d. Change the training logic

The full production trainer arrives in **Lesson 8** (early stopping, LR scheduling, checkpointing, history logging, learning curves). Today's `train_one` is the minimal loop. To add your own training behavior (e.g. custom LR schedule), you'd extend `trainer.py` — but Lesson 8 will do exactly that, so it's worth waiting.

---

## 5. How to test whether the model predicts correctly

### 5a. The honest definition of "correct"

In financial prediction, "correct" is NOT "high accuracy." Daily stock returns are
mostly noise (efficient markets). The honest bar:

- **`beats_naive = True`**: the model's test RMSE beats the naive "always predict zero" baseline. (Currently: NO model passes this.)
- **`directional_accuracy > 50%`**: it predicts UP/DOWN correctly more often than a coin flip. (Currently: ~49–52% — essentially no edge.)

**Honest current status: the models do NOT reliably predict correctly.** This is the expected, documented finding — not a bug. See [dl-baseline-findings](../C:/Users/YEHOSHUVA/.claude/projects/c--stockM/memory/dl-baseline-findings.md).

### 5b. How to run a prediction test (today, ML models)

```bash
PYTHONPATH=src python -c "
from models.prediction import predict_single_stock, predict_batch
print(predict_single_stock('RELIANCE.NS'))          # latest test row
print(predict_batch(['RELIANCE.NS','TCS.NS','INFY.NS']))
"
```
This loads the deployed (optimized) model and predicts on the latest unseen test row, returning `predicted_return`, `signal` (BUY/HOLD/SELL), `realised_return`, and `correct_direction`.

### 5c. How to run a DL model test

```bash
PYTHONPATH=src python src/deep_learning/transformer.py RELIANCE.NS   # trains all 4, prints comparison
PYTHONPATH=src python src/deep_learning/lstm.py RELIANCE.NS          # trains just the LSTM
```

### 5d. How to evaluate properly (the metrics)

Full evaluation arrives in **Lesson 9** (MAE, MSE, RMSE, R², MAPE, training/inference time, GPU/CPU usage) and **Lesson 10** (the comparison report across all 7 ML models + 4 DL models). The baselines are already evaluated in [reports/training/](../reports/training/) and [reports/training/global_leaderboard.json](../reports/training/global_leaderboard.json).

---

## 6. When do we evaluate?

- **Baselines: already evaluated** (Phase 5/6). Per-ticker reports in `reports/training/`, global leaderboard in `reports/training/global_leaderboard.json`, optimization reports in `reports/optimization/`.
- **Deep learning: Lessons 9–10** (this phase). Lesson 9 = the full metric suite per DL model. Lesson 10 = the unified comparison report (LinearRegression, Ridge, RandomForest, XGBoost, LightGBM, LSTM, GRU, CNN, Transformer) with a recommendation.

---

## 7. What's implemented vs. what's left

### ✅ Implemented

- Full data pipeline: download → validate → feature-engineer → EDA → prepared splits (51 tickers).
- 7 classical ML models per ticker, trained, evaluated, saved.
- Hyperparameter optimization (Optuna + TimeSeriesSplit), tuned models deployed.
- Production prediction for ML models (`predict_single_stock`, `predict_batch`).
- 4 deep-learning architectures (LSTM, GRU, CNN, Transformer), all built, trained, and compared on RELIANCE.
- Sequence builder, PyTorch framework, shared trainer.
- Backtest config skeleton (`configs/backtest_config.yaml`).

### ⏳ Left to implement (Phase 7, Lessons 8–14)

- **Lesson 8**: Production training pipeline (early stopping, LR scheduling, model checkpointing, training-history logging, learning-curve plots).
- **Lesson 9**: Full DL evaluation suite (MAE/MSE/RMSE/R²/MAPE, timing, resource usage).
- **Lesson 10**: Unified comparison report across all models; recommend the best for StockM.
- **Lesson 11**: Save DL models (weights, full model, optimizer state, metadata, dataset version).
- **Lesson 12**: Wire DL into `src/models/prediction.py` (`load_deep_learning_model`, `predict`, `predict_single_stock`, `predict_batch`).
- **Lesson 13**: Error analysis (residuals, market regimes, hard cases).
- **Lesson 14**: Final architecture review.

### 🔮 Future phases (beyond this session)

- **Backtesting engine** (event-driven, walk-forward) — config exists, code not built. `src/stockM/backtesting/` is an empty scaffold.
- **Paper trading** — see §8.
- **Richer features**: `technical`/`fundamental`/`text_nlp`/`macro`/`graph` groups (reserved in feature_config).
- **Reinforcement learning**, **multimodal learning**, **AI-powered trading assistants** (long-term vision).

---

## 8. When do we go to paper trading?

**Not yet — and not in this 14-lesson session.** Paper trading is a future phase.
Here's the honest prerequisite chain:

1. **A model with real edge.** Today, NO model beats the naive zero-predictor or
   clearly exceeds 50% directional accuracy. Paper-trading a no-edge model would
   just bleed transaction costs. The realistic levers to get edge first:
   classification on direction, longer horizons, richer features, pooling tickers.
2. **The DL prediction pipeline** (Lesson 12) — so the chosen model can produce
   live signals.
3. **A backtesting engine** — `configs/backtest_config.yaml` is already written
   (event-driven, walk-forward, 0.1% commission, 0.05% slippage, Sharpe/Sortino/
   max-drawdown metrics, SPY benchmark). The engine itself (`src/stockM/backtesting/`)
   is an empty scaffold — it needs to be built.
4. **Risk management** — position sizing, max drawdown limits, kill switches.

**Realistic path to paper trading:** finish Phase 7 (Lessons 8–14) → build the
backtesting engine → run walk-forward backtests on the best model → if the
backtest is profitable net of costs AND the edge survives out-of-sample → connect
to a paper-trading broker API (e.g. a paper account on a broker that supports
NIFTY equities) for live signal generation with no real money.

**Honest estimate:** paper trading is at least one full phase away. The honest
priority right now is finding a model/target/feature-set that has real,
out-of-sample edge — not deploying a no-edge model to paper.

---

## 9. The honest bottom line

The system is **architecturally complete through the DL model zoo** and
**honestly reports that it has no real predictive edge yet** on daily NIFTY
returns from OHLCV features alone. That's not a failure — it's the correct,
expected finding for efficient markets at the daily horizon. The value built so
far is a **production-grade, leakage-safe, honestly-evaluated ML/DL platform**
that will surface real edge the moment the data/target/features provide it —
rather than hiding behind an overfit, gamed metric.

The next high-leverage step is **Lesson 8 (training pipeline)**, which lets the
Transformer (and all models) train to proper convergence — the Transformer's
52.2% directional accuracy hint needs a fair, warmed-up re-evaluation before we
can say whether attention found something real.

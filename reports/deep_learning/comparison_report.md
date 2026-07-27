# StockM - Model Comparison Report: RELIANCE.NS

Target: `target_next_return` (next-day log return regression).
Evaluated on the held-out TEST split (DL aligned to 735 windowed test
dates; ML on the full 764 test rows). Same metrics as the Phase-5
baselines (`models.evaluation`) - apples-to-apples.

## Unified comparison (sorted by test RMSE, best first)

| rank | model | family | test_rmse | test_r2 | dir_acc | beats_naive | train_s | infer_s | params | interpretability |
|------|-------|--------|-----------|---------|---------|-------------|---------|----------|--------|------------------|
| 1 | naive_zero | naive | 0.0130 | -0.000 | - | yes | 0.0 | 0.000 | - | N/A (predicts zero) |
| 2 | lightgbm_optimized | classical_optimized | 0.0132 | -0.006 | 0.4938 | no | - | 0.029 | - | Medium (tree importance + SHAP) |
| 3 | random_forest | classical_baseline | 0.0141 | -0.168 | 0.5265 | no | 12.7 | - | - | Medium (feature importance, many trees) |
| 4 | gru | deep_learning | 0.0145 | -0.215 | 0.4952 | no | 63.5 | 0.148 | 45,377 | Low (opaque hidden state) |
| 5 | gradient_boosting | classical_baseline | 0.0159 | -0.491 | 0.4947 | no | 1.7 | - | - | Medium (feature importance) |
| 6 | lstm | deep_learning | 0.0162 | -0.517 | 0.4924 | no | 15.1 | 0.086 | 60,481 | Low (opaque cell state) |
| 7 | cnn | deep_learning | 0.0178 | -0.827 | 0.5048 | no | 6.9 | 0.038 | 20,161 | Low-Medium (filters somewhat readable) |
| 8 | linear_regression | classical_baseline | 0.0182 | -0.951 | 0.4801 | no | 0.0 | - | - | High (coefficients = factor loadings) |
| 9 | decision_tree | classical_baseline | 0.0388 | -7.834 | 0.5119 | no | 0.3 | - | - | High (human-readable splits) |
| 10 | transformer | deep_learning | 0.0438 | -10.090 | 0.5048 | no | 137.8 | 0.080 | 102,657 | Medium (attention weights introspectable) |

## Recommendation

**Best model by test RMSE: `lightgbm_optimized`** (RMSE 0.0132, R2 -0.006, directional accuracy 0.4938).

**Verdict: no edge - best of a no-edge field; not deployable for trading.** The winner does NOT beat the naive zero-predictor - it is merely the least-bad of a no-edge field.

**Honest caveat:** None of the models clears the two-part deployability bar
(beats_naive=True AND directional accuracy clearly >50%). The optimized
LightGBM is the best StockM currently has, but it has no real trading
edge on this task. Deploying it to live/paper trading would, at best, lose
money on transaction costs. The bottleneck is the SIGNAL (negative R2
everywhere), not the architecture.

## Trade-off analysis (accuracy vs speed vs interpretability vs cost)

**Accuracy (test RMSE / R2):** Every model has NEGATIVE R2 - all are worse
than predicting the mean. The optimized LightGBM is closest to the naive
floor (R2 -0.006); the Transformer is catastrophically overfit (R2 -10).
Negative R2 across the board means there is essentially no signal to model
in daily next-day returns from 40 OHLCV features on one ticker.

**Speed:** Classical ML trains in seconds and infers in ~30 ms. The CNN is
the DL speed champion (7 s train, 38 ms infer). The Transformer is the
slowest (138 s train) and the GRU is anomalously slow on this CPU build
(63 s). For end-of-day trading across 50 tickers, ML or CNN inference cost
is negligible; recurrent/Transformer cost is acceptable but not free.

**Interpretability:** linear_regression and decision_tree are fully
interpretable (coefficients / splits). Tree ensembles (RF, GBM, LightGBM)
offer feature importance + SHAP. The CNN's filters are somewhat readable;
the Transformer's attention weights are introspectable (a Lesson-13 lever);
LSTM/GRU cell states are opaque. For a regulated/auditable trading system,
the ML models win on explainability.

**Computational cost:** params range 0 (naive) to 103k (Transformer). A
rule of thumb: params >> training samples (3540) invites overfitting -
exactly what the Transformer's R2 -10 shows. The CNN (20k params) is the
best-parametrized DL model and correspondingly overfit least among DL.

**Cost-benefit verdict:** The optimized ML model gives the best accuracy
AND the best speed AND the best interpretability AND the lowest cost. DL
loses on every axis for THIS task. DL's potential value is forward-looking
(richer features, pooled tickers, multimodal) - not for beating ML on
single-ticker OHLCV daily returns.

## Path to real edge (non-architecture levers)

1. **Classification on `target_direction`** (UP/DOWN) instead of return
   magnitude - the configured primary; matches the model_config.yaml
   `output_size: 1 binary: UP probability` intent.
2. **Longer horizon** (`target_return_5d`) - 5-day returns are less noisy
   than daily.
3. **Richer features** - enable the `technical`/`fundamental`/`macro`
   groups (only `ohlcv` is active); more signal per row.
4. **Pool tickers** - train one model across 50 NIFTY names for ~50x more
   data; DL's capacity advantage emerges at scale.
5. **Tune signal thresholds vs transaction costs** - even a 51% directional
   model can be profitable if you only act on high-confidence calls.

## Artifacts

- ML baselines: `reports/training/<SYM>_baseline_report.json`
- Optimized ML: `models/optimized/<SYM>/` (deployed via best_optimized.json)
- DL checkpoints: `models/checkpoints/<SYM>_<model>.pt` (Lesson 8)
- DL learning curves: `reports/deep_learning/<SYM>_<model>_curves.png`
- This report: `reports/deep_learning/comparison_report.md` (+ .json)
"""
StockM v1.0 - Phase 5: Machine Learning Baseline Models
=======================================================

Trains, evaluates, compares, saves, and serves classical baseline models on
the prepared datasets from Phase 4. These baselines are the benchmark that
future Deep Learning models (Phase 6) must beat.

Package layout (one responsibility per module - SOLID):
    data_loader.py         - load prepared train/val/test, split X/y (leakage-safe)
    baseline_models.py     - model factory: Linear / DecisionTree / RandomForest /
                             GradientBoosting (XGBoost + LightGBM optional)
    evaluation.py          - MAE / MSE / RMSE / R^2 / MAPE + naive baseline
    feature_importance.py  - per-model + consensus feature importance
    error_analysis.py      - residual distribution, bias, variance, autocorr
    comparison.py          - model comparison table + best-model recommendation
    model_storage.py       - save/load models + versioned metadata
    prediction.py          - load_model / predict / predict_single_stock / predict_batch
    training_pipeline.py   - orchestrator: load -> train -> evaluate -> compare -> save

Design contract
---------------
- Target (default): ``target_next_return`` (regression). Returns are stationary,
  unlike raw price. The target is configurable.
- X/y separation is enforced in code: no ``target_*`` column ever enters X.
- Prepared data is ALREADY scaled (StandardScaler, fit on train in Phase 4);
  no rescaling happens here. Inference reuses the saved scaler params.
- Model selection uses the VALIDATION split; the TEST split is touched once
  for the honest report.
- Per-ticker models: each ticker gets its own model + scaler + feature list,
  self-contained under models/saved_models/<SYMBOL>/.

Boosting note
-------------
XGBoost and LightGBM are optional. If installed, they are auto-included; if
not, sklearn's HistGradientBoostingRegressor is the dependency-free boosting
baseline. This keeps the pipeline runnable without extra installs.
"""

from models.training_pipeline import BaselineTrainer
from models.prediction import predict_single_stock, predict_batch, load_model

__all__ = [
    "BaselineTrainer",
    "predict_single_stock",
    "predict_batch",
    "load_model",
]

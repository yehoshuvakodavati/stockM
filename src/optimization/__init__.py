"""
StockM v1.0 - Phase 6: Hyperparameter Optimization & Time-Series Validation
============================================================================

Optimizes the Phase 5 baseline models, validates them correctly for
time-series data (never random K-Fold), and selects a production-ready
optimized model with reproducible experiments.

Package layout (one responsibility per module - SOLID):
    search_spaces.py          - configurable per-model hyperparameter spaces
    time_series_validation.py - TimeSeriesSplit / expanding / rolling window CV
    hyperparameter_optimizer.py - grid / random / Bayesian (optuna) search
    validation_curves.py      - learning + validation curves (bias/variance)
    model_selection.py        - baseline vs optimized comparison + best pick
    robustness.py             - regime-segment evaluation (bull/bear/sideways)
    experiment_tracker.py     - reproducible experiment logs (CSV + JSON)
    model_saver.py            - save optimized models + versioned metadata
    optimization_pipeline.py  - orchestrator tying it all together

Anti-leakage contract (the spine of this phase)
-----------------------------------------------
- Hyperparameters are chosen using TIME-SERIES CV or the held-out
  validation split, never the test set. Tuning on test = overfitting to test.
- Every CV split keeps train strictly before validation in time.
- The test split is touched ONCE for the honest report, never for selection.
- Every experiment is logged with seed + params + data version so a run is
  reproducible months later.

Bayesian optimization uses optuna when installed (auto-detected); if not,
the pipeline falls back to randomized search and logs that it did.
"""

from optimization.optimization_pipeline import OptimizationPipeline

__all__ = ["OptimizationPipeline"]

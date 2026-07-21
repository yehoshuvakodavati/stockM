"""
StockM v1.0 - Phase 4, Lesson 10
Feature Scaling
===============

Scaling puts features on a comparable numeric footing. Whether it matters
depends on the model:

Requires scaling
----------------
- Gradient-based models (neural nets, logistic regression): features with
  large ranges dominate the loss landscape and gradients become ill-conditioned.
- Distance-based models (KNN, SVM with RBF, k-means): distance is dominated
  by the largest-scale feature.
- Regularised linear models (Ridge/Lasso/ElasticNet): the penalty is applied
  per-coefficient, so unscaled features get penalised unfairly.

Does NOT require scaling
------------------------
- Tree-based models (Random Forest, XGBoost, LightGBM): splits are
  threshold-based and monotonic - scale is irrelevant. (They are, however,
  fine to feed scaled data.)

Three scalers
-------------
StandardScaler : (x - mean) / std. Assumes ~normal; sensitive to outliers.
MinMaxScaler   : (x - min) / (max - min) -> [0, 1]. Sensitive to outliers
                 (one extreme sets the range). Good for bounded signals.
RobustScaler   : (x - median) / IQR. Robust to outliers - the right default
                 for fat-tailed financial data.

Iron rule (anti-leakage)
------------------------
The scaler is FIT ON TRAIN ONLY. The mean/std/min/max/IQR are computed from
the training split and then *applied* (transform) to val and test. Fitting on
the full dataset leaks future statistics into training. This module enforces
that by construction: you call ``fit`` on train, then ``transform`` on each
split. We persist the fitted parameters so the same transform can be applied
at inference time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

_SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class FeatureScaler:
    """Fit-on-train scaler that remembers its parameters for reuse.

    Wraps an sklearn scaler so we can (a) exclude targets from scaling,
    (b) persist the parameters as JSON, and (c) guarantee fit-on-train-only.
    """

    def __init__(self, method: str = "robust") -> None:
        if method not in _SCALER_MAP:
            raise ValueError(f"Unknown scaler: {method}. Use one of {list(_SCALER_MAP)}.")
        self.method = method
        self._scaler = _SCALER_MAP[method]()
        self._columns: list[str] = []
        self._fitted = False

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "FeatureScaler":
        """Fit the scaler on the TRAIN split's feature columns only."""
        self._columns = list(feature_cols)
        # Fit on the selected feature columns; targets are never scaled.
        self._scaler.fit(train_df[feature_cols].to_numpy())
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transform; returns a new frame with scaled features."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit(train_df) first.")
        out = df.copy()
        out[self._columns] = self._scaler.transform(df[self._columns].to_numpy())
        return out

    def fit_transform(self, train_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        return self.fit(train_df, feature_cols).transform(train_df)

    def params(self) -> dict[str, Any]:
        """Return the fitted parameters as a JSON-serialisable dict.

        Lets us persist the scaler alongside the prepared dataset and rebuild
        it at inference time without re-fitting (which would require the
        training data).
        """
        if not self._fitted:
            raise RuntimeError("Scaler not fitted.")
        p: dict[str, Any] = {"method": self.method, "columns": self._columns}
        s = self._scaler
        if self.method == "standard":
            p["mean_"] = [float(v) for v in s.mean_]
            p["scale_"] = [float(v) for v in s.scale_]
        elif self.method == "minmax":
            p["min_"] = [float(v) for v in s.min_]
            p["scale_"] = [float(v) for v in s.scale_]  # = (max - min)
        elif self.method == "robust":
            p["center_"] = [float(v) for v in s.center_]
            p["scale_"] = [float(v) for v in s.scale_]
        return p

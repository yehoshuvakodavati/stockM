"""
StockM v1.0 - Phase 5, Lesson 13
Error Analysis
==============

Metrics tell you *how wrong* the model is on average. Error analysis tells
you *where and how* it is wrong - the difference between "RMSE = 0.018" and
"the model systematically overshoots on down-days and misses 40% of the
biggest moves." This is how professionals debug ML models.

We analyse RESIDUALS = y_true - y_pred (the signed error).

What each diagnostic means
--------------------------
residual_mean (bias)   Mean residual. ~0 = unbiased. Strongly negative =>
                       model systematically over-predicts (predicts higher
                       returns than realised). Non-zero bias is a calibration
                       bug, often a regularisation or target-centring issue.
residual_std           Dispersion of errors. The "typical" error magnitude.
                       Compare to the target's own std: if residual_std ~= y_std,
                       the model explains nothing.
residual_skew / kurt   Shape. Fat-tailed residuals (kurt>3) mean a few
                       spectacularly bad predictions - usually crash days.
autocorr_lag1          Do errors cluster? If today's error predicts tomorrow's
                       (|autocorr| high), there is structure the model missed -
                       a missing feature or a regime it can't handle. Near-zero
                       residuals that look like white noise = the model
                       extracted what it could.
high/low error rows    The specific days the model got most wrong / most right.
                       Inspecting these by hand (what was the market doing?)
                       is where the real learning happens.
error_by_sign          Is the model worse on UP days or DOWN days? Asymmetric
                       error is common (markets fall faster than they rise).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats as sp_stats
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def residual_analysis(
    y_true, y_pred, X: pd.DataFrame | None = None, n_examples: int = 5
) -> dict[str, Any]:
    """Full residual diagnostic for one model's predictions.

    Args:
        y_true:     Ground-truth target.
        y_pred:     Model predictions.
        X:          Optional feature frame (for high/low-error row context).
        n_examples: Number of high/low-error rows to surface.

    Returns:
        Dict of bias, dispersion, shape, autocorrelation, and example rows.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    residuals = y_true - y_pred

    out: dict[str, Any] = {}

    # --- Bias & dispersion ------------------------------------------------
    out["residual_mean"] = float(np.mean(residuals))            # bias
    out["residual_std"] = float(np.std(residuals, ddof=1))
    out["residual_min"] = float(np.min(residuals))
    out["residual_max"] = float(np.max(residuals))
    out["target_std"] = float(np.std(y_true, ddof=1))
    # Variance-explained sanity (squared correlation, independent of bias).
    if np.std(y_pred) > 0:
        out["corr_pred_actual"] = float(np.corrcoef(y_pred, y_true)[0, 1])
    else:
        out["corr_pred_actual"] = 0.0

    # --- Shape (skew / kurtosis) -----------------------------------------
    if len(residuals) >= 8 and np.std(residuals) > 0:
        out["residual_skew"] = float(pd.Series(residuals).skew())
        out["residual_kurtosis"] = float(pd.Series(residuals).kurtosis())
    else:
        out["residual_skew"] = None
        out["residual_kurtosis"] = None

    # --- Error autocorrelation (lag 1) -----------------------------------
    if len(residuals) >= 10:
        r = residuals - residuals.mean()
        denom = np.sum(r * r)
        if denom > 0:
            out["autocorr_lag1"] = float(np.sum(r[1:] * r[:-1]) / denom)
        else:
            out["autocorr_lag1"] = 0.0
    else:
        out["autocorr_lag1"] = None

    # --- Asymmetric error by true sign -----------------------------------
    up = y_true > 0
    out["mae_up_days"] = float(np.mean(np.abs(residuals[up]))) if up.any() else None
    out["mae_down_days"] = (
        float(np.mean(np.abs(residuals[~up]))) if (~up).any() else None
    )

    # --- Extreme-error examples ------------------------------------------
    order = np.argsort(np.abs(residuals))[::-1]  # worst first
    worst_idx = order[:n_examples]
    best_idx = order[-n_examples:][::-1]

    def _row(i: int) -> dict[str, Any]:
        rec = {
            "index": str(X.index[i]) if (X is not None and hasattr(X, "index")) else int(i),
            "y_true": float(y_true[i]),
            "y_pred": float(y_pred[i]),
            "residual": float(residuals[i]),
        }
        return rec

    out["highest_error_rows"] = [_row(i) for i in worst_idx]
    out["lowest_error_rows"] = [_row(i) for i in best_idx]

    # --- Normality of residuals (optional) -------------------------------
    if _HAVE_SCIPY and len(residuals) >= 20:
        try:
            _, p = sp_stats.normaltest(residuals)
            out["residuals_normal_p"] = float(p)
        except Exception:  # pragma: no cover
            out["residuals_normal_p"] = None

    return out

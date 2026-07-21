"""
StockM v1.0 - Phase 4: Dataset Preparation & EDA
================================================

Transforms the AI-ready feature dataset (from Phase 3) into a clean,
analysed, ML-ready dataset split into train / validation / test.

Package layout (one module per EDA concern - Single Responsibility):
    dataset_overview.py   - shape, dtypes, missing, duplicates, memory
    statistics.py         - descriptive statistics (mean/median/std/quartiles)
    distributions.py      - skewness, kurtosis, normality
    correlation.py        - Pearson/Spearman, multicollinearity, redundancy
    outliers.py           - IQR + Z-score detection
    target_analysis.py    - target distribution & class imbalance
    leakage_detection.py  - future/target/feature leakage audits
    feature_selection.py  - variance, mutual info, correlation, tree importance
    scaling.py            - Standard/MinMax/Robust scalers (fit on train only)
    splitting.py          - chronological train/val/test split
    eda_pipeline.py       - orchestrator: analyse -> select -> scale -> split -> save

Anti-leakage contract
---------------------
Every *decision* (which features to keep, scaler parameters, outlier
thresholds used for removal) is fit on the TRAIN split only and then applied
to val/test. Exploratory statistics may be computed on the full dataset for
*understanding*, but anything that becomes a preprocessing parameter is
train-derived. This honours data_config.yaml: scaling.fit_on: train_only,
splits.gap_days: 1.
"""

from eda.eda_pipeline import EDAPipeline

__all__ = ["EDAPipeline"]

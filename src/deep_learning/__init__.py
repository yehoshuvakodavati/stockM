"""
StockM v1.0 - Phase 7 (Deep Learning)
=====================================

Sequence models for financial time-series forecasting.

This package is a *parallel modeling track* to ``src/models`` (classical ML).
It consumes the SAME prepared datasets (``data/prepared/<SYM>/``) produced by
the EDA pipeline, reshaped here into temporal windows for recurrent /
convolutional / attention architectures.

Layering (one responsibility per module):
    sequence_builder.py  - row -> window reshape (leakage-safe, config-driven)
    (later) architectures - LSTM / GRU / CNN / Transformer model factories
    (later) training      - early stopping, checkpointing, LR scheduling
    (later) inference     - load + predict, integrated with src/prediction

Design contract with the rest of StockM
---------------------------------------
* Identical data: sequences are built from ``models.data_loader.load_dataset``,
  so DL trains on the exact rows/target/split the baselines used.
* Identical north stars: a DL model is deployed only if it beats the naive
  zero-predictor AND clears ~50% directional accuracy on the TEST split.
"""

from __future__ import annotations

from deep_learning.sequence_builder import (
    SequenceConfig,
    build_sequences,
)
from deep_learning.framework import (
    count_parameters,
    device_summary,
    get_device,
    load_checkpoint,
    load_training_config,
    make_dataloader,
    make_tensor_dataset,
    save_checkpoint,
    set_deterministic,
    set_global_seed,
)

__all__ = [
    # sequence building (Lesson 2)
    "SequenceConfig",
    "build_sequences",
    # framework utilities (Lesson 3)
    "count_parameters",
    "device_summary",
    "get_device",
    "load_checkpoint",
    "load_training_config",
    "make_dataloader",
    "make_tensor_dataset",
    "save_checkpoint",
    "set_deterministic",
    "set_global_seed",
]

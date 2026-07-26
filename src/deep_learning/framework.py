"""
StockM v1.0 - Phase 7, Lesson 3
PyTorch Framework Setup & Reusable Utilities
============================================

The shared foundation every deep-learning module in StockM builds on:
reproducibility, device selection, sequence -> DataLoader plumbing, and
checkpoint save/load. Deliberately *primitive* - the training loop, early
stopping, LR scheduling and history logging arrive in Lesson 8; the evaluation
metrics in Lesson 9; the prediction API in Lesson 12. This module owns only the
boilerplate that does not depend on a specific architecture.

Framework choice: PyTorch (NOT TensorFlow/Keras)
-------------------------------------------------
The roadmap defaulted to TensorFlow/Keras, but the runtime is Python 3.14,
for which no TensorFlow wheel exists. PyTorch 2.13 ships cp314 wheels and is
what requirements.txt / pyproject.toml already declare. PyTorch covers every
architecture in the roadmap (LSTM/GRU/CNN/Transformer) natively, so the switch
is environment-forced and lossless. The model_config fields (hidden_size,
num_layers, dropout, bidirectional) map directly onto torch.nn modules.

Regression vs classification (the Lesson-2 fork, resolved here)
---------------------------------------------------------------
The framework is target-agnostic at the data layer (a DataLoader carries
whatever y the sequence builder produced). The OUTPUT HEAD + LOSS are chosen
per architecture in Lessons 4-7. Default decision for the primary DL models:

    * target_direction   (binary UP/DOWN)  -> sigmoid head + BCE loss
    * target_next_return (regression)      -> linear head + MSE loss  (switch)

Rationale: data_config.yaml declares target type=binary_direction; the features
were selected against target_direction (feature_metadata.json); and the honest
baseline finding (ML barely beats naive-zero on next_return, ~50% directional
floor) recommends predicting direction over magnitude. Directional accuracy -
which BOTH regression (sign of pred) and classification (threshold of prob) can
produce - remains the unifying north star for the Lesson-10 comparison.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Optional dependency, auto-detected at import (same pattern as baseline_models
# uses for xgboost/lightgbm). The module imports cleanly without torch so that
# config/inspection code works on machines where torch is not yet installed;
# only the torch-dependent calls raise a clear error.
try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training_config.yaml"

logger = logging.getLogger("stockm.deep_learning.framework")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed: int = 42) -> None:
    """Seed every RNG that affects DL training (python, numpy, torch).

    Why professionals do this: a "win" you cannot reproduce is noise. DL
    weight initialisation is random; without a fixed seed a 0.001 RMSE
    improvement is indistinguishable from luck. Mirrors the seed discipline
    the classical baselines already follow (random_state=42 everywhere).

    Args:
        seed: Integer seed. 42 matches the rest of StockM.
    """
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover (CPU env)
            torch.cuda.manual_seed_all(seed)
    logger.info("global seed set: %d", seed)


def set_deterministic(enabled: bool = True) -> None:
    """Toggle torch's deterministic-algorithms mode (full reproducibility).

    Deterministic mode trades a little speed for bit-exact reproducibility -
    useful when comparing architectures. Some CUDA ops have no deterministic
    implementation and will then raise; on CPU (StockM's default) it is free.
    """
    if not _HAS_TORCH:
        return
    torch.use_deterministic_algorithms(enabled, warn_only=True)
    if enabled:
        # cudnn determinism flag (no-op on CPU, harmless).
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device(preference: str = "auto") -> str:
    """Return the torch device string to train on.

    Args:
        preference: "auto" (cuda if available else cpu), "cpu", or "cuda".

    Returns:
        One of "cuda" / "cpu". Falls back to "cpu" if torch is absent or no GPU.
    """
    if not _HAS_TORCH or preference == "cpu":
        return "cpu"
    if preference == "cuda" and torch.cuda.is_available():  # pragma: no cover
        return "cuda"
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"  # pragma: no cover
    logger.warning("unknown device preference %r, falling back to cpu", preference)
    return "cpu"


def device_summary(device: str | None = None) -> dict[str, Any]:
    """Report the runtime device + torch/CUDA versions for logging/metadata."""
    dev = device or get_device()
    info: dict[str, Any] = {"device": dev, "torch_available": _HAS_TORCH}
    if _HAS_TORCH:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()  # pragma: no cover
        if torch.cuda.is_available():  # pragma: no cover
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


# ---------------------------------------------------------------------------
# Sequence -> DataLoader plumbing (the "dataset loading" utility)
# ---------------------------------------------------------------------------
def _require_torch() -> None:
    """Raise a clear, actionable error if torch is not installed."""
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is not installed. Install it with:\n"
            "  python -m pip install --user torch "
            "--index-url https://download.pytorch.org/whl/cpu"
        )


def make_tensor_dataset(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    dtype_x: Any = None,
    dtype_y: Any = None,
) -> "TensorDataset":
    """Wrap (X_seq, y_seq) numpy arrays into a torch TensorDataset.

    Casts features to float32 and targets to a dtype appropriate to their
    semantics: float32 for regression (target_next_return), int64 for
    classification (target_direction). The caller can override both.

    Args:
        X_seq:   (n_windows, lookback, n_features) feature array.
        y_seq:   (n_windows,) target array.
        dtype_x: Override feature dtype (default float32).
        dtype_y: Override target dtype. If None, inferred: int64 when y is
                 binary {0,1}, else float32.

    Returns:
        A TensorDataset of (features, targets) tensors.
    """
    _require_torch()
    fx = dtype_x if dtype_x is not None else torch.float32
    if dtype_y is not None:
        fy = dtype_y
    else:
        # Infer: binary 0/1 -> classification (long); else regression (float).
        uniq = set(np.unique(y_seq).tolist())
        fy = torch.long if uniq.issubset({0, 1}) else torch.float32

    X_t = torch.as_tensor(np.asarray(X_seq), dtype=fx)
    y_t = torch.as_tensor(np.asarray(y_seq), dtype=fy)
    # Targets stay 1-D (n_windows,): MSE/BCEWithLogits both accept 1-D targets;
    # classification heads reshape internally if they need a class dim.
    return TensorDataset(X_t, y_t)


def make_dataloader(
    dataset: "TensorDataset",
    batch_size: int = 64,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> "DataLoader":
    """Build a DataLoader with StockM-safe defaults.

    Args:
        dataset:     A TensorDataset from make_tensor_dataset.
        batch_size:  Mini-batch size (training_config.yaml -> loop.batch_size).
        shuffle:     True ONLY for train. Never shuffle val/test (walk-forward
                     order must be preserved for honest evaluation).
        drop_last:   Drop the final incomplete batch (train only; keeps batch
                     statistics stable, slight data loss).
        num_workers: 0 on Windows (multiprocessing spawn overhead > benefit at
                     StockM scale). >0 only for large pooled datasets.

    Returns:
        A configured torch DataLoader.
    """
    _require_torch()
    device = get_device()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),  # pragma: no cover (CPU env)
    )


# ---------------------------------------------------------------------------
# Checkpointing primitive (Lesson 11 extends with full metadata/versioning)
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer | None" = None,
    epoch: int = 0,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a training checkpoint (weights + optimizer state + progress).

    Saves state_dicts (not the whole object) so a checkpoint reloads even if
    the model class moves - the architecture is rebuilt from code, then
    weights are restored. This is the standard PyTorch contract and the
    reason we save state_dict rather than pickle the model.

    Args:
        path:       Destination .pt/.pth file.
        model:      The nn.Module whose weights to save.
        optimizer:  Optimizer whose state to save (needed to resume training
                    exactly; None for inference-only checkpoints).
        epoch:      Epoch index reached (for resuming).
        extra:      Arbitrary metadata (config, metrics, target_col, ...).

    Returns:
        The resolved path written.
    """
    _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    logger.info("checkpoint saved -> %s (epoch %d)", path, epoch)
    return path


def load_checkpoint(
    path: str | Path,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer | None" = None,
    map_location: str | None = None,
) -> dict[str, Any]:
    """Load a checkpoint into an already-constructed model (+optimizer).

    The model architecture MUST be rebuilt in code first (same class + same
    hyperparameters); this only restores weights. Returns the checkpoint's
    metadata dict (epoch, extra, ...) for logging/resumption.

    Args:
        path:          Checkpoint file to read.
        model:         nn.Module with the SAME architecture as when saved.
        optimizer:     If given, restore its state too (resume training).
        map_location:  Device to map tensors to (default: current device).

    Returns:
        {"epoch": int, "extra": dict, ...} from the checkpoint.
    """
    _require_torch()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    loc = map_location or get_device()
    payload = torch.load(path, map_location=loc, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    logger.info("checkpoint loaded <- %s (epoch %s)", path, payload.get("epoch"))
    return {"epoch": payload.get("epoch", 0), "extra": payload.get("extra", {})}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def count_parameters(model: "torch.nn.Module", trainable_only: bool = True) -> int:
    """Return the parameter count of a model (capacity vs. data sanity check).

    A model with more parameters than training samples is almost guaranteed to
    memorise noise - the key overfitting risk for StockM's ~3500 train windows.
    """
    _require_torch()
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Loss selection (task-based, shared by every architecture).
# ---------------------------------------------------------------------------
def get_loss(task: str) -> "torch.nn.Module":
    """Return the loss module matching the forecasting task.

    Architecture-agnostic: LSTM / GRU / CNN / Transformer all call this, so the
    task -> loss mapping lives here, not in each model file.

    regression     -> MSELoss (same yardstick as the classical baselines).
    classification -> BCEWithLogitsLoss (stable; expects float targets in {0,1}).
    """
    _require_torch()
    if task == "regression":
        return torch.nn.MSELoss()
    if task == "classification":
        return torch.nn.BCEWithLogitsLoss()
    raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")


# ---------------------------------------------------------------------------
# Config loading: the single source of truth for trainer hyperparameters.
# ---------------------------------------------------------------------------
def load_training_config(config_path: Path | None = None) -> dict[str, Any]:
    """Read training_config.yaml with graceful defaults.

    Returns a flat-ish dict with the keys the framework/trainer needs
    (device, batch_size, epochs, lr, checkpoint dir, early-stopping, ...).
    Missing keys fall back to sensible defaults so a trimmed config still runs.
    """
    path = config_path or TRAINING_CONFIG
    cfg: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    trainer = cfg.get("trainer", {})
    loop = cfg.get("loop", {})
    opt = cfg.get("optimization", {})
    ckpt = cfg.get("checkpoints", {})
    es = cfg.get("early_stopping", {})
    return {
        "device": trainer.get("device", "auto"),
        "mixed_precision": trainer.get("mixed_precision", False),
        "epochs": int(loop.get("epochs", 50)),
        "batch_size": int(loop.get("batch_size", 64)),
        "gradient_clip_norm": float(loop.get("gradient_clip_norm", 1.0)),
        "optimizer": opt.get("optimizer", "adam"),
        "learning_rate": float(opt.get("learning_rate", 1e-3)),
        "weight_decay": float(opt.get("weight_decay", 1e-4)),
        "lr_scheduler": opt.get("lr_scheduler", "reduce_on_plateau"),
        "scheduler_params": opt.get("scheduler_params", {"factor": 0.5, "patience": 5}),
        "checkpoint_dir": PROJECT_ROOT / ckpt.get("save_dir", "models/checkpoints"),
        "early_stopping": {
            "enabled": bool(es.get("enabled", True)),
            "monitor": es.get("monitor", "val_loss"),
            "mode": es.get("mode", "min"),
            "patience": int(es.get("patience", 10)),
            "min_delta": float(es.get("min_delta", 1e-4)),
        },
    }


# ---------------------------------------------------------------------------
# Smoke test: confirm torch imports + a tensor round-trips, using the real
# RELIANCE sequences from Lesson 2. Run:
#   python src/deep_learning/framework.py
# (needs `src/` on sys.path, same as the other runners)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    print("=== framework smoke test ===")
    print("torch_available:", _HAS_TORCH)
    if not _HAS_TORCH:
        print("torch NOT installed - install it, then re-run.")
        sys.exit(1)

    set_global_seed(42)
    print("device_summary:", device_summary())

    from deep_learning.sequence_builder import build_sequences  # noqa: E402

    seq = build_sequences("RELIANCE.NS")
    # Regression target (float) path:
    ds_reg = make_tensor_dataset(seq["X_train_seq"], seq["y_train_seq"])
    dl_reg = make_dataloader(ds_reg, batch_size=64, shuffle=True, drop_last=True)
    xb, yb = next(iter(dl_reg))
    print(f"train batch: X={tuple(xb.shape)} dtype={xb.dtype} | y={tuple(yb.shape)} dtype={yb.dtype}")

    # Classification target (int) path - switch the fork:
    seq_c = build_sequences("RELIANCE.NS", target_col="target_direction")
    ds_cls = make_tensor_dataset(seq_c["X_train_seq"], seq_c["y_train_seq"])
    xb_c, yb_c = next(iter(make_dataloader(ds_cls, batch_size=64)))
    print(
        f"classification batch: y dtype={yb_c.dtype} "
        f"(binary classes: {yb_c.unique().tolist()})"
    )

    # Checkpoint round-trip on a tiny model:
    model = torch.nn.Linear(seq["n_features"], 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt = save_checkpoint(
        PROJECT_ROOT / "models" / "checkpoints" / "_framework_smoke.pt",
        model, opt, epoch=0,
        extra={"target_col": "target_next_return", "n_features": seq["n_features"]},
    )
    meta = load_checkpoint(ckpt, model, opt)
    print("checkpoint round-trip OK | meta:", meta)
    print("linear params:", count_parameters(model))
    print("\nFRAMEWORK SMOKE TEST PASSED.")

"""
StockM v1.0 - Phase 7, Lesson 11
Deep-Learning Model Storage & Versioning
=======================================

The DL analogue of models/model_storage.py: makes every trained neural net a
self-describing, reloadable artifact. A saved model is traceable to its data,
target, features, hyperparameters, and metrics - months later, you can tell
exactly what produced a given weights file.

Layout
------
    models/deep_learning/<SYMBOL>/
        <model_name>.pt                  # state_dict + optimizer state + arch config
        <model_name>_metadata.json       # full provenance + metrics
        <model_name>_history.json        # per-epoch training history
        best_dl_model.json               # name of the deployed DL model

What we save (the roadmap's six artifacts)
------------------------------------------
1. Model weights      -> the state_dict inside the .pt (the portable contract)
2. Complete model     -> arch_type + arch_kwargs in metadata, rebuilt on load.
                         (Modern PyTorch = arch config + state_dict, NOT a pickle.
                         Pickling the whole model breaks when code moves; we save
                         the rebuild recipe instead.)
3. Optimizer state    -> inside the .pt (needed to resume training exactly)
4. Training history   -> <name>_history.json (for reproducibility + re-plotting)
5. Metadata           -> <name>_metadata.json (provenance: data, target, features,
                         hyperparams, metrics, scaler ref, dataset version)
6. Dataset version    -> embedded in metadata (split row counts, date ranges,
                         method) + a scaler_ref path to the prepared scaler params

Why state_dict, not pickle
--------------------------
torch.save(model) pickles the class - reload breaks if the class moves/renames
or its signature changes. Saving state_dict + arch config means the architecture
is rebuilt FROM CODE (always current), then weights are restored. This is the
standard PyTorch contract and what framework.save_checkpoint already does.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from deep_learning.framework import load_checkpoint, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DL_MODELS_DIR = PROJECT_ROOT / "models" / "deep_learning"

logger = logging.getLogger("stockm.deep_learning.storage")

STOCKM_VERSION = "1.0"

# Architecture registry -> the factory that rebuilds each model type from kwargs.
# Lazy import inside the loader to avoid forcing torch at import time of this
# module's consumers that only build metadata.
_ARCH_TYPES = ("lstm", "gru", "cnn", "transformer")


def _safe_filename(symbol: str) -> str:
    return symbol.replace(".", "_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(obj: Any) -> Any:
    """Coerce numpy/torch types to JSON-serialisable Python types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if v != v else v  # NaN -> None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float):
        return None if obj != obj else obj
    return obj


def build_dl_metadata(
    *,
    model_name: str,
    symbol: str,
    arch_type: str,
    arch_kwargs: dict[str, Any],
    target_col: str,
    feature_names: list[str],
    training: dict[str, Any],
    metrics: dict[str, Any],
    dataset_info: dict[str, Any],
    scaler_ref: str | None = None,
    is_best: bool = False,
    model_version: str = "dl_v1",
) -> dict[str, Any]:
    """Assemble the full provenance/metadata record for a trained DL model.

    Mirrors models.model_storage.build_metadata, with DL-specific fields
    (arch_type, arch_kwargs, training history ref, optimizer-saved flag).
    """
    if arch_type not in _ARCH_TYPES:
        raise ValueError(f"arch_type must be one of {_ARCH_TYPES}, got {arch_type!r}")
    return {
        "model_name": model_name,
        "symbol": symbol,
        "model_version": model_version,
        "stockm_version": STOCKM_VERSION,
        "trained_at": _now_iso(),
        "arch_type": arch_type,                # how to rebuild the architecture
        "arch_kwargs": _jsonable(arch_kwargs),  # the rebuild recipe
        "target_col": target_col,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "training": _jsonable(training),       # epochs_run, best_val_loss, best_epoch, train_time_s
        "metrics": _jsonable(metrics),         # val + test rmse/r2/mae/dir_acc/beats_naive
        "dataset": _jsonable(dataset_info),    # row_counts, date_ranges, split_method, dataset_version
        "scaler_ref": scaler_ref,
        "optimizer_state_saved": True,
        "is_best": is_best,
    }


def dataset_info_from_sequences(seq: dict[str, Any]) -> dict[str, Any]:
    """Extract the dataset-version block from a build_sequences() result.

    Captures the exact data the model saw: row counts per split, date ranges,
    lookback/horizon, and a dataset_version string (date-range hash).
    """
    counts = seq.get("window_counts", {})
    ranges = seq.get("date_ranges", {})
    return {
        "lookback": seq.get("lookback"),
        "horizon": seq.get("horizon"),
        "stride": seq.get("stride"),
        "window_counts": counts,
        "date_ranges": ranges,
        "split_method": "chronological_gap1",
        "dataset_version": f"{ranges.get('train', ('?',))[0]}_{ranges.get('test', ('?',))[1]}",
    }


def save_dl_model(
    model: "torch.nn.Module",
    *,
    symbol: str,
    model_name: str,
    arch_type: str,
    arch_kwargs: dict[str, Any],
    history: list[dict[str, float]],
    metadata: dict[str, Any],
    optimizer: "torch.optim.Optimizer | None" = None,
    output_dir: Path | None = None,
) -> Path:
    """Persist a trained DL model + its provenance. Returns the checkpoint path.

    Writes three files under models/deep_learning/<SYMBOL>/:
        <model_name>.pt                 (state_dict + optimizer + arch config)
        <model_name>_metadata.json       (provenance)
        <model_name>_history.json        (per-epoch training history)
    """
    base = output_dir or DL_MODELS_DIR
    out_dir = base / _safe_filename(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / f"{model_name}.pt"
    meta_path = out_dir / f"{model_name}_metadata.json"
    hist_path = out_dir / f"{model_name}_history.json"

    # 1) weights + optimizer state + arch config (the portable "complete model")
    save_checkpoint(
        ckpt_path, model, optimizer=optimizer, epoch=len(history),
        extra={"arch_type": arch_type, "arch_kwargs": _jsonable(arch_kwargs),
               "metadata_ref": str(meta_path)},
    )
    # 2) metadata (provenance)
    meta_path.write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    # 3) training history
    hist_path.write_text(json.dumps(_jsonable(history), indent=2), encoding="utf-8")
    logger.info("saved DL model %s / %s -> %s", symbol, model_name, ckpt_path)
    return ckpt_path


def mark_best_dl(symbol: str, model_name: str, output_dir: Path | None = None) -> Path:
    """Record which DL model is deployed for a symbol (read by the DL predictor)."""
    base = output_dir or DL_MODELS_DIR
    out_dir = base / _safe_filename(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_dl_model.json"
    path.write_text(
        json.dumps({"symbol": symbol, "best_model": model_name,
                    "chosen_by": "val_loss", "updated_at": _now_iso()}, indent=2),
        encoding="utf-8",
    )
    logger.info("marked best DL model %s / %s", symbol, model_name)
    return path


def _rebuild_arch(arch_type: str, arch_kwargs: dict[str, Any]) -> "torch.nn.Module":
    """Rebuild a model from its saved arch_type + arch_kwargs (the recipe)."""
    if arch_type == "lstm":
        from deep_learning.lstm import build_lstm
        return build_lstm(input_size=arch_kwargs["input_size"],
                          **{k: v for k, v in arch_kwargs.items() if k != "input_size"})
    if arch_type == "gru":
        from deep_learning.gru import build_gru
        return build_gru(input_size=arch_kwargs["input_size"],
                         **{k: v for k, v in arch_kwargs.items() if k != "input_size"})
    if arch_type == "cnn":
        from deep_learning.cnn import build_cnn
        return build_cnn(input_size=arch_kwargs["input_size"],
                         **{k: v for k, v in arch_kwargs.items() if k != "input_size"})
    if arch_type == "transformer":
        from deep_learning.transformer import build_transformer
        return build_transformer(input_size=arch_kwargs["input_size"],
                                  **{k: v for k, v in arch_kwargs.items() if k != "input_size"})
    raise ValueError(f"unknown arch_type {arch_type!r}")


def load_dl_model(
    symbol: str,
    model_name: str | None = None,
    output_dir: Path | None = None,
    device: str | None = None,
) -> tuple["torch.nn.Module", dict[str, Any]]:
    """Load a saved DL model + metadata. Rebuilds the architecture from the
    saved arch_kwargs, then restores the weights -> ready for inference.

    Args:
        symbol:     Ticker.
        model_name:  Specific model, or None to load the deployed (best) one.
        output_dir:  Override the default models/deep_learning directory.
        device:      Map tensors to this device (default: current device).

    Returns:
        (model in eval mode, metadata).
    """
    base = output_dir or DL_MODELS_DIR
    out_dir = base / _safe_filename(symbol)

    if model_name is None:
        best_path = out_dir / "best_dl_model.json"
        if not best_path.exists():
            raise FileNotFoundError(f"No best_dl_model.json for {symbol} at {best_path}")
        model_name = json.loads(best_path.read_text(encoding="utf-8"))["best_model"]

    ckpt_path = out_dir / f"{model_name}.pt"
    meta_path = out_dir / f"{model_name}_metadata.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DL checkpoint not found: {ckpt_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    # Rebuild the architecture from the saved recipe, then restore weights.
    model = _rebuild_arch(metadata["arch_type"], metadata["arch_kwargs"])
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()
    logger.info("loaded DL model %s / %s (%s)", symbol, model_name, metadata.get("arch_type"))
    return model, metadata


if __name__ == "__main__":  # pragma: no cover
    # Roundtrip smoke test: train a tiny LSTM, save, reload, verify identical preds.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    from deep_learning.framework import (set_global_seed, make_dataloader, make_tensor_dataset,
                                          get_device, load_training_config)
    from deep_learning.sequence_builder import build_sequences
    from deep_learning.lstm import build_lstm
    from deep_learning.trainer import Trainer, TrainSettings

    import numpy as np

    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    set_global_seed(42)
    seq = build_sequences(sym, target_col="target_next_return")
    tcfg = load_training_config()
    train_ds = make_tensor_dataset(seq["X_train_seq"], seq["y_train_seq"])
    val_ds = make_tensor_dataset(seq["X_validation_seq"], seq["y_validation_seq"])
    test_ds = make_tensor_dataset(seq["X_test_seq"], seq["y_test_seq"])
    train_dl = make_dataloader(train_ds, tcfg["batch_size"], shuffle=True, drop_last=True)
    val_dl = make_dataloader(val_ds, tcfg["batch_size"], shuffle=False)
    test_dl = make_dataloader(test_ds, tcfg["batch_size"], shuffle=False)

    arch_kwargs = {"input_size": seq["n_features"], "hidden_size": 64, "num_layers": 2,
                   "dropout": 0.2, "task": "regression"}
    model = build_lstm(**arch_kwargs)
    trainer = Trainer(TrainSettings(epochs=5, lr=0.001, weight_decay=1e-4, grad_clip=1.0,
                                    early_stopping=False, save_best=False, label="lstm"), device="cpu")
    res = trainer.train(model, train_dl, val_dl, test_dl, task="regression")

    meta = build_dl_metadata(
        model_name="lstm_v1", symbol=sym, arch_type="lstm", arch_kwargs=arch_kwargs,
        target_col="target_next_return", feature_names=seq["feature_names"],
        training={"epochs_run": res["epochs_run"], "best_val_loss": res["best_val_loss"],
                  "best_epoch": res["best_epoch"], "train_time_s": res["train_time_s"]},
        metrics={"test_rmse": res["test_rmse"], "test_mae": res["test_mae"],
                 "directional_accuracy": res["directional_accuracy"],
                 "beats_naive": res["beats_naive"]},
        dataset_info=dataset_info_from_sequences(seq),
        scaler_ref=str(Path("data/prepared") / sym.replace(".", "_") / "scaler_params.json"),
    )
    save_dl_model(trainer.last_model, symbol=sym, model_name="lstm_v1", arch_type="lstm",
                  arch_kwargs=arch_kwargs, history=res["history"], metadata=meta,
                  optimizer=None, output_dir=Path("models/deep_learning"))
    mark_best_dl(sym, "lstm_v1", Path("models/deep_learning"))

    # Reload and verify predictions are IDENTICAL to the saved model.
    model2, meta2 = load_dl_model(sym, model_name="lstm_v1", device="cpu")
    # Compare preds on the test set: the restored model should match the saved one.
    saved_preds, reloaded_preds = [], []
    with torch.no_grad():
        for xb, _ in test_dl:
            saved_preds.append(trainer.last_model(xb).squeeze(-1).cpu().numpy())
            reloaded_preds.append(model2(xb).squeeze(-1).cpu().numpy())
    a = np.concatenate(saved_preds)
    b = np.concatenate(reloaded_preds)
    print("ROUNDTRIP OK" if np.allclose(a, b, atol=1e-6) else "ROUNDTRIP FAIL")
    print("max abs diff:", float(np.max(np.abs(a - b))) if len(a) else 0)
    print("reloaded arch:", meta2.get("arch_type"), "test_rmse:", meta2["metrics"]["test_rmse"])

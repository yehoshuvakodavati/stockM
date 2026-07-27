"""
StockM v1.0 - Phase 7, Lesson 6
1D CNN for Time-Series Forecasting
==================================

A sequence-to-one 1D convolutional network: slides small kernels across the
time axis to detect LOCAL temporal motifs (a 3-day momentum burst, a
reversal setup), then pools them into a single next-day forecast.

Roadmap concepts taught here
----------------------------
* Convolution (1D)   - a kernel sliding over the time axis
* Filters            - one kernel = one learned local pattern detector
* Kernel size        - how many consecutive days one filter spans
* Pooling            - max-pool (local) + global average pool (summary)
* Feature maps       - the (batch, filters, time) tensor of detector outputs

How CNNs differ from the recurrent models
-----------------------------------------
* PARALLEL over time. A conv computes all timesteps at once (no recurrence),
  so training is much faster than LSTM/GRU per sample.
* LOCAL, not global. A kernel of size k sees only k consecutive days. Stacking
  layers grows the receptive field, but a 2-layer k=3 CNN sees ~5 days, not the
  full 30. It excels at short motifs; the LSTM excels at long-range context.
* Translation-invariant. A pattern learned at days 5-7 fires at days 25-27 too
  - useful when a setup can appear anywhere in the window.

Config: cnn_v1 in configs/model_config.yaml
    filters: 64, kernel_size: 3, num_layers: 2, dropout: 0.2
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model_config.yaml"

logger = logging.getLogger("stockm.deep_learning.cnn")


class CNNModel(nn.Module):
    """Sequence-to-one 1D CNN forecaster.

    Input  : (batch, lookback, input_size)   - one window per row
    Output : (batch, 1)                       - next-day forecast

    Architecture
    ------------
        (batch, lookback, input_size)
            -> transpose to (batch, input_size, lookback)   [channels=features, length=time]
            -> [Conv1d -> ReLU -> MaxPool1d] x num_layers   [local motifs + downsampling]
            -> AdaptiveAvgPool1d(1)                          [global summary: (batch, filters, 1)]
            -> flatten -> Dropout -> Linear(filters, 1)     [forecast head]

    Parameters
    ----------
    input_size : int
        Number of features per timestep (= n_features). Becomes the input
        channel count for the first Conv1d.
    filters : int
        Conv channels (number of learned filters) per layer.
    kernel_size : int
        Width of each conv filter in DAYS. 3 = 3-day local patterns.
    num_layers : int
        Stacked conv blocks. Receptive field grows ~ (kernel_size-1)*num_layers + 1.
    dropout : float
        Regularisation before the linear head.
    task : str
        "regression" (linear head + MSE) or "classification" (logits + BCEWithLogits).
    """

    def __init__(
        self,
        input_size: int,
        filters: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.2,
        task: str = "regression",
    ) -> None:
        super().__init__()
        if task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.task = task
        self.input_size = input_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        layers: list[nn.Module] = []
        in_ch = input_size
        for i in range(num_layers):
            # padding = kernel_size // 2 keeps the time length stable before pooling
            # ("same" padding) so the sequence length is predictable.
            layers.append(nn.Conv1d(in_ch, filters, kernel_size,
                                    padding=kernel_size // 2))
            layers.append(nn.ReLU())
            # MaxPool halves the time length each block (ceil_mode guards odd lengths).
            layers.append(nn.MaxPool1d(kernel_size=2, ceil_mode=True))
            in_ch = filters
        self.conv = nn.Sequential(*layers)

        # Global average pool collapses whatever time-length remains to 1,
        # producing a fixed-size (batch, filters) summary regardless of input
        # length. This makes the model robust to different lookback values.
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a window through the CNN and emit one forecast per window.

        Args:
            x: (batch, lookback, input_size) feature tensor.

        Returns:
            (batch, 1) forecasts (raw logits for classification).
        """
        # Conv1d expects (batch, channels, length). Our tensor is
        # (batch, lookback=channels?, features). We treat FEATURES as channels
        # and LOOKBACK as the length, so transpose axes 1 and 2.
        x = x.transpose(1, 2)               # (batch, input_size, lookback)
        x = self.conv(x)                    # (batch, filters, time')
        x = self.global_pool(x)             # (batch, filters, 1)
        x = x.squeeze(-1)                   # (batch, filters)
        return self.fc(self.head_dropout(x))  # (batch, 1)


def _load_cnn_config(config_path: Path | None = None) -> dict[str, Any]:
    """Read the ``cnn_v1`` block from model_config.yaml (config-driven)."""
    path = config_path or MODEL_CONFIG
    cfg: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
        block = full.get("cnn_v1", {})
        cfg = {
            "filters": int(block.get("filters", 64)),
            "kernel_size": int(block.get("kernel_size", 3)),
            "num_layers": int(block.get("num_layers", 2)),
            "dropout": float(block.get("dropout", 0.2)),
        }
    return cfg


def build_cnn(input_size: int, config_path: Path | None = None, **overrides: Any) -> CNNModel:
    """Factory: build a CNNModel from config + overrides (mirrors build_lstm/build_gru)."""
    cfg = _load_cnn_config(config_path)
    cfg.update(overrides)
    return CNNModel(input_size=input_size, **cfg)


# ---------------------------------------------------------------------------
# 3-way comparison: CNN vs LSTM vs GRU under identical conditions.
# Run:  PYTHONPATH=src python src/deep_learning/cnn.py RELIANCE.NS
# ---------------------------------------------------------------------------
def compare_cnn(symbol: str, task: str = "regression", epochs: int = 20) -> dict[str, Any]:
    """Train CNN, LSTM, and GRU under identical conditions and compare."""
    from deep_learning.framework import (
        get_device,
        make_dataloader,
        make_tensor_dataset,
        set_global_seed,
        load_training_config,
    )
    from deep_learning.sequence_builder import build_sequences
    from deep_learning.lstm import build_lstm
    from deep_learning.gru import build_gru
    from deep_learning.trainer import train_one

    tcfg = load_training_config()
    device = get_device(tcfg["device"])
    target_col = "target_next_return" if task == "regression" else "target_direction"
    seq = build_sequences(symbol, target_col=target_col)

    dtype_y = torch.float32 if task == "classification" else None
    train_ds = make_tensor_dataset(seq["X_train_seq"], seq["y_train_seq"], dtype_y=dtype_y)
    val_ds = make_tensor_dataset(seq["X_validation_seq"], seq["y_validation_seq"], dtype_y=dtype_y)
    test_ds = make_tensor_dataset(seq["X_test_seq"], seq["y_test_seq"], dtype_y=dtype_y)
    train_dl = make_dataloader(train_ds, tcfg["batch_size"], shuffle=True, drop_last=True)
    val_dl = make_dataloader(val_ds, tcfg["batch_size"], shuffle=False)
    test_dl = make_dataloader(test_ds, tcfg["batch_size"], shuffle=False)

    common = dict(task=task, epochs=epochs, lr=tcfg["learning_rate"],
                  weight_decay=tcfg["weight_decay"], grad_clip=tcfg["gradient_clip_norm"],
                  device=device)
    n_feat = seq["n_features"]

    results: dict[str, dict[str, Any]] = {}
    for name, factory in (("lstm", lambda: build_lstm(n_feat, task=task)),
                          ("gru", lambda: build_gru(n_feat, task=task)),
                          ("cnn", lambda: build_cnn(n_feat, task=task))):
        logger.info("=== %s ===", name.upper())
        set_global_seed(42)  # reset before each for a controlled comparison
        results[name] = train_one(factory().to(device), train_dl, val_dl, test_dl,
                                  label=name.upper(), **common)

    logger.info(
        "COMPARE3 | %s %s | metric         LSTM       GRU       CNN\n"
        "  params         %8d  %8d  %8d\n"
        "  train_time_s   %8.2f  %8.2f  %8.2f\n"
        "  val_loss        %8.6f  %8.6f  %8.6f\n"
        "  test_rmse       %8.6f  %8.6f  %8.6f\n"
        "  dir_accuracy    %8.4f  %8.4f  %8.4f\n"
        "  beats_naive      %8s  %8s  %8s",
        symbol, task,
        results["lstm"]["params"], results["gru"]["params"], results["cnn"]["params"],
        results["lstm"]["train_time_s"], results["gru"]["train_time_s"], results["cnn"]["train_time_s"],
        results["lstm"]["final_val_loss"], results["gru"]["final_val_loss"], results["cnn"]["final_val_loss"],
        results["lstm"].get("test_rmse"), results["gru"].get("test_rmse"), results["cnn"].get("test_rmse"),
        results["lstm"]["directional_accuracy"], results["gru"]["directional_accuracy"], results["cnn"]["directional_accuracy"],
        results["lstm"]["beats_naive"], results["gru"]["beats_naive"], results["cnn"]["beats_naive"],
    )
    return {"symbol": symbol, "task": task, "epochs": epochs, **results}


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    res = compare_cnn(sym, task="regression", epochs=20)
    print("\n=== CNN vs LSTM vs GRU ===")
    print(f"symbol: {res['symbol']}  task: {res['task']}  epochs: {res['epochs']}")
    for mname in ("lstm", "gru", "cnn"):
        m = res[mname]
        print(f"\n--- {mname.upper()} ---")
        for k, v in m.items():
            if k == "history":
                continue
            print(f"  {k}: {v}")

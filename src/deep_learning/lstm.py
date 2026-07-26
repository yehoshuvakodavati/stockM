"""
StockM v1.0 - Phase 7, Lesson 4
LSTM Model for Next-Day Forecasting
==================================

A sequence-to-one LSTM: ingests a window of `lookback` trading days and emits
a single next-day forecast. The first real architecture on the Lesson-2/3
foundation.

Roadmap concepts taught here
----------------------------
* Recurrent Neural Networks (RNNs)        - weight sharing across timesteps
* Vanishing gradient problem              - why plain RNNs forget the far past
* Cell state (C_t) and hidden state (H_t) - the LSTM's two memory tracks
* Forget / Input / Output gates           - learned controllers of memory flow

Task choice for this lesson
---------------------------
The module supports BOTH heads via ``task``:
    task="regression"     -> linear output + MSELoss, target=target_next_return
    task="classification" -> raw logits + BCEWithLogitsLoss, target=target_direction
                             (this is what model_config.yaml's "output_size: 1,
                             binary: UP probability" implies, and the Lesson-3
                             recommended primary).

Lesson 4 DEMOS regression (target_next_return) for two honest reasons:
  1. It is the gentlest first LSTM (linear head + MSE map directly onto the
     single neuron of Lesson 1).
  2. It gives a direct, apples-to-apples RMSE comparison to the classical
     baselines on the IDENTICAL target - "does sequence-awareness beat
     random_forest's val_rmse?" Classification is Exercise 1 (one flag).

Config: lstm_baseline_v1 in configs/model_config.yaml
    hidden_size: 64, num_layers: 2, dropout: 0.2, bidirectional: false
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

# torch is required for this module (it defines an nn.Module). Import directly;
# framework.py already guards the import for the package as a whole.
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model_config.yaml"

logger = logging.getLogger("stockm.deep_learning.lstm")


class LSTMModel(nn.Module):
    """Sequence-to-one LSTM forecaster.

    Input  : (batch, lookback, input_size)   - one window per row
    Output : (batch, 1)                       - next-day forecast
             regression     -> a real value (return)
             classification -> a raw logit (apply sigmoid for UP-probability)

    Parameters
    ----------
    input_size : int
        Number of features per timestep (= n_features from the prepared data).
    hidden_size : int
        Width of the LSTM hidden state. 64 is a modest start for ~3500 windows.
    num_layers : int
        Stacked LSTM layers. >1 enables inter-layer dropout.
    dropout : float
        Regularisation. Applied (a) between stacked LSTM layers via nn.LSTM's
        own dropout, and (b) before the linear head.
    task : str
        "regression" or "classification" - selects the output head + the loss
        used by get_loss().
    bidirectional : bool
        Read the window forwards (and backwards). Safe for sequence-to-one
        (the whole window is in the past relative to the target), but default
        false to match config and keep the model simple.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        task: str = "regression",
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
        self.task = task
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # nn.LSTM's dropout only acts BETWEEN stacked layers, so it is a no-op
        # for num_layers=1. Gate it to avoid a PyTorch warning.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,            # (batch, seq, feature) - matches DataLoader
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        out_factor = 2 if bidirectional else 1
        self.head_dropout = nn.Dropout(dropout)            # regularise the head
        self.fc = nn.Linear(hidden_size * out_factor, 1)   # -> (batch, 1)
        # NOTE: no sigmoid on the classification path - BCEWithLogitsLoss is
        # numerically stable and expects raw logits.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a window through the LSTM and emit one forecast per window.

        Args:
            x: (batch, lookback, input_size) feature tensor.

        Returns:
            (batch, 1) forecasts.
        """
        # out: (batch, lookback, hidden_size * dirs) - hidden state at every step
        # (h_n, c_n): final hidden/cell states, shape (num_layers * dirs, batch, hidden)
        out, _ = self.lstm(x)
        last = out[:, -1, :]                  # take the LAST timestep's hidden state
        return self.fc(self.head_dropout(last))


def _load_lstm_config(config_path: Path | None = None) -> dict[str, Any]:
    """Read the ``lstm_baseline_v1`` block from model_config.yaml (config-driven).

    Returns architecture kwargs only; ``task`` is a call-time decision (see
    module docstring) and is NOT inferred from output_size, so the same config
    serves both the regression demo and the classification primary.
    """
    path = config_path or MODEL_CONFIG
    cfg: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
        block = full.get("lstm_baseline_v1", {})
        cfg = {
            "hidden_size": int(block.get("hidden_size", 64)),
            "num_layers": int(block.get("num_layers", 2)),
            "dropout": float(block.get("dropout", 0.2)),
            "bidirectional": bool(block.get("bidirectional", False)),
        }
    return cfg


def build_lstm(input_size: int, config_path: Path | None = None, **overrides: Any) -> LSTMModel:
    """Factory: build an LSTMModel from config + overrides.

    Args:
        input_size:  Number of features per timestep (from the prepared data).
        config_path: Override the model_config.yaml location.
        **overrides: Any constructor kwarg (hidden_size, num_layers, dropout,
                     task, bidirectional). ``task`` is the usual override.

    Returns:
        A constructed (untrained) LSTMModel.
    """
    cfg = _load_lstm_config(config_path)
    cfg.update(overrides)
    return LSTMModel(input_size=input_size, **cfg)


# ---------------------------------------------------------------------------
# Minimal training demo (NOT the production trainer - that is Lesson 8).
# Proves the architecture trains end-to-end on real StockM data and reports a
# baseline-comparable result. Run:
#   PYTHONPATH=src python src/deep_learning/lstm.py RELIANCE.NS
# ---------------------------------------------------------------------------
def _train_demo(symbol: str, task: str = "regression", epochs: int = 20) -> dict[str, Any]:
    """Train an LSTM on one ticker with a minimal loop and report metrics.

    Uses the framework utilities (seed, dataloaders, param count) and the
    sequence builder. No early stopping / LR scheduling / checkpointing - those
    arrive in Lesson 8. Deliberately small so it runs in a few minutes on CPU.
    """
    from deep_learning.framework import (
        count_parameters,
        get_device,
        get_loss,
        make_dataloader,
        make_tensor_dataset,
        set_global_seed,
        load_training_config,
    )
    from deep_learning.sequence_builder import build_sequences

    set_global_seed(42)
    tcfg = load_training_config()
    device = get_device(tcfg["device"])

    target_col = "target_next_return" if task == "regression" else "target_direction"
    seq = build_sequences(symbol, target_col=target_col)

    dtype_y = None  # let make_tensor_dataset infer (float for reg, long for cls)
    train_ds = make_tensor_dataset(seq["X_train_seq"], seq["y_train_seq"], dtype_y=dtype_y)
    val_ds = make_tensor_dataset(seq["X_validation_seq"], seq["y_validation_seq"], dtype_y=dtype_y)
    test_ds = make_tensor_dataset(seq["X_test_seq"], seq["y_test_seq"], dtype_y=dtype_y)
    # BCEWithLogitsLoss needs float targets, so force float for classification.
    if task == "classification":
        train_ds = make_tensor_dataset(seq["X_train_seq"], seq["y_train_seq"], dtype_y=torch.float32)
        val_ds = make_tensor_dataset(seq["X_validation_seq"], seq["y_validation_seq"], dtype_y=torch.float32)
        test_ds = make_tensor_dataset(seq["X_test_seq"], seq["y_test_seq"], dtype_y=torch.float32)

    train_dl = make_dataloader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True)
    val_dl = make_dataloader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)
    test_dl = make_dataloader(test_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model = build_lstm(input_size=seq["n_features"], task=task).to(device)
    loss_fn = get_loss(task)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"],
                           weight_decay=tcfg["weight_decay"])
    clip = tcfg["gradient_clip_norm"]

    logger.info(
        "LSTM demo | symbol=%s task=%s params=%d device=%s epochs=%d bs=%d lr=%g",
        symbol, task, count_parameters(model), device, epochs, tcfg["batch_size"], tcfg["learning_rate"],
    )

    history: list[tuple[int, float, float]] = []
    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        n = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)            # (batch,)
            loss = loss_fn(pred, yb.float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            train_loss += loss.item() * len(yb)
            n += len(yb)
        train_loss /= max(n, 1)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(-1)
                val_loss += loss_fn(pred, yb.float()).item() * len(yb)
                nv += len(yb)
        val_loss /= max(nv, 1)
        history.append((epoch, train_loss, val_loss))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            logger.info("epoch %2d/%d | train_loss=%.6f val_loss=%.6f", epoch, epochs, train_loss, val_loss)

    # --- test metrics ---
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            out = model(xb).squeeze(-1).cpu()
            if task == "classification":
                out = torch.sigmoid(out)           # logit -> UP probability
            preds.append(out)
            ys.append(yb)
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(ys).numpy().astype(np.float64)

    if task == "regression":
        mse = float(np.mean((y_pred - y_true) ** 2))
        rmse = float(np.sqrt(mse))
        # Directional accuracy: sign of predicted return vs sign of actual.
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        # Naive zero-predictor RMSE = sqrt(mean(y^2)); the floor to beat.
        naive_rmse = float(np.sqrt(np.mean(y_true ** 2)))
        beats_naive = rmse < naive_rmse
        result = {
            "task": task, "symbol": symbol, "params": count_parameters(model),
            "test_mse": mse, "test_rmse": rmse, "naive_test_rmse": naive_rmse,
            "beats_naive": beats_naive, "directional_accuracy": dir_acc,
            "final_val_loss": history[-1][2], "history": history,
        }
    else:
        y_hat = (y_pred > 0.5).astype(np.float64)
        acc = float(np.mean(y_hat == y_true))
        majority = float(np.mean(y_true == 1))  # all-UP naive floor
        result = {
            "task": task, "symbol": symbol, "params": count_parameters(model),
            "test_accuracy": acc, "naive_majority_acc": max(majority, 1 - majority),
            "beats_naive": acc > max(majority, 1 - majority),
            "directional_accuracy": acc, "final_val_loss": history[-1][2], "history": history,
        }
    logger.info("DONE | %s", {k: v for k, v in result.items() if k != "history"})
    return result


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    res = _train_demo(sym, task="regression", epochs=20)
    print("\n=== LSTM demo result ===")
    for k, v in res.items():
        if k == "history":
            print(f"history (epoch, train_loss, val_loss):")
            for row in v[:3]:
                print("   ", row)
            print("    ...")
            for row in v[-3:]:
                print("   ", row)
        else:
            print(f"{k}: {v}")

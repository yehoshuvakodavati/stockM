"""
StockM v1.0 - Phase 7, Lesson 5
GRU Model + LSTM-vs-GRU Comparison
=================================

A sequence-to-one GRU: the simplified recurrent cousin of the LSTM. Same
interface as LSTMModel (input window -> one next-day forecast), so the two can
be trained and compared under identical conditions.

Roadmap concepts taught here
----------------------------
* Update gate (z_t)  - merges the LSTM's forget + input gates into one
* Reset gate (r_t)   - controls how much past info feeds the candidate
* Why GRU is simpler - 3 gates vs 4, no separate cell state, fewer params

Comparison dimensions (the Lesson-5 deliverable)
------------------------------------------------
For an identical setup (same data, seed, batch, epochs, optimizer) we report:
    1. Training time     - wall-clock seconds (GRU is cheaper per step)
    2. Validation metrics- val loss / RMSE (does the simplification cost accuracy?)
    3. Memory usage      - parameter count (the capacity + memory proxy)
    4. Prediction accuracy- test RMSE, directional accuracy, beats_naive

Architecture parity with the LSTM
---------------------------------
Same config source (model_config.yaml), same hidden_size/num_layers/dropout,
same head (Linear(hidden->1)), same loss (framework.get_loss). The ONLY
difference is nn.GRU instead of nn.LSTM - a controlled comparison.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model_config.yaml"

logger = logging.getLogger("stockm.deep_learning.gru")


class GRUModel(nn.Module):
    """Sequence-to-one GRU forecaster (same I/O contract as LSTMModel).

    Input  : (batch, lookback, input_size)
    Output : (batch, 1)

    A GRU has NO separate cell state - only a hidden state h_t. The update gate
    directly interpolates between the old hidden state and a new candidate,
    which still yields a near-identity gradient path (so GRU, like LSTM,
    mitigates the vanishing gradient) but with 3 gates instead of 4.
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

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )
        out_factor = 2 if bidirectional else 1
        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * out_factor, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a window through the GRU and emit one forecast per window.

        Args:
            x: (batch, lookback, input_size) feature tensor.

        Returns:
            (batch, 1) forecasts (raw logits for classification).
        """
        # GRU returns (output, h_n) - NO cell state, unlike LSTM's (out, (h_n, c_n)).
        out, _ = self.gru(x)                # out: (batch, lookback, hidden*dirs)
        last = out[:, -1, :]                # last timestep's hidden state
        return self.fc(self.head_dropout(last))


def _load_gru_config(config_path: Path | None = None) -> dict[str, Any]:
    """Read GRU architecture kwargs from model_config.yaml.

    Reuses the ``lstm_baseline_v1`` block (same hidden_size/num_layers/dropout)
    so the comparison is controlled - identical capacity, only the cell type
    differs. A dedicated ``gru_v1`` block can override later without code change.
    """
    path = config_path or MODEL_CONFIG
    cfg: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
        block = full.get("gru_v1") or full.get("lstm_baseline_v1", {})
        cfg = {
            "hidden_size": int(block.get("hidden_size", 64)),
            "num_layers": int(block.get("num_layers", 2)),
            "dropout": float(block.get("dropout", 0.2)),
            "bidirectional": bool(block.get("bidirectional", False)),
        }
    return cfg


def build_gru(input_size: int, config_path: Path | None = None, **overrides: Any) -> GRUModel:
    """Factory: build a GRUModel from config + overrides (mirrors build_lstm)."""
    cfg = _load_gru_config(config_path)
    cfg.update(overrides)
    return GRUModel(input_size=input_size, **cfg)


# ---------------------------------------------------------------------------
# Shared minimal trainer used for the FAIR LSTM-vs-GRU comparison.
# Same loop as lstm._train_demo; isolated here so both models train under
# byte-identical conditions. (The production trainer is Lesson 8.)
# ---------------------------------------------------------------------------
def _train_one(
    model: nn.Module,
    train_dl,
    val_dl,
    test_dl,
    task: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    device: str,
) -> dict[str, Any]:
    """Train one model, return metrics + wall-clock training time."""
    from deep_learning.framework import count_parameters, get_loss

    loss_fn = get_loss(task)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    params = count_parameters(model)

    t0 = time.perf_counter()
    history: list[tuple[int, float, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb.float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_loss += loss.item() * len(yb)
            n += len(yb)
        tr_loss /= max(n, 1)

        model.eval()
        va_loss, nv = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += loss_fn(model(xb).squeeze(-1), yb.float()).item() * len(yb)
                nv += len(yb)
        va_loss /= max(nv, 1)
        history.append((epoch, tr_loss, va_loss))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            logger.info("  epoch %2d/%d | train=%.6f val=%.6f", epoch, epochs, tr_loss, va_loss)
    train_time = time.perf_counter() - t0

    # Test metrics
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            out = model(xb.to(device)).squeeze(-1).cpu()
            if task == "classification":
                out = torch.sigmoid(out)
            preds.append(out)
            ys.append(yb)
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(ys).numpy().astype(np.float64)

    if task == "regression":
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        naive_rmse = float(np.sqrt(np.mean(y_true ** 2)))
        return {
            "params": params, "train_time_s": round(train_time, 2),
            "final_val_loss": history[-1][2], "test_rmse": rmse,
            "naive_test_rmse": naive_rmse, "beats_naive": rmse < naive_rmse,
            "directional_accuracy": dir_acc, "history": history,
        }
    y_hat = (y_pred > 0.5).astype(np.float64)
    acc = float(np.mean(y_hat == y_true))
    maj = float(np.mean(y_true == 1))
    return {
        "params": params, "train_time_s": round(train_time, 2),
        "final_val_loss": history[-1][2], "test_accuracy": acc,
        "naive_majority_acc": max(maj, 1 - maj), "beats_naive": acc > max(maj, 1 - maj),
        "directional_accuracy": acc, "history": history,
    }


def compare_rnn(
    symbol: str,
    task: str = "regression",
    epochs: int = 20,
) -> dict[str, Any]:
    """Train LSTM and GRU under identical conditions and compare them.

    Args:
        symbol: Ticker to train on.
        task:   "regression" or "classification".
        epochs: Training epochs (same for both - controlled comparison).

    Returns:
        {"lstm": {...}, "gru": {...}} metrics dicts + a comparison summary.
    """
    from deep_learning.framework import (
        get_device,
        make_dataloader,
        make_tensor_dataset,
        set_global_seed,
        load_training_config,
    )
    from deep_learning.sequence_builder import build_sequences
    from deep_learning.lstm import build_lstm

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

    logger.info("=== LSTM ===")
    set_global_seed(42)  # reset before each so both see the same data order + init scale
    lstm_res = _train_one(build_lstm(seq["n_features"], task=task).to(device),
                          train_dl, val_dl, test_dl, **common)
    logger.info("=== GRU ===")
    set_global_seed(42)
    gru_res = _train_one(build_gru(seq["n_features"], task=task).to(device),
                         train_dl, val_dl, test_dl, **common)

    summary = {
        "symbol": symbol, "task": task, "epochs": epochs,
        "lstm": lstm_res, "gru": gru_res,
    }
    logger.info(
        "COMPARE | %s %s | metric        LSTM       GRU\n"
        "  params        %8d  %8d\n"
        "  train_time_s  %8.2f  %8.2f\n"
        "  val_loss      %8.6f  %8.6f\n"
        "  test_rmse     %8.6f  %8.6f\n"
        "  dir_accuracy  %8.4f  %8.4f\n"
        "  beats_naive   %8s  %8s",
        symbol, task,
        lstm_res["params"], gru_res["params"],
        lstm_res["train_time_s"], gru_res["train_time_s"],
        lstm_res["final_val_loss"], gru_res["final_val_loss"],
        lstm_res.get("test_rmse", lstm_res.get("test_accuracy")),
        gru_res.get("test_rmse", gru_res.get("test_accuracy")),
        lstm_res["directional_accuracy"], gru_res["directional_accuracy"],
        lstm_res["beats_naive"], gru_res["beats_naive"],
    )
        gru_res.get("test_rmse", gru_res.get("test_accuracy")),
        lstm_res["directional_accuracy"], gru_res["directional_accuracy"],
        lstm_res["beats_naive"], gru_res["beats_naive"],
    )
    return summary


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    res = compare_rnn(sym, task="regression", epochs=20)
    print("\n=== LSTM vs GRU comparison ===")
    print(f"symbol: {res['symbol']}  task: {res['task']}  epochs: {res['epochs']}")
    for mname in ("lstm", "gru"):
        m = res[mname]
        print(f"\n--- {mname.upper()} ---")
        for k, v in m.items():
            if k == "history":
                continue
            print(f"  {k}: {v}")

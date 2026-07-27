"""
StockM v1.0 - Phase 7, Lesson 7
Transformer for Time-Series Forecasting
=======================================

A sequence-to-one Transformer encoder. Where the LSTM recurses across time and
the CNN convolves locally, the Transformer lets EVERY day in the window attend
directly to EVERY other day - global, parallel, no recurrence.

Roadmap concepts taught here
----------------------------
* Self-attention        - each position weights every other position by relevance
* Multi-head attention  - h parallel attention heads = h learned "what matters" views
* Positional encoding   - sinusoidal signals that re-inject order (attention alone is
                          permutation-invariant)
* Feed-forward network  - per-position MLP that refines each attended representation
* Residual connections  - add x to sublayer(x) so gradients flow through deep stacks
* Layer normalization   - stabilise training (pre-LN = norm_first=True)

Why Transformers in financial AI
--------------------------------
* Global receptive field at layer 1 (vs CNN's local kernels; vs LSTM's sequential
  unrolling that costs O(time) to connect distant days).
* Parallel over time -> faster than RNNs on GPU (less so on CPU).
* Attention weights are introspectable ("which days did the model look at?") - a
  foot-in-the-door for interpretability (Lesson 13).
* SOTA on many sequence tasks; increasingly adopted for forecasting.

Honest expectation
------------------
Per [[dl-baseline-findings]], LSTM/GRU/CNN all landed at ~50% directional /
beats_naive=False on RELIANCE. Attention is unlikely to manufacture signal where
the SNR is this low - but it may surface a DIFFERENT temporal structure (direct
long-range day-to-day links) the others missed. The run will tell.

Config: transformer_v1 in configs/model_config.yaml
    d_model: 64, n_heads: 4, num_layers: 2, dropout: 0.1
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model_config.yaml"

logger = logging.getLogger("stockm.deep_learning.transformer")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017).

    Attention is permutation-invariant - it would treat the window as a bag of
    days with no order. We ADD a fixed, position-dependent signal to each day's
    embedding so the model can tell day 5 from day 25. Sinusoids are used (not
    learned) so they generalise to window lengths unseen in training.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sinusoidal PE, got {d_model}")
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # 10000^(2i/d_model) computed in log space for numerical stability.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dims -> sin
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims  -> cos
        # register_buffer: part of state (saved/loaded with the model) but NOT
        # a learned parameter.
        self.register_buffer("pe", pe.unsqueeze(0))    # (1, max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to (batch, seq, d_model)."""
        x = x + self.pe[:, : x.size(1)]                # broadcast over batch
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Sequence-to-one Transformer encoder forecaster.

    Input  : (batch, lookback, input_size)
    Output : (batch, 1)

    Architecture
    ------------
        (batch, lookback, input_size)
            -> Linear(input_size -> d_model)            [project features to d_model]
            -> + PositionalEncoding                      [re-inject order]
            -> TransformerEncoder x num_layers           [self-attn + FFN, pre-LN]
            -> mean over the time axis                   [(batch, d_model) global summary]
            -> Dropout -> Linear(d_model -> 1)           [forecast head]

    Parameters
    ----------
    input_size : int
        Features per timestep (projected to d_model).
    d_model : int
        Internal model width. Must be even (sinusoidal PE).
    n_heads : int
        Parallel attention heads. d_model must be divisible by n_heads.
    num_layers : int
        Stacked encoder layers.
    dim_feedforward : int | None
        Width of the per-position FFN. Default 4*d_model (the Transformer standard).
    dropout : float
        Regularisation (PE, attention, FFN, head).
    task : str
        "regression" (linear head + MSE) or "classification" (logits + BCEWithLogits).
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        task: str = "regression",
    ) -> None:
        super().__init__()
        if task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.task = task
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout,
            activation="relu",
            batch_first=True,        # (batch, seq, d_model) - matches our tensors
            norm_first=True,         # pre-LN: more stable without LR warmup
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a window through the Transformer and emit one forecast per window.

        Args:
            x: (batch, lookback, input_size) feature tensor.

        Returns:
            (batch, 1) forecasts (raw logits for classification).
        """
        x = self.input_proj(x)          # (batch, lookback, d_model)
        x = self.pos_enc(x)             # + positional encoding
        x = self.encoder(x)             # (batch, lookback, d_model)
        x = x.mean(dim=1)               # mean-pool over time -> (batch, d_model)
        return self.fc(self.head_dropout(x))


def _load_transformer_config(config_path: Path | None = None) -> dict[str, Any]:
    """Read the ``transformer_v1`` block from model_config.yaml (config-driven)."""
    path = config_path or MODEL_CONFIG
    cfg: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
        block = full.get("transformer_v1", {})
        cfg = {
            "d_model": int(block.get("d_model", 64)),
            "n_heads": int(block.get("n_heads", 4)),
            "num_layers": int(block.get("num_layers", 2)),
            "dropout": float(block.get("dropout", 0.1)),
        }
    return cfg


def build_transformer(
    input_size: int, config_path: Path | None = None, **overrides: Any
) -> TransformerModel:
    """Factory: build a TransformerModel from config + overrides."""
    cfg = _load_transformer_config(config_path)
    cfg.update(overrides)
    return TransformerModel(input_size=input_size, **cfg)


# ---------------------------------------------------------------------------
# 4-way comparison: LSTM / GRU / CNN / Transformer, identical conditions.
# Run:  PYTHONPATH=src python src/deep_learning/transformer.py RELIANCE.NS
# ---------------------------------------------------------------------------
def compare_all(symbol: str, task: str = "regression", epochs: int = 20) -> dict[str, Any]:
    """Train all four architectures under identical conditions and compare."""
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
    from deep_learning.cnn import build_cnn
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

    factories = {
        "lstm": lambda: build_lstm(n_feat, task=task),
        "gru": lambda: build_gru(n_feat, task=task),
        "cnn": lambda: build_cnn(n_feat, task=task),
        "transformer": lambda: build_transformer(n_feat, task=task),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, factory in factories.items():
        logger.info("=== %s ===", name.upper())
        set_global_seed(42)  # reset before each -> controlled, reproducible comparison
        results[name] = train_one(factory().to(device), train_dl, val_dl, test_dl,
                                  label=name.upper(), **common)

    logger.info(
        "COMPARE4 | %s %s | metric          LSTM       GRU       CNN    TRANS\n"
        "  params          %8d  %8d  %8d  %8d\n"
        "  train_time_s    %8.2f  %8.2f  %8.2f  %8.2f\n"
        "  val_loss         %8.6f  %8.6f  %8.6f  %8.6f\n"
        "  test_rmse        %8.6f  %8.6f  %8.6f  %8.6f\n"
        "  dir_accuracy     %8.4f  %8.4f  %8.4f  %8.4f\n"
        "  beats_naive       %8s  %8s  %8s  %8s",
        symbol, task,
        results["lstm"]["params"], results["gru"]["params"],
        results["cnn"]["params"], results["transformer"]["params"],
        results["lstm"]["train_time_s"], results["gru"]["train_time_s"],
        results["cnn"]["train_time_s"], results["transformer"]["train_time_s"],
        results["lstm"]["final_val_loss"], results["gru"]["final_val_loss"],
        results["cnn"]["final_val_loss"], results["transformer"]["final_val_loss"],
        results["lstm"].get("test_rmse"), results["gru"].get("test_rmse"),
        results["cnn"].get("test_rmse"), results["transformer"].get("test_rmse"),
        results["lstm"]["directional_accuracy"], results["gru"]["directional_accuracy"],
        results["cnn"]["directional_accuracy"], results["transformer"]["directional_accuracy"],
        results["lstm"]["beats_naive"], results["gru"]["beats_naive"],
        results["cnn"]["beats_naive"], results["transformer"]["beats_naive"],
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
    res = compare_all(sym, task="regression", epochs=20)
    print("\n=== 4-way comparison ===")
    print(f"symbol: {res['symbol']}  task: {res['task']}  epochs: {res['epochs']}")
    for mname in ("lstm", "gru", "cnn", "transformer"):
        m = res[mname]
        print(f"\n--- {mname.upper()} ---")
        for k, v in m.items():
            if k == "history":
                continue
            print(f"  {k}: {v}")

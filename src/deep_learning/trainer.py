"""
StockM v1.0 - Phase 7, Lesson 8
Production Training Pipeline
=============================

Architecture-agnostic trainer for the deep-learning track (LSTM / GRU / CNN /
Transformer). One `Trainer` trains any `nn.Module`; comparisons stay FAIR
because only the model differs.

Production features (the Lesson-8 upgrade over the minimal loop)
----------------------------------------------------------------
    * Early stopping          - halt when val_loss plateaus; restore best weights
    * LR scheduling           - ReduceLROnPlateau + optional linear warmup
                                (the warmup is what the Transformer was missing
                                in Lesson 7 - it under-trained without it)
    * Model checkpointing     - save the best model to disk (state_dict + extra)
    * Training-history logging - per-epoch train/val loss + lr, returned + saved
    * Learning-curve plotting  - separate plotting.plot_learning_curves()

Config-driven: reads training_config.yaml (early_stopping, optimization,
checkpoints blocks). The Trainer honours those settings; callers can override.

Why restore best weights
------------------------
Early stopping without restoring the best weights reports metrics from the LAST
epoch - which may be worse than the peak. We track the best val_loss, checkpoint
at it, and load those weights back before evaluation. This is the difference
between "honest test metrics" and "test metrics from a model that already
started overfitting."
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

logger = logging.getLogger("stockm.deep_learning.trainer")


@dataclass
class TrainSettings:
    """All knobs for one training run, in one serialisable place.

    Defaults mirror training_config.yaml; callers (or the config loader) override.
    """

    epochs: int = 50
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    # Early stopping (monitor=val_loss, mode=min is the only supported combo here)
    early_stopping: bool = True
    es_patience: int = 10
    es_min_delta: float = 0.0001
    # LR scheduling
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    warmup_epochs: int = 0          # linear warmup before ReduceLROnPlateau
    # Checkpointing
    checkpoint_dir: Path | None = None   # None = don't save to disk
    save_best: bool = True
    # Logging
    log_every: int = 5
    label: str = ""


class Trainer:
    """Train one nn.Module with early stopping, LR scheduling, and checkpointing.

    Usage:
        settings = TrainSettings(epochs=50, warmup_epochs=5)
        trainer = Trainer(settings, device="cpu")
        result = trainer.train(model, train_dl, val_dl, test_dl, task="regression")
    """

    def __init__(self, settings: TrainSettings | None = None, device: str = "cpu") -> None:
        self.s = settings or TrainSettings()
        self.device = device

    # ------------------------------------------------------------------ train
    def train(
        self,
        model: "torch.nn.Module",
        train_dl,
        val_dl,
        test_dl,
        task: str = "regression",
        checkpoint_name: str | None = None,
    ) -> dict[str, Any]:
        """Train `model`, restore best weights, and return metrics + history.

        Args:
        model:          nn.Module with forward (batch, lookback, features) -> (batch, 1).
        train/val/test_dl: DataLoaders yielding (X, y).
        task:           "regression" or "classification".
        checkpoint_name: If set (and save_best), checkpoint to
                        <checkpoint_dir>/<checkpoint_name>.pt. None -> no disk save.

        Returns:
            Dict with params, train_time_s, final_val_loss, best_val_loss,
            stopped_epoch, history (per-epoch), and task-specific test metrics.
        """
        from deep_learning.framework import count_parameters, get_loss, save_checkpoint, load_checkpoint

        s = self.s
        model = model.to(self.device)
        loss_fn = get_loss(task)
        opt = torch.optim.Adam(model.parameters(), lr=s.lr, weight_decay=s.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=s.scheduler_factor, patience=s.scheduler_patience
        )
        params = count_parameters(model)
        tag = f"[{s.label}] " if s.label else ""

        history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        ckpt_path = (
            Path(s.checkpoint_dir) / f"{checkpoint_name}.pt"
            if (s.save_best and s.checkpoint_dir and checkpoint_name)
            else None
        )
        if ckpt_path:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        stopped_epoch = 0
        for epoch in range(1, s.epochs + 1):
            # --- linear warmup (manual; avoids SequentialLR's metric-forwarding bug) ---
            if s.warmup_epochs > 0 and epoch <= s.warmup_epochs:
                warmup_lr = s.lr * (epoch / s.warmup_epochs)
                for g in opt.param_groups:
                    g["lr"] = warmup_lr

            # --- train ---
            model.train()
            tr_loss, n = 0.0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb.float())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), s.grad_clip)
                opt.step()
                tr_loss += loss.item() * len(yb)
                n += len(yb)
            tr_loss /= max(n, 1)

            # --- validate ---
            model.eval()
            va_loss, nv = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    va_loss += loss_fn(model(xb).squeeze(-1), yb.float()).item() * len(yb)
                    nv += len(yb)
            va_loss /= max(nv, 1)

            current_lr = opt.param_groups[0]["lr"]
            history.append(
                {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "lr": current_lr}
            )

            # --- LR scheduling (only after warmup; ReduceLROnPlateau needs the metric) ---
            if s.warmup_epochs == 0 or epoch > s.warmup_epochs:
                scheduler.step(va_loss)

            # --- checkpoint best + early-stopping bookkeeping ---
            improved = va_loss < best_val_loss - s.es_min_delta
            if improved:
                best_val_loss = va_loss
                best_epoch = epoch
                epochs_no_improve = 0
                if ckpt_path:
                    save_checkpoint(
                        ckpt_path, model, optimizer=opt, epoch=epoch,
                        extra={"val_loss": va_loss, "task": task, "label": s.label},
                    )
                # Always keep an in-memory copy of the best weights (cheap restore).
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epoch == 1 or epoch % s.log_every == 0 or epoch == s.epochs:
                logger.info(
                    "  %sepoch %2d/%d | train=%.6f val=%.6f lr=%.2e %s",
                    tag, epoch, s.epochs, tr_loss, va_loss, current_lr,
                    "(best)" if improved else f"({epochs_no_improve}/{s.es_patience})",
                )

            # --- early stopping ---
            if s.early_stopping and epochs_no_improve >= s.es_patience:
                logger.info("  %searly stopping at epoch %d (best=%d, val=%.6f)",
                            tag, epoch, best_epoch, best_val_loss)
                stopped_epoch = epoch
                break
            stopped_epoch = epoch

        train_time = time.perf_counter() - t0

        # --- restore best weights (in-memory preferred; fall back to disk checkpoint) ---
        if "best_state" in locals():
            model.load_state_dict(best_state)
        elif ckpt_path and ckpt_path.exists():
            load_checkpoint(ckpt_path, model)

        # Expose the trained (best-weights-restored) model for external eval.
        self.last_model = model

        # --- test metrics (on the restored best model) ---
        test_metrics = self._evaluate(model, test_dl, task)
        result = {
            "params": params,
            "train_time_s": round(train_time, 2),
            "final_val_loss": history[-1]["val_loss"],
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "epochs_run": len(history),
            "early_stopped": stopped_epoch < s.epochs,
            "history": history,
            "checkpoint": str(ckpt_path) if ckpt_path else None,
            **test_metrics,
        }
        logger.info(
            "  %sdone | best_val=%.6f @ep%d | ran %d/%d epochs (%s) | test_rmse=%.6f dir_acc=%.4f",
            tag, best_val_loss, best_epoch, len(history), s.epochs,
            "early-stop" if result["early_stopped"] else "full",
            result.get("test_rmse", float("nan")), result["directional_accuracy"],
        )
        return result

    # ------------------------------------------------------------- evaluation
    def _evaluate(self, model, test_dl, task: str) -> dict[str, Any]:
        """Compute test metrics on the (restored best) model."""
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                out = model(xb.to(self.device)).squeeze(-1).cpu()
                if task == "classification":
                    out = torch.sigmoid(out)
                preds.append(out)
                ys.append(yb)
        y_pred = torch.cat(preds).numpy()
        y_true = torch.cat(ys).numpy().astype(np.float64)

        if task == "regression":
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            mae = float(np.mean(np.abs(y_pred - y_true)))
            dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
            naive_rmse = float(np.sqrt(np.mean(y_true ** 2)))
            return {
                "test_rmse": rmse, "test_mae": mae,
                "naive_test_rmse": naive_rmse, "beats_naive": rmse < naive_rmse,
                "directional_accuracy": dir_acc,
            }
        y_hat = (y_pred > 0.5).astype(np.float64)
        acc = float(np.mean(y_hat == y_true))
        maj = float(np.mean(y_true == 1))
        return {
            "test_accuracy": acc, "naive_majority_acc": max(maj, 1 - maj),
            "beats_naive": acc > max(maj, 1 - maj), "directional_accuracy": acc,
        }


# ---------------------------------------------------------------------------
# Save training history as JSON (for reproducibility + later plotting).
# ---------------------------------------------------------------------------
def save_history(history: list[dict[str, float]], path: str | Path) -> Path:
    """Persist the per-epoch training history to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("history saved -> %s", path)
    return path


# ---------------------------------------------------------------------------
# Backward-compatible convenience wrapper.
# The comparison runners (gru.compare_rnn, cnn.compare_cnn, transformer.compare_all)
# call train_one(...); this delegates to Trainer so they automatically gain
# early stopping + LR scheduling. Defaults are conservative (warmup_epochs=0,
# no checkpoint dir) so comparison behaviour stays controlled unless overridden.
# ---------------------------------------------------------------------------
def train_one(
    model: "torch.nn.Module",
    train_dl,
    val_dl,
    test_dl,
    task: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    device: str,
    log_every: int = 5,
    label: str = "",
) -> dict[str, Any]:
    """Thin wrapper: TrainSettings -> Trainer.train(). Kept for backward compat."""
    settings = TrainSettings(
        epochs=epochs, lr=lr, weight_decay=weight_decay, grad_clip=grad_clip,
        log_every=log_every, label=label,
        # Conservative defaults for fair comparison runs: no warmup, no disk
        # checkpoint, early stopping ON with a generous patience so all models
        # get a comparable amount of training. Callers wanting the full
        # production treatment construct Trainer(TrainSettings(...)) directly.
        warmup_epochs=0, checkpoint_dir=None, save_best=False,
        early_stopping=True, es_patience=10, es_min_delta=1e-4,
    )
    return Trainer(settings, device=device).train(
        model, train_dl, val_dl, test_dl, task=task, checkpoint_name=None
    )

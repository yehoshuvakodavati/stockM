"""
StockM v1.0 - Phase 7, Lesson 8
Learning-Curve Visualization
============================

Plots training vs validation loss across epochs - the single most useful DL
diagnostic. A widening gap = overfitting; both flat = under-fitting; val rising
while train falls = overfitting; a healthy run = both descend and converge.

Also plots the LR schedule when warmup/scheduling is active, since the LR
trajectory explains a lot of loss behaviour (e.g. the Transformer's Lesson-7
under-training was an LR-warmup story).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no display needed, writes PNG to disk
import matplotlib.pyplot as plt

logger = logging.getLogger("stockm.deep_learning.plotting")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DL_REPORTS_DIR = PROJECT_ROOT / "reports" / "deep_learning"


def plot_learning_curves(
    history: list[dict[str, float]],
    save_path: str | Path | None = None,
    title: str = "Learning Curves",
    show_lr: bool = True,
) -> Path:
    """Plot train/val loss (and LR) vs epoch and save to PNG.

    Args:
        history:   Per-epoch records from Trainer.train() (list of dicts with
                   epoch, train_loss, val_loss, lr).
        save_path: PNG destination. None -> reports/deep_learning/learning_curve.png.
        title:     Plot title.
        show_lr:   Also plot the LR schedule on a secondary y-axis.

    Returns:
        The path written.
    """
    if not history:
        raise ValueError("empty history - nothing to plot")

    epochs = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]
    va = [h["val_loss"] for h in history]

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(epochs, tr, "o-", label="train loss", linewidth=1.5, markersize=3)
    ax_loss.plot(epochs, va, "s-", label="val loss", linewidth=1.5, markersize=3)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(title)
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    if show_lr:
        ax_lr = ax_loss.twinx()
        ax_lr.plot(epochs, [h["lr"] for h in history], ":", color="gray",
                   linewidth=1, label="learning rate")
        ax_lr.set_ylabel("learning rate", color="gray")
        ax_lr.tick_params(axis="y", labelcolor="gray")
        ax_lr.legend(loc="center right")

    save_path = Path(save_path) if save_path else (DL_REPORTS_DIR / "learning_curve.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("learning curve saved -> %s", save_path)
    return save_path

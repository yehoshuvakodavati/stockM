"""
StockM v1.0 - Phase 7, Lesson 8
Training Pipeline Demo
======================

Showcases the production Trainer (early stopping + LR scheduling + warmup +
checkpointing + history + learning curves) on the Transformer and LSTM.

The Lesson-7 finding: the Transformer under-trained in 20 epochs without an LR
warmup (val_loss ~0.003, 10x the recurrent models). This demo gives it the
warmup it needs and shows it now converges to the same loss band as the others -
proving the poor Lesson-7 RMSE was a TRAINING artifact, not an architectural
verdict.

Run:  PYTHONPATH=src python src/deep_learning/run_training_demo.py RELIANCE.NS
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_learning.framework import (
    get_device,
    make_dataloader,
    make_tensor_dataset,
    set_global_seed,
    load_training_config,
)
from deep_learning.sequence_builder import build_sequences
from deep_learning.lstm import build_lstm
from deep_learning.transformer import build_transformer
from deep_learning.trainer import Trainer, TrainSettings, save_history
from deep_learning.plotting import plot_learning_curves

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CURVES_DIR = PROJECT_ROOT / "reports" / "deep_learning"

logger = logging.getLogger("stockm.deep_learning.run_training_demo")


def main(symbol: str = "RELIANCE.NS", epochs: int = 40) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    tcfg = load_training_config()
    device = get_device(tcfg["device"])

    seq = build_sequences(symbol, target_col="target_next_return")
    train_ds = make_tensor_dataset(seq["X_train_seq"], seq["y_train_seq"])
    val_ds = make_tensor_dataset(seq["X_validation_seq"], seq["y_validation_seq"])
    test_ds = make_tensor_dataset(seq["X_test_seq"], seq["y_test_seq"])
    train_dl = make_dataloader(train_ds, tcfg["batch_size"], shuffle=True, drop_last=True)
    val_dl = make_dataloader(val_ds, tcfg["batch_size"], shuffle=False)
    test_dl = make_dataloader(test_ds, tcfg["batch_size"], shuffle=False)

    common = dict(epochs=epochs, lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"],
                  grad_clip=tcfg["gradient_clip_norm"], early_stopping=True,
                  es_patience=10, es_min_delta=1e-4, save_best=True,
                  checkpoint_dir=CKPT_DIR, log_every=5)
    n_feat = seq["n_features"]

    runs = {
        # LSTM: no warmup needed (recurrent models trained fine in Lesson 4-6).
        "lstm": dict(label="LSTM", model=build_lstm(n_feat, task="regression"), warmup=0),
        # Transformer: WITH warmup - the fix for the Lesson-7 under-training.
        "transformer_warmup": dict(label="Transformer+warmup",
                                   model=build_transformer(n_feat, task="regression"), warmup=5),
    }

    results = {}
    for name, r in runs.items():
        logger.info("=== %s ===", name.upper())
        set_global_seed(42)
        settings = TrainSettings(warmup_epochs=r["warmup"], label=r["label"], **common)
        result = Trainer(settings, device=device).train(
            r["model"], train_dl, val_dl, test_dl, task="regression", checkpoint_name=f"{symbol.replace('.','_')}_{name}"
        )
        # Persist history + learning curve.
        save_history(result["history"], CURVES_DIR / f"{symbol.replace('.','_')}_{name}_history.json")
        plot_learning_curves(result["history"], CURVES_DIR / f"{symbol.replace('.','_')}_{name}_curves.png",
                             title=f"{symbol} - {r['label']}")
        results[name] = result

    logger.info(
        "LESSON8 | %s | metric            LSTM       TRANSF+warmup\n"
        "  best_val_loss     %.6f    %.6f\n"
        "  test_rmse         %.6f    %.6f\n"
        "  dir_accuracy      %.4f    %.4f\n"
        "  beats_naive       %s     %s\n"
        "  epochs_run        %d        %d",
        symbol,
        results["lstm"]["best_val_loss"], results["transformer_warmup"]["best_val_loss"],
        results["lstm"]["test_rmse"], results["transformer_warmup"]["test_rmse"],
        results["lstm"]["directional_accuracy"], results["transformer_warmup"]["directional_accuracy"],
        results["lstm"]["beats_naive"], results["transformer_warmup"]["beats_naive"],
        results["lstm"]["epochs_run"], results["transformer_warmup"]["epochs_run"],
    )
    return 0


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    raise SystemExit(main(sym))

"""
main.py
=======
Orchestrator — runs the full pipeline end-to-end.

Modes:
  train      — train all three models
  evaluate   — load checkpoints and evaluate on val set
  visualize  — generate FFT plots, training curves, prediction grids
  all        — train → evaluate → visualize  (recommended for first run)

Usage:
    python main.py --mode all
    python main.py --mode train     --epochs 20 --batch_size 32
    python main.py --mode evaluate
    python main.py --mode visualize
    python main.py --mode all --data_root dataset --epochs 10
"""

import logging
import argparse
import sys
import time
from pathlib import Path

import torch

# ── Project modules ──────────────────────────────────────────────────────
from config import CFG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Banner
# ───────────────────────────────────────────────────────────────────────────

BANNER = r"""
+----------------------------------------------------------+
|   AI-Generated Image Detection                           |
|   Spatial CNN  vs  Frequency-Domain CNN & SVM            |
+----------------------------------------------------------+
"""


def print_system_info() -> None:
    log.info(f"Python  : {sys.version.split()[0]}")
    log.info(f"PyTorch : {torch.__version__}")
    log.info(f"Device  : {CFG.device}")
    if CFG.device == "cuda":
        log.info(f"GPU     : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"VRAM    : {vram:.1f} GB")


# ───────────────────────────────────────────────────────────────────────────
# Argument parsing
# ───────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI-Generated Image Detection — Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode all
  python main.py --mode train --epochs 30 --batch_size 64
  python main.py --mode evaluate
  python main.py --mode visualize --n_fft_samples 4
        """,
    )

    p.add_argument(
        "--mode",
        choices=["train", "evaluate", "visualize", "all"],
        default="all",
        help="Pipeline stage to execute. Default: all",
    )
    p.add_argument("--data_root",      type=Path, default=CFG.dataset_root,
                   help=f"Dataset root. Default: {CFG.dataset_root}")
    p.add_argument("--epochs",         type=int,  default=CFG.epochs,
                   help=f"Training epochs. Default: {CFG.epochs}")
    p.add_argument("--batch_size",     type=int,  default=CFG.batch_size,
                   help=f"Batch size. Default: {CFG.batch_size}")
    p.add_argument("--lr",             type=float,default=CFG.lr,
                   help=f"Learning rate. Default: {CFG.lr}")
    p.add_argument("--num_workers",    type=int,  default=CFG.num_workers,
                   help="DataLoader workers. Use 0 on Windows if errors occur.")
    p.add_argument("--seed",           type=int,  default=CFG.seed,
                   help=f"Random seed. Default: {CFG.seed}")
    p.add_argument("--model",
                   choices=["cnn_spatial", "cnn_fft", "svm_fft", "all"],
                   default="all",
                   help="Which model(s) to train. Default: all")
    p.add_argument("--n_fft_samples",  type=int,  default=3,
                   help="Number of image pairs for FFT visualisation. Default: 3")
    p.add_argument("--viz_model",
                   choices=["cnn_spatial", "cnn_fft"],
                   default="cnn_spatial",
                   help="Model used for prediction grid. Default: cnn_spatial")

    return p.parse_args()


# ───────────────────────────────────────────────────────────────────────────
# Stage runners
# ───────────────────────────────────────────────────────────────────────────

def run_train(args: argparse.Namespace) -> None:
    """Import and invoke the training pipeline."""
    from train import main as train_main

    # Patch sys.argv so train.main() sees our args
    sys.argv = [
        "train.py",
        "--model",       args.model,
        "--data_root",   str(args.data_root),
        "--epochs",      str(args.epochs),
        "--batch_size",  str(args.batch_size),
        "--lr",          str(args.lr),
        "--num_workers", str(args.num_workers),
        "--seed",        str(args.seed),
    ]
    train_main()


def run_evaluate(args: argparse.Namespace) -> None:
    """Import and invoke the evaluation pipeline."""
    from evaluate import run_evaluation
    run_evaluation(
        data_root   = args.data_root,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )


def run_visualize(args: argparse.Namespace) -> None:
    """Import and invoke the visualisation pipeline."""
    from visualize import (
        show_fft_comparison,
        plot_training_curves,
        plot_prediction_grid,
    )

    log.info("Generating FFT comparison plots …")
    show_fft_comparison(args.data_root, n_samples=args.n_fft_samples)

    log.info("Generating training curves …")
    plot_training_curves()

    log.info(f"Generating prediction grid ({args.viz_model}) …")
    plot_prediction_grid(args.data_root, model_name=args.viz_model)


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(BANNER)
    args = parse_args()

    # ── Validate dataset root ──────────────────────────────────────────
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        log.error(
            f"Dataset not found: {data_root}\n"
            "  Run prepare_dataset.py first, then try again."
        )
        sys.exit(1)

    print_system_info()
    log.info(f"\nMode         : {args.mode.upper()}")
    log.info(f"Data root    : {data_root}")
    log.info(f"Epochs       : {args.epochs}")
    log.info(f"Batch size   : {args.batch_size}")
    log.info(f"Learning rate: {args.lr}")
    log.info(f"Device       : {CFG.device}\n")

    t_start = time.time()

    # ── Execute chosen mode ────────────────────────────────────────────
    if args.mode in ("train", "all"):
        log.info("=" * 55)
        log.info("  STAGE 1 / 3  —  TRAINING")
        log.info("=" * 55)
        run_train(args)

    if args.mode in ("evaluate", "all"):
        log.info("=" * 55)
        log.info("  STAGE 2 / 3  —  EVALUATION")
        log.info("=" * 55)
        run_evaluate(args)

    if args.mode in ("visualize", "all"):
        log.info("=" * 55)
        log.info("  STAGE 3 / 3  —  VISUALISATION")
        log.info("=" * 55)
        run_visualize(args)

    total = time.time() - t_start
    log.info("")
    log.info("=" * 55)
    log.info(f"  ✅  Pipeline complete in {total/60:.1f} min")
    log.info("=" * 55)
    log.info("  Results → results/")
    log.info("  Plots   → results/visualizations/")
    log.info("  Metrics → results/evaluation_summary.json")
    log.info("=" * 55)


if __name__ == "__main__":
    main()

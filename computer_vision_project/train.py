"""
train.py
========
Training loop for all three model variants:

  cnn_spatial  — Model A: SimpleCNN on raw RGB images
  cnn_fft      — Model B (CNN): SimpleCNN on FFT magnitude spectrum
  svm_fft      — Model B (SVM): RBF SVM on hand-crafted FFT features

Usage:
    # Train all models
    python train.py --model all

    # Train one model
    python train.py --model cnn_spatial --epochs 20 --batch_size 32
    python train.py --model cnn_fft     --epochs 20 --batch_size 32
    python train.py --model svm_fft

    # Override dataset root
    python train.py --model all --data_root dataset
"""

import os
import csv
import time
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config     import CFG
from dataset    import get_dataloaders
from preprocess import get_spatial_transforms, get_fft_transforms
from model_cnn  import build_spatial_cnn, build_fft_cnn, build_resnet18
from model_svm  import SVMClassifier
from fft_features import extract_dataset_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: nn.Module, path: Path, epoch: int, val_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"  : epoch,
        "val_acc": val_acc,
        "model"  : model.state_dict(),
    }, path)
    log.info(f"Checkpoint saved → {path}  (epoch={epoch}, val_acc={val_acc:.4f})")


class MetricLogger:
    """Appends epoch metrics to a CSV file."""

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file  = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ───────────────────────────────────────────────────────────────────────────
# CNN training loop
# ───────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model    : nn.Module,
    loader   : torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device   : torch.device,
    desc     : str = "Train",
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Returns:
        (avg_loss, accuracy)  both floats
    """
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs            = labels.size(0)
        running_loss += loss.item() * bs
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model    : nn.Module,
    loader   : torch.utils.data.DataLoader,
    criterion: nn.Module,
    device   : torch.device,
    desc     : str = "Val  ",
) -> tuple[float, float]:
    """
    Evaluate model on validation set.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for imgs, labels in tqdm(loader, desc=desc, leave=False, unit="batch"):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        bs            = labels.size(0)
        running_loss += loss.item() * bs
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += bs

    return running_loss / total, correct / total


def train_cnn(
    model_name  : str,
    model       : nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader  : torch.utils.data.DataLoader,
    epochs      : int,
    lr          : float,
    device      : torch.device,
) -> nn.Module:
    """
    Full training loop for a CNN model.

    Saves:
        checkpoints/<model_name>_best.pth   — best val-accuracy checkpoint
        logs/<model_name>_metrics.csv       — per-epoch metrics

    Returns the model with best weights loaded.
    """
    log.info(f"\n{'='*55}")
    log.info(f"  Training {model_name}  |  device={device}  |  epochs={epochs}")
    log.info(f"  Parameters: {model.num_parameters():,}")
    log.info(f"{'='*55}")

    model    = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=CFG.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    ckpt_path = CFG.checkpoint_dir / f"{model_name}_best.pth"

    best_val_acc  = 0.0
    best_epoch    = 0
    t0            = time.time()

    with MetricLogger(
        CFG.log_dir / f"{model_name}_metrics.csv",
        ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "elapsed_s"],
    ) as logger:
        for epoch in range(1, epochs + 1):
            t_epoch = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                desc=f"[{model_name}] E{epoch:02d} Train"
            )
            val_loss, val_acc = validate(
                model, val_loader, criterion, device,
                desc=f"[{model_name}] E{epoch:02d} Val  "
            )
            scheduler.step()
            elapsed = time.time() - t_epoch
            cur_lr  = scheduler.get_last_lr()[0]

            log.info(
                f"  Epoch {epoch:02d}/{epochs}  |  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  |  "
                f"lr={cur_lr:.2e}  ({elapsed:.1f}s)"
            )

            logger.log({
                "epoch"     : epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc" : f"{train_acc:.6f}",
                "val_loss"  : f"{val_loss:.6f}",
                "val_acc"   : f"{val_acc:.6f}",
                "lr"        : f"{cur_lr:.6e}",
                "elapsed_s" : f"{elapsed:.1f}",
            })

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch   = epoch
                save_checkpoint(model, ckpt_path, epoch, val_acc)

    total_time = time.time() - t0

    log.info(f"\n  Training complete in {total_time/60:.1f} min")
    log.info(f"  Best val_acc = {best_val_acc:.4f}  (epoch {best_epoch})")
    log.info(f"  Checkpoint   → {ckpt_path}")

    # Reload best weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    return model


# ───────────────────────────────────────────────────────────────────────────
# SVM training
# ───────────────────────────────────────────────────────────────────────────

def train_svm(data_root: Path) -> SVMClassifier:
    """
    Extract FFT features from all training images and fit the RBF SVM.

    The feature dataloader uses the raw spatial transform (ToTensor only,
    no normalisation) so pixel values are in [0, 1] — the FFT extractor
    handles its own normalisation internally.
    """
    log.info(f"\n{'='*55}")
    log.info("  Training svm_fft")
    log.info(f"{'='*55}")

    from torchvision import transforms as T
    raw_transform = T.Compose([
        T.Resize((CFG.image_size, CFG.image_size)),
        T.ToTensor(),  # no normalise — FFT extractor uses raw [0,1] values
    ])

    train_loader, val_loader, _ = get_dataloaders(
        root=data_root,
        train_transform=raw_transform,
        val_transform=raw_transform,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        seed=CFG.seed,
    )

    log.info("Extracting training features …")
    X_train, y_train = extract_dataset_features(
        train_loader, CFG.radial_bins, CFG.high_freq_pct
    )

    log.info("Extracting validation features …")
    X_val, y_val = extract_dataset_features(
        val_loader, CFG.radial_bins, CFG.high_freq_pct
    )

    clf = SVMClassifier(
        kernel=CFG.svm_kernel,
        C=CFG.svm_C,
        gamma=CFG.svm_gamma,
        seed=CFG.seed,
    )
    clf.train(X_train, y_train)

    val_acc = clf.score(X_val, y_val)
    log.info(f"  SVM val accuracy: {val_acc:.4f}")

    ckpt_path = CFG.checkpoint_dir / "svm_fft_best.pkl"
    clf.save(ckpt_path)

    return clf


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AI-image detection models.")
    p.add_argument(
        "--model", choices=["cnn_spatial", "cnn_fft", "svm_fft", "resnet18", "all"],
        default="all", help="Which model to train. Default: all"
    )
    p.add_argument("--data_root",  type=Path, default=CFG.dataset_root)
    p.add_argument("--epochs",     type=int,  default=CFG.epochs)
    p.add_argument("--batch_size", type=int,  default=CFG.batch_size)
    p.add_argument("--lr",         type=float,default=CFG.lr)
    p.add_argument("--seed",       type=int,  default=CFG.seed)
    p.add_argument(
        "--num_workers", type=int, default=CFG.num_workers,
        help="DataLoader worker processes. Use 0 on Windows if errors occur."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(CFG.device)
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU   : {torch.cuda.get_device_name(0)}")

    models_to_train = (
        ["cnn_spatial", "cnn_fft", "svm_fft", "resnet18"]
        if args.model == "all" else [args.model]
    )

    trained: dict = {}

    for model_name in models_to_train:

        # ── CNN Spatial ────────────────────────────────────────────────
        if model_name == "cnn_spatial":
            train_loader, val_loader, _ = get_dataloaders(
                root=args.data_root,
                train_transform=get_spatial_transforms("train", CFG.image_size, CFG.norm_mean, CFG.norm_std),
                val_transform  =get_spatial_transforms("val",   CFG.image_size, CFG.norm_mean, CFG.norm_std),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
            )
            model = build_spatial_cnn(CFG.cnn_channels, CFG.dropout)
            trained["cnn_spatial"] = train_cnn(
                "cnn_spatial", model, train_loader, val_loader,
                args.epochs, args.lr, device
            )

        # ── CNN FFT ────────────────────────────────────────────────────
        elif model_name == "cnn_fft":
            train_loader, val_loader, _ = get_dataloaders(
                root=args.data_root,
                train_transform=get_fft_transforms("train", CFG.image_size),
                val_transform  =get_fft_transforms("val",   CFG.image_size),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
            )
            model = build_fft_cnn(CFG.cnn_channels, CFG.dropout)
            trained["cnn_fft"] = train_cnn(
                "cnn_fft", model, train_loader, val_loader,
                args.epochs, args.lr, device
            )

        # ── SVM FFT ────────────────────────────────────────────────────
        elif model_name == "svm_fft":
            trained["svm_fft"] = train_svm(args.data_root)

        # ── ResNet-18  (Model C — transfer learning) ───────────────────
        elif model_name == "resnet18":
            train_loader, val_loader, _ = get_dataloaders(
                root=args.data_root,
                train_transform=get_spatial_transforms("train", CFG.image_size, CFG.norm_mean, CFG.norm_std),
                val_transform  =get_spatial_transforms("val",   CFG.image_size, CFG.norm_mean, CFG.norm_std),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
            )
            model = build_resnet18(pretrained=True, dropout=CFG.dropout)
            trained["resnet18"] = train_cnn(
                "resnet18", model, train_loader, val_loader,
                args.epochs, args.lr, device
            )

    log.info("\n✅  All requested models trained.")


if __name__ == "__main__":
    main()

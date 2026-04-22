"""
dataset.py
==========
PyTorch Dataset for the prepared AI-image detection dataset.

Expected folder layout (produced by prepare_dataset.py):

    dataset/
        train/
            real/   *.jpg
            fake/   *.jpg
        val/
            real/   *.jpg
            fake/   *.jpg

Label convention:
    0  →  real
    1  →  fake / AI-generated

Usage:
    from dataset import get_dataloaders
    from preprocess import get_spatial_transforms

    train_loader, val_loader = get_dataloaders(
        root="dataset",
        train_transform=get_spatial_transforms("train"),
        val_transform=get_spatial_transforms("val"),
        batch_size=32,
    )
"""

import random
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

log = logging.getLogger(__name__)

LABEL_MAP = {"real": 0, "fake": 1}
IMG_EXTS  = {".jpg", ".jpeg", ".png"}


class AIImageDataset(Dataset):
    """
    Binary image dataset: real (0) vs AI-generated / fake (1).

    Args:
        root      : path to split folder, e.g. ``dataset/train``
        transform : callable applied to each PIL Image before returning
        seed      : shuffle seed for reproducibility (applied once at build time)
    """

    def __init__(self, root: Path | str, transform=None, seed: int = 42):
        self.root      = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        self._build_index(seed)

    # ── Index building ────────────────────────────────────────────────

    def _build_index(self, seed: int) -> None:
        """Walk real/ and fake/ sub-folders and collect (path, label) pairs."""
        for label_name, label_idx in LABEL_MAP.items():
            folder = self.root / label_name
            if not folder.is_dir():
                log.warning(f"Label folder not found, skipping: {folder}")
                continue

            paths = [
                p for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            ]
            for p in paths:
                self.samples.append((p, label_idx))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root}. "
                "Run prepare_dataset.py first."
            )

        # Shuffle with fixed seed so real/fake are interleaved
        rng = random.Random(seed)
        rng.shuffle(self.samples)

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        log.info(
            f"[Dataset] {self.root.name}: "
            f"{len(self.samples)} images  (real={n_real}, fake={n_fake})"
        )

    # ── Dataset protocol ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        # Open and convert to RGB (handles palette, RGBA, grayscale sources)
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    # ── Helper ────────────────────────────────────────────────────────

    def class_counts(self) -> dict[str, int]:
        """Return {'real': N, 'fake': M} for display / class-weighting."""
        return {
            "real": sum(1 for _, l in self.samples if l == 0),
            "fake": sum(1 for _, l in self.samples if l == 1),
        }


# ───────────────────────────────────────────────────────────────────────────
# Factory function
# ───────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    root: str | Path,
    train_transform,
    val_transform,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train and val DataLoaders from the prepared dataset root.

    Args:
        root            : path to dataset root (contains train/ and val/)
        train_transform : transform applied during training
        val_transform   : transform applied during validation
        batch_size      : images per batch
        num_workers     : DataLoader worker processes (use 0 on Windows if errors occur)
        seed            : passed to AIImageDataset for shuffle reproducibility
        pin_memory      : pin host memory for faster GPU transfer

    Returns:
        (train_loader, val_loader, test_loader)
    """
    root = Path(root)

    train_dataset = AIImageDataset(root / "train", transform=train_transform, seed=seed)
    val_dataset   = AIImageDataset(root / "val",   transform=val_transform,   seed=seed)
    test_dataset  = AIImageDataset(root / "test",  transform=val_transform,   seed=seed) # Use val_transform (usually eval mode) for test

    # Use a Generator for reproducible DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        generator=g,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )

    log.info(
        f"[DataLoader] Train: {len(train_loader)} batches | "
        f"Val: {len(val_loader)} batches | "
        f"Test: {len(test_loader)} batches"
    )
    return train_loader, val_loader, test_loader

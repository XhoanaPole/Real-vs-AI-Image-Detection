"""
config.py
=========
Central configuration for the AI image detection project.
Edit values here; all other modules import from this file.
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────
    dataset_root : Path = Path("dataset")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir       : Path = Path("logs")
    results_dir   : Path = Path("results")

    # ── Image settings ────────────────────────────────────────────────
    image_size : int = 256          # images are resized to image_size × image_size
    in_channels_spatial: int = 3    # RGB
    in_channels_fft    : int = 1    # grayscale FFT magnitude

    # ── Training ──────────────────────────────────────────────────────
    batch_size  : int   = 32
    epochs      : int   = 20
    lr          : float = 1e-4
    weight_decay: float = 1e-4
    num_workers : int   = 0         # 0 = required on Windows CPU; set higher on Linux/GPU
    val_ratio   : float = 0.20

    # ── CNN architecture ──────────────────────────────────────────────
    cnn_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    dropout     : float = 0.5

    # ── SVM ───────────────────────────────────────────────────────────
    svm_kernel  : str  = "rbf"
    svm_C       : float = 1.0
    svm_gamma   : str  = "scale"

    # ── FFT features ──────────────────────────────────────────────────
    radial_bins  : int = 16         # bins for radial frequency histogram
    high_freq_pct: float = 0.25     # outer fraction = high-frequency region

    # ── Device ────────────────────────────────────────────────────────
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── ImageNet normalisation (used for spatial CNN) ─────────────────
    norm_mean: tuple = (0.485, 0.456, 0.406)
    norm_std : tuple = (0.229, 0.224, 0.225)

    # ── Reproducibility ───────────────────────────────────────────────
    seed: int = 42

    def __post_init__(self):
        # Create output directories on init
        for d in (self.checkpoint_dir, self.log_dir, self.results_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        lines = ["=" * 50, "  Configuration", "=" * 50]
        for k, v in self.__dict__.items():
            lines.append(f"  {k:<20}: {v}")
        lines.append("=" * 50)
        return "\n".join(lines)


# Singleton — import this everywhere
CFG = Config()

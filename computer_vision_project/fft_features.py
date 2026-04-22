"""
fft_features.py
===============
Frequency-domain feature extraction for the SVM pipeline (Model B, Option 2).

Three features are extracted from the log-magnitude FFT spectrum of each image:

  1. mean_energy         — average log-magnitude across the entire spectrum
  2. high_freq_ratio     — fraction of energy in the outer 25 % radial ring
                           (high-frequency noise patterns typical of GAN/diffusion)
  3. radial_distribution — 16-bin normalised histogram of energy w.r.t. radial
                           distance from the DC component (centre of spectrum)

Combined feature vector length: 2 + radial_bins  =  18 (default)

The same FFTTransform used for the CNN branch is applied here, ensuring
consistency across all frequency-domain code paths.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ───────────────────────────────────────────────────────────────────────────

def _image_to_magnitude(img_array: np.ndarray) -> np.ndarray:
    """
    Convert a numpy image array to its centred log-magnitude FFT spectrum.

    Args:
        img_array : uint8 or float array (H, W) grayscale  or  (H, W, 3) BGR/RGB

    Returns:
        magnitude : float64 array (H, W)  log(1 + |FFT|),  centre = DC component
    """
    # Ensure grayscale
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.astype(np.float64)

    fft_shifted = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    magnitude   = np.log1p(np.abs(fft_shifted))
    return magnitude


def _radial_mask(shape: tuple[int, int], r_min: float, r_max: float) -> np.ndarray:
    """
    Boolean mask for pixels whose radial distance from the image centre
    falls in [r_min, r_max).

    Args:
        shape         : (H, W)
        r_min, r_max  : radial band boundaries (pixels)

    Returns:
        bool array (H, W)
    """
    H, W    = shape
    cy, cx  = H / 2.0, W / 2.0
    Y, X    = np.ogrid[:H, :W]
    r       = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    return (r >= r_min) & (r < r_max)


def _max_radius(shape: tuple[int, int]) -> float:
    """Maximum possible radial distance from image centre to corner."""
    H, W = shape
    return np.sqrt((H / 2.0) ** 2 + (W / 2.0) ** 2)


# ───────────────────────────────────────────────────────────────────────────
# Main feature extractor
# ───────────────────────────────────────────────────────────────────────────

def extract_fft_features(
    img_array: np.ndarray,
    radial_bins: int = 16,
    high_freq_pct: float = 0.25,
) -> dict:
    """
    Extract hand-crafted frequency-domain features from a single image.

    Args:
        img_array     : numpy array (H, W) grayscale  or  (H, W, 3) BGR
        radial_bins   : number of radial histogram bins
        high_freq_pct : fraction of the radius counted as 'high frequency'
                        (0.25 = outer 25 % of the radial spectrum)

    Returns:
        dict with keys:
            "mean_energy"          : float  — mean log-magnitude
            "high_freq_ratio"      : float  — high-freq energy / total energy
            "radial_distribution"  : list[float]  — normalised radial histogram
    """
    magnitude  = _image_to_magnitude(img_array)
    total_e    = magnitude.sum()
    r_max      = _max_radius(magnitude.shape)

    # ── Feature 1: mean energy ─────────────────────────────────────────
    mean_energy = magnitude.mean()

    # ── Feature 2: high-frequency energy ratio ─────────────────────────
    cutoff_r    = r_max * (1.0 - high_freq_pct)   # inner boundary of high-freq ring
    hf_mask     = _radial_mask(magnitude.shape, cutoff_r, r_max)
    high_freq_e = magnitude[hf_mask].sum()
    high_freq_ratio = float(high_freq_e / (total_e + 1e-12))

    # ── Feature 3: radial distribution (histogram) ─────────────────────
    bin_edges  = np.linspace(0, r_max, radial_bins + 1)
    radial_hist = np.zeros(radial_bins, dtype=np.float64)

    H, W = magnitude.shape
    cy, cx = H / 2.0, W / 2.0
    Y, X = np.ogrid[:H, :W]
    r_grid = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    for i in range(radial_bins):
        mask = (r_grid >= bin_edges[i]) & (r_grid < bin_edges[i + 1])
        radial_hist[i] = magnitude[mask].sum()

    # Normalise so the histogram sums to 1
    hist_sum = radial_hist.sum()
    if hist_sum > 1e-12:
        radial_hist /= hist_sum

    return {
        "mean_energy"         : float(mean_energy),
        "high_freq_ratio"     : high_freq_ratio,
        "radial_distribution" : radial_hist.tolist(),
    }


def features_to_vector(features: dict) -> np.ndarray:
    """
    Flatten the feature dict to a 1-D numpy array suitable for SVM input.

    Shape: (2 + radial_bins,)  →  default (18,)
    """
    return np.array(
        [features["mean_energy"], features["high_freq_ratio"]]
        + features["radial_distribution"],
        dtype=np.float32,
    )


# ───────────────────────────────────────────────────────────────────────────
# Dataset-level extraction  (used to build SVM training matrix)
# ───────────────────────────────────────────────────────────────────────────

def extract_dataset_features(
    dataloader: DataLoader,
    radial_bins: int = 16,
    high_freq_pct: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterate over *dataloader* and extract FFT features for every image.

    The DataLoader should return raw (un-normalised) tensors — use the
    FFT or *spatial* pipeline with ToTensor() only (no Normalize).

    Args:
        dataloader    : a DataLoader whose images are tensors (B, C, H, W)
                        or (B, 1, H, W) in range [0, 1]
        radial_bins   : passed to extract_fft_features
        high_freq_pct : passed to extract_fft_features

    Returns:
        X : float32 array (N, feature_dim)
        y : int array     (N,)            label 0=real, 1=fake
    """
    import torch

    X_list: list[np.ndarray] = []
    y_list: list[int]        = []

    log.info("Extracting FFT features for SVM …")
    for batch_idx, (imgs, labels) in enumerate(dataloader):
        # imgs: (B, C, H, W) float tensor in [0, 1]
        imgs_np = (imgs.numpy() * 255).astype(np.uint8)  # (B, C, H, W) uint8

        for i in range(imgs_np.shape[0]):
            # Convert (C, H, W) → (H, W, C) then grayscale via cv2
            img_chw = imgs_np[i]                        # (C, H, W)
            img_hwc = np.transpose(img_chw, (1, 2, 0))  # (H, W, C)

            feats = extract_fft_features(img_hwc, radial_bins, high_freq_pct)
            vec   = features_to_vector(feats)
            X_list.append(vec)
            y_list.append(int(labels[i]))

        if (batch_idx + 1) % 10 == 0:
            log.info(f"  Processed {(batch_idx + 1) * dataloader.batch_size} images …")

    X = np.stack(X_list, axis=0)  # (N, feature_dim)
    y = np.array(y_list, dtype=np.int64)

    log.info(f"Feature matrix shape: {X.shape}  |  Labels shape: {y.shape}")
    return X, y


# ───────────────────────────────────────────────────────────────────────────
# Visualisation — FFT spectrum comparison
# ───────────────────────────────────────────────────────────────────────────

def visualize_fft(
    real_img:  np.ndarray,
    fake_img:  np.ndarray,
    save_path: Path | str | None = None,
    title:     str = "FFT Spectrum: Real vs AI-Generated",
) -> None:
    """
    Plot side-by-side comparison of a real and a fake image alongside
    their centred log-magnitude FFT spectra.

    Args:
        real_img  : (H, W, 3) uint8 RGB array
        fake_img  : (H, W, 3) uint8 RGB array
        save_path : save figure to this path if provided
        title     : overall figure title
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    for row, (img, label) in enumerate([(real_img, "Real"), (fake_img, "AI-Generated (Fake)")]):
        # --- Column 0: original image ---
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{label}\n(Original)")
        axes[row, 0].axis("off")

        # Grayscale for FFT
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # --- Column 1: FFT log magnitude (colour-mapped) ---
        magnitude = _image_to_magnitude(gray)
        mn, mx    = magnitude.min(), magnitude.max()
        magnitude_norm = (magnitude - mn) / (mx - mn + 1e-12)

        im = axes[row, 1].imshow(magnitude_norm, cmap="inferno", origin="upper")
        axes[row, 1].set_title(f"{label}\nLog-Magnitude FFT Spectrum")
        axes[row, 1].axis("off")
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)

        # --- Column 2: radial energy distribution ---
        feats = extract_fft_features(gray)
        radial = feats["radial_distribution"]
        bins   = range(len(radial))
        color  = "#2196F3" if label == "Real" else "#F44336"
        axes[row, 2].bar(bins, radial, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
        axes[row, 2].set_title(f"{label}\nRadial Energy Distribution")
        axes[row, 2].set_xlabel("Frequency bin (inner → outer)")
        axes[row, 2].set_ylabel("Normalised energy")
        axes[row, 2].text(
            0.98, 0.95,
            f"Mean energy: {feats['mean_energy']:.3f}\n"
            f"High-freq ratio: {feats['high_freq_ratio']:.3f}",
            transform=axes[row, 2].transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"FFT comparison saved → {save_path}")

    plt.close(fig)

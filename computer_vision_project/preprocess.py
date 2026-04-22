"""
preprocess.py
=============
Image transforms for both pipelines:

  Pipeline A  — Spatial CNN
      Raw RGB image → normalised tensor  (3 × 256 × 256)

  Pipeline B  — Frequency CNN / SVM
      Grayscale → 2-D FFT → log-magnitude → normalised tensor  (1 × 256 × 256)

The same AIImageDataset class is used for both pipelines — only the
transform changes.  This guarantees identical preprocessing for real
and fake images.
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ───────────────────────────────────────────────────────────────────────────
# Custom transform: PIL Image → FFT magnitude tensor
# ───────────────────────────────────────────────────────────────────────────

class FFTTransform:
    """
    Convert a PIL Image to its 2-D FFT log-magnitude spectrum.

    Steps:
      1. Convert PIL Image to grayscale numpy array   [H, W]  float64 in [0, 255]
      2. Apply 2-D FFT with numpy.fft.fft2
      3. Shift zero-frequency component to centre     numpy.fft.fftshift
      4. Compute log-scaled magnitude: log(1 + |F|)
      5. Min-max normalise to [0, 1]
      6. Convert to float32 torch tensor  [1, H, W]

    The result is visually interpretable: bright spots in the centre are
    low-frequency (smooth) components; high-frequency patterns appears at the
    edges.  AI-generated images often show characteristic grid artefacts.
    """

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Step 1: grayscale
        gray = np.array(img.convert("L"), dtype=np.float64)   # (H, W)

        # Step 2 + 3: FFT and shift
        fft_shifted = np.fft.fftshift(np.fft.fft2(gray))

        # Step 4: log magnitude
        magnitude = np.log1p(np.abs(fft_shifted))             # log(1 + |F|)

        # Step 5: min-max normalise to [0, 1]
        mn, mx = magnitude.min(), magnitude.max()
        if mx - mn > 1e-8:
            magnitude = (magnitude - mn) / (mx - mn)
        else:
            magnitude = np.zeros_like(magnitude)

        # Step 6: to torch tensor  (1, H, W)  float32
        tensor = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)
        return tensor


# ───────────────────────────────────────────────────────────────────────────
# Spatial transforms  (Pipeline A)
# ───────────────────────────────────────────────────────────────────────────

def get_spatial_transforms(split: str, image_size: int = 256,
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)) -> transforms.Compose:
    """
    Return torchvision transform pipeline for raw RGB images.

    Training augmentations are intentionally conservative — we do NOT
    want to destroy forensic artefacts with heavy augmentation.

    Args:
        split      : "train" or "val"
        image_size : target spatial resolution (default 256)
        mean / std : normalisation statistics (ImageNet defaults)

    Returns:
        torchvision.transforms.Compose
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),                       # → [0, 1]  (3, H, W)
            transforms.Normalize(mean=mean, std=std),    # ImageNet stats
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ───────────────────────────────────────────────────────────────────────────
# FFT-magnitude transforms  (Pipeline B — CNN branch)
# ───────────────────────────────────────────────────────────────────────────

def get_fft_transforms(split: str, image_size: int = 256) -> transforms.Compose:
    """
    Return transform pipeline that outputs an FFT magnitude tensor.

    No heavy augmentation — we preserve frequency-domain structure.
    A small random horizontal flip is still safe because the magnitude
    spectrum is symmetric.

    Args:
        split      : "train" or "val"
        image_size : resize before FFT (default 256)

    Returns:
        torchvision.transforms.Compose  producing tensors of shape (1, H, W)
    """
    base = [
        transforms.Resize((image_size, image_size)),
    ]

    if split == "train":
        base.append(transforms.RandomHorizontalFlip(p=0.5))

    base.append(FFTTransform())   # PIL → FFT magnitude tensor (1, H, W)

    return transforms.Compose(base)

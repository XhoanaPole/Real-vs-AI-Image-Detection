"""
model_cnn.py
============
SimpleCNN — a lightweight 4-block convolutional neural network.

Used for BOTH pipelines:
  - Model A  (spatial)  : in_channels=3  (RGB image)
  - Model B  (FFT-CNN)  : in_channels=1  (FFT magnitude spectrum)

Architecture:

    Input  (C × 256 × 256)
      ↓
    ConvBlock 1:  C  → 32    Conv(3×3) → BN → ReLU → MaxPool(2×2)   128×128
      ↓
    ConvBlock 2:  32 → 64    Conv(3×3) → BN → ReLU → MaxPool(2×2)    64×64
      ↓
    ConvBlock 3:  64 → 128   Conv(3×3) → BN → ReLU → MaxPool(2×2)    32×32
      ↓
    ConvBlock 4:  128 → 256  Conv(3×3) → BN → ReLU → MaxPool(2×2)    16×16
      ↓
    Global Average Pooling   → (256,)
      ↓
    FC(256 → 128) → ReLU → Dropout(p)
      ↓
    FC(128 → 2)              (logits for CrossEntropyLoss)

Total params ≈ 1.5 M — designed to fit comfortably in 6 GB VRAM at batch_size=32.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ───────────────────────────────────────────────────────────────────────────
# Building block
# ───────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Conv2d → BatchNorm2d → ReLU → MaxPool2d

    Args:
        in_ch  : number of input feature maps
        out_ch : number of output feature maps
        kernel : convolution kernel size (default 3)
        pool   : max-pool kernel / stride size (default 2)
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool, stride=pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ───────────────────────────────────────────────────────────────────────────
# Main model
# ───────────────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    Lightweight CNN for binary real vs AI-generated image classification.

    Args:
        in_channels  : 3 for RGB spatial pipeline, 1 for FFT magnitude pipeline
        channels     : list of 4 output channel counts for each ConvBlock
                       default [32, 64, 128, 256]
        dropout      : dropout probability in the classifier head
        num_classes  : number of output classes (2 for binary classification)
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: list[int] | None = None,
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 256]

        assert len(channels) == 4, "Expected exactly 4 channel values."

        # ── Convolutional backbone ────────────────────────────────────
        self.features = nn.Sequential(
            ConvBlock(in_channels, channels[0]),  # 256 → 128
            ConvBlock(channels[0], channels[1]),  # 128 →  64
            ConvBlock(channels[1], channels[2]),  #  64 →  32
            ConvBlock(channels[2], channels[3]),  #  32 →  16
        )

        # ── Global Average Pooling ────────────────────────────────────
        # Output: (batch, channels[3], 1, 1)  → flatten to (batch, channels[3])
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[3], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        # ── Weight initialisation ─────────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal for Conv layers, constant for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W) tensor
        Returns:
            logits : (B, num_classes) tensor  — raw scores (no softmax)
        """
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ───────────────────────────────────────────────────────────────────────────
# Quick model factory
# ───────────────────────────────────────────────────────────────────────────

def build_spatial_cnn(channels=None, dropout=0.5) -> SimpleCNN:
    """Build Model A: 3-channel spatial CNN."""
    return SimpleCNN(in_channels=3, channels=channels, dropout=dropout)


def build_fft_cnn(channels=None, dropout=0.5) -> SimpleCNN:
    """Build Model B (CNN branch): 1-channel FFT magnitude CNN."""
    return SimpleCNN(in_channels=1, channels=channels, dropout=dropout)


def build_resnet18(pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    """
    Build Model C: pretrained ResNet-18 fine-tuned for binary classification.

    The final FC layer is replaced with Dropout → Linear(512 → 2).
    All earlier layers start from ImageNet weights (transfer learning).

    Args:
        pretrained : use ImageNet weights (strongly recommended)
        dropout    : dropout probability before the final linear layer

    Returns:
        nn.Module with .num_parameters() helper attached
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model   = resnet18(weights=weights)

    in_features = model.fc.in_features          # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 2),
    )

    # Attach the same helper as SimpleCNN so training code is uniform
    model.num_parameters = lambda: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return model


# ───────────────────────────────────────────────────────────────────────────
# Self-test
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    # Spatial model
    m_spatial = build_spatial_cnn()
    x_spatial = torch.randn(4, 3, 256, 256)
    out = m_spatial(x_spatial)
    assert out.shape == (4, 2), f"Unexpected output shape: {out.shape}"
    print(f"✓  Spatial CNN  — params: {m_spatial.num_parameters():,}  | output: {out.shape}")

    # FFT model
    m_fft = build_fft_cnn()
    x_fft = torch.randn(4, 1, 256, 256)
    out   = m_fft(x_fft)
    assert out.shape == (4, 2)
    print(f"✓  FFT CNN      — params: {m_fft.num_parameters():,}  | output: {out.shape}")

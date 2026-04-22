"""
visualize.py
============
Visualisation utilities:

  1. FFT spectrum comparison — real vs fake side-by-side
  2. Training curve plots   — loss + accuracy over epochs
  3. Sample prediction grid — model predictions on random val images

All outputs are saved to results/visualizations/.

Usage:
    python visualize.py
    python visualize.py --data_root dataset --n_samples 4
"""

import logging
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from PIL import Image

from config      import CFG
from fft_features import visualize_fft

log = logging.getLogger(__name__)

VIZ_DIR = CFG.results_dir / "visualizations"


# ───────────────────────────────────────────────────────────────────────────
# 0. GradCAM
# ───────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for any CNN.

    Usage:
        cam = GradCAM(model, target_layer=model.layer4[-1])  # ResNet-18
        heatmap, class_idx = cam(input_tensor)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,  # (1, C, H, W)
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Compute GradCAM heatmap.

        Returns:
            heatmap   : float32 (H, W) in [0, 1]
            class_idx : predicted class used for backward pass
        """
        self.model.eval()
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1).squeeze()  # (h, w)
        cam     = torch.relu(cam).cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _overlay_heatmap(
    img_rgb: np.ndarray,   # (H, W, 3) uint8
    cam:     np.ndarray,   # (h, w) float [0,1]
    alpha:   float = 0.45,
) -> np.ndarray:
    """Resize CAM to image size and blend as a jet colourmap overlay."""
    H, W = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    heatmap     = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended     = (alpha * heatmap_rgb + (1 - alpha) * img_rgb).astype(np.uint8)
    return blended


def plot_gradcam(
    dataset_root: Path,
    model_name:   str  = "resnet18",
    n_images:     int  = 4,
    save_dir:     Path = VIZ_DIR,
) -> None:
    """
    Generate GradCAM heatmaps for *n_images* images (half real, half fake).

    Supported models: 'resnet18', 'cnn_spatial'.

    For each image produces a 3-column figure:
      Original | GradCAM overlay | FFT spectrum
    """
    from preprocess import get_spatial_transforms
    from model_cnn  import build_spatial_cnn, build_resnet18

    device = torch.device(CFG.device)

    if model_name == "resnet18":
        ckpt_path    = CFG.checkpoint_dir / "resnet18_best.pth"
        model        = build_resnet18(pretrained=False, dropout=CFG.dropout)
        target_layer = model.layer4[-1]           # last ResNet block
    elif model_name == "cnn_spatial":
        ckpt_path    = CFG.checkpoint_dir / "cnn_spatial_best.pth"
        model        = build_spatial_cnn(CFG.cnn_channels, CFG.dropout)
        target_layer = model.features[-1].block[-2]  # last BN before pool
    else:
        log.warning(f"GradCAM not supported for: {model_name}")
        return

    if not ckpt_path.exists():
        log.warning(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    cam_gen   = GradCAM(model, target_layer)
    transform = get_spatial_transforms("val", CFG.image_size, CFG.norm_mean, CFG.norm_std)

    real_dir   = dataset_root / "val" / "real"
    fake_dir   = dataset_root / "val" / "fake"
    half       = n_images // 2
    real_paths = random.sample(list(real_dir.glob("*.jpg")), min(half, len(list(real_dir.glob("*.jpg")))))
    fake_paths = random.sample(list(fake_dir.glob("*.jpg")), min(half, len(list(fake_dir.glob("*.jpg")))))
    samples    = [(p, 0, "Real") for p in real_paths] + [(p, 1, "Fake") for p in fake_paths]

    save_dir.mkdir(parents=True, exist_ok=True)
    label_names = {0: "Real", 1: "Fake"}

    for img_path, true_label, label_str in samples:
        orig_rgb = np.array(Image.open(img_path).convert("RGB"))

        tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        tensor.requires_grad_(True)

        heatmap, pred_idx = cam_gen(tensor)
        pred_label = label_names[pred_idx]
        correct    = pred_idx == true_label

        overlay = _overlay_heatmap(orig_rgb, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        title_color = "#4CAF50" if correct else "#F44336"
        fig.suptitle(
            f"GradCAM — {model_name}  |  GT: {label_str}  →  Pred: {pred_label}  "
            f"({'✓' if correct else '✗'})",
            fontsize=13, fontweight="bold", color=title_color,
        )

        axes[0].imshow(orig_rgb);     axes[0].set_title("Original");         axes[0].axis("off")
        axes[1].imshow(overlay);      axes[1].set_title("GradCAM Overlay");  axes[1].axis("off")

        from fft_features import _image_to_magnitude
        mag = _image_to_magnitude(orig_rgb)
        mn, mx = mag.min(), mag.max()
        mag_norm = (mag - mn) / (mx - mn + 1e-8)
        axes[2].imshow(mag_norm, cmap="inferno"); axes[2].set_title("FFT Spectrum"); axes[2].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        stem = img_path.stem
        out  = save_dir / f"gradcam_{model_name}_{label_str.lower()}_{stem}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"GradCAM → {out}")

    cam_gen.remove_hooks()
    log.info(f"GradCAM plots saved → {save_dir}")


# ───────────────────────────────────────────────────────────────────────────
# 1. FFT Spectrum Comparison
# ───────────────────────────────────────────────────────────────────────────

def show_fft_comparison(
    dataset_root: Path,
    n_samples   : int = 3,
    save_dir    : Path = VIZ_DIR,
) -> None:
    """
    Pick *n_samples* random real and fake images and plot their FFT spectra.

    For each pair, produces one figure with 2 rows × 3 cols:
      Row 0 = real image  │ FFT spectrum │ radial distribution
      Row 1 = fake image  │ FFT spectrum │ radial distribution

    Saves to save_dir/fft_comparison_<i>.png
    """
    real_dir = dataset_root / "val" / "real"
    fake_dir = dataset_root / "val" / "fake"

    real_paths = list(real_dir.glob("*.jpg"))
    fake_paths = list(fake_dir.glob("*.jpg"))

    if not real_paths or not fake_paths:
        log.warning(
            "Val real/fake images not found. "
            "Run prepare_dataset.py first."
        )
        return

    n = min(n_samples, len(real_paths), len(fake_paths))
    real_sample = random.sample(real_paths, n)
    fake_sample = random.sample(fake_paths, n)

    save_dir.mkdir(parents=True, exist_ok=True)

    for i, (rp, fp) in enumerate(zip(real_sample, fake_sample)):
        real_img = np.array(Image.open(rp).convert("RGB"))
        fake_img = np.array(Image.open(fp).convert("RGB"))

        out_path = save_dir / f"fft_comparison_{i+1}.png"
        visualize_fft(real_img, fake_img, save_path=out_path)

    log.info(f"FFT comparison plots saved → {save_dir}")


# ───────────────────────────────────────────────────────────────────────────
# 2. Training Curves
# ───────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    log_dir: Path = CFG.log_dir,
    save_dir: Path = VIZ_DIR,
) -> None:
    """
    Read <model>_metrics.csv files from log_dir and plot:
      - Loss curves (train + val)
      - Accuracy curves (train + val)

    Creates one figure per model plus a combined accuracy comparison.
    """
    csv_files = list(log_dir.glob("*_metrics.csv"))
    if not csv_files:
        log.warning(f"No metrics CSV files found in {log_dir}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    colours = {
        "cnn_spatial": {"train": "#1565C0", "val": "#42A5F5"},
        "cnn_fft"    : {"train": "#2E7D32", "val": "#66BB6A"},
        "svm_fft"    : {"train": "#BF360C", "val": "#FF7043"},
    }

    all_val_accs = {}

    for csv_path in csv_files:
        model_name = csv_path.stem.replace("_metrics", "")
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if df.empty:
            continue

        col = colours.get(model_name, {"train": "blue", "val": "orange"})
        epochs = df["epoch"].values

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Training Curves — {model_name}", fontsize=14, fontweight="bold")

        # Loss
        ax_loss.plot(epochs, df["train_loss"], color=col["train"], lw=2, label="Train loss")
        ax_loss.plot(epochs, df["val_loss"],   color=col["val"],   lw=2, linestyle="--", label="Val loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)

        # Accuracy
        ax_acc.plot(epochs, df["train_acc"], color=col["train"], lw=2, label="Train acc")
        ax_acc.plot(epochs, df["val_acc"],   color=col["val"],   lw=2, linestyle="--", label="Val acc")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Accuracy")
        ax_acc.set_ylim(0, 1)
        ax_acc.legend()
        ax_acc.grid(alpha=0.3)

        plt.tight_layout()
        out = save_dir / f"training_curves_{model_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Training curve → {out}")

        all_val_accs[model_name] = (epochs, df["val_acc"].values)

    # Combined val accuracy comparison
    if len(all_val_accs) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Validation Accuracy — All Models", fontsize=14, fontweight="bold")

        for model_name, (epochs, val_acc) in all_val_accs.items():
            col = colours.get(model_name, {}).get("val", "grey")
            label_map = {
                "cnn_spatial": "Model A — Spatial CNN",
                "cnn_fft"    : "Model B — FFT CNN",
                "svm_fft"    : "Model B — FFT SVM",
            }
            ax.plot(epochs, val_acc, color=col, lw=2,
                    label=label_map.get(model_name, model_name))

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="red", linestyle=":", lw=1.2, alpha=0.5, label="Random baseline")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out = save_dir / "val_accuracy_all_models.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Combined val accuracy → {out}")


# ───────────────────────────────────────────────────────────────────────────
# 3. Sample Prediction Grid
# ───────────────────────────────────────────────────────────────────────────

def plot_prediction_grid(
    dataset_root: Path,
    model_name  : str = "cnn_spatial",
    n_images    : int = 8,
    save_dir    : Path = VIZ_DIR,
) -> None:
    """
    Show a grid of val images with model predictions vs ground truth.

    Green border = correct prediction.
    Red border   = wrong prediction.
    """
    from preprocess import get_spatial_transforms, get_fft_transforms
    from model_cnn  import build_spatial_cnn, build_fft_cnn, build_resnet18

    device = torch.device(CFG.device)

    # Load model
    if model_name == "cnn_spatial":
        ckpt_path = CFG.checkpoint_dir / "cnn_spatial_best.pth"
        model     = build_spatial_cnn(CFG.cnn_channels, CFG.dropout)
        transform = get_spatial_transforms("val")
    elif model_name == "cnn_fft":
        ckpt_path = CFG.checkpoint_dir / "cnn_fft_best.pth"
        model     = build_fft_cnn(CFG.cnn_channels, CFG.dropout)
        transform = get_fft_transforms("val")
    elif model_name == "resnet18":
        ckpt_path = CFG.checkpoint_dir / "resnet18_best.pth"
        model     = build_resnet18(pretrained=False, dropout=CFG.dropout)
        transform = get_spatial_transforms("val")
    else:
        log.warning(f"Prediction grid not supported for: {model_name}")
        return

    if not ckpt_path.exists():
        log.warning(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    # Gather sample paths: half real, half fake
    real_dir  = dataset_root / "val" / "real"
    fake_dir  = dataset_root / "val" / "fake"
    half      = n_images // 2
    real_paths = random.sample(list(real_dir.glob("*.jpg")), min(half, len(list(real_dir.glob("*.jpg")))))
    fake_paths = random.sample(list(fake_dir.glob("*.jpg")), min(half, len(list(fake_dir.glob("*.jpg")))))

    samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    random.shuffle(samples)

    nrow = 2
    ncol = max(len(samples) // nrow, 1)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3.5))
    axes = np.array(axes).reshape(nrow, ncol)
    fig.suptitle(f"Predictions — {model_name}", fontsize=14, fontweight="bold")

    label_names = {0: "Real", 1: "Fake"}

    with torch.no_grad():
        for idx, (path, true_label) in enumerate(samples):
            row, col = divmod(idx, ncol)
            if row >= nrow:
                break
            ax = axes[row, col]

            # Display original RGB
            orig = np.array(Image.open(path).convert("RGB"))
            ax.imshow(orig)
            ax.axis("off")

            # Predict
            pil_img = Image.open(path).convert("RGB")
            tensor  = transform(pil_img).unsqueeze(0).to(device)
            logits  = model(tensor)
            pred    = int(logits.argmax(dim=1).item())
            conf    = float(torch.softmax(logits, dim=1)[0, pred].item())

            correct = pred == true_label
            color   = "#4CAF50" if correct else "#F44336"

            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

            ax.set_title(
                f"GT: {label_names[true_label]}\n"
                f"Pred: {label_names[pred]} ({conf:.0%})",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    # Hide any unused axes
    for idx in range(len(samples), nrow * ncol):
        row, col = divmod(idx, ncol)
        if row < nrow:
            axes[row, col].axis("off")

    green_patch = mpatches.Patch(color="#4CAF50", label="Correct")
    red_patch   = mpatches.Patch(color="#F44336", label="Incorrect")
    fig.legend(handles=[green_patch, red_patch], loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"predictions_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Prediction grid → {out}")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate visualizations.")
    p.add_argument("--data_root",  type=Path, default=CFG.dataset_root)
    p.add_argument("--n_samples",  type=int,  default=3,
                   help="Number of image pairs for FFT comparison. Default: 3")
    p.add_argument("--model_name", type=str,  default="resnet18",
                   choices=["cnn_spatial", "cnn_fft", "resnet18"],
                   help="Model to use for prediction grid and GradCAM.")
    p.add_argument("--gradcam_n",  type=int,  default=4,
                   help="Number of images for GradCAM (half real, half fake).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    log.info("Generating FFT comparison plots …")
    show_fft_comparison(args.data_root, n_samples=args.n_samples)

    log.info("Generating training curves …")
    plot_training_curves()

    log.info(f"Generating prediction grid ({args.model_name}) …")
    plot_prediction_grid(args.data_root, model_name=args.model_name)

    log.info(f"Generating GradCAM heatmaps ({args.model_name}) …")
    plot_gradcam(args.data_root, model_name=args.model_name, n_images=args.gradcam_n)

    log.info(f"\n✅  All visualizations saved → {VIZ_DIR}")


if __name__ == "__main__":
    main()

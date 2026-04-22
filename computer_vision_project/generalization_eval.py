"""
generalization_eval.py
======================
Cross-generator generalization study.

Research question:
    Does a model trained on images from Generator A still detect images
    from Generator B (which it has never seen)?

This answers the most important open question in AI image detection:
    "Is the detector learning general AI artifacts, or just memorizing
    one generator's specific patterns?"

Expected dataset layout
-----------------------
    dataset/
        generators/
            stable_diffusion/
                real/   *.jpg   (same real photos, symlinked or copied)
                fake/   *.jpg   (SD-generated images)
            dalle/
                real/   *.jpg
                fake/   *.jpg
            midjourney/
                real/   *.jpg
                fake/   *.jpg

Each generator folder must contain real/ and fake/ sub-folders.
Real images can be shared across generators (identical folder or symlinks).

Usage
-----
    # Evaluate resnet18 across all generators
    python generalization_eval.py --model resnet18

    # Evaluate cnn_spatial across all generators
    python generalization_eval.py --model cnn_spatial

    # Specify custom generator root
    python generalization_eval.py --model resnet18 --gen_root dataset/generators
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from PIL import Image

from config     import CFG
from preprocess import get_spatial_transforms, get_fft_transforms
from model_cnn  import build_spatial_cnn, build_fft_cnn, build_resnet18

log = logging.getLogger(__name__)

LABEL_MAP = {"real": 0, "fake": 1}
IMG_EXTS  = {".jpg", ".jpeg", ".png"}


# ───────────────────────────────────────────────────────────────────────────
# Data helpers
# ───────────────────────────────────────────────────────────────────────────

def load_generator_images(
    gen_folder: Path,
    transform,
    device: torch.device,
    max_per_class: int = 500,
) -> tuple[torch.Tensor, np.ndarray] | None:
    """
    Load all real + fake images from a generator folder into tensors.

    Args:
        gen_folder    : path like dataset/generators/stable_diffusion
        transform     : torchvision transform to apply to each PIL image
        device        : torch device
        max_per_class : cap per class to keep evaluation fast

    Returns:
        (tensors, labels) or None if folder is missing / empty
    """
    all_tensors = []
    all_labels  = []

    for label_name, label_idx in LABEL_MAP.items():
        folder = gen_folder / label_name
        if not folder.is_dir():
            log.warning(f"  Missing: {folder}")
            return None

        paths = [p for p in sorted(folder.iterdir())
                 if p.is_file() and p.suffix.lower() in IMG_EXTS][:max_per_class]

        if not paths:
            log.warning(f"  No images in {folder}")
            return None

        for p in paths:
            try:
                img    = Image.open(p).convert("RGB")
                tensor = transform(img)
                all_tensors.append(tensor)
                all_labels.append(label_idx)
            except Exception as e:
                log.debug(f"  Skipping {p.name}: {e}")

    if not all_tensors:
        return None

    X = torch.stack(all_tensors)   # (N, C, H, W)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


# ───────────────────────────────────────────────────────────────────────────
# Model loader
# ───────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, device: torch.device) -> torch.nn.Module | None:
    """Load a trained model from checkpoints/."""
    ckpt_map = {
        "cnn_spatial": (CFG.checkpoint_dir / "cnn_spatial_best.pth", lambda: build_spatial_cnn(CFG.cnn_channels, CFG.dropout)),
        "cnn_fft"    : (CFG.checkpoint_dir / "cnn_fft_best.pth",     lambda: build_fft_cnn(CFG.cnn_channels, CFG.dropout)),
        "resnet18"   : (CFG.checkpoint_dir / "resnet18_best.pth",     lambda: build_resnet18(pretrained=False, dropout=CFG.dropout)),
    }

    if model_name not in ckpt_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(ckpt_map)}")

    ckpt_path, builder = ckpt_map[model_name]
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}  (train it first)")
        return None

    model = builder()
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    log.info(f"Loaded {model_name} from {ckpt_path}")
    return model


# ───────────────────────────────────────────────────────────────────────────
# Evaluation
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_generator(
    model:      torch.nn.Module,
    X:          torch.Tensor,
    y:          np.ndarray,
    device:     torch.device,
    batch_size: int = 64,
) -> dict:
    """Run model over (X, y) in batches and return accuracy + AUC."""
    all_preds = []
    all_probs = []

    for start in tqdm(range(0, len(X), batch_size), desc="  Evaluating", leave=False):
        batch  = X[start:start + batch_size].to(device)
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_probs.append(probs)

    preds_all = np.concatenate(all_preds)
    probs_all = np.concatenate(all_probs)

    acc = float(accuracy_score(y, preds_all))
    try:
        auc = float(roc_auc_score(y, probs_all))
    except ValueError:
        auc = float("nan")

    n_real = int((y == 0).sum())
    n_fake = int((y == 1).sum())
    return {"accuracy": acc, "auc": auc, "n_real": n_real, "n_fake": n_fake}


# ───────────────────────────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────────────────────────

def plot_generalization_heatmap(
    results: dict[str, dict],
    model_name: str,
    save_path: Path,
) -> None:
    """
    Heatmap: rows = generators evaluated on, columns = accuracy / AUC.
    """
    generators = list(results.keys())
    accs = [results[g]["accuracy"] * 100 for g in generators]
    aucs = [results[g]["auc"] for g in generators]

    data = np.array([accs, aucs]).T   # (n_gen, 2)

    fig, ax = plt.subplots(figsize=(6, max(3, len(generators) * 0.8 + 1)))
    sns.heatmap(
        data,
        annot=True, fmt=".2f",
        xticklabels=["Accuracy (%)", "ROC AUC"],
        yticklabels=generators,
        cmap="RdYlGn", vmin=50, vmax=100,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(
        f"Cross-Generator Generalization — {model_name}",
        fontsize=13, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Generalization heatmap → {save_path}")


def plot_generalization_bars(
    results: dict[str, dict],
    model_name: str,
    save_path: Path,
) -> None:
    """Bar chart of accuracy per generator."""
    generators = list(results.keys())
    accs = [results[g]["accuracy"] * 100 for g in generators]

    fig, ax = plt.subplots(figsize=(max(6, len(generators) * 1.5), 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(generators)))
    bars = ax.bar(generators, accs, color=colors, edgecolor="black", linewidth=0.8)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.axhline(50, color="red", linestyle="--", lw=1.5, alpha=0.7, label="Random baseline")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Cross-Generator Accuracy — {model_name}\n"
        "(model trained on original dataset, tested on each generator separately)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Generalization bar chart → {save_path}")


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def run_generalization_eval(
    model_name: str,
    gen_root:   Path,
    max_per_class: int = 500,
) -> dict:
    """
    Evaluate *model_name* on every generator sub-folder inside *gen_root*.

    Returns:
        dict[generator_name, metrics_dict]
    """
    device = torch.device(CFG.device)

    model = load_model(model_name, device)
    if model is None:
        return {}

    # Pick transform matching the model type
    if model_name == "cnn_fft":
        transform = get_fft_transforms("val", CFG.image_size)
    else:
        transform = get_spatial_transforms("val", CFG.image_size, CFG.norm_mean, CFG.norm_std)

    if not gen_root.is_dir():
        log.error(
            f"Generator root not found: {gen_root}\n"
            "Create sub-folders for each generator:\n"
            "  dataset/generators/stable_diffusion/real/  …fake/\n"
            "  dataset/generators/dalle/real/              …fake/\n"
            "  dataset/generators/midjourney/real/         …fake/"
        )
        return {}

    generator_folders = sorted(
        d for d in gen_root.iterdir() if d.is_dir()
    )

    if not generator_folders:
        log.error(f"No generator sub-folders found in {gen_root}")
        return {}

    results: dict[str, dict] = {}

    for gen_folder in generator_folders:
        gen_name = gen_folder.name
        log.info(f"\nEvaluating on generator: {gen_name}")

        data = load_generator_images(gen_folder, transform, device, max_per_class)
        if data is None:
            log.warning(f"  Skipping {gen_name} (missing or empty)")
            continue

        X, y = data
        metrics = evaluate_on_generator(model, X, y, device)
        results[gen_name] = metrics

        log.info(
            f"  {gen_name:<20}  acc={metrics['accuracy']:.4f}  "
            f"auc={metrics['auc']:.4f}  "
            f"(real={metrics['n_real']}, fake={metrics['n_fake']})"
        )

    if results:
        # Console table
        print("\n" + "=" * 60)
        print(f"  CROSS-GENERATOR RESULTS — {model_name}")
        print("=" * 60)
        print(f"  {'Generator':<20} {'Accuracy':>10} {'AUC':>10}")
        print("-" * 60)
        for gen, m in results.items():
            print(f"  {gen:<20} {m['accuracy']:>10.4f} {m['auc']:>10.4f}")
        print("=" * 60)

        # Save plots
        viz_dir = CFG.results_dir / "generalization"
        plot_generalization_heatmap(results, model_name, viz_dir / f"gen_heatmap_{model_name}.png")
        plot_generalization_bars(results, model_name, viz_dir / f"gen_bars_{model_name}.png")

        # Save JSON
        out_json = viz_dir / f"gen_results_{model_name}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved → {out_json}")

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-generator generalization evaluation.")
    p.add_argument(
        "--model", choices=["cnn_spatial", "cnn_fft", "resnet18"],
        default="resnet18", help="Which trained model to evaluate."
    )
    p.add_argument(
        "--gen_root", type=Path,
        default=CFG.dataset_root / "generators",
        help="Root folder containing one sub-folder per generator.",
    )
    p.add_argument(
        "--max_per_class", type=int, default=500,
        help="Max images per class per generator (keeps eval fast).",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_generalization_eval(args.model, args.gen_root, args.max_per_class)

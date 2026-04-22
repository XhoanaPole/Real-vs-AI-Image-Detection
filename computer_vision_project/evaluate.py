"""
evaluate.py
===========
Evaluation module for all three model variants.

Computes:
  - Accuracy
  - Confusion matrix
  - Precision / Recall / F1  (per class + macro)
  - ROC curve + AUC
  - Side-by-side model comparison chart

Saves all results to results/ directory.

Usage:
    python evaluate.py
    python evaluate.py --data_root dataset --batch_size 32
"""

import logging
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc,
)
from tqdm import tqdm

from config     import CFG
from dataset    import get_dataloaders
from preprocess import get_spatial_transforms, get_fft_transforms
from model_cnn  import build_spatial_cnn, build_fft_cnn, build_resnet18
from model_svm  import SVMClassifier
from fft_features import extract_dataset_features

log = logging.getLogger(__name__)

LABEL_NAMES = ["Real", "Fake"]


# ───────────────────────────────────────────────────────────────────────────
# CNN evaluation
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_cnn(
    model    : nn.Module,
    loader   : torch.utils.data.DataLoader,
    device   : torch.device,
) -> dict:
    """
    Run model over loader and collect predictions + probabilities.

    Returns dict with: all_labels, all_preds, all_probs (numpy arrays)
    """
    model.eval()
    model.to(device)

    all_labels = []
    all_preds  = []
    all_probs  = []

    for imgs, labels in tqdm(loader, desc="Evaluating", unit="batch", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)

        all_labels.append(labels.numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_probs.append(probs[:, 1].cpu().numpy())  # P(fake)

    return {
        "all_labels": np.concatenate(all_labels),
        "all_preds" : np.concatenate(all_preds),
        "all_probs" : np.concatenate(all_probs),
    }


# ───────────────────────────────────────────────────────────────────────────
# Metrics computation
# ───────────────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    acc    = float(accuracy_score(labels, preds))
    cm     = confusion_matrix(labels, preds).tolist()
    report = classification_report(labels, preds, target_names=LABEL_NAMES)
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = float(auc(fpr, tpr))
    return {
        "accuracy"              : acc,
        "confusion_matrix"      : cm,
        "classification_report" : report,
        "roc_auc"               : roc_auc,
        "fpr"                   : fpr.tolist(),
        "tpr"                   : tpr.tolist(),
    }


# ───────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ───────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list[list[int]], title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        np.array(cm),
        annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Confusion matrix → {save_path}")


def plot_roc_curves(results: dict[str, dict], save_path: Path) -> None:
    """Plot overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 5))

    colours = {"cnn_spatial": "#2196F3", "cnn_fft": "#4CAF50", "svm_fft": "#FF5722", "resnet18": "#9C27B0"}
    labels  = {
        "cnn_spatial": "Model A — Spatial CNN",
        "cnn_fft"    : "Model B — FFT CNN",
        "svm_fft"    : "Model B — FFT SVM",
        "resnet18"   : "Model C — ResNet-18 (pretrained)",
    }

    for name, res in results.items():
        if "fpr" not in res:
            continue
        roc_auc = res["roc_auc"]
        ax.plot(
            res["fpr"], res["tpr"],
            label=f"{labels.get(name, name)} (AUC={roc_auc:.3f})",
            color=colours.get(name, "grey"), lw=2,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"ROC curves → {save_path}")


def plot_accuracy_comparison(results: dict[str, dict], save_path: Path) -> None:
    """Bar chart comparing validation accuracy of all models."""
    names  = []
    accs   = []
    labels_map = {
        "cnn_spatial": "Model A\nSpatial CNN",
        "cnn_fft"    : "Model B\nFFT CNN",
        "svm_fft"    : "Model B\nFFT SVM",
        "resnet18"   : "Model C\nResNet-18",
    }
    colours = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    for name, res in results.items():
        if "accuracy" in res:
            names.append(labels_map.get(name, name))
            accs.append(res["accuracy"] * 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, accs, color=colours[:len(names)], edgecolor="black",
                  linewidth=0.8, width=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison\n(Spatial vs Frequency Domain)",
                 fontsize=13, fontweight="bold")
    ax.axhline(50, color="red", linestyle="--", lw=1.2, alpha=0.6, label="Random baseline")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Accuracy comparison → {save_path}")


# ───────────────────────────────────────────────────────────────────────────
# Main evaluation orchestrator
# ───────────────────────────────────────────────────────────────────────────

def compare_models(results: dict[str, dict]) -> None:
    """Print a formatted comparison table to console."""
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<22} {'Accuracy':>10} {'ROC AUC':>10}")
    print("-" * 60)
    for name, res in results.items():
        acc = res.get("accuracy", float("nan"))
        roc = res.get("roc_auc", float("nan"))
        print(f"  {name:<22} {acc:>10.4f} {roc:>10.4f}")
    print("=" * 60 + "\n")


def run_evaluation(data_root: Path, batch_size: int, num_workers: int) -> dict:
    """
    Load all available checkpoints and evaluate them on the val set.

    Returns:
        dict[model_name, metrics_dict]
    """
    device = torch.device(CFG.device)
    results: dict[str, dict] = {}

    # ── CNN Spatial ────────────────────────────────────────────────────
    ckpt_spatial = CFG.checkpoint_dir / "cnn_spatial_best.pth"
    if ckpt_spatial.exists():
        log.info("Evaluating cnn_spatial …")
        model = build_spatial_cnn(CFG.cnn_channels, CFG.dropout)
        ckpt  = torch.load(ckpt_spatial, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])

        _, _, test_loader = get_dataloaders(
            root=data_root,
            train_transform=get_spatial_transforms("train"),
            val_transform  =get_spatial_transforms("val"),
            batch_size=batch_size, num_workers=num_workers,
        )
        raw = evaluate_cnn(model, test_loader, device)
        results["cnn_spatial"] = compute_metrics(
            raw["all_labels"], raw["all_preds"], raw["all_probs"]
        )
        plot_confusion_matrix(
            results["cnn_spatial"]["confusion_matrix"],
            "Model A — Spatial CNN",
            CFG.results_dir / "cm_cnn_spatial.png",
        )
    else:
        log.warning(f"Checkpoint not found, skipping: {ckpt_spatial}")

    # ── CNN FFT ────────────────────────────────────────────────────────
    ckpt_fft = CFG.checkpoint_dir / "cnn_fft_best.pth"
    if ckpt_fft.exists():
        log.info("Evaluating cnn_fft …")
        model = build_fft_cnn(CFG.cnn_channels, CFG.dropout)
        ckpt  = torch.load(ckpt_fft, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])

        _, _, test_loader = get_dataloaders(
            root=data_root,
            train_transform=get_fft_transforms("train"),
            val_transform  =get_fft_transforms("val"),
            batch_size=batch_size, num_workers=num_workers,
        )
        raw = evaluate_cnn(model, test_loader, device)
        results["cnn_fft"] = compute_metrics(
            raw["all_labels"], raw["all_preds"], raw["all_probs"]
        )
        plot_confusion_matrix(
            results["cnn_fft"]["confusion_matrix"],
            "Model B — FFT CNN",
            CFG.results_dir / "cm_cnn_fft.png",
        )
    else:
        log.warning(f"Checkpoint not found, skipping: {ckpt_fft}")

    # ── SVM FFT ────────────────────────────────────────────────────────
    ckpt_svm = CFG.checkpoint_dir / "svm_fft_best.pkl"
    if ckpt_svm.exists():
        log.info("Evaluating svm_fft …")
        from torchvision import transforms as T
        raw_tf = T.Compose([
            T.Resize((CFG.image_size, CFG.image_size)),
            T.ToTensor(),
        ])
        _, _, test_loader = get_dataloaders(
            root=data_root,
            train_transform=raw_tf,
            val_transform  =raw_tf,
            batch_size=batch_size, num_workers=num_workers,
        )

        X_test, y_test = extract_dataset_features(
            test_loader, CFG.radial_bins, CFG.high_freq_pct
        )
        clf   = SVMClassifier.load(ckpt_svm)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        results["svm_fft"] = compute_metrics(y_test, preds, probs)
        plot_confusion_matrix(
            results["svm_fft"]["confusion_matrix"],
            "Model B — FFT SVM",
            CFG.results_dir / "cm_svm_fft.png",
        )
    else:
        log.warning(f"Checkpoint not found, skipping: {ckpt_svm}")

    # ── ResNet-18  (Model C) ───────────────────────────────────────────
    ckpt_resnet = CFG.checkpoint_dir / "resnet18_best.pth"
    if ckpt_resnet.exists():
        log.info("Evaluating resnet18 …")
        model = build_resnet18(pretrained=False, dropout=CFG.dropout)
        ckpt  = torch.load(ckpt_resnet, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])

        _, _, test_loader = get_dataloaders(
            root=data_root,
            train_transform=get_spatial_transforms("train"),
            val_transform  =get_spatial_transforms("val"),
            batch_size=batch_size, num_workers=num_workers,
        )
        raw = evaluate_cnn(model, test_loader, device)
        results["resnet18"] = compute_metrics(
            raw["all_labels"], raw["all_preds"], raw["all_probs"]
        )
        plot_confusion_matrix(
            results["resnet18"]["confusion_matrix"],
            "Model C — ResNet-18 (pretrained)",
            CFG.results_dir / "cm_resnet18.png",
        )
    else:
        log.warning(f"Checkpoint not found, skipping: {ckpt_resnet}")

    # ── Comparative plots ──────────────────────────────────────────────
    if results:
        plot_roc_curves(results, CFG.results_dir / "roc_curves.png")
        plot_accuracy_comparison(results, CFG.results_dir / "accuracy_comparison.png")
        compare_models(results)

        # Print classification reports
        for name, res in results.items():
            print(f"\n{'-'*50}")
            print(f"  {name}  —  Classification Report")
            print(f"{'-'*50}")
            print(res.get("classification_report", "N/A"))

        # Save JSON summary (excluding fpr/tpr arrays to keep it readable)
        summary = {
            k: {kk: vv for kk, vv in v.items() if kk not in ("fpr", "tpr")}
            for k, v in results.items()
        }
        out = CFG.results_dir / "evaluation_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"\nEvaluation summary → {out}")

    return results


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained models.")
    p.add_argument("--data_root",   type=Path, default=CFG.dataset_root)
    p.add_argument("--batch_size",  type=int,  default=CFG.batch_size)
    p.add_argument("--num_workers", type=int,  default=CFG.num_workers)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_evaluation(args.data_root, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()

# AI-Generated Image Detection
### Spatial CNN vs Frequency-Domain CNN & SVM

A complete PyTorch project that compares **spatial** and **frequency-domain** approaches for detecting AI-generated images.

---

## Project Structure

```
computer_vision_project/
│
├── dataset.py            # PyTorch Dataset + DataLoader factory
├── preprocess.py         # Spatial & FFT transforms
├── fft_features.py       # FFT feature extraction + visualisation
├── model_cnn.py          # SimpleCNN (shared by Model A and B-CNN)
├── model_svm.py          # RBF SVM wrapper (Model B-SVM)
├── train.py              # Training loops (CNN + SVM)
├── evaluate.py           # Metrics, confusion matrix, ROC, comparison
├── visualize.py          # FFT plots, training curves, prediction grid
├── config.py             # Central hyperparameters (edit here)
├── main.py               # Pipeline orchestrator
│
├── prepare_dataset.py    # Dataset sampling + preprocessing
├── verify_dataset.py     # Dataset integrity check
├── explore_dataset.py    # Inspect GenImage before sampling
├── merge_custom.py       # Add custom real/fake images
│
├── requirements.txt           # Full training requirements
├── requirements_dataset.txt   # Lightweight dataset-prep requirements
├── README.md
└── README_dataset.md          # Dataset preparation guide
```

---

## Pipelines

| Pipeline | Input | Model | Description |
|---|---|---|---|
| **Model A** | Raw RGB (3×256×256) | SimpleCNN | Baseline spatial CNN |
| **Model B — CNN** | FFT magnitude (1×256×256) | SimpleCNN | CNN on log-magnitude spectrum |
| **Model B — SVM** | 18 FFT features | RBF SVM | Hand-crafted frequency features |

### FFT Features (Model B — SVM)
| Feature | Size | Description |
|---|---|---|
| `mean_energy` | 1 | Average log-magnitude across spectrum |
| `high_freq_ratio` | 1 | Energy in outer 25% radial ring / total |
| `radial_distribution` | 16 | Normalised radial energy histogram |

---

## Quick Start

### 1. Activate your Environment

You already have PyTorch installed in your Anaconda environment named `xhoan1`. All required project dependencies have been installed there automatically!

Open your **Anaconda Prompt** and run:

```powershell
conda activate xhoan1
cd C:\Users\xhoan\Downloads\computer_vision_project
```

### 2. Prepare Dataset

> See `README_dataset.md` for full dataset preparation guide.

```powershell
# Step 1: Explore your GenImage download
python explore_dataset.py --genimage_root D:\GenImage

# Step 2: Sample and preprocess 1000 images per class
python prepare_dataset.py `
    --genimage_root D:\GenImage `
    --output_root   dataset `
    --generators    Midjourney Stable_Diffusion_V1_4 `
    --num_samples   1000

# Step 3: Verify the prepared dataset
python verify_dataset.py --dataset_root dataset
```

**Expected output structure:**
```
dataset/
├── train/
│   ├── real/   (800 images)
│   └── fake/   (800 images)
└── val/
    ├── real/   (200 images)
    └── fake/   (200 images)
```

### 3. Run the Full Pipeline

```powershell
# Train all models → Evaluate → Visualize
python main.py --mode all
```

---

## Detailed Commands

### Train individual models

```powershell
# Model A — Spatial CNN
python train.py --model cnn_spatial --epochs 20 --batch_size 32

# Model B — FFT CNN
python train.py --model cnn_fft --epochs 20 --batch_size 32

# Model B — FFT SVM (no epochs needed)
python train.py --model svm_fft
```

### Evaluate

```powershell
python evaluate.py --data_root dataset --batch_size 32
```

Produces:
- `results/cm_cnn_spatial.png` — confusion matrix
- `results/cm_cnn_fft.png`
- `results/cm_svm_fft.png`
- `results/roc_curves.png` — overlaid ROC curves
- `results/accuracy_comparison.png` — bar chart
- `results/evaluation_summary.json` — full metrics

### Visualize

```powershell
python visualize.py --data_root dataset --n_samples 3
```

Produces:
- `results/visualizations/fft_comparison_*.png`
- `results/visualizations/training_curves_*.png`
- `results/visualizations/val_accuracy_all_models.png`
- `results/visualizations/predictions_cnn_spatial.png`

---

## Configuration

All key hyperparameters are in **`config.py`**. Edit there — no need to touch other files:

```python
CFG.epochs      = 20       # training epochs
CFG.batch_size  = 32       # images per batch (RTX 4050 6GB: safe at 32)
CFG.lr          = 1e-4     # learning rate (AdamW + CosineAnnealingLR)
CFG.image_size  = 256      # spatial resolution
CFG.radial_bins = 16       # FFT radial histogram bins
CFG.num_workers = 4        # DataLoader workers (set 0 if Windows errors)
```

---

## CNN Architecture

```
Input  (C × 256 × 256)          C = 3 (spatial) or 1 (FFT)
  ↓
ConvBlock 1:  C  → 32    Conv(3×3) → BN → ReLU → MaxPool    → 128×128
ConvBlock 2:  32 → 64    Conv(3×3) → BN → ReLU → MaxPool    →  64×64
ConvBlock 3:  64 → 128   Conv(3×3) → BN → ReLU → MaxPool    →  32×32
ConvBlock 4:  128 → 256  Conv(3×3) → BN → ReLU → MaxPool    →  16×16
  ↓
Global Average Pooling  →  (256,)
  ↓
FC(256 → 128) → ReLU → Dropout(0.5)
  ↓
FC(128 → 2)   — logits
```

**~1.5M parameters** — fits comfortably in 6 GB VRAM at `batch_size=32`.

---

## Expected Results

These are typical ranges on a balanced 1000-sample-per-class subset:

| Model | Val Accuracy | ROC AUC |
|---|---|---|
| Model A — Spatial CNN | 75–90% | 0.83–0.95 |
| Model B — FFT CNN | 70–85% | 0.78–0.92 |
| Model B — FFT SVM | 60–75% | 0.65–0.82 |

> Results vary significantly by generator. Midjourney images are harder to detect than BigGAN.

---

## Output Files

```
checkpoints/
├── cnn_spatial_best.pth      # best CNN spatial checkpoint
├── cnn_fft_best.pth          # best CNN FFT checkpoint
└── svm_fft_best.pkl          # fitted SVM pipeline

logs/
├── cnn_spatial_metrics.csv
└── cnn_fft_metrics.csv

results/
├── accuracy_comparison.png
├── roc_curves.png
├── cm_cnn_spatial.png
├── cm_cnn_fft.png
├── cm_svm_fft.png
├── evaluation_summary.json
└── visualizations/
    ├── fft_comparison_1.png
    ├── fft_comparison_2.png
    ├── fft_comparison_3.png
    ├── training_curves_cnn_spatial.png
    ├── training_curves_cnn_fft.png
    ├── val_accuracy_all_models.png
    └── predictions_cnn_spatial.png
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA out of memory` | Reduce `--batch_size` to 16 |
| `DataLoader worker error` | Add `--num_workers 0` |
| `No images found` | Run `prepare_dataset.py` first |
| SVM is very slow | Normal for large datasets — uses all CPU cores |
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |

---

## Hardware

Optimised for:

- **GPU**: NVIDIA GeForce RTX 4050 (6 GB VRAM)
- **CUDA**: 12.x
- **Python**: 3.10+
- **OS**: Windows 11

---

## Research Goal

Compare:
- **Spatial domain (Model A)**: Can a CNN learn texture/artefact patterns from raw pixels?
- **Frequency domain (Model B)**: Do FFT spectral signatures better reveal AI generation artefacts?

Key hypothesis: AI-generated images exhibit characteristic high-frequency patterns in the FFT spectrum (grid artefacts from upsampling, spectral peaks from periodic structures) that simple spatial CNNs may miss.

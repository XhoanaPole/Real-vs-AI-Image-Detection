# Real vs AI Image Detection
### A Dual-Domain Approach: Spatial CNNs, FFT Analysis, and Transfer Learning

A complete PyTorch and Streamlit project that detects AI-generated images by analyzing both standard visual features (spatial domain) and invisible mathematical artifacts (frequency domain).

---

## Dataset Overview

The models were trained and validated on a curated dataset containing images from five state-of-the-art diffusion models paired with real photographs:
1. **Stable Diffusion**
2. **DALL-E 3**
3. **GLIDE**
4. **Google Imagen**
5. **Google Gemini (Imagen 3)** — *used as a zero-shot/generalization benchmark*

**Dataset Access:**
The `Gemini_dataset` sample (60 images: 30 real, 30 fake) is available in this repository under `Gemini_dataset_sample/`. The full 13,000+ image dataset is hosted externally.

---

## Project Structure

```
computer_vision_project/
│
├── dataset.py            # PyTorch Dataset + DataLoader factory
├── preprocess.py         # Spatial & FFT transforms
├── fft_features.py       # FFT feature extraction + mathematical analysis
├── model_cnn.py          # SimpleCNN & ResNet-18 backbone architecture
├── model_svm.py          # RBF SVM baseline model
├── train.py              # Training loops for all 4 models
├── evaluate.py           # Metrics calculation (Accuracy, ROC-AUC, Recall)
├── visualize.py          # FFT plots, training curves, Grad-CAM maps
├── config.py             # Central hyperparameters (epochs, batch size, lr)
├── generalization_eval.py# Cross-generator accuracy breakdown
└── app.py                # Interactive Streamlit Web Interface
```

---

## The Four Investigators (Models)

| Pipeline | Approach | Input | Model Architecture |
|---|---|---|---|
| **Model A** | 👁️ Visual (Spatial) | Raw RGB (3×256×256) | Custom lightweight CNN |
| **Model B** | 📡 Frequency (FFT) | Log-magnitude spectrum (1×256×256) | Custom lightweight CNN |
| **Model C** | 🎯 Classical baseline| Hand-crafted FFT features (18) | RBF SVM |
| **Model D** | 🧬 Transfer Learning | Raw RGB (3×256×256) | Pre-trained ResNet-18 |

---

## Test Set Results (2,727 Unseen Images)

| Model | Accuracy | Recall (Fake) | ROC-AUC |
|---|---|---|---|
| **ResNet-18 (Model D)** | **94.7%** | **95.0%** | **0.987** |
| CNN Spatial (Model A) | 84.3% | 85.0% | 0.919 |
| CNN FFT (Model B) | 78.0% | 83.0% | 0.863 |
| SVM FFT (Model C) | 73.2% | 67.0% | 0.801 |

> *Note: These models were evaluated against modern diffusion models, which are significantly harder to detect than older GANs.*

---

## Quick Start: Interactive Dashboard

The easiest way to experience the detection system is through the live web interface.

```powershell
# 1. Activate your conda environment
conda activate venv_gpu

# 2. Run the Streamlit app
streamlit run app.py
```
This will open a browser window at `http://localhost:8501/` where you can upload any image and see the spatial, frequency, and ResNet-18 predictions in real time.

---

## Detailed Commands for Researchers

### 1. Training the Models
All hyperparameters are managed in `config.py`.

```powershell
# Train the custom Spatial CNN
python train.py --model cnn_spatial

# Train the Frequency (FFT) CNN
python train.py --model cnn_fft

# Train the ResNet-18 Transfer Learning model
python train.py --model resnet18
```

### 2. Evaluation & Metrics
```powershell
python evaluate.py
```
Produces:
- `results/cm_*.png` — Confusion matrices
- `results/roc_curves.png` — Overlaid ROC curves comparing all models

### 3. Visualizations
```powershell
python visualize.py
```
Produces:
- Visual comparisons of real vs AI FFT spectra.
- Training loss and accuracy curves.

---

## Why Frequency Analysis (FFT)?

AI image generators (like Diffusion Models) construct images using mathematical grids and iterative denoising steps. While the final image looks perfect to the human eye, these underlying mathematical processes leave behind invisible recurring high-frequency artifacts (such as grid patterns and artificial symmetry). 

By transforming the image into the frequency domain using a Fast Fourier Transform (FFT), this system strips away the visual content to examine the raw mathematical signature of the image, exposing AI-generation patterns that spatial models might miss.

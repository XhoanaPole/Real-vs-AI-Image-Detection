"""
app.py
======
Streamlit web demo for AI-generated image detection.

Features:
  - Single image upload with Real/Fake/Uncertain verdict
  - Confidence threshold: flags as Uncertain when score is 30–70 %
  - Generator identification: predicts which AI made the fake
  - Batch test: upload multiple images and get a summary table
  - FFT spectrum + GradCAM heatmap visualisation

Run:
    streamlit run app.py
"""

import io
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config       import CFG
from preprocess   import get_spatial_transforms, get_fft_transforms, FFTTransform
from model_cnn    import build_spatial_cnn, build_fft_cnn, build_resnet18
from fft_features import _image_to_magnitude, extract_fft_features, features_to_vector
from visualize    import GradCAM, _overlay_heatmap


# ── Constants ────────────────────────────────────────────────────────────────

UNCERTAIN_LOW  = 0.30   # below → Real
UNCERTAIN_HIGH = 0.70   # above → Fake   (between = Uncertain)

LABEL_NAMES  = {0: "Real", 1: "AI-Generated (Fake)", -1: "Uncertain"}
LABEL_COLORS = {0: "#4CAF50", 1: "#F44336", -1: "#FF9800"}
LABEL_ICONS  = {0: "🟢", 1: "🔴", -1: "🟡"}

GENERATOR_COLORS = {
    "DALL-E 3":        "#2196F3",
    "GLIDE":           "#9C27B0",
    "Imagen":          "#FF5722",
    "Stable Diffusion":"#4CAF50",
    "Gemini":          "#1565C0",
}

MODEL_INFO = {
    "ResNet-18 (pretrained) — Model C": {
        "key":       "resnet18",
        "ckpt":      CFG.checkpoint_dir / "resnet18_best.pth",
        "builder":   lambda: build_resnet18(pretrained=False, dropout=CFG.dropout),
        "transform": get_spatial_transforms("val", CFG.image_size, CFG.norm_mean, CFG.norm_std),
        "gradcam":   True,
    },
    "Spatial CNN — Model A": {
        "key":       "cnn_spatial",
        "ckpt":      CFG.checkpoint_dir / "cnn_spatial_best.pth",
        "builder":   lambda: build_spatial_cnn(CFG.cnn_channels, CFG.dropout),
        "transform": get_spatial_transforms("val", CFG.image_size, CFG.norm_mean, CFG.norm_std),
        "gradcam":   True,
    },
    "FFT CNN — Model B": {
        "key":       "cnn_fft",
        "ckpt":      CFG.checkpoint_dir / "cnn_fft_best.pth",
        "builder":   lambda: build_fft_cnn(CFG.cnn_channels, CFG.dropout),
        "transform": get_fft_transforms("val", CFG.image_size),
        "gradcam":   False,
    },
}


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide",
)


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_model(model_key: str):
    info = next(v for v in MODEL_INFO.values() if v["key"] == model_key)
    ckpt_path = info["ckpt"]
    if not Path(ckpt_path).exists():
        return None, f"Checkpoint not found: {ckpt_path}\nTrain with: python train.py --model {model_key}"
    model = info["builder"]()
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, None


@st.cache_resource(show_spinner="Loading generator classifier …")
def load_generator_classifier():
    ckpt = CFG.checkpoint_dir / "generator_classifier.pkl"
    if not ckpt.exists():
        return None
    with open(ckpt, "rb") as f:
        return pickle.load(f)


# ── Inference helpers ─────────────────────────────────────────────────────────

def predict(model, transform, pil_img: Image.Image):
    tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    fake_prob = float(probs[1].item())
    if fake_prob < UNCERTAIN_LOW:
        verdict = 0
    elif fake_prob > UNCERTAIN_HIGH:
        verdict = 1
    else:
        verdict = -1
    conf = float(probs[verdict].item()) if verdict >= 0 else abs(fake_prob - 0.5) * 2
    return verdict, conf, fake_prob


def predict_generator(clf, pil_img: Image.Image):
    img_np = np.array(pil_img.convert("L"), dtype=np.float64)
    vec    = features_to_vector(extract_fft_features(img_np)).reshape(1, -1)
    label  = clf.predict(vec)[0]
    probs  = clf.predict_proba(vec)[0]
    classes = clf.classes_
    prob_dict = {c: float(p) for c, p in zip(classes, probs)}
    return label, prob_dict


def compute_gradcam(model, model_key, transform, pil_img):
    if model_key == "resnet18":
        target_layer = model.layer4[-1]
    elif model_key == "cnn_spatial":
        target_layer = model.features[-1].block[-2]
    else:
        return None
    cam_gen = GradCAM(model, target_layer)
    tensor  = transform(pil_img).unsqueeze(0)
    tensor.requires_grad_(True)
    heatmap, _ = cam_gen(tensor)
    cam_gen.remove_hooks()
    return heatmap


def fft_figure(pil_img: Image.Image) -> plt.Figure:
    gray = np.array(pil_img.convert("L"), dtype=np.float64)
    mag  = _image_to_magnitude(gray)
    mn, mx = mag.min(), mag.max()
    mag_norm = (mag - mn) / (mx - mn + 1e-8)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mag_norm, cmap="inferno")
    ax.set_title("FFT Log-Magnitude Spectrum", fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    return fig


# ── Shared sidebar ────────────────────────────────────────────────────────────

st.title("🔍 AI Image Detector")
st.markdown(
    "Detect whether an image was taken by a camera or generated by AI.  \n"
    "Results below **70 % confidence** are flagged as **Uncertain**."
)
st.divider()

with st.sidebar:
    st.header("Settings")
    model_label  = st.selectbox("Model", list(MODEL_INFO.keys()), index=0)
    show_gradcam = st.checkbox("Show GradCAM heatmap", value=True)
    show_fft     = st.checkbox("Show FFT spectrum",    value=True)
    st.divider()
    st.markdown("**Confidence threshold**")
    st.markdown(f"- < {int(UNCERTAIN_LOW*100)}% fake → **Real**")
    st.markdown(f"- {int(UNCERTAIN_LOW*100)}–{int(UNCERTAIN_HIGH*100)}% → **Uncertain**")
    st.markdown(f"- > {int(UNCERTAIN_HIGH*100)}% fake → **Fake**")
    st.divider()
    st.markdown("**Models**")
    st.markdown(
        "- **ResNet-18**: highest accuracy\n"
        "- **Spatial CNN**: custom RGB model\n"
        "- **FFT CNN**: frequency-domain model"
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_single, tab_batch = st.tabs(["Single Image", "Batch Test"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Image
# ═══════════════════════════════════════════════════════════════════════════

with tab_single:
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="single_upload",
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        img_np  = np.array(pil_img)

        info      = MODEL_INFO[model_label]
        model_key = info["key"]
        transform = info["transform"]
        gradcam_ok = info["gradcam"]

        model, err = load_model(model_key)
        if err:
            st.error(err)
            st.stop()

        verdict, conf, fake_prob = predict(model, transform, pil_img)
        label_str   = LABEL_NAMES[verdict]
        label_color = LABEL_COLORS[verdict]
        label_icon  = LABEL_ICONS[verdict]

        # ── Verdict ───────────────────────────────────────────────────────
        st.markdown(
            f"<h2 style='color:{label_color}; text-align:center;'>"
            f"{label_icon} {label_str}</h2>",
            unsafe_allow_html=True,
        )

        col_real, col_fake = st.columns(2)
        col_real.metric("Real probability",         f"{(1 - fake_prob) * 100:.1f}%")
        col_fake.metric("AI-Generated probability", f"{fake_prob * 100:.1f}%")
        st.progress(int(fake_prob * 100), text=f"Fake score: {fake_prob:.2%}")



        st.divider()

        # ── Image columns ─────────────────────────────────────────────────
        ncols = 1 + (1 if show_gradcam and gradcam_ok else 0) + (1 if show_fft else 0)
        cols  = st.columns(ncols)
        col_idx = 0

        with cols[col_idx]:
            st.image(pil_img, caption="Uploaded image", use_container_width=True)
        col_idx += 1

        if show_gradcam and gradcam_ok:
            with st.spinner("Computing GradCAM …"):
                heatmap = compute_gradcam(model, model_key, transform, pil_img)
            if heatmap is not None:
                overlay = _overlay_heatmap(img_np, heatmap, alpha=0.45)
                with cols[col_idx]:
                    st.image(overlay, caption="GradCAM — where the model looks",
                             use_container_width=True)
                col_idx += 1

        if show_fft:
            fig = fft_figure(pil_img)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)
            with cols[col_idx]:
                st.image(buf, caption="FFT Log-Magnitude Spectrum",
                         use_container_width=True)

        with st.expander("Technical details"):
            st.markdown(f"""
| Property | Value |
|---|---|
| Model | `{model_key}` |
| Input size | `{CFG.image_size} × {CFG.image_size}` |
| Verdict | `{label_str}` |
| Confidence | `{conf:.4f}` |
| P(fake) | `{fake_prob:.4f}` |
| Uncertain zone | `{UNCERTAIN_LOW:.0%} – {UNCERTAIN_HIGH:.0%}` |
| Device | `{CFG.device}` |
            """)
    else:
        st.info("Upload an image above to get started.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Test
# ═══════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.markdown("Upload multiple images to test them all at once.")

    batch_files = st.file_uploader(
        "Upload images (multiple allowed)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="batch_upload",
    )

    if batch_files:
        info      = MODEL_INFO[model_label]
        model_key = info["key"]
        transform = info["transform"]

        model, err = load_model(model_key)
        if err:
            st.error(err)
            st.stop()

        gen_clf = load_generator_classifier()

        rows = []
        progress = st.progress(0, text="Analysing images …")

        for i, f in enumerate(batch_files):
            pil_img = Image.open(f).convert("RGB")
            verdict, conf, fake_prob = predict(model, transform, pil_img)
            label_str = LABEL_NAMES[verdict]

            rows.append({
                "File":         f.name,
                "Verdict":      label_str,
                "P(fake)":      f"{fake_prob*100:.1f}%",
                "P(real)":      f"{(1-fake_prob)*100:.1f}%",
            })
            progress.progress((i + 1) / len(batch_files),
                               text=f"Analysing {i+1}/{len(batch_files)} …")

        progress.empty()

        df = pd.DataFrame(rows)

        # ── Summary stats ─────────────────────────────────────────────────
        n_real      = (df["Verdict"] == "Real").sum()
        n_fake      = (df["Verdict"] == "AI-Generated (Fake)").sum()
        n_uncertain = (df["Verdict"] == "Uncertain").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total",     len(df))
        c2.metric("Real 🟢",   n_real)
        c3.metric("Fake 🔴",   n_fake)
        c4.metric("Uncertain 🟡", n_uncertain)

        st.divider()

        # ── Colour-coded table ────────────────────────────────────────────
        def color_verdict(val):
            colors = {
                "Real":                 "color: #4CAF50; font-weight:bold",
                "AI-Generated (Fake)":  "color: #F44336; font-weight:bold",
                "Uncertain":            "color: #FF9800; font-weight:bold",
            }
            return colors.get(val, "")

        styled = df.style.map(color_verdict, subset=["Verdict"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── CSV download ──────────────────────────────────────────────────
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv,
            file_name="batch_results.csv",
            mime="text/csv",
        )

        # ── Thumbnail grid ────────────────────────────────────────────────
        with st.expander("Preview all images"):
            cols = st.columns(4)
            for i, (f, row) in enumerate(zip(batch_files, rows)):
                f.seek(0)
                pil = Image.open(f).convert("RGB")
                color = LABEL_COLORS[
                    0 if row["Verdict"] == "Real"
                    else 1 if row["Verdict"] == "AI-Generated (Fake)"
                    else -1
                ]
                with cols[i % 4]:
                    st.image(pil, use_container_width=True)
                    st.markdown(
                        f"<p style='color:{color}; font-size:12px; text-align:center;'>"
                        f"{row['Verdict']}<br>{row['P(fake)']} fake</p>",
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Upload one or more images above to run batch analysis.")

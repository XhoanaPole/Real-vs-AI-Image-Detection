"""
train_generator_classifier.py
==============================
Train a multi-class SVM to identify which AI generator made a fake image.
Uses FFT features from fake images in the training set.

Labels:  DALL-E 3 | GLIDE | Imagen | Stable Diffusion

Run:
    python train_generator_classifier.py
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from config import CFG
from fft_features import extract_fft_features, features_to_vector

log = logging.getLogger(__name__)

DATASET_TO_LABEL = {
    "DALLE_dataset":   "DALL-E 3",
    "GLIDE_dataset":   "GLIDE",
    "IMAGEN_dataset":  "Imagen",
    "SD_dataset":      "Stable Diffusion",
    "Gemini_dataset":  "Gemini",
    # compDataset has no known single generator — excluded from classifier
}


def label_from_filename(name: str):
    for prefix, label in DATASET_TO_LABEL.items():
        if name.startswith(prefix):
            return label
    return None


def load_features(fake_dir: Path, max_per_class: int = 500):
    X, y, counts = [], [], {}
    for img_path in sorted(fake_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = label_from_filename(img_path.name)
        if label is None:
            continue
        if counts.get(label, 0) >= max_per_class:
            continue
        try:
            img = np.array(Image.open(img_path).convert("L"), dtype=np.float64)
            X.append(features_to_vector(extract_fft_features(img)))
            y.append(label)
            counts[label] = counts.get(label, 0) + 1
        except Exception as e:
            log.warning(f"Skipping {img_path.name}: {e}")

    log.info(f"  Loaded {len(X)} images from {fake_dir.name}/")
    for lbl, cnt in counts.items():
        log.info(f"    {lbl}: {cnt}")
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    log.info("Loading train fake images …")
    X_train, y_train = load_features(CFG.dataset_root / "train" / "fake", max_per_class=500)

    log.info("Loading test fake images …")
    X_test, y_test = load_features(CFG.dataset_root / "test" / "fake", max_per_class=200)

    log.info("Training multi-class SVM (rbf, C=10) …")
    clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    log.info(f"Test accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

    out = CFG.checkpoint_dir / "generator_classifier.pkl"
    with open(out, "wb") as f:
        pickle.dump(clf, f)
    log.info(f"Saved → {out}")


if __name__ == "__main__":
    main()

"""
model_svm.py
============
RBF-SVM classifier wrapper for Model B (Option 2 — SVM on FFT features).

Wraps sklearn.svm.SVC with:
  - Probability calibration (predict_proba available)
  - Standard scaling of features (important for RBF kernel)
  - joblib save / load

Usage:
    from model_svm import SVMClassifier

    clf = SVMClassifier()
    clf.train(X_train, y_train)
    accuracy = clf.score(X_val, y_val)
    clf.save("checkpoints/svm_best.pkl")

    clf2 = SVMClassifier.load("checkpoints/svm_best.pkl")
"""

import logging
from pathlib import Path

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

log = logging.getLogger(__name__)


class SVMClassifier:
    """
    RBF SVM with feature standardisation for FFT-feature classification.

    The classifier is a sklearn Pipeline:
        StandardScaler  →  SVC(kernel='rbf', probability=True)

    Scaling is critical for RBF kernels — features on different scales
    will otherwise distort the kernel distance computation.

    Args:
        kernel  : SVM kernel (default 'rbf')
        C       : regularisation parameter (default 1.0)
        gamma   : kernel coefficient ('scale' = 1 / (n_features × X.var()))
        seed    : random state for SVM (affects training convergence)
    """

    def __init__(
        self,
        kernel: str  = "rbf",
        C:      float = 1.0,
        gamma:  str  = "scale",
        seed:   int  = 42,
    ):
        self.kernel = kernel
        self.C      = C
        self.gamma  = gamma
        self.seed   = seed

        self._pipeline: Pipeline | None = None
        self._is_fitted: bool = False

    # ── Build ─────────────────────────────────────────────────────────

    def _build(self) -> Pipeline:
        svc = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,          # enables predict_proba
            random_state=self.seed,
            class_weight="balanced",   # handles minor class imbalance
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc",    svc),
        ])

    # ── Training ──────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the SVM on the provided feature matrix and labels.

        Args:
            X : (N, feature_dim) float array
            y : (N,)             int array  (0=real, 1=fake)
        """
        log.info(
            f"Training SVM  (kernel={self.kernel}, C={self.C}, gamma={self.gamma}) "
            f"on {X.shape[0]:,} samples × {X.shape[1]} features …"
        )
        self._pipeline = self._build()
        self._pipeline.fit(X, y)
        self._is_fitted = True

        train_acc = accuracy_score(y, self._pipeline.predict(X))
        log.info(f"SVM training accuracy: {train_acc:.4f}")

    # ── Inference ─────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (0 or 1) for feature matrix X.
        """
        self._check_fitted()
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.  Shape: (N, 2).
        Column 0 = P(real), Column 1 = P(fake).
        """
        self._check_fitted()
        return self._pipeline.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on (X, y)."""
        self._check_fitted()
        return accuracy_score(y, self.predict(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute full evaluation metrics.

        Returns dict with keys:
            accuracy, confusion_matrix, classification_report
        """
        self._check_fitted()
        preds = self.predict(X)
        return {
            "accuracy"              : float(accuracy_score(y, preds)),
            "confusion_matrix"      : confusion_matrix(y, preds).tolist(),
            "classification_report" : classification_report(
                y, preds, target_names=["Real", "Fake"]
            ),
        }

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise the fitted pipeline to disk with joblib."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        log.info(f"SVM model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SVMClassifier":
        """Load a previously saved SVM from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SVM checkpoint not found: {path}")
        pipeline = joblib.load(path)

        # Reconstruct wrapper
        obj = cls.__new__(cls)
        obj._pipeline  = pipeline
        obj._is_fitted = True
        # Recover hyper-params from pipeline if possible
        svc = pipeline.named_steps.get("svc")
        obj.kernel = getattr(svc, "kernel", "rbf")
        obj.C      = getattr(svc, "C",      1.0)
        obj.gamma  = getattr(svc, "gamma",  "scale")
        obj.seed   = getattr(svc, "random_state", 42)
        log.info(f"SVM model loaded  ← {path}")
        return obj

    # ── Helpers ───────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError(
                "SVMClassifier is not fitted yet. Call .train(X, y) first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"SVMClassifier(kernel={self.kernel!r}, C={self.C}, "
            f"gamma={self.gamma!r}, status={status})"
        )

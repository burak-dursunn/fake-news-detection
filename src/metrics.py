"""Tüm model deneylerinde ortak kullanılan sınıflandırma metrikleri."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    """İki sınıflı ham model skorlarını sahte haber olasılığına dönüştürür."""
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    """Accuracy, precision, recall, F1 ve ROC-AUC değerlerini hesaplar."""
    labels = np.asarray(labels)
    fake_probability = probabilities_from_logits(logits)
    predictions = (fake_probability >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(labels, fake_probability)),
    }

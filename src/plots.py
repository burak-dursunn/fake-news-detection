"""Sunumda kullanılacak Türkçe başlıklı deney grafiklerini üretir."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from src.metrics import probabilities_from_logits


def _save(figure: plt.Figure, output_path: Path) -> None:
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_training_history(log_history: list[dict], output_dir: Path) -> None:
    """Trainer günlüğünden eğitim/doğrulama kaybı ve F1 grafikleri üretir."""
    train_logs = [entry for entry in log_history if "loss" in entry and "eval_loss" not in entry]
    eval_logs = [entry for entry in log_history if "eval_loss" in entry]

    if train_logs or eval_logs:
        figure, axis = plt.subplots(figsize=(9, 5))
        if train_logs:
            axis.plot(
                [entry["step"] for entry in train_logs],
                [entry["loss"] for entry in train_logs],
                label="Eğitim kaybı",
                color="#2563EB",
            )
        if eval_logs:
            axis.plot(
                [entry["step"] for entry in eval_logs],
                [entry["eval_loss"] for entry in eval_logs],
                marker="o",
                label="Doğrulama kaybı",
                color="#DC2626",
            )
        axis.set(title="Eğitim ve Doğrulama Kaybı", xlabel="Eğitim adımı", ylabel="Kayıp")
        axis.grid(alpha=0.25)
        axis.legend()
        _save(figure, output_dir / "01_Egitim_Dogrulama_Kaybi.png")

    f1_logs = [entry for entry in eval_logs if "eval_f1" in entry]
    accuracy_logs = [entry for entry in eval_logs if "eval_accuracy" in entry]
    if f1_logs or accuracy_logs:
        figure, axis = plt.subplots(figsize=(9, 5))
        if accuracy_logs:
            axis.plot(
                [entry["epoch"] for entry in accuracy_logs],
                [entry["eval_accuracy"] for entry in accuracy_logs],
                marker="o",
                label="Doğruluk",
                color="#2563EB",
            )
        if f1_logs:
            axis.plot(
                [entry["epoch"] for entry in f1_logs],
                [entry["eval_f1"] for entry in f1_logs],
                marker="o",
                label="F1 skoru",
                color="#16A34A",
            )
        axis.set(title="Doğrulama Doğruluğu ve F1 Skoru", xlabel="Epoch", ylabel="Skor")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
        _save(figure, output_dir / "02_Dogruluk_ve_F1_Egrisi.png")


def plot_test_curves_and_matrix(
    labels: np.ndarray,
    logits: np.ndarray,
    output_dir: Path,
) -> None:
    """Nihai test tahminlerinden ROC, PR ve karışıklık matrisi üretir."""
    labels = np.asarray(labels)
    fake_probability = probabilities_from_logits(logits)
    predictions = (fake_probability >= 0.5).astype(int)

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    RocCurveDisplay.from_predictions(labels, fake_probability, name="Model", ax=axis)
    axis.lines[0].set_label(f"Model (AUC = {roc_auc_score(labels, fake_probability):.4f})")
    axis.legend(loc="lower right")
    axis.set(title="ROC Eğrisi", xlabel="Yanlış pozitif oranı", ylabel="Doğru pozitif oranı")
    axis.grid(alpha=0.25)
    _save(figure, output_dir / "03_ROC_Egrisi.png")

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    PrecisionRecallDisplay.from_predictions(labels, fake_probability, name="Model", ax=axis)
    axis.lines[0].set_label(f"Model (AP = {average_precision_score(labels, fake_probability):.4f})")
    axis.legend(loc="lower left")
    axis.set(title="Kesinlik ve Duyarlılık Eğrisi", xlabel="Duyarlılık (sahte haber)", ylabel="Kesinlik (sahte haber)")
    axis.grid(alpha=0.25)
    _save(figure, output_dir / "04_Precision_Recall_Egrisi.png")

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    ConfusionMatrixDisplay(matrix, display_labels=["Gerçek", "Sahte"]).plot(
        ax=axis, colorbar=False, cmap="Blues"
    )
    axis.set(title="Karışıklık Matrisi", xlabel="Tahmin edilen sınıf", ylabel="Gerçek etiket")
    _save(figure, output_dir / "05_Karisiklik_Matrisi.png")

"""Config dosyasına göre WELFake üzerinde tek bir model deneyi çalıştırır."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import ExperimentConfig, load_config
from src.data import load_prepared_splits
from src.metrics import classification_metrics
from src.plots import plot_test_curves_and_matrix, plot_training_history


class NewsDataset(Dataset):
    """Başlık ve metni tokenizer'ın çift-girdi desteğiyle modele sunar."""

    def __init__(self, frame, tokenizer, max_length: int) -> None:
        self.labels = frame["label"].astype(int).tolist()
        self.encodings = tokenizer(
            frame["title"].fillna("").tolist(),
            frame["text"].fillna("").tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tek bir WELFake model deneyi çalıştırır.")
    parser.add_argument("--config", type=Path, required=True, help="Deney JSON dosyası.")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="prepare_data çıktılarının klasörü."
    )
    parser.add_argument(
        "--output-root", type=Path, required=True, help="Deney sonuç kök klasörü."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help=(
            "Epoch checkpointlerinin yazılacağı kök klasör. Verilmezse "
            "--output-root kullanılır. Colab'da kesintiye karşı Drive yolu verilebilir."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Aynı adlı deney klasörünün üzerine yazar."
    )
    return parser.parse_args()


def _write_experiment_summary(
    output_dir: Path,
    config: ExperimentConfig,
    test_metrics: dict[str, float],
) -> None:
    lines = [
        f"# {config.experiment_name}",
        "",
        "## Deney ayarları",
        f"- Model: `{config.model_name}`",
        f"- Maksimum token: {config.max_length}",
        f"- Öğrenme oranı: {config.learning_rate}",
        f"- Etkin batch boyutu: {config.effective_batch_size}",
        f"- Rastgele tohum: {config.seed}",
        "",
        "## Nihai test sonuçları",
    ]
    lines.extend(f"- {name}: {value:.4f}" for name, value in test_metrics.items())
    (output_dir / "deney_ozeti.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_training_arguments(
    config: ExperimentConfig,
    checkpoint_dir: Path,
    train_size: int,
    use_cuda: bool,
) -> TrainingArguments:
    """Yüklü Transformers sürümüyle uyumlu TrainingArguments üretir.

    Colab imajları farklı Transformers sürümleriyle gelebilir. Bazı eski
    sürümlerde ``warmup_ratio`` ve ``eval_strategy`` yerine sırasıyla
    ``warmup_steps`` ve ``evaluation_strategy`` kullanılır.
    """
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    steps_per_epoch = math.ceil(
        train_size / (config.per_device_train_batch_size * config.gradient_accumulation_steps)
    )
    estimated_total_steps = math.ceil(steps_per_epoch * config.num_train_epochs)
    arguments: dict[str, object] = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": config.logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "fp16": use_cuda,
        "report_to": "none",
        "seed": config.seed,
    }
    if "warmup_ratio" in parameters:
        arguments["warmup_ratio"] = config.warmup_ratio
    else:
        arguments["warmup_steps"] = round(estimated_total_steps * config.warmup_ratio)
    if "eval_strategy" in parameters:
        arguments["eval_strategy"] = "epoch"
    else:
        arguments["evaluation_strategy"] = "epoch"
    if "data_seed" in parameters:
        arguments["data_seed"] = config.seed
    return TrainingArguments(**arguments)


def run_experiment(
    config: ExperimentConfig,
    config_path: Path,
    data_dir: Path,
    output_root: Path,
    checkpoint_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, float]:
    output_dir = output_root / config.experiment_name
    if output_dir.exists():
        completed = (output_dir / "test_metrikleri.json").is_file()
        if completed and not overwrite:
            raise FileExistsError(
                f"Deney klasörü zaten var: {output_dir}. Yeniden yazmak için --overwrite kullanın."
            )
        if not completed:
            print("Tamamlanmamış önceki deney çıktısı temizlenerek eğitim yeniden başlatılıyor.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "kullanilan_config.json")
    checkpoint_dir = (checkpoint_root or output_root) / config.experiment_name / "checkpointler"

    set_seed(config.seed)
    splits = load_prepared_splits(data_dir, seed=config.seed)
    required_splits = {"train", "validation", "test"}
    if set(splits) != required_splits:
        raise ValueError(f"Eksik veya beklenmeyen veri bölmesi: {sorted(splits)}")

    print(f"Model indiriliyor/yükleniyor: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    datasets = {
        name: NewsDataset(frame, tokenizer, config.max_length) for name, frame in splits.items()
    }

    use_cuda = torch.cuda.is_available()
    arguments = build_training_arguments(
        config=config,
        checkpoint_dir=checkpoint_dir,
        train_size=len(datasets["train"]),
        use_cuda=use_cuda,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=lambda prediction: classification_metrics(
            prediction.label_ids, prediction.predictions
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    trainer.train()
    prediction = trainer.predict(datasets["test"], metric_key_prefix="test")
    test_metrics = classification_metrics(prediction.label_ids, prediction.predictions)
    (output_dir / "test_metrikleri.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "egitim_gecmisi.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    plot_training_history(trainer.state.log_history, output_dir)
    plot_test_curves_and_matrix(prediction.label_ids, prediction.predictions, output_dir)
    _write_experiment_summary(output_dir, config, test_metrics)
    trainer.save_model(output_dir / "en_iyi_model")
    tokenizer.save_pretrained(output_dir / "en_iyi_model")
    print(f"Deney tamamlandı: {output_dir}")
    return test_metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics = run_experiment(
        config=config,
        config_path=args.config,
        data_dir=args.data_dir,
        output_root=args.output_root,
        checkpoint_root=args.checkpoint_root,
        overwrite=args.overwrite,
    )
    print("Nihai test metrikleri:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()

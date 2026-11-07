"""Deney yapılandırmalarını doğrulayan küçük JSON yardımcıları."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    model_name: str
    max_length: int
    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    seed: int = 42
    early_stopping_patience: int = 2
    logging_steps: int = 100
    dataset_name: str = "welfake"

    def validate(self) -> None:
        if self.dataset_name not in {"welfake", "isot", "combined_corpus"}:
            raise ValueError("dataset_name welfake, isot veya combined_corpus olmalıdır.")
        if not self.experiment_name.strip():
            raise ValueError("experiment_name boş olamaz.")
        if self.max_length not in {128, 256, 384, 512}:
            raise ValueError("max_length 128, 256, 384 veya 512 olmalıdır.")
        if self.max_length > 512:
            raise ValueError("Bu modeller en fazla 512 token destekler.")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs pozitif olmalıdır.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate pozitif olmalıdır.")
        if self.per_device_train_batch_size <= 0 or self.per_device_eval_batch_size <= 0:
            raise ValueError("Batch size değerleri pozitif olmalıdır.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps pozitif olmalıdır.")

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict:
        return asdict(self) | {"effective_batch_size": self.effective_batch_size}


def load_config(path: Path) -> ExperimentConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Config dosyası bulunamadı: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Config JSON biçimi geçersiz: {path}") from error
    config = ExperimentConfig(**raw)
    config.validate()
    return config

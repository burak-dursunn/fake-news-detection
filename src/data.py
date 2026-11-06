"""WELFake verisini güvenli ve tekrar üretilebilir biçimde hazırlama araçları."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split


LABELS: Final[set[int]] = {0, 1}


@dataclass(frozen=True)
class CleaningReport:
    """Ham veride yapılan her elemenin sayısal özeti."""

    raw_rows: int
    rows_with_empty_text: int
    rows_with_invalid_label: int
    rows_with_conflicting_duplicate_content: int
    duplicate_rows_removed: int
    final_rows: int
    final_label_counts: dict[str, int]


def _normalise_text(series: pd.Series) -> pd.Series:
    """Model için metni bozmadan boşluk ve Unicode tutarsızlıklarını düzeltir."""
    return (
        series.fillna("")
        .astype("string")
        .str.normalize("NFKC")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _content_id(title: str, text: str) -> str:
    """Başlık ve metinden sabit, kişisel veri içermeyen bir kimlik üretir."""
    payload = f"{title}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_and_clean_welfake(input_path: Path) -> tuple[pd.DataFrame, CleaningReport]:
    """WELFake CSV'sini okur ve eğitim için temiz tabloyu döndürür.

    Ham dosya değiştirilmez. Bir kaydın eğitimde kullanılabilmesi için `text`
    alanı boş olmamalı ve etiketi 0 ya da 1 olmalıdır. Başlık boşsa kayıt
    korunur; model o haberi yalnızca metniyle görür.
    """
    raw = pd.read_csv(input_path, encoding="utf-8", low_memory=False)
    required_columns = {"title", "text", "label"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise ValueError(
            "WELFake dosyasında zorunlu sütunlar eksik: "
            f"{', '.join(sorted(missing_columns))}"
        )

    raw_rows = len(raw)
    frame = raw[["title", "text", "label"]].copy()
    frame.insert(0, "source_row_id", raw.index.astype("int64"))
    frame["title"] = _normalise_text(frame["title"])
    frame["text"] = _normalise_text(frame["text"])

    numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
    valid_label = numeric_labels.isin(LABELS)
    empty_text = frame["text"].eq("")
    invalid_label_count = int((~valid_label).sum())
    empty_text_count = int(empty_text.sum())

    frame = frame.loc[valid_label & ~empty_text].copy()
    frame["label"] = numeric_labels.loc[frame.index].astype("int8")
    frame["article_id"] = [
        _content_id(title, text)
        for title, text in zip(frame["title"], frame["text"], strict=True)
    ]

    # Aynı başlık+metin için farklı etiket bulunuyorsa kayıtlar güvenilir
    # değildir; tüm çelişkili kopyalar çıkarılır ve rapora yazılır.
    label_counts = frame.groupby("article_id")["label"].nunique()
    conflicting_ids = label_counts.index[label_counts.gt(1)]
    conflicting_mask = frame["article_id"].isin(conflicting_ids)
    conflicting_count = int(conflicting_mask.sum())
    frame = frame.loc[~conflicting_mask].copy()

    before_deduplication = len(frame)
    frame = frame.drop_duplicates(subset=["article_id"], keep="first").copy()
    duplicate_count = before_deduplication - len(frame)
    frame = frame[["source_row_id", "article_id", "title", "text", "label"]]
    frame = frame.reset_index(drop=True)

    report = CleaningReport(
        raw_rows=raw_rows,
        rows_with_empty_text=empty_text_count,
        rows_with_invalid_label=invalid_label_count,
        rows_with_conflicting_duplicate_content=conflicting_count,
        duplicate_rows_removed=duplicate_count,
        final_rows=len(frame),
        final_label_counts={
            str(label): int(count)
            for label, count in frame["label"].value_counts().sort_index().items()
        },
    )
    return frame, report


def create_split_manifest(
    clean_data: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Makale düzeyinde sabit %70/%15/%15 train/validation/test bölmesi üretir."""
    train_ids, temporary_ids = train_test_split(
        clean_data["article_id"],
        test_size=0.30,
        random_state=seed,
        stratify=clean_data["label"],
    )
    temporary = clean_data.set_index("article_id").loc[temporary_ids]
    validation_ids, test_ids = train_test_split(
        temporary_ids,
        test_size=0.50,
        random_state=seed,
        stratify=temporary["label"],
    )
    split = pd.concat(
        [
            pd.DataFrame({"article_id": train_ids, "split": "train"}),
            pd.DataFrame({"article_id": validation_ids, "split": "validation"}),
            pd.DataFrame({"article_id": test_ids, "split": "test"}),
        ],
        ignore_index=True,
    )
    return split.sort_values("article_id").reset_index(drop=True)


def save_prepared_data(
    clean_data: pd.DataFrame,
    report: CleaningReport,
    output_dir: Path,
    seed: int = 42,
    overwrite: bool = False,
) -> None:
    """Temiz veri, sabit split manifesti ve raporları hedef klasöre yazar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "clean_data": output_dir / "welfake_temiz.csv",
        "split_manifest": output_dir / f"welfake_split_seed{seed}.csv",
        "report": output_dir / "welfake_temizleme_raporu.json",
        "split_report": output_dir / "welfake_split_ozeti.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        existing_names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Hazır veri dosyaları zaten var: {existing_names}. "
            "Üzerine yazmak için --overwrite kullanın."
        )

    split_manifest = create_split_manifest(clean_data, seed=seed)
    split_with_label = split_manifest.merge(
        clean_data[["article_id", "label"]], on="article_id", validate="one_to_one"
    )
    split_summary = {
        "seed": seed,
        "split_counts": {
            name: int(count)
            for name, count in split_manifest["split"].value_counts().sort_index().items()
        },
        "label_counts_by_split": {
            split_name: {
                str(label): int(count)
                for label, count in group["label"].value_counts().sort_index().items()
            }
            for split_name, group in split_with_label.groupby("split", sort=True)
        },
    }

    clean_data.to_csv(paths["clean_data"], index=False, encoding="utf-8")
    split_manifest.to_csv(paths["split_manifest"], index=False, encoding="utf-8")
    paths["report"].write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["split_report"].write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_prepared_splits(data_dir: Path, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Temiz veriyle manifesti birleştirip eğitim kümelerini döndürür.

    Bu fonksiyon modelden bağımsızdır: aynı manifest her model, token uzunluğu
    ve hiperparametre denemesinde yeniden kullanılır.
    """
    clean_path = data_dir / "welfake_temiz.csv"
    manifest_path = data_dir / f"welfake_split_seed{seed}.csv"
    if not clean_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Hazırlanmış veri bulunamadı. Önce `python -m src.prepare_data` "
            "komutunu çalıştırın."
        )
    clean_data = pd.read_csv(clean_path, encoding="utf-8", low_memory=False)
    manifest = pd.read_csv(manifest_path, encoding="utf-8")
    merged = manifest.merge(clean_data, on="article_id", how="inner", validate="one_to_one")
    if len(merged) != len(clean_data):
        raise ValueError("Split manifesti ile temiz veri birbiriyle uyuşmuyor.")
    return {
        split_name: group.reset_index(drop=True)
        for split_name, group in merged.groupby("split", sort=False)
    }

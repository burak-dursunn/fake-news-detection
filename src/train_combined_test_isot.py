"""Combined Corpus eğitim/doğrulama ve ISOT test bölmesiyle ayrı deney."""
from __future__ import annotations

print("[1/7] Program başladı; eğitim paketleri yükleniyor...", flush=True)

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainerCallback, set_seed)
from transformers.trainer_callback import PrinterCallback

from src.config import load_config
from src.metrics import classification_metrics, probabilities_from_logits
from src.plots import plot_test_curves_and_matrix, plot_training_history
from src.run_experiment import NewsDataset, build_training_arguments


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_progress(series: pd.Series, function, name: str) -> pd.Series:
    total = len(series)
    count = 0

    def apply(value):
        nonlocal count
        result = function(value)
        count += 1
        if count % 10000 == 0 or count == total:
            print(f"  {name}: {count:,}/{total:,}", flush=True)
        return result

    return series.map(apply)


class LiveProgress(TrainerCallback):
    """Colab alt süreç çıktısında okunabilir periyodik eğitim durumu."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()
        print(f"[5/7] Eğitim başladı: toplam {state.max_steps:,} adım.", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs and state.global_step:
            elapsed = time.monotonic() - self.started
            remaining = elapsed / state.global_step * (state.max_steps - state.global_step)
            percent = 100 * state.global_step / state.max_steps
            print(f"  Eğitim: {state.global_step:,}/{state.max_steps:,} adım (%{percent:.1f}) | "
                  f"tur: {state.epoch:.2f} | kayıp: {logs['loss']:.4f} | "
                  f"geçen: {elapsed / 60:.1f} dk | tahmini kalan: {remaining / 60:.1f} dk", flush=True)
        elif "eval_loss" in logs:
            print(f"  Doğrulama tamamlandı: kayıp={logs['eval_loss']:.4f}, "
                  f"F1={logs.get('eval_f1', 0):.4f}", flush=True)


def prepare_combined(folder: Path, seed: int):
    paths = sorted(folder.glob("*.csv"))
    if len(paths) != 29:
        raise ValueError(f"Tam Train klasöründe 29 CSV bekleniyor; bulunan: {len(paths)}")
    # Kaynak parçalar satır ortasından bile bölünebiliyor ve her parçaya
    # sütun başlığı eklenmiş. Sadece eklenen başlıkları kaldırıp baytları
    # sırayla birleştirmek, quoted çok satırlı kayıtları kurtarır.
    print("[2/7] 29 CSV parçası özgün sırayla birleştiriliyor...", flush=True)
    with tempfile.TemporaryDirectory(prefix="combined_train_") as temp:
        joined = Path(temp) / "train.csv"
        expected_header = None
        with joined.open("wb") as out:
            for index, path in enumerate(paths):
                with path.open("rb") as part:
                    header = part.readline()
                    if index == 0:
                        expected_header = header
                        out.write(header)
                    elif header != expected_header:
                        raise ValueError(f"Parça başlığı uyuşmuyor: {path.name}")
                    shutil.copyfileobj(part, out, length=1024 * 1024)
                print(f"  Birleştirildi: {index + 1}/{len(paths)} ({path.name})", flush=True)
        print("  Birleştirilen CSV okunuyor; uzun metinler nedeniyle birkaç dakika sürebilir...", flush=True)
        data = pd.read_csv(joined, encoding="utf-8-sig", low_memory=False)
    print(f"  Okunan haber: {len(data):,}", flush=True)
    cols = {str(x).strip().casefold(): x for x in data.columns}
    if not {"label", "statement"}.issubset(cols):
        raise ValueError("Birleştirilen veride Label/Statement eksik.")
    data = data[[cols["label"], cols["statement"]]].copy()
    data.columns = ["label", "text"]
    data["kaynak_dosya"] = "siralariyla_birlestirilmis_29_parca"
    data["kaynak_satir"] = np.arange(2, len(data) + 2)
    initial = len(data)
    data["text"] = map_progress(data.text.fillna("").astype(str), norm, "Metinler hazırlanıyor")
    labels = pd.to_numeric(data.label, errors="coerce")
    bad = (~labels.isin([0, 1])) | data.text.eq("")
    bad_count = int(bad.sum())
    data = data.loc[~bad].copy()
    data["label"] = labels.loc[~bad].astype(int)
    data["text_key"] = map_progress(data.text, lambda v: hashlib.sha256(v.casefold().encode("utf-8")).hexdigest(),
                                    "Tekrar kontrolü")
    conflicting = data.groupby("text_key").label.transform("nunique").gt(1)
    conflict_count = int(conflicting.sum())
    data = data.loc[~conflicting].copy()
    before = len(data)
    data = data.drop_duplicates("text_key").reset_index(drop=True)
    if set(data.label.unique()) != {0, 1}:
        raise ValueError("Eğitim verisinde iki etiket de bulunmalıdır.")
    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=0.1,
                                          random_state=seed, stratify=data.label)
    data["split"] = "train"
    data.loc[val_idx, "split"] = "validation"
    data["title"] = ""
    info = {"girdi_satir_sayisi": initial, "bos_gecersiz_satir": bad_count,
            "celiskili_tekrar_satirlari": conflict_count, "tekrar_satirlari": before - len(data),
            "egitim_sayisi": len(train_idx), "dogrulama_sayisi": len(val_idx),
            "etiket_sayilari": {str(k): int(v) for k, v in data.label.value_counts().sort_index().items()},
            "not": "Kaynak Train bölmesinden %10 doğrulama ayrıldı. Etiket anlamı kullanıcı varsayımıdır; kaynak tarafından doğrulanmadı."}
    return data, info


def load_isot_test(folder: Path, seed: int, sample_size: int = 10_000):
    clean = folder / "isot_temiz.csv"
    if not clean.is_file():
        raise FileNotFoundError(f"ISOT temiz CSV bulunamadı: {clean}")
    data = pd.read_csv(clean, low_memory=False)
    if not {"article_id", "text", "label"}.issubset(data.columns):
        raise ValueError("ISOT verisinde article_id, text ve label sütunları gerekli.")
    if data.article_id.isna().any() or data.article_id.duplicated().any():
        raise ValueError("ISOT haber kimlikleri boş veya tekrarlı.")
    labels = pd.to_numeric(data.label, errors="coerce")
    if (data.text.isna().any() or data.text.astype(str).str.strip().eq("").any()
            or not labels.isin([0, 1]).all() or set(labels.unique()) != {0, 1}):
        raise ValueError("ISOT test verisi boş veya geçersiz.")
    if len(data) <= sample_size:
        raise ValueError(f"{sample_size} haber seçmek için yeterli ISOT kaydı yok: {len(data)}")
    data["label"] = labels.astype(int)
    _, indices = train_test_split(np.arange(len(data)), test_size=sample_size,
                                   random_state=seed, stratify=data.label)
    test = data.iloc[indices][["article_id", "text", "label"]].copy()
    test["text"] = test.text.astype(str).map(norm)
    test["title"] = ""  # Combined Corpus'ta başlık yok; her iki veri için aynı girdi.
    return test.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined-train", type=Path, required=True)
    p.add_argument("--isot-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    a = p.parse_args()
    config = load_config(a.config)
    if a.output.exists():
        raise FileExistsError(f"Sonuç klasörü mevcut, korunuyor: {a.output}")
    set_seed(config.seed)
    data, info = prepare_combined(a.combined_train, config.seed)
    print("[3/7] ISOT verisinden 10.000 test haberi seçiliyor...", flush=True)
    test = load_isot_test(a.isot_dir, config.seed)
    if set(test.label.unique()) != {0, 1}:
        raise ValueError("ISOT test bölmesinde iki sınıf da olmalı.")
    info.update({"isot_test_sayisi": len(test), "isot_klasoru": str(a.isot_dir),
                 "isot_temiz_sha256": sha256(a.isot_dir / "isot_temiz.csv"),
                 "isot_secimi": {"yontem": "sınıf oranları korunarak rastgele örnekleme",
                                  "rastgele_tohum": config.seed, "kaynak": "temiz ISOT verisinin tamamı"},
                 "isot_test_etiket_sayilari": {str(k): int(v) for k, v in test.label.value_counts().sort_index().items()},
                 "model_girdisi": "Her iki veri kümesinde yalnız haber metni; başlık boş",
                 "etiket_varsayimi": {"0": "gerçek", "1": "sahte"},
                 "sinir": "Combined Corpus ve ISOT arasında benzerlik kontrolü yapılmadı."})
    a.output.mkdir(parents=True)
    (a.output / "veri_ozeti.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.output / "ayarlar.json").write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = a.combined_train / "indirme_kaynagi.json"
    if manifest_path.exists():
        (a.output / "combined_kaynak.json").write_bytes(manifest_path.read_bytes())
    data[["kaynak_dosya", "kaynak_satir", "label", "text_key", "split"]].to_csv(a.output / "combined_egitim_dogrulama_bolmesi.csv", index=False)
    print("Veri özeti:", json.dumps(info, ensure_ascii=False), flush=True)
    print(f"[4/7] Model indiriliyor/yükleniyor: {config.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2,
                 id2label={0: "Gercek", 1: "Sahte"}, label2id={"Gercek": 0, "Sahte": 1})
    print("  Eğitim metinleri tokenlara dönüştürülüyor...", flush=True)
    train_data = NewsDataset(data[data.split.eq("train")], tokenizer, config.max_length)
    print("  Doğrulama metinleri tokenlara dönüştürülüyor...", flush=True)
    valid_data = NewsDataset(data[data.split.eq("validation")], tokenizer, config.max_length)
    print("  ISOT test metinleri tokenlara dönüştürülüyor...", flush=True)
    test_data = NewsDataset(test, tokenizer, config.max_length)
    arguments = build_training_arguments(config, a.checkpoint_root / a.output.name,
                                          len(train_data), torch.cuda.is_available())
    arguments.disable_tqdm = True
    trainer = Trainer(model=model, args=arguments, train_dataset=train_data, eval_dataset=valid_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=lambda x: classification_metrics(x.label_ids, x.predictions),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience), LiveProgress()])
    trainer.remove_callback(PrinterCallback)
    print("  Model eğitime hazırlanıyor...", flush=True)
    trainer.train()
    print("[6/7] ISOT test haberleri değerlendiriliyor...", flush=True)
    result = trainer.predict(test_data, metric_key_prefix="test")
    probabilities = probabilities_from_logits(result.predictions)
    pred = (probabilities >= 0.5).astype(int)
    test[["article_id", "label"]].assign(tahmin=pred, sahte_olasiligi=probabilities).to_csv(
        a.output / "ISOT_Test_Tahminleri.csv", index=False, encoding="utf-8-sig")
    results = {"metrikler": classification_metrics(result.label_ids, result.predictions),
        "siniflar": classification_report(result.label_ids, pred, labels=[0, 1],
                    target_names=["gercek", "sahte"], output_dict=True, zero_division=0),
        "karisiklik_matrisi": confusion_matrix(result.label_ids, pred, labels=[0, 1]).tolist()}
    (a.output / "ISOT_Test_Metrikleri.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.output / "Egitim_Gecmisi.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("[7/7] Grafikler ve model kaydediliyor...", flush=True)
    plot_training_history(trainer.state.log_history, a.output)
    plot_test_curves_and_matrix(result.label_ids, result.predictions, a.output)
    trainer.save_model(a.output / "en_iyi_model")
    tokenizer.save_pretrained(a.output / "en_iyi_model")
    print("Deney tamamlandı:", a.output, flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

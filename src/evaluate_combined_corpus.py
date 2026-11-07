"""ISOT modelini Combined Corpus haberleri üzerinde yeniden eğitmeden değerlendirir."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.metrics import classification_metrics, probabilities_from_logits
from src.plots import plot_test_curves_and_matrix


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\(\s*reuters\s*\)|\breuters\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV dosyası bulunamadı: {data_dir}")
    frames, file_info = [], []
    for path in files:
        print(f"Okunuyor: {path.name}", flush=True)
        raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        columns = {str(col).strip().casefold(): col for col in raw.columns}
        if not {"label", "statement"}.issubset(columns):
            raise ValueError(f"{path.name}: Label ve Statement sütunları gerekli.")
        frame = raw[[columns["label"], columns["statement"]]].copy()
        frame.columns = ["label", "text"]
        frame.insert(0, "kaynak_satir", np.arange(2, len(frame) + 2))
        frame.insert(0, "kaynak_dosya", path.name)
        labels = pd.to_numeric(frame.label, errors="coerce")
        if not labels.isin([0, 1]).all():
            raise ValueError(f"{path.name}: Etiketler sadece 0 veya 1 olmalı.")
        frame["label"] = labels.astype(int)
        frame["text"] = frame.text.fillna("").astype(str).map(clean_text)
        if frame.text.eq("").any():
            raise ValueError(f"{path.name}: Temizlik sonrası boş metin bulundu.")
        frames.append(frame)
        file_info.append({"dosya": path.name, "satir": len(frame),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.concat(frames, ignore_index=True), file_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 4:
        parser.error("Batch size pozitif, max length en az 4 olmalı.")
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"Kaydedilmiş model klasörü bulunamadı: {args.model}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Sonuç klasörü boş değil: {args.output}")
    data, files = load_data(args.data_dir)
    if set(data.label.unique()) != {0, 1}:
        raise ValueError("ROC ve PR için her iki etiket de gerekli.")
    print(f"{len(data)} kayıt hazır. Model yükleniyor: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
    if model.config.num_labels != 2:
        raise ValueError("Modelin iki sınıflı olması gerekir.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Test başlıyor; aygıt: {device}", flush=True)
    all_logits, lengths = [], []
    with torch.inference_mode():
        for start in range(0, len(data), args.batch_size):
            texts = data.text.iloc[start:start + args.batch_size].tolist()
            # Eğitimde kullanılan çift girdi düzeni: başlık + metin.
            # Bu dosyalarda başlık bulunmadığı için başlık bilinçli olarak boştur.
            uncut = tokenizer([""] * len(texts), texts, truncation=False, padding=False,
                              verbose=False)
            lengths.extend(len(ids) for ids in uncut["input_ids"])
            inputs = tokenizer([""] * len(texts), texts, truncation=True,
                               max_length=args.max_length, padding=True, return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
            all_logits.append(logits.cpu().numpy())
            if start == 0 or (start // args.batch_size) % 50 == 0 or start + len(texts) == len(data):
                print(f"Test: {start + len(texts)}/{len(data)}", flush=True)
    logits = np.concatenate(all_logits)
    labels = data.label.to_numpy()
    probs = probabilities_from_logits(logits)
    pred = (probs >= 0.5).astype(int)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "model_klasoru": str(args.model), "etiketler": {"0": "gerçek", "1": "sahte"},
        "girdi": "Boş başlık + haber metni; Reuters ifadesi kaldırıldı",
        "max_token": args.max_length, "esik": 0.5,
        "dosyalar": files, "kayit_sayisi": len(data),
        "etiket_sayilari": {str(k): int(v) for k, v in data.label.value_counts().sort_index().items()},
        "token_sinirini_asan_kayit_sayisi": int(np.sum(np.asarray(lengths) > args.max_length)),
        "ayni_metin_tekrar_satirlari": int(data.text.str.casefold().duplicated().sum()),
        "metrikler": classification_metrics(labels, logits),
        "sinif_bazinda": classification_report(labels, pred, labels=[0, 1],
            target_names=["gercek", "sahte"], output_dict=True, zero_division=0),
        "karisiklik_matrisi": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
        "not": "Tüm satırlar değerlendirilmiştir; ISOT ile benzerlik kontrolü yapılmadı.",
    }
    pd.DataFrame({"kaynak_dosya": data.kaynak_dosya, "kaynak_satir": data.kaynak_satir,
        "gercek_etiket": labels, "tahmin": pred, "sahte_olasiligi": probs,
        "kesilmeden_token_sayisi": lengths}).to_csv(args.output / "Combined_Corpus_Tahminler.csv", index=False, encoding="utf-8-sig")
    (args.output / "Combined_Corpus_Test_Raporu.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_test_curves_and_matrix(labels, logits, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Sonuçlar kaydedildi: {args.output}", flush=True)


if __name__ == "__main__":
    main()

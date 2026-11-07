"""WELFake ile eğitilmiş sınıflandırıcıyı ISOT üzerinde, örtüşmeyi bildirerek değerlendir."""
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
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.metrics import classification_metrics, probabilities_from_logits
from src.plots import plot_test_curves_and_matrix


def text_key(value: str) -> str:
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_isot(fake_path: Path, true_path: Path) -> pd.DataFrame:
    frames = []
    for path, label in [(fake_path, 1), (true_path, 0)]:
        data = pd.read_csv(path, usecols=["title", "text"], low_memory=False).fillna("")
        data["label"] = label
        frames.append(data)
    data = pd.concat(frames, ignore_index=True)
    data = data.loc[data["text"].str.strip().ne("")].copy()
    data["text_key"] = data["text"].map(text_key)
    return data.reset_index(drop=True)


def mark_overlap(isot: pd.DataFrame, prepared_dir: Path) -> pd.DataFrame:
    welfake = pd.read_csv(prepared_dir / "welfake_temiz.csv", usecols=["article_id", "text"])
    manifest = pd.read_csv(prepared_dir / "welfake_split_seed42.csv")
    welfake = welfake.merge(manifest, on="article_id", validate="one_to_one")
    welfake["text_key"] = welfake["text"].fillna("").map(text_key)
    if welfake["text_key"].duplicated().any():
        raise ValueError("WELFake v2 içinde tekrar eden metin bulundu; split kontrolü gerekli.")
    result = isot.merge(welfake[["text_key", "split"]], on="text_key", how="left", validate="many_to_one")
    result["split"] = result["split"].fillna("welfake_disinda")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", type=Path, required=True)
    parser.add_argument("--true", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Sonuç klasörü boş değil: {args.output}")
    print("ISOT okunuyor; WELFake bölmeleriyle karşılaştırılıyor...", flush=True)
    data = mark_overlap(load_isot(args.fake, args.true), args.prepared_dir)
    print("ISOT kayıtlarının WELFake bölmeleri:", data["split"].value_counts().to_dict(), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model yükleniyor: {args.model} ({device})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(data), args.batch_size):
            batch = data.iloc[start:start + args.batch_size]
            inputs = tokenizer(batch.title.tolist(), batch.text.tolist(), truncation=True,
                               max_length=args.max_length, padding=True, return_tensors="pt")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            scores.append(model(**inputs).logits.cpu().numpy())
            if start % (args.batch_size * 100) == 0:
                print(f"Değerlendiriliyor: {min(start + len(batch), len(data))}/{len(data)}", flush=True)
    logits = np.concatenate(scores)
    labels = data["label"].to_numpy()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    results = {"uyari": "ISOT, WELFake ile örtüşür; tamamı bağımsız dış test değildir.",
               "etiketler": {"0": "gerçek", "1": "sahte"},
               "max_length": args.max_length, "gruplar": {}}
    for name, mask in [("tum_isot", np.ones(len(data), dtype=bool))] + [
        (split, data["split"].eq(split).to_numpy())
        for split in ["train", "validation", "test", "welfake_disinda"]
    ]:
        count = int(mask.sum())
        entry = {"kayit_sayisi": count}
        if count and len(np.unique(labels[mask])) == 2:
            entry["metrikler"] = classification_metrics(labels[mask], logits[mask])
            entry["karisiklik_matrisi"] = confusion_matrix(labels[mask], logits[mask].argmax(axis=1), labels=[0, 1]).tolist()
            if name in {"tum_isot", "test", "welfake_disinda"}:
                graph_dir = output / name
                graph_dir.mkdir()
                plot_test_curves_and_matrix(labels[mask], logits[mask], graph_dir)
        results["gruplar"][name] = entry
    data[["label", "split"]].assign(sahte_olasiligi=probabilities_from_logits(logits)).to_csv(
        output / "ISOT_tahminleri.csv", index=False
    )
    (output / "ISOT_degerlendirme_raporu.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print(f"Sonuçlar: {output}", flush=True)


if __name__ == "__main__":
    main()

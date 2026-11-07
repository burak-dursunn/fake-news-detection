"""Yalnızca ISOT için temizlik, benzer haber gruplaması ve sabit bölme.

MinHash yalnızca aday bulur; birleştirmeden önce tam kelime parçalarının
Jaccard benzerliği ölçülür. Yaklaşık arama tüm yakın tekrarları garanti etmez.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedGroupKFold

from src.data import audit_split, clean_news_frame

SCHEMA_VERSION = 1


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(value: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def shingles(text: str) -> set[bytes]:
    words = normalized(text).split()
    return {" ".join(words[i:i + 5]).encode("utf-8") for i in range(len(words) - 4)}


def group_similar_articles(frame: pd.DataFrame) -> tuple[list[int], dict]:
    """Başlık, giriş ve yüksek metin benzerliğinin bağlı bileşenlerini bulur."""
    parents = list(range(len(frame)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    seen = {}
    index = MinHashLSH(threshold=0.75, num_perm=64)
    texts = frame["text"].tolist()
    near_pairs = 0
    for i, (title, text) in enumerate(zip(frame["title"], texts, strict=True)):
        # Eski audit_split ile de uyumlu ham-normalize giriş anahtarı tutulur.
        keys = [("prefix", text.casefold()[:300])]
        norm_title = normalized(title)
        if norm_title:
            keys.append(("title", norm_title))
        for kind, value in keys:
            key = (kind, hashlib.sha256(value.encode("utf-8")).digest())
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
        chunks = shingles(text)
        if chunks:
            sketch = MinHash(num_perm=64, seed=42)
            # Uzun belgelerde büyük geçici NumPy dizileri oluşturma.
            values = sorted(chunks)
            for offset in range(0, len(values), 256):
                sketch.update_batch(values[offset:offset + 256])
            for candidate in sorted(index.query(sketch)):
                other = shingles(texts[candidate])
                similarity = len(chunks & other) / len(chunks | other)
                if similarity >= 0.85:
                    union(i, candidate)
                    near_pairs += 1
            index.insert(i, sketch)
        if (i + 1) % 1000 == 0 or i + 1 == len(frame):
            print(f"Benzerlik taraması: {i + 1}/{len(frame)}", flush=True)
    groups = [root(i) for i in range(len(frame))]
    return groups, {
        "method": "normalized_title_or_300char_prefix_or_5word_jaccard_0.85",
        "candidate_search": "MinHashLSH threshold=0.75 num_perm=64 seed=42",
        "verified_near_duplicate_pairs": near_pairs,
        "group_count": len(set(groups)),
        "largest_group": int(pd.Series(groups).value_counts().max()),
        "limitation": "Yaklaşık aday araması tüm benzer haberleri veya aynı olayı yakalamaz.",
    }


def split_groups(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    manifest = frame[["article_id", "group_id"]].copy()
    manifest["split"] = ""
    for fold, (_, indices) in enumerate(splitter.split(frame, frame["label"], frame["group_id"])):
        manifest.loc[indices, "split"] = "train" if fold < 14 else "validation" if fold < 17 else "test"
    return manifest


def verify_prepared_isot(output_dir: Path, seed: int = 42) -> dict:
    """Tekrar metin benzerliği hesaplamadan dosya sürümünü ve grupları denetler."""
    report = json.loads((output_dir / "isot_hazirlama_raporu.json").read_text(encoding="utf-8"))
    if report.get("dataset") != "isot" or report.get("schema_version") != SCHEMA_VERSION or report.get("seed") != seed:
        raise ValueError("ISOT veri sürümü/tohumu uyuşmuyor; yeniden hazırlayın.")
    expected = {"isot_temiz.csv", f"isot_split_seed{seed}.csv"}
    if set(report.get("prepared_sha256", {})) != expected:
        raise ValueError("ISOT dosya özeti listesi eksik/geçersiz.")
    for name in expected:
        if file_hash(output_dir / name) != report["prepared_sha256"][name]:
            raise ValueError(f"Hazır ISOT dosyası değişmiş veya eksik kopyalanmış: {name}")
    manifest = pd.read_csv(output_dir / f"isot_split_seed{seed}.csv")
    if manifest[["article_id", "group_id", "split"]].isna().any().any():
        raise ValueError("ISOT manifestinde boş kimlik/bölme var.")
    if manifest.groupby("group_id")["split"].nunique().gt(1).any():
        raise ValueError("Benzer ISOT haberleri farklı bölmelerde bulundu.")
    return report


def prepare_isot(true_path: Path, fake_path: Path, output_dir: Path, seed: int = 42) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Hedef boş değil; önceki veri korunuyor: {output_dir}")
    frames = []
    for path, label in ((true_path, 0), (fake_path, 1)):
        print(f"Ham dosya okunuyor: {path.name} -> etiket {label}", flush=True)
        frame = pd.read_csv(path, usecols=["title", "text"], keep_default_na=False)
        frame["label"] = label
        frames.append(frame)
    raw_counts = {"true": len(frames[0]), "fake": len(frames[1])}
    clean, cleaning = clean_news_frame(pd.concat(frames, ignore_index=True))
    del frames
    print(f"Temizlik tamamlandı: {len(clean)} haber. Benzer haberler gruplanıyor...", flush=True)
    groups, grouping = group_similar_articles(clean)
    clean["group_id"] = groups
    manifest = split_groups(clean, seed)
    audit = audit_split(clean, manifest[["article_id", "split"]])
    audit["similar_groups_across_splits"] = int(manifest.groupby("group_id")["split"].nunique().gt(1).sum())
    joined = clean.merge(manifest[["article_id", "split"]], validate="one_to_one", on="article_id")
    for name in ("train", "validation", "test"):
        if set(joined.loc[joined["split"].eq(name), "label"]) != {0, 1}:
            raise ValueError(f"{name} her iki sınıfı da içermiyor.")
    output_dir.mkdir(parents=True, exist_ok=True)
    clean.drop(columns="group_id").to_csv(output_dir / "isot_temiz.csv", index=False)
    manifest.to_csv(output_dir / f"isot_split_seed{seed}.csv", index=False)
    report = {
        "dataset": "isot", "schema_version": SCHEMA_VERSION, "seed": seed,
        "label_mapping": {"0": "Gerçek (True.csv)", "1": "Sahte (Fake.csv)"},
        "raw_counts": raw_counts, "cleaning": asdict(cleaning), "grouping": grouping,
        "audit": audit, "split_counts": joined["split"].value_counts().to_dict(),
        "label_counts_by_split": {name: {str(k): int(v) for k, v in group["label"].value_counts().items()}
                                  for name, group in joined.groupby("split")},
        "raw_sha256": {"true": file_hash(true_path), "fake": file_hash(fake_path)},
        "prepared_sha256": {name: file_hash(output_dir / name)
                            for name in ("isot_temiz.csv", f"isot_split_seed{seed}.csv")},
        "limitations": ["Etiketler kaynak dosyadan alınır; haberlerin doğruluğu tek tek doğrulanmaz.",
                        "Kaynak farkı (Reuters vb.) başarıyı şişirebilir.",
                        "Bu, kaynaklardan bağımsız veya tarih sıralı bir test değildir."],
    }
    (output_dir / "isot_hazirlama_raporu.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    verify_prepared_isot(output_dir, seed)
    return report


def main():
    parser = argparse.ArgumentParser(description="Yalnızca ISOT'u bir kez hazırlar; model indirmez.")
    parser.add_argument("--true", type=Path, required=True)
    parser.add_argument("--fake", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prepare_isot(args.true, args.fake, args.output_dir, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

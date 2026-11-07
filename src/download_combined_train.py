"""Yalnızca Combined Corpus Train dosyalarını sabit GitHub sürümünden indir."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMMIT = "aa17a5a3ec98b1dd3a8ecc54587cda6ac2809264"
ROOT = "Dataset/Combined_Corpus/Train"
REPO = "JunaedYounusKhan51/FakeNewsDetection"


def request(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fake-news-detection-research"}), timeout=120)


def git_blob_hash(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(output: Path) -> None:
    url = f"https://api.github.com/repos/{REPO}/contents/{ROOT}?ref={COMMIT}"
    with request(url) as response:
        items = json.load(response)
    items = [item for item in items if item["type"] == "file" and item["name"].endswith(".csv")]
    if len(items) != 29:
        raise ValueError(f"Kaynakta beklenen 29 CSV yerine {len(items)} dosya bulundu.")
    output.mkdir(parents=True, exist_ok=True)

    def one(item):
        target = output / item["name"]
        if target.exists():
            if git_blob_hash(target) != item["sha"]:
                raise ValueError(f"Mevcut dosya kaynaktan farklı; üzerine yazılmadı: {target}")
            return f"Zaten mevcut: {target.name}"
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=output, suffix=".part", delete=False) as stream:
                tmp = Path(stream.name)
                with request(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{ROOT}/{item['name']}") as response:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
            if git_blob_hash(tmp) != item["sha"]:
                raise ValueError(f"İndirme doğrulaması başarısız: {item['name']}")
            os.replace(tmp, target)
            return f"İndirildi: {target.name}"
        finally:
            if tmp and tmp.exists():
                tmp.unlink()

    with ThreadPoolExecutor(max_workers=4) as pool:
        for message in pool.map(one, items):
            print(message, flush=True)
    manifest = {"repo": REPO, "commit": COMMIT, "klasor": ROOT,
                "dosyalar": [{"ad": x["name"], "git_blob_sha1": x["sha"], "bayt": x["size"]} for x in items]}
    (output / "indirme_kaynagi.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    download(parser.parse_args().output)

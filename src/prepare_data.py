"""Komut satırından WELFake temizliği ve sabit veri bölmesi oluşturur."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_and_clean_welfake, save_prepared_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WELFake verisini temizler ve train/validation/test bölmesi üretir."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Ham WELFake_Dataset.csv dosyasının yolu.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Temiz verinin ve raporların yazılacağı klasör.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Bölme için rastgele tohum.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Hedef klasördeki hazırlanmış veri dosyalarının üzerine yazar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Ham veri bulunamadı: {args.input}")

    print("WELFake ham verisi okunuyor ve temizleniyor...")
    clean_data, report = load_and_clean_welfake(args.input)
    save_prepared_data(
        clean_data=clean_data,
        report=report,
        output_dir=args.output_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Temiz veri hazır: {report.final_rows:,} kayıt")
    print(f"Boş text nedeniyle çıkarılan kayıt: {report.rows_with_empty_text:,}")
    print(f"Tekrarlı kayıt nedeniyle çıkarılan kayıt: {report.duplicate_rows_removed:,}")
    print(f"Çıktı klasörü: {args.output_dir}")


if __name__ == "__main__":
    main()

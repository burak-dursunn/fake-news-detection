import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import create_split_manifest, load_and_clean_welfake


class WELFakeCleaningTests(unittest.TestCase):
    def test_cleaning_removes_empty_invalid_and_duplicate_rows(self) -> None:
        rows = pd.DataFrame(
            {
                "title": [" Başlık ", "Başlık", "Boş", "Hatalı"],
                "text": [" Metin\n alanı ", "Metin alanı", " ", "Geçerli metin"],
                "label": [1, 1, 0, 7],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "welfake.csv"
            rows.to_csv(path, index=False)
            clean, report = load_and_clean_welfake(path)

        self.assertEqual(len(clean), 1)
        self.assertEqual(report.rows_with_empty_text, 1)
        self.assertEqual(report.rows_with_invalid_label, 1)
        self.assertEqual(report.duplicate_rows_removed, 1)
        self.assertEqual(clean.iloc[0]["text"], "Metin alanı")

    def test_split_keeps_each_article_in_one_partition(self) -> None:
        frame = pd.DataFrame(
            {
                "article_id": [f"id-{index}" for index in range(20)],
                "label": [0, 1] * 10,
            }
        )
        manifest = create_split_manifest(frame, seed=42)
        self.assertEqual(len(manifest), len(frame))
        self.assertEqual(manifest["article_id"].nunique(), len(frame))
        self.assertEqual(set(manifest["split"]), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()

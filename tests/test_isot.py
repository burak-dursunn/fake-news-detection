import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import clean_news_frame, load_prepared_splits
from src.prepare_isot import file_hash, group_similar_articles, split_groups, verify_prepared_isot


class ISOTTests(unittest.TestCase):
    def test_conflicting_texts_are_removed_not_relabelled(self):
        raw = pd.DataFrame({"title": ["A", "B", "C"], "text": ["same", "SAME", "valid"], "label": [0, 1, 0]})
        clean, report = clean_news_frame(raw)
        self.assertEqual(clean["text"].tolist(), ["valid"])
        self.assertEqual(report.rows_with_conflicting_duplicate_content, 2)

    def test_similarity_and_title_connections_are_transitive(self):
        original = " ".join(f"word{i}" for i in range(200))
        changed = original.replace("word100", "replacement")
        frame = pd.DataFrame({"title": ["First headline", "New headline", "NEW HEADLINE!", "Other"],
                              "text": [original, "extra " + changed, "Unrelated text", "Separate item"]})
        groups, report = group_similar_articles(frame)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])
        self.assertGreaterEqual(report["verified_near_duplicate_pairs"], 1)

    def test_groups_never_cross_splits(self):
        frame = pd.DataFrame({"article_id": [str(i) for i in range(80)],
                              "group_id": [i // 2 for i in range(80)], "label": [i % 2 for i in range(80)]})
        manifest = split_groups(frame, 42)
        self.assertEqual(set(manifest["split"]), {"train", "validation", "test"})
        self.assertEqual(manifest.groupby("group_id")["split"].nunique().max(), 1)

    def test_hash_mismatch_blocks_training_even_without_text_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            clean = pd.DataFrame({"article_id": ["a", "b", "c"], "title": ["A", "B", "C"],
                                  "text": ["one", "two", "three"], "label": [0, 1, 0]})
            manifest = pd.DataFrame({"article_id": ["a", "b", "c"], "group_id": [0, 1, 2],
                                     "split": ["train", "validation", "test"]})
            clean.to_csv(folder / "isot_temiz.csv", index=False)
            manifest.to_csv(folder / "isot_split_seed42.csv", index=False)
            report = {"dataset": "isot", "schema_version": 1, "seed": 42,
                      "prepared_sha256": {p.name: file_hash(p) for p in folder.glob("*.csv")}}
            (folder / "isot_hazirlama_raporu.json").write_text(json.dumps(report), encoding="utf-8")
            verify_prepared_isot(folder)
            self.assertEqual(set(load_prepared_splits(folder, audit=False, dataset_name="isot")),
                             {"train", "validation", "test"})
            with (folder / "isot_temiz.csv").open("a") as stream:
                stream.write("corrupted")
            with self.assertRaisesRegex(ValueError, "değişmiş"):
                load_prepared_splits(folder, audit=False, dataset_name="isot")


if __name__ == "__main__":
    unittest.main()

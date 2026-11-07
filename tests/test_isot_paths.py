import tempfile
import unittest
from pathlib import Path

from src.isot_paths import PREPARED_NAMES, find_prepared_isot


class ISOTPathTests(unittest.TestCase):
    def test_finds_prepared_data_in_raw_folder_with_case_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploaded = root / 'data' / 'raw'
            uploaded.mkdir(parents=True)
            for name in PREPARED_NAMES:
                (uploaded / name.upper()).touch()
            found = find_prepared_isot([root / 'data/ISOT/hazir_v1', uploaded])
            self.assertEqual(set(found), set(PREPARED_NAMES))
            self.assertTrue(all(path.parent == uploaded for path in found.values()))

    def test_partial_copies_are_not_combined(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'first', Path(tmp) / 'second'
            first.mkdir()
            second.mkdir()
            (first / PREPARED_NAMES[0]).touch()
            for name in PREPARED_NAMES[1:]:
                (second / name).touch()
            with self.assertRaisesRegex(FileNotFoundError, 'eksik'):
                find_prepared_isot([first, second])

    def test_complete_copy_takes_precedence_over_partial_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'first', Path(tmp) / 'second'
            first.mkdir()
            second.mkdir()
            (first / PREPARED_NAMES[0]).touch()
            for name in PREPARED_NAMES:
                (second / name).touch()
            found = find_prepared_isot([first, second])
            self.assertTrue(all(path.parent == second for path in found.values()))

    def test_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_prepared_isot([Path(tmp)]))


if __name__ == '__main__':
    unittest.main()

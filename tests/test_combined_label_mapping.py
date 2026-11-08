"""Combined Corpus ve ISOT etiketlerinin eğitimde aynı anlama gelmesi."""
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.train_combined_test_isot import prepare_combined


def test_combined_labels_are_reversed_after_existing_split(tmp_path: Path):
    # Dosya parçaları, kaynakta olduğu gibi her parçada aynı başlığı içerir.
    for number in range(29):
        with (tmp_path / f"part_{number:03}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Statement", "Label"])
            for j in range(4):
                writer.writerow([f"Unique article {number} number {j}", j % 2])
    data, info = prepare_combined(tmp_path, 42)
    assert len(data) == 116
    assert (data.label == 1 - data.kaynak_etiket).all()
    _, expected_validation = train_test_split(np.arange(len(data)), test_size=0.1,
                                               random_state=42, stratify=data.kaynak_etiket)
    assert set(data.index[data.split.eq("validation")]) == set(expected_validation)
    assert info["kaynak_etiket_sayilari"] == {"0": 58, "1": 58}

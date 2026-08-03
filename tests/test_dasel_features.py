from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baseline_ml.dasel_features import extract_windows  # noqa: E402


class DaselFeatureTests(unittest.TestCase):
    def test_extracts_262_columns_and_removes_exact_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "DASEL_preprocessed_train.csv"
            output = root / "features"
            rows = [
                {"timestamp": "2023-04-10T14:21:46+09:00", "mac_address": 1, "RSSI": -70, "room": "a"},
                {"timestamp": "2023-04-10T14:21:46+09:00", "mac_address": 1, "RSSI": -70, "room": "a"},
                {"timestamp": "2023-04-10T14:21:46+09:00", "mac_address": 2, "RSSI": -80, "room": "a"},
                {"timestamp": "2023-04-10T14:21:47+09:00", "mac_address": 1, "RSSI": -60, "room": "b"},
            ]
            pd.DataFrame(rows).to_csv(source, index=False)

            manifest = extract_windows(source, output, windows=[1, 3])
            one_second = pd.read_csv(output / "rf_window_features_1s.csv")
            three_second = pd.read_csv(output / "rf_window_features_3s.csv")
            saved_manifest = json.loads((output / "feature_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(one_second.shape, (2, 262))
            self.assertEqual(three_second.shape, (1, 262))
            self.assertEqual(int(one_second["total_detections"].sum()), 3)
            self.assertEqual(manifest["deduplicated_detection_rows"], 3)
            self.assertEqual(saved_manifest["source_file"], source.name)
            self.assertFalse(saved_manifest["source_path_published"])
            self.assertNotIn(str(root), json.dumps(saved_manifest))


if __name__ == "__main__":
    unittest.main()

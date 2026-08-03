from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baseline_ml.strong_inference import (  # noqa: E402
    align_window_predictions,
    extract_unlabeled_features,
    load_unlabeled_detections,
    map_predictions_to_source_rows,
)


class StrongInferenceTests(unittest.TestCase):
    def test_challenge_schema_is_normalized_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.csv"
            frame = pd.DataFrame(
                {
                    "Unnamed: 0": [10, 11, 12],
                    "user_id": [90, 90, 90],
                    "timestamp": ["2023-04-14 10:00:00"] * 3,
                    "mac address": [1, 1, 2],
                    "RSSI": [-90, -90, -95],
                    "power": [-2_147_483_648] * 3,
                }
            )
            frame.to_csv(path, index=False)
            normalized, detections, audit = load_unlabeled_detections(path)
            self.assertEqual(len(normalized), 3)
            self.assertEqual(len(detections), 2)
            self.assertEqual(audit["exact_duplicate_rows_removed"], 1)
            self.assertEqual(audit["n_users"], 1)
            self.assertNotIn("users", audit)
            self.assertEqual(set(detections["mac_address"]), {1, 2})

            features = extract_unlabeled_features(detections, 1)
            self.assertEqual(features.shape, (1, 263))
            self.assertEqual(features.loc[0, "total_detections"], 2)
            self.assertEqual(features.loc[0, "room"], "__unlabeled__")

    def test_subsecond_packets_are_not_collapsed_as_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.csv"
            pd.DataFrame(
                {
                    "user_id": [90, 90],
                    "timestamp": [
                        "2023-04-14 10:00:00.100",
                        "2023-04-14 10:00:00.900",
                    ],
                    "mac address": [1, 1],
                    "RSSI": [-90, -90],
                }
            ).to_csv(path, index=False)
            _, detections, audit = load_unlabeled_detections(path)
            self.assertEqual(len(detections), 2)
            self.assertEqual(audit["exact_duplicate_rows_removed"], 0)
            features = extract_unlabeled_features(detections, 1)
            self.assertEqual(features.loc[0, "total_detections"], 2)

    @staticmethod
    def _predictions(window_seconds: int, timestamps: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "window_seconds": window_seconds,
                "window_ts_local": timestamps,
                "window_ts_utc": timestamps,
                "predicted_room_raw": ["501"] * len(timestamps),
                "predicted_room_causal": ["501"] * len(timestamps),
                "predicted_room_fixed_decoder": ["501"] * len(timestamps),
                "confidence_raw": [0.8] * len(timestamps),
                "entropy_raw": [0.5] * len(timestamps),
            }
        )

    def test_1s_3s_alignment_and_probability_class_order(self) -> None:
        timestamps_1s = [
            "2023-04-14T01:00:00+00:00",
            "2023-04-14T01:00:01+00:00",
            "2023-04-14T01:00:02+00:00",
        ]
        timestamps_3s = ["2023-04-14T01:00:00+00:00"]
        predictions_1s = self._predictions(1, timestamps_1s)
        predictions_3s = self._predictions(3, timestamps_3s)
        probabilities_1s = pd.DataFrame(
            {"prob_raw__501": [0.8] * 3, "prob_raw__502": [0.2] * 3}
        )
        probabilities_3s = pd.DataFrame(
            {"prob_raw__501": [0.8], "prob_raw__502": [0.2]}
        )
        aligned = align_window_predictions(
            predictions_1s, probabilities_1s, predictions_3s, probabilities_3s
        )
        self.assertEqual(len(aligned), 3)
        self.assertTrue(aligned["agree_fixed_decoder"].all())
        np.testing.assert_allclose(aligned["raw_js_divergence"], 0.0, atol=1e-12)

        reversed_probabilities = probabilities_3s[["prob_raw__502", "prob_raw__501"]]
        with self.assertRaisesRegex(AssertionError, "class orders"):
            align_window_predictions(
                predictions_1s,
                probabilities_1s,
                predictions_3s,
                reversed_probabilities,
            )

    def test_source_rows_map_many_to_one_to_window_prediction(self) -> None:
        source = pd.DataFrame(
            {
                "source_id": [1, 2],
                "source_row": [0, 1],
                "user_id": [90, 90],
                "timestamp": ["a", "b"],
                "timestamp_local": pd.to_datetime(
                    ["2023-04-14 10:00:00.1", "2023-04-14 10:00:00.9"]
                ).tz_localize("Asia/Seoul"),
                "mac_address": [1, 2],
                "RSSI": [-90, -95],
            }
        )
        prediction = self._predictions(1, ["2023-04-14 10:00:00+09:00"])
        prediction["margin_raw"] = 0.6
        prediction["confidence_causal"] = 0.8
        prediction["margin_causal"] = 0.6
        prediction["entropy_causal"] = 0.5
        mapped = map_predictions_to_source_rows(source, prediction, 1)
        self.assertEqual(len(mapped), 2)
        self.assertTrue(mapped["predicted_room_fixed_decoder"].eq("501").all())


if __name__ == "__main__":
    unittest.main()

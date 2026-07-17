from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baseline_ml.strong_ml import (  # noqa: E402
    boundary_mask,
    causal_probability_smoothing,
    feature_set,
    fit_markov_model,
    strict_fold,
    viterbi_decode,
)


class StrongMLProtocolTests(unittest.TestCase):
    def test_strict_fold_filters_both_sides_to_intersection(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["d1", "d1", "d2", "d2", "d2"],
                "window_ts": pd.to_datetime(
                    [
                        "2023-04-10 00:00:00Z",
                        "2023-04-10 00:00:01Z",
                        "2023-04-11 00:00:00Z",
                        "2023-04-11 00:00:01Z",
                        "2023-04-11 00:00:02Z",
                    ]
                ),
                "room": ["shared", "train_only", "shared", "test_only", "test_only"],
            }
        )
        train, test, audit = strict_fold(frame, "d2")
        self.assertEqual(set(train["room"]), {"shared"})
        self.assertEqual(set(test["room"]), {"shared"})
        self.assertEqual(audit["n_common_classes"], 1)
        self.assertEqual(audit["n_train_dropped"], 1)
        self.assertEqual(audit["n_test_dropped"], 2)

    def test_feature_variants_have_expected_contract(self) -> None:
        frame = pd.DataFrame(
            {**{f"freq_B{i:02d}": [1.0] for i in range(1, 24)}, **{f"f{i}": [1.0] for i in range(120)}}
        )
        selected = [f"f{i}" for i in range(120)]
        self.assertEqual(len(feature_set("M2", selected, frame)), 100)
        m5 = feature_set("M5", selected, frame)
        self.assertEqual(len(m5), 100)
        self.assertEqual(m5[:23], [f"freq_B{i:02d}" for i in range(1, 24)])

    def test_fixed_decoder_respects_probability_and_sequence_shapes(self) -> None:
        timestamps = pd.date_range("2023-04-10", periods=5, freq="s", tz="UTC")
        probabilities = np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.1, 0.9],
            ]
        )
        smoothed = causal_probability_smoothing(probabilities, timestamps, 1, 3)
        self.assertEqual(smoothed.shape, probabilities.shape)
        self.assertTrue(np.allclose(smoothed.sum(axis=1), 1.0))

        train = pd.DataFrame(
            {
                "date": ["d1"] * 5,
                "window_ts": timestamps,
                "room": ["a", "a", "a", "b", "b"],
            }
        )
        encoder = LabelEncoder().fit(["a", "b"])
        initial, transition = fit_markov_model(train, encoder)
        decoded = viterbi_decode(smoothed, timestamps, 1, initial, transition)
        self.assertEqual(decoded.shape, (5,))
        self.assertTrue(set(decoded).issubset({0, 1}))

    def test_boundary_mask_marks_transition_neighborhood(self) -> None:
        timestamps = pd.date_range("2023-04-10", periods=7, freq="s", tz="UTC")
        mask = boundary_mask(["a", "a", "a", "b", "b", "b", "b"], timestamps, 1.0)
        self.assertEqual(mask.tolist(), [False, False, True, True, True, False, False])


if __name__ == "__main__":
    unittest.main()

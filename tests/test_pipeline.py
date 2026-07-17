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

from baseline_ml.pipeline import (  # noqa: E402
    CLASS_AUDIT_COLUMNS,
    PREDICTION_COLUMNS,
    RESULT_COLUMNS,
    SUMMARY_COLUMNS,
    main,
    stage_extract,
    stage_process,
    train_lodo,
)


def _config(aligned_path: Path, raw_path: Path, label_path: Path) -> dict[str, object]:
    return {
        "project_name": "synthetic pipeline test",
        "raw": {
            "ble_path": str(raw_path),
            "label_path": str(label_path),
        },
        "processing": {
            "build_aligned_if_missing": True,
            "label_start_col": "started_at",
            "label_end_col": "finished_at",
            "label_max_gap_minutes": 30,
            "label_metadata_cols": ["room", "floor"],
        },
        "processed": {
            "aligned_cache_path": str(aligned_path),
            "date_col": "date",
            "time_col": "timestamp",
            "user_col": "user_id",
            "label_col": "room",
        },
        "features": {
            "window_seconds": 30,
            "rssi_prefix": "RSSI_",
            "missing_rssi": -110.0,
            "replace_zero_rssi": True,
            "aggregations": ["mean", "std", "min", "max", "last"],
            "include_time_features": False,
        },
        "training": {
            "evaluation": "leave_one_day_out",
            "closed_set": True,
            "class_protocol": "dasel_intersection",
            "model": "rf",
            "random_state": 7,
            "xgb_params": {},
            "rf_params": {
                "n_estimators": 20,
                "max_depth": 4,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
            },
        },
    }


def _synthetic_aligned() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day_index, day in enumerate(pd.date_range("2025-01-01", periods=3, freq="D")):
        for label_index, room in enumerate(("room_a", "room_b")):
            for window_index in range(4):
                window = day + pd.Timedelta(minutes=window_index * 2 + label_index)
                # Append the later observation first to ensure extraction must sort.
                for seconds, sample_delta in ((20, 2), (5, -2)):
                    rows.append(
                        {
                            "user_id": 1,
                            "timestamp": (window + pd.Timedelta(seconds=seconds)).isoformat(),
                            "room": room,
                            "RSSI_1": -42 - 24 * label_index + window_index + sample_delta,
                            "RSSI_2": -82 + 20 * label_index - window_index - sample_delta,
                            "RSSI_3": -110 if window_index % 2 else -70 - day_index,
                        }
                    )
    return pd.DataFrame(rows)


class PipelineTests(unittest.TestCase):
    def test_all_stage_writes_valid_lodo_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned_path = root / "aligned.csv"
            raw_path = root / "raw.csv"
            label_path = root / "labels.csv"
            config_path = root / "config.json"
            output_dir = root / "run"

            _synthetic_aligned().to_csv(aligned_path, index=False)
            pd.DataFrame({"user_id": [1], "timestamp": ["2025-01-01"]}).to_csv(
                raw_path, index=False
            )
            pd.DataFrame({"user_id": [1], "room": ["room_a"]}).to_csv(
                label_path, index=False
            )
            config_path.write_text(
                json.dumps(_config(aligned_path, raw_path, label_path)), encoding="utf-8"
            )

            self.assertEqual(
                main(
                    [
                        "--config",
                        str(config_path),
                        "--stage",
                        "all",
                        "--output-dir",
                        str(output_dir),
                    ]
                ),
                0,
            )

            training_dir = output_dir / "training"
            results = pd.read_csv(training_dir / "lodo_results.csv")
            class_audit = pd.read_csv(training_dir / "lodo_class_audit.csv")
            predictions = pd.read_csv(training_dir / "lodo_predictions.csv")
            summary = pd.read_csv(training_dir / "lodo_summary.csv")

            self.assertEqual(list(results.columns), list(RESULT_COLUMNS))
            self.assertEqual(list(class_audit.columns), list(CLASS_AUDIT_COLUMNS))
            self.assertEqual(list(predictions.columns), list(PREDICTION_COLUMNS))
            self.assertEqual(list(summary.columns), list(SUMMARY_COLUMNS))
            self.assertEqual(len(results), 3)
            self.assertEqual(len(summary), 1)
            self.assertEqual(int(summary.loc[0, "total_test"]), int(results["n_test"].sum()))
            self.assertAlmostEqual(
                float(summary.loc[0, "mean_macro_f1"]), float(results["macro_f1"].mean())
            )

    def test_last_aggregation_is_chronological(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned_path = root / "aligned.csv"
            raw_path = root / "raw.csv"
            label_path = root / "labels.csv"
            config = _config(aligned_path, raw_path, label_path)
            config["features"]["aggregations"] = ["last"]  # type: ignore[index]

            pd.DataFrame(
                [
                    {
                        "user_id": 1,
                        "timestamp": "2025-01-01T00:00:20",
                        "room": "room_a",
                        "RSSI_1": -40,
                    },
                    {
                        "user_id": 1,
                        "timestamp": "2025-01-01T00:00:05",
                        "room": "room_a",
                        "RSSI_1": -80,
                    },
                ]
            ).to_csv(aligned_path, index=False)

            features = stage_extract(config, root / "features")
            self.assertEqual(len(features), 1)
            self.assertEqual(float(features.loc[0, "RSSI_1_last"]), -40.0)

    def test_room_transition_does_not_split_a_time_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned_path = root / "aligned.csv"
            config = _config(aligned_path, root / "raw.csv", root / "labels.csv")

            pd.DataFrame(
                [
                    {"user_id": 1, "timestamp": "2025-01-01T00:00:05", "room": "a", "RSSI_1": -70},
                    {"user_id": 1, "timestamp": "2025-01-01T00:00:10", "room": "b", "RSSI_1": -60},
                    {"user_id": 1, "timestamp": "2025-01-01T00:00:15", "room": "a", "RSSI_1": -50},
                    {"user_id": 1, "timestamp": "2025-01-01T00:00:20", "room": "b", "RSSI_1": -40},
                ]
            ).to_csv(aligned_path, index=False)

            features = stage_extract(config, root / "features")
            self.assertEqual(len(features), 1)
            self.assertEqual(features.loc[0, "room"], "b")

    def test_process_builds_missing_cache_from_raw_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned_path = root / "aligned.csv"
            raw_path = root / "raw.csv"
            label_path = root / "labels.csv"
            config = _config(aligned_path, raw_path, label_path)

            pd.DataFrame(
                [
                    {"user_id": 90, "timestamp": "2025-01-01T00:00:40", "RSSI_1": -60},
                    {"user_id": 90, "timestamp": "2025-01-01T00:01:20", "RSSI_1": -50},
                    {"user_id": 90, "timestamp": "2025-01-01T02:00:00", "RSSI_1": -40},
                ]
            ).to_csv(raw_path, index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": 97,
                        "started_at": "2025-01-01T00:00:00",
                        "finished_at": "2025-01-01T00:00:00",
                        "room": "room_a",
                        "floor": "1",
                    },
                    {
                        "user_id": 97,
                        "started_at": "2025-01-01T00:01:00",
                        "finished_at": "2025-01-01T00:01:30",
                        "room": "room_b",
                        "floor": "1",
                    },
                ]
            ).to_csv(label_path, index=False)

            processed = stage_process(config, root / "process")
            self.assertTrue(aligned_path.is_file())
            self.assertEqual(processed["room"].tolist(), ["room_a", "room_b"])

    def test_limited_raw_fallback_does_not_persist_private_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned_path = root / "aligned.csv"
            raw_path = root / "raw.csv"
            label_path = root / "labels.csv"
            config = _config(aligned_path, raw_path, label_path)

            pd.DataFrame(
                [
                    {"user_id": 1, "timestamp": "2025-01-01T00:00:05", "RSSI_1": -60},
                    {"user_id": 1, "timestamp": "2025-01-01T00:00:10", "RSSI_1": -55},
                ]
            ).to_csv(raw_path, index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": 1,
                        "started_at": "2025-01-01T00:00:00",
                        "finished_at": "2025-01-01T00:01:00",
                        "room": "room_a",
                    }
                ]
            ).to_csv(label_path, index=False)

            processed = stage_process(config, root / "process", nrows=1)
            self.assertEqual(len(processed), 1)
            self.assertFalse(aligned_path.exists())

    def test_multi_user_alignment_rejects_mismatched_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned_path = root / "aligned.csv"
            raw_path = root / "raw.csv"
            label_path = root / "labels.csv"
            config = _config(aligned_path, raw_path, label_path)

            pd.DataFrame(
                [
                    {"user_id": 1, "timestamp": "2025-01-01T00:00:05", "RSSI_1": -60},
                    {"user_id": 2, "timestamp": "2025-01-01T00:00:05", "RSSI_1": -50},
                ]
            ).to_csv(raw_path, index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": 1,
                        "started_at": "2025-01-01T00:00:00",
                        "finished_at": "2025-01-01T00:01:00",
                        "room": "room_a",
                    },
                    {
                        "user_id": 3,
                        "started_at": "2025-01-01T00:00:00",
                        "finished_at": "2025-01-01T00:01:00",
                        "room": "room_b",
                    },
                ]
            ).to_csv(label_path, index=False)

            with self.assertRaisesRegex(ValueError, "user IDs do not match"):
                stage_process(config, root / "process")

    def test_open_set_unseen_labels_fail_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root / "aligned.csv", root / "raw.csv", root / "labels.csv")
            config["training"]["closed_set"] = False  # type: ignore[index]
            config["training"]["class_protocol"] = "train_label_space"  # type: ignore[index]
            features = pd.DataFrame(
                {
                    "date": ["2025-01-01", "2025-01-02"],
                    "user_id": [1, 1],
                    "room": ["room_a", "room_b"],
                    "window_start": ["2025-01-01", "2025-01-02"],
                    "RSSI_1_mean": [-60.0, -40.0],
                }
            )

            with self.assertRaisesRegex(ValueError, "absent from training"):
                train_lodo(config, features, root / "training")

    def test_dasel_intersection_filters_both_sides_before_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root / "aligned.csv", root / "raw.csv", root / "labels.csv")
            rows = []
            class_sets = {
                "2025-01-01": ("shared_a", "shared_b", "day1_only"),
                "2025-01-02": ("shared_a", "shared_b", "day2_only"),
                "2025-01-03": ("shared_a", "shared_b"),
            }
            for day, rooms in class_sets.items():
                for room_index, room in enumerate(rooms):
                    for sample in range(3):
                        rows.append(
                            {
                                "date": day,
                                "user_id": 1,
                                "room": room,
                                "window_start": f"{day} 00:00:0{sample}",
                                "RSSI_1_mean": float(room_index * 10 + sample),
                                "RSSI_2_mean": float(room_index * -5 + sample),
                            }
                        )

            results = train_lodo(config, pd.DataFrame(rows), root / "training")
            day1 = results.loc[results["test_day"].eq("2025-01-01")].iloc[0]
            self.assertEqual(day1["class_protocol"], "dasel_intersection")
            self.assertEqual(int(day1["n_train_classes_raw"]), 3)
            self.assertEqual(int(day1["n_test_classes_raw"]), 3)
            self.assertEqual(int(day1["n_common_classes"]), 2)
            self.assertEqual(int(day1["n_classes"]), 2)
            self.assertEqual(int(day1["n_train_only_classes"]), 1)
            self.assertEqual(int(day1["n_test_only_classes"]), 1)
            self.assertEqual(int(day1["n_train_dropped"]), 3)
            self.assertEqual(int(day1["n_test_dropped"]), 3)

    def test_single_day_smoke_output_keeps_csv_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned_path = root / "aligned.csv"
            raw_path = root / "raw.csv"
            label_path = root / "labels.csv"
            config_path = root / "config.json"
            output_dir = root / "run"

            _synthetic_aligned().to_csv(aligned_path, index=False)
            pd.DataFrame({"x": [1]}).to_csv(raw_path, index=False)
            pd.DataFrame({"x": [1]}).to_csv(label_path, index=False)
            config_path.write_text(
                json.dumps(_config(aligned_path, raw_path, label_path)), encoding="utf-8"
            )

            main(
                [
                    "--config",
                    str(config_path),
                    "--stage",
                    "all",
                    "--output-dir",
                    str(output_dir),
                    "--limit-rows",
                    "8",
                ]
            )

            training_dir = output_dir / "training"
            self.assertEqual(
                list(pd.read_csv(training_dir / "lodo_results.csv").columns),
                list(RESULT_COLUMNS),
            )
            self.assertEqual(
                list(pd.read_csv(training_dir / "lodo_summary.csv").columns),
                list(SUMMARY_COLUMNS),
            )


if __name__ == "__main__":
    unittest.main()

"""DASEL-compatible long-form BLE feature extraction for 1s/3s research runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_WINDOWS = (1, 3)
BEACONS = tuple(range(1, 26))
REQUIRED_COLUMNS = ("timestamp", "mac_address", "RSSI", "room")


def _mad(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    center = np.median(array)
    return float(np.median(np.abs(array - center)))


def _room_per_window(data: pd.DataFrame) -> pd.Series:
    room_counts = (
        data.groupby(["window_ts", "room"], sort=False)
        .agg(n=("RSSI", "size"), first_order=("_order", "min"))
        .reset_index()
    )
    return (
        room_counts.sort_values(
            ["window_ts", "n", "first_order"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("window_ts")
        .set_index("window_ts")["room"]
    )


def _top_features(counts: np.ndarray, means: np.ndarray) -> dict[str, np.ndarray]:
    n_rows = counts.shape[0]
    ids = np.arange(1, 26)
    result: dict[str, np.ndarray] = {}

    count_order = np.argsort(-counts, axis=1, kind="stable")
    active_count = (counts > 0).sum(axis=1)
    for rank in range(3):
        column_index = count_order[:, rank]
        valid = active_count > rank
        result[f"top{rank + 1}_beacon_count_id"] = np.where(
            valid, ids[column_index], 0
        )
        result[f"top{rank + 1}_beacon_count"] = np.where(
            valid, counts[np.arange(n_rows), column_index], 0.0
        )

    rssi_sort_values = np.where(np.isfinite(means), means, -np.inf)
    rssi_order = np.argsort(-rssi_sort_values, axis=1, kind="stable")
    active_rssi = np.isfinite(means).sum(axis=1)
    for rank in range(3):
        column_index = rssi_order[:, rank]
        valid = active_rssi > rank
        result[f"top{rank + 1}_beacon_rssi_id"] = np.where(
            valid, ids[column_index], 0
        )
        result[f"top{rank + 1}_beacon_rssi"] = np.where(
            valid, means[np.arange(n_rows), column_index], np.nan
        )
    return result


def extract_one_window(data: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """Create the historical 262-column tumbling-window feature table."""
    if seconds <= 0:
        raise ValueError("Window length must be positive")
    work = data.copy()
    work["window_ts"] = work["timestamp"].dt.floor(f"{seconds}s")
    grouped = work.groupby("window_ts", sort=True, observed=True)
    index = grouped.size().index

    global_features = grouped["RSSI"].agg(
        total_detections="size",
        rssi_mean_all="mean",
        rssi_std_all="std",
        rssi_min_all="min",
        rssi_max_all="max",
        rssi_median_all="median",
    )
    global_features["active_beacon_count"] = grouped["mac_address"].nunique()
    global_features["rssi_range_all"] = (
        global_features["rssi_max_all"] - global_features["rssi_min_all"]
    )
    quantiles = grouped["RSSI"].quantile([0.25, 0.75]).unstack()
    global_features["rssi_iqr_all"] = quantiles[0.75] - quantiles[0.25]
    global_features["rssi_mad_all"] = grouped["RSSI"].agg(_mad)
    # The historical source exposes no calibrated transmit-power channel.
    # Keep these compatibility fields constant so downstream selectors can
    # explicitly remove them using training data only.
    global_features["power_mean_all"] = -2_147_483_648.0
    global_features["power_std_all"] = 0.0

    beacon = grouped.apply(
        lambda frame: frame.groupby("mac_address", observed=True)["RSSI"].agg(
            ["count", "mean", "std", "min", "max", "median"]
        ),
        include_groups=False,
    )
    beacon.index.names = ["window_ts", "mac_address"]

    matrices: dict[str, pd.DataFrame] = {}
    for metric in ("count", "mean", "std", "min", "max", "median"):
        matrices[metric] = (
            beacon[metric].unstack("mac_address").reindex(index=index, columns=BEACONS)
        )

    count_matrix = matrices["count"].fillna(0.0)
    freq_matrix = count_matrix.div(global_features["total_detections"], axis=0).fillna(0.0)
    present_matrix = (count_matrix > 0).astype(int)
    range_matrix = matrices["max"] - matrices["min"]

    result = pd.DataFrame(index=index)
    result["date"] = result.index.strftime("%Y-%m-%d")
    result["room"] = _room_per_window(work).reindex(index).astype(str)
    result["hour"] = result.index.hour
    result["minute"] = result.index.minute
    result["second"] = result.index.second
    result["dayofweek"] = result.index.dayofweek

    global_order = (
        "total_detections",
        "active_beacon_count",
        "rssi_mean_all",
        "rssi_std_all",
        "rssi_min_all",
        "rssi_max_all",
        "rssi_median_all",
        "power_mean_all",
        "power_std_all",
        "rssi_range_all",
        "rssi_iqr_all",
        "rssi_mad_all",
    )
    for column in global_order:
        result[column] = global_features[column]

    matrix_specs = (
        ("count", count_matrix),
        ("freq", freq_matrix),
        ("present", present_matrix),
        ("rssi_mean", matrices["mean"]),
        ("rssi_std", matrices["std"]),
        ("rssi_min", matrices["min"]),
        ("rssi_max", matrices["max"]),
        ("rssi_median", matrices["median"]),
        ("rssi_range", range_matrix),
    )
    for prefix, matrix in matrix_specs:
        for beacon_id in BEACONS:
            result[f"{prefix}_B{beacon_id:02d}"] = matrix[beacon_id]

    top = _top_features(
        count_matrix.to_numpy(dtype=float),
        matrices["mean"].to_numpy(dtype=float),
    )
    for column, values in top.items():
        result[column] = values

    frequency_values = freq_matrix.to_numpy(dtype=float)
    entropy = -np.sum(
        np.where(
            frequency_values > 0,
            frequency_values * np.log(np.maximum(frequency_values, 1e-12)),
            0.0,
        ),
        axis=1,
    )
    active = global_features["active_beacon_count"].to_numpy(dtype=float)
    entropy_norm = np.divide(
        entropy,
        np.log(np.maximum(active, 2.0)),
        out=np.zeros_like(entropy),
        where=active > 1,
    )
    sorted_frequency = np.sort(frequency_values, axis=1)
    result["freq_entropy"] = entropy
    result["freq_entropy_norm"] = entropy_norm
    result["dominant_beacon_freq"] = sorted_frequency[:, -1]
    result["dominant_beacon_margin"] = sorted_frequency[:, -1] - sorted_frequency[:, -2]

    seconds_of_day = result.index.hour * 3600 + result.index.minute * 60 + result.index.second
    phase = 2.0 * np.pi * seconds_of_day / 86_400.0
    result["time_sin"] = np.sin(phase)
    result["time_cos"] = np.cos(phase)

    result.index.name = "window_ts"
    result = result.reset_index()
    if result.shape[1] != 262:
        raise RuntimeError(
            f"Window {seconds}s produced {result.shape[1]} columns, expected 262"
        )
    return result


def load_preprocessed(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(
        path,
        dtype={"mac_address": "int16", "RSSI": "int16", "room": "string"},
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise KeyError(f"DASEL preprocessed input is missing columns: {missing}")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="raise")
    raw = raw.sort_values("timestamp", kind="stable").reset_index(drop=True)
    raw["_order"] = np.arange(len(raw), dtype=np.int64)
    before = len(raw)
    data = raw.drop_duplicates(REQUIRED_COLUMNS, keep="first").copy()
    invalid_beacons = sorted(set(data["mac_address"].astype(int)) - set(BEACONS))
    if invalid_beacons:
        raise ValueError(f"Beacon IDs outside the DASEL 1..25 range: {invalid_beacons}")
    return data, {
        "source_file": path.name,
        "source_path_published": False,
        "source_rows": int(before),
        "deduplicated_detection_rows": int(len(data)),
        "deduplication_key": list(REQUIRED_COLUMNS),
    }


def extract_windows(
    input_path: Path,
    output_dir: Path,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data, manifest = load_preprocessed(input_path)
    window_manifest: dict[str, object] = {}
    for seconds in windows:
        features = extract_one_window(data, int(seconds))
        output = output_dir / f"rf_window_features_{seconds}s.csv"
        features.to_csv(output, index=False)
        window_manifest[str(seconds)] = {
            "rows": int(len(features)),
            "columns": int(features.shape[1]),
            "detection_sum": int(features["total_detections"].sum()),
            "days": {
                str(key): int(value)
                for key, value in features["date"].value_counts().sort_index().items()
            },
            "rooms": int(features["room"].nunique()),
            "output_file": output.name,
        }
    manifest["windows"] = window_manifest
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract the historical 262-column DASEL 1s/3s feature matrices."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    args = parser.parse_args(argv)
    manifest = extract_windows(args.input, args.output_dir, args.windows)
    for window, item in manifest["windows"].items():
        print(f"{window}s: {item['rows']} rows x {item['columns']} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strong, DASEL-class-matched 1s/3s ML training and explanation pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import platform
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import DMatrix, XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
META_COLUMNS = {
    "room",
    "date",
    "window_ts",
    "timestamp",
    "window_start",
    "window_end",
    "start_time",
    "end_time",
    "day_group",
    "fold",
    "test_day",
    "y_true",
    "y_pred",
    "_source_file",
    "_visit_id",
    "_visit_len",
    "_visit_pos",
    "_visit_stratum",
    "_is_augmented",
}
TIME_FEATURES = {"hour", "minute", "second", "dayofweek", "time_sin", "time_cos"}
POWER_FEATURES = {
    "power_mean_all",
    "power_std_all",
    "power_min_all",
    "power_max_all",
    "power_count_all",
}
DIAGNOSTIC_MARKERS = (
    "label_reliability",
    "label_confidence",
    "label_duration",
    "boundary_distance",
    "boundary_ratio",
    "is_boundary",
    "annotation",
    "overlap",
    "groundtruth",
    "ground_truth",
    "prev_room_gt",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_strong_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol") != "dasel_intersection":
        raise ValueError("Strong public runs require protocol=dasel_intersection")
    return config


def load_feature_frame(features_dir: Path, window: int) -> pd.DataFrame:
    path = features_dir / f"rf_window_features_{window}s.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing private feature matrix: {path}")
    frame = pd.read_csv(path)
    required = {"date", "window_ts", "room"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path.name} is missing columns: {missing}")
    frame["date"] = frame["date"].astype(str)
    frame["room"] = frame["room"].astype(str)
    frame["window_ts"] = pd.to_datetime(frame["window_ts"], errors="raise", utc=True)
    return frame.sort_values(["date", "window_ts"], kind="stable").reset_index(drop=True)


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    sums = values.sum(axis=1, keepdims=True)
    bad = sums[:, 0] <= eps
    if np.any(bad):
        values[bad] = 1.0 / values.shape[1]
        sums = values.sum(axis=1, keepdims=True)
    return values / np.maximum(sums, eps)


def split_segments(
    timestamps: Iterable[Any],
    window_seconds: int,
    max_gap_multiplier: float = 2.5,
) -> list[np.ndarray]:
    series = pd.Series(pd.to_datetime(list(timestamps), utc=True)).reset_index(drop=True)
    if series.empty:
        return []
    max_gap = max(window_seconds * max_gap_multiplier, window_seconds + 1.0)
    groups = (series.diff().dt.total_seconds().fillna(0.0) > max_gap).cumsum()
    return [np.asarray(index, dtype=int) for index in groups.groupby(groups).groups.values()]


def strict_fold(
    frame: pd.DataFrame,
    test_day: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_raw = frame.loc[~frame["date"].eq(test_day)].copy()
    test_raw = frame.loc[frame["date"].eq(test_day)].copy()
    train_classes = set(train_raw["room"].astype(str))
    test_classes = set(test_raw["room"].astype(str))
    common = sorted(train_classes & test_classes)
    train = train_raw.loc[train_raw["room"].isin(common)].copy()
    test = test_raw.loc[test_raw["room"].isin(common)].copy()
    if set(train["room"].astype(str)) != set(common) or set(test["room"].astype(str)) != set(common):
        raise AssertionError(f"Fold {test_day} failed strict shared-class filtering")
    audit = {
        "test_day": test_day,
        "class_protocol": "dasel_intersection",
        "n_train_raw": int(len(train_raw)),
        "n_test_raw": int(len(test_raw)),
        "n_train_classes_raw": int(len(train_classes)),
        "n_test_classes_raw": int(len(test_classes)),
        "n_common_classes": int(len(common)),
        "n_train_dropped": int(len(train_raw) - len(train)),
        "n_test_dropped": int(len(test_raw) - len(test)),
        "n_train_only_classes": int(len(train_classes - test_classes)),
        "n_test_only_classes": int(len(test_classes - train_classes)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }
    return (
        train.sort_values(["date", "window_ts"], kind="stable").reset_index(drop=True),
        test.sort_values(["date", "window_ts"], kind="stable").reset_index(drop=True),
        audit,
    )


def add_visit_metadata(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    out = frame.sort_values(["date", "window_ts"], kind="stable").reset_index(drop=True).copy()
    max_gap = max(window * 2.5, window + 1.0)
    new_visit = (
        out["date"].ne(out["date"].shift())
        | out["room"].ne(out["room"].shift())
        | out["window_ts"].diff().dt.total_seconds().fillna(0.0).gt(max_gap)
    )
    out["_visit_id"] = new_visit.cumsum().astype(int)
    out["_visit_len"] = out.groupby("_visit_id")["_visit_id"].transform("size").astype(int)
    out["_visit_pos"] = out.groupby("_visit_id").cumcount().astype(int)
    fraction = (out["_visit_pos"] + 0.5) / out["_visit_len"]
    out["_visit_stratum"] = np.select(
        [fraction.lt(0.2), fraction.ge(0.8)],
        ["entry", "exit"],
        default="middle",
    )
    return out


def add_relative_rssi_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    mean_cols = [f"rssi_mean_B{i:02d}" for i in range(1, 24) if f"rssi_mean_B{i:02d}" in out]
    if not mean_cols:
        return out
    matrix = out[mean_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    has_signal = np.isfinite(matrix).any(axis=1)
    strongest = np.full(len(out), np.nan, dtype=float)
    strongest[has_signal] = np.nanmax(matrix[has_signal], axis=1)
    relative = {
        f"rssi_rel_B{i + 1:02d}": matrix[:, i] - strongest
        for i in range(len(mean_cols))
    }
    return pd.concat([out, pd.DataFrame(relative, index=out.index)], axis=1)


def add_fingerprint_features(frame: pd.DataFrame, fingerprints: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    freq_cols = [f"freq_B{i:02d}" for i in range(1, 24) if f"freq_B{i:02d}" in out]
    if fingerprints.empty or not freq_cols:
        return out
    p = normalize_rows(out[freq_cols].to_numpy(dtype=float))
    q = normalize_rows(fingerprints[freq_cols].to_numpy(dtype=float))
    cosine = (p @ q.T) / np.maximum(
        np.linalg.norm(p, axis=1, keepdims=True) * np.linalg.norm(q, axis=1, keepdims=True).T,
        1e-12,
    )
    js = np.empty((len(p), len(q)), dtype=float)
    l1 = np.empty_like(js)
    for index in range(len(q)):
        room = q[index][None, :]
        midpoint = 0.5 * (p + room)
        kl_p = np.sum(
            np.where(p > 0, p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(midpoint, 1e-12))), 0.0),
            axis=1,
        )
        kl_q = np.sum(
            np.where(room > 0, room * (np.log(np.maximum(room, 1e-12)) - np.log(np.maximum(midpoint, 1e-12))), 0.0),
            axis=1,
        )
        js[:, index] = np.sqrt(np.maximum(0.5 * (kl_p + kl_q), 0.0))
        l1[:, index] = np.abs(p - room).sum(axis=1)
    cosine_sorted = np.sort(cosine, axis=1)
    js_sorted = np.sort(js, axis=1)
    l1_sorted = np.sort(l1, axis=1)
    best_cos = cosine.argmax(axis=1)
    best_js = js.argmin(axis=1)
    features = pd.DataFrame(
        {
            "fp_cos_best": cosine_sorted[:, -1],
            "fp_cos_second": cosine_sorted[:, -2] if cosine.shape[1] > 1 else cosine_sorted[:, -1],
            "fp_js_best": js_sorted[:, 0],
            "fp_js_second": js_sorted[:, 1] if js.shape[1] > 1 else js_sorted[:, 0],
            "fp_l1_best": l1_sorted[:, 0],
            "fp_l1_second": l1_sorted[:, 1] if l1.shape[1] > 1 else l1_sorted[:, 0],
            "fp_metric_agreement": (best_cos == best_js).astype(float),
        },
        index=out.index,
    )
    features["fp_cos_margin"] = features["fp_cos_best"] - features["fp_cos_second"]
    features["fp_js_margin"] = features["fp_js_second"] - features["fp_js_best"]
    features["fp_l1_margin"] = features["fp_l1_second"] - features["fp_l1_best"]
    features["fp_match_confidence"] = features["fp_cos_margin"] * features["fp_js_margin"]
    return pd.concat([out, features], axis=1)


def engineer_fold(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_relative = add_relative_rssi_features(train)
    test_relative = add_relative_rssi_features(test)
    freq_cols = [f"freq_B{i:02d}" for i in range(1, 24) if f"freq_B{i:02d}" in train_relative]
    grouped = train_relative.groupby("room", observed=True)[freq_cols].mean()
    fingerprints = pd.DataFrame(
        normalize_rows(grouped.to_numpy(dtype=float)),
        index=grouped.index.astype(str),
        columns=freq_cols,
    )
    return (
        add_fingerprint_features(train_relative, fingerprints),
        add_fingerprint_features(test_relative, fingerprints),
        fingerprints,
    )


def candidate_columns(frame: pd.DataFrame) -> list[str]:
    result: list[str] = []
    for column in frame.columns:
        lower = column.lower()
        if column in META_COLUMNS or column in TIME_FEATURES or column in POWER_FEATURES:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        if lower.endswith("_id") or "_id_" in lower:
            continue
        if re.search(r"(?:^|_)B(?:24|25)(?:_|$)", column):
            continue
        if any(marker in lower for marker in DIAGNOSTIC_MARKERS):
            continue
        if re.fullmatch(r"rssi_std_B\d{2}", column):
            continue
        result.append(column)
    return result


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float))
    span = values.max() - values.min()
    return np.zeros_like(values) if span <= 1e-12 else (values - values.min()) / span


def _discrete_feature(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("present_b") or bool(
        re.fullmatch(r"top[123]_beacon_count", lower)
    )


def fit_selector(train: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    candidates = candidate_columns(train)
    raw = train[candidates].replace([np.inf, -np.inf], np.nan)
    usable = [
        column
        for column in candidates
        if not raw[column].isna().all()
        and raw[column].isna().mean() <= 0.98
        and raw[column].nunique(dropna=True) > 1
    ]
    medians = raw[usable].median(numeric_only=True)
    matrix = raw[usable].fillna(medians).fillna(0.0).astype(float)
    usable = [
        column
        for column in usable
        if matrix[column].value_counts(dropna=False).iloc[0] / max(len(matrix), 1) < 0.9995
    ]
    matrix = matrix[usable]
    keep: list[str] = []
    signatures: dict[int, str] = {}
    for column in matrix.columns:
        signature = int(pd.util.hash_pandas_object(matrix[column], index=False).sum())
        if signature not in signatures or not matrix[column].equals(matrix[signatures[signature]]):
            signatures[signature] = column
            keep.append(column)
    matrix = matrix[keep]

    max_rows = int(config["max_rank_rows"])
    if len(matrix) > max_rows:
        selected_indices: list[int] = []
        per_day = max(1, max_rows // train["date"].nunique())
        for offset, day in enumerate(sorted(train["date"].unique())):
            indices = train.index[train["date"].eq(day)]
            selected_indices.extend(
                pd.Series(indices).sample(min(per_day, len(indices)), random_state=42 + offset).tolist()
            )
        rank_index = selected_indices[:max_rows]
    else:
        rank_index = train.index.tolist()
    rank_x = matrix.loc[rank_index]
    room_y = LabelEncoder().fit_transform(train.loc[rank_index, "room"].astype(str))
    day_y = LabelEncoder().fit_transform(train.loc[rank_index, "date"].astype(str))
    discrete = np.asarray([_discrete_feature(column) for column in matrix.columns], dtype=bool)
    mi_room = mutual_info_classif(rank_x, room_y, discrete_features=discrete, random_state=42)
    mi_day = (
        mutual_info_classif(rank_x, day_y, discrete_features=discrete, random_state=42)
        if len(np.unique(day_y)) > 1
        else np.zeros(len(matrix.columns))
    )
    trees = ExtraTreesClassifier(
        n_estimators=int(config["extra_trees"]),
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=2,
    )
    trees.fit(rank_x, room_y)
    ranking = pd.DataFrame(
        {
            "feature": matrix.columns,
            "mi_room": mi_room,
            "mi_day": mi_day,
            "extratrees_importance": trees.feature_importances_,
            "missing_ratio": [raw[column].isna().mean() for column in matrix.columns],
        }
    )
    ranking["strong_day_dominant"] = (
        (ranking["mi_day"] >= float(config["day_mi_threshold"]))
        & (ranking["mi_day"] > float(config["day_dominance_ratio"]) * ranking["mi_room"])
    )
    ranking["priority_score"] = (
        0.45 * _minmax(mi_room)
        + 0.45 * _minmax(trees.feature_importances_)
        - 0.25 * _minmax(mi_day)
        - 0.10 * ranking["missing_ratio"]
    )
    eligible = ranking.loc[~ranking["strong_day_dominant"]].sort_values(
        ["priority_score", "mi_room", "extratrees_importance"],
        ascending=False,
    )
    correlation = matrix[eligible["feature"]].corr(method="spearman").abs()
    selected: list[str] = []
    dropped: list[dict[str, Any]] = []
    threshold = float(config["correlation_threshold"])
    for feature in eligible["feature"]:
        conflict = next(
            (
                kept
                for kept in selected
                if np.isfinite(correlation.at[feature, kept]) and correlation.at[feature, kept] >= threshold
            ),
            None,
        )
        if conflict is None:
            selected.append(feature)
        else:
            dropped.append(
                {
                    "dropped_feature": feature,
                    "kept_feature": conflict,
                    "abs_spearman_corr": float(correlation.at[feature, conflict]),
                }
            )
    selected = selected[: int(config["top_k"])]
    ranking["selected_top_k"] = ranking["feature"].isin(selected)
    return {
        "features": selected,
        "ranking": ranking.sort_values("priority_score", ascending=False),
        "correlation_dropped": pd.DataFrame(dropped),
        "n_candidates": len(candidates),
        "n_usable": len(matrix.columns),
    }


def feature_set(variant: str, selected: list[str], frame: pd.DataFrame) -> list[str]:
    if variant == "M2":
        return [feature for feature in selected if feature in frame][:100]
    if variant == "M5":
        frequency = [f"freq_B{i:02d}" for i in range(1, 24) if f"freq_B{i:02d}" in frame]
        non_frequency = [
            feature
            for feature in selected
            if feature in frame and not feature.startswith(("freq_B", "present_B"))
        ]
        return list(dict.fromkeys(frequency + non_frequency[:77]))[:100]
    raise ValueError(f"Unsupported strong variant: {variant}")


def _selected_rssi_columns(features: list[str]) -> list[str]:
    prefixes = (
        "rssi_mean_B",
        "rssi_median_B",
        "rssi_min_B",
        "rssi_max_B",
        "rssi_rel_B",
        "rssi_mean_all",
        "rssi_median_all",
        "rssi_min_all",
        "rssi_max_all",
    )
    return [feature for feature in features if feature.startswith(prefixes)]


def _recompute_relative(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    means = [f"rssi_mean_B{i:02d}" for i in range(1, 24) if f"rssi_mean_B{i:02d}" in out]
    if not means:
        return out
    matrix = out[means].to_numpy(dtype=float)
    valid = np.isfinite(matrix).any(axis=1)
    strongest = np.full(len(out), np.nan)
    strongest[valid] = np.nanmax(matrix[valid], axis=1)
    for index, source in enumerate(means):
        target = f"rssi_rel_{source.rsplit('_', 1)[-1]}"
        if target in out:
            out[target] = matrix[:, index] - strongest
    return out


def augment_training(
    train: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    counts = train["room"].value_counts()
    target = int(np.quantile(counts.to_numpy(), float(config["target_quantile"])))
    jitter = _selected_rssi_columns(features)
    scale = (
        train[jitter].std(numeric_only=True).replace(0, np.nan).fillna(1.0)
        if jitter
        else pd.Series(dtype=float)
    )
    parts = [train.assign(_is_augmented=0)]
    audit: list[dict[str, Any]] = []
    for offset, (room, original_value) in enumerate(counts.items()):
        original = int(original_value)
        n_new = min(
            max(target - original, 0),
            int(config["max_new_per_class"]),
            max(int(float(config["max_multiplier"]) * original) - original, 0),
        )
        audit.append({"room": room, "original_n": original, "augmented_n": n_new, "target_n": target})
        if n_new <= 0:
            continue
        sampled = train.loc[train["room"].eq(room)].sample(
            n=n_new,
            replace=True,
            random_state=seed + offset,
        ).copy()
        if jitter:
            values = sampled[jitter].to_numpy(dtype=float)
            noise = rng.normal(
                0.0,
                scale.reindex(jitter).to_numpy(dtype=float) * float(config["rssi_noise_fraction"]),
                size=values.shape,
            )
            valid = np.isfinite(values)
            values[valid] += noise[valid]
            sampled[jitter] = values
            sampled = _recompute_relative(sampled)
        sampled["_is_augmented"] = 1
        parts.append(sampled)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audit)


def training_weights(
    variant: str,
    augmented: pd.DataFrame,
    window: int,
    config: dict[str, Any],
) -> np.ndarray:
    class_weight = compute_sample_weight("balanced", augmented["room"].astype(str)).astype(float)
    class_weight /= max(class_weight.mean(), 1e-12)
    if variant == "M2":
        visit_factor = np.minimum(
            1.0,
            float(config["visit_cap_seconds"])
            / (augmented["_visit_len"].to_numpy(dtype=float) * window),
        )
        visit_factor /= max(visit_factor.mean(), 1e-12)
        weights = (
            float(config["m2_class_fraction"]) * class_weight
            + float(config["m2_visit_fraction"]) * visit_factor
        )
    else:
        weights = class_weight
    return weights / max(weights.mean(), 1e-12)


def fit_markov_model(
    train: pd.DataFrame,
    encoder: LabelEncoder,
    pseudocount: float = 0.001,
    window_seconds: int = 1,
    max_gap_multiplier: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an initial distribution and transition matrix using outer-train only."""

    n_classes = len(encoder.classes_)
    initial = np.full(n_classes, pseudocount, dtype=float)
    transition = np.full((n_classes, n_classes), pseudocount, dtype=float)
    ordered = train.sort_values(["date", "window_ts"], kind="stable")
    encoded = encoder.transform(ordered["room"].astype(str))
    for value in encoded:
        initial[value] += 1.0
    max_gap = max(window_seconds * max_gap_multiplier, window_seconds + 1.0)
    for day in sorted(ordered["date"].unique()):
        day_rows = ordered.loc[ordered["date"].eq(day)]
        day_y = encoder.transform(day_rows["room"].astype(str))
        timestamps = day_rows["window_ts"].reset_index(drop=True)
        if len(day_y) < 2:
            continue
        gaps = timestamps.diff().dt.total_seconds().to_numpy()
        for index in range(1, len(day_y)):
            if np.isfinite(gaps[index]) and gaps[index] <= max_gap:
                transition[day_y[index - 1], day_y[index]] += 1.0
    initial /= initial.sum()
    transition /= transition.sum(axis=1, keepdims=True)
    return initial, transition


def causal_probability_smoothing(
    probabilities: np.ndarray,
    timestamps: Sequence[Any],
    window_seconds: int,
    steps: int,
    max_gap_multiplier: float = 2.5,
) -> np.ndarray:
    """Apply the historical causal fixed decoder kernel within contiguous segments."""

    probabilities = normalize_rows(probabilities)
    if steps <= 1 or len(probabilities) <= 1:
        return probabilities
    base = np.linspace(1.0, float(steps), steps)
    base /= base.sum()
    smoothed = probabilities.copy()
    for segment in split_segments(timestamps, window_seconds, max_gap_multiplier):
        segment_probs = probabilities[segment]
        for local_index, global_index in enumerate(segment):
            start = max(0, local_index - steps + 1)
            history = segment_probs[start : local_index + 1]
            weights = base[-len(history) :]
            weights = weights / weights.sum()
            smoothed[global_index] = np.average(history, axis=0, weights=weights)
    return normalize_rows(smoothed)


def viterbi_decode(
    probabilities: np.ndarray,
    timestamps: Sequence[Any],
    window_seconds: int,
    initial: np.ndarray,
    transition: np.ndarray,
    markov_lambda: float = 1.0,
    max_gap_multiplier: float = 2.5,
) -> np.ndarray:
    """Decode each contiguous sequence independently with a train-only Markov prior."""

    probabilities = normalize_rows(probabilities)
    decoded = np.empty(len(probabilities), dtype=int)
    log_initial = np.log(np.maximum(initial, 1e-12))
    log_transition = markov_lambda * np.log(np.maximum(transition, 1e-12))
    for segment in split_segments(timestamps, window_seconds, max_gap_multiplier):
        emissions = np.log(np.maximum(probabilities[segment], 1e-12))
        scores = np.empty_like(emissions)
        paths = np.zeros_like(emissions, dtype=int)
        scores[0] = log_initial + emissions[0]
        for index in range(1, len(segment)):
            candidates = scores[index - 1][:, None] + log_transition
            paths[index] = candidates.argmax(axis=0)
            scores[index] = candidates.max(axis=0) + emissions[index]
        state = int(scores[-1].argmax())
        local = np.empty(len(segment), dtype=int)
        local[-1] = state
        for index in range(len(segment) - 1, 0, -1):
            local[index - 1] = paths[index, local[index]]
        decoded[segment] = local
    return decoded


def boundary_mask(
    labels: Sequence[str],
    timestamps: Sequence[Any],
    radius_seconds: float = 15.0,
) -> np.ndarray:
    """Mark samples within ``radius_seconds`` of a ground-truth room transition."""

    labels_array = np.asarray(labels, dtype=str)
    timestamp_ns = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True)).as_unit("ns").asi8
    mask = np.zeros(len(labels_array), dtype=bool)
    if len(labels_array) < 2:
        return mask
    changes = np.flatnonzero(labels_array[1:] != labels_array[:-1]) + 1
    radius_ns = int(radius_seconds * 1_000_000_000)
    for index in changes:
        boundary_time = timestamp_ns[index]
        left = np.searchsorted(timestamp_ns, boundary_time - radius_ns, side="left")
        right = np.searchsorted(timestamp_ns, boundary_time + radius_ns, side="right")
        mask[left:right] = True
    return mask


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] | None = None,
) -> dict[str, float | int]:
    if len(y_true) == 0:
        return {
            "n_samples": 0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
        }
    metric_labels = list(labels) if labels is not None else sorted(np.unique(y_true).tolist())
    macro_recall = float(
        recall_score(y_true, y_pred, labels=metric_labels, average="macro", zero_division=0)
    )
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": macro_recall,
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=metric_labels, average="macro", zero_division=0)
        ),
        "macro_recall": macro_recall,
        "macro_f1": float(f1_score(y_true, y_pred, labels=metric_labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    encoder: LabelEncoder,
    decoder: str,
) -> pd.DataFrame:
    labels = np.arange(len(encoder.classes_))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "decoder": decoder,
            "room": encoder.classes_,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )


def native_tree_shap(
    models: Sequence[XGBClassifier],
    matrix: pd.DataFrame,
    max_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute exact TreeSHAP contributions with XGBoost's native predictor."""

    if len(matrix) > max_samples:
        sample = matrix.sample(max_samples, random_state=seed)
    else:
        sample = matrix
    seed_importance: list[np.ndarray] = []
    dmatrix = DMatrix(sample, feature_names=list(sample.columns))
    for model in models:
        contributions = model.get_booster().predict(
            dmatrix,
            pred_contribs=True,
            strict_shape=True,
        )
        values = np.asarray(contributions)
        if values.ndim == 3:
            values = values[:, :, :-1]
            importance = np.abs(values).mean(axis=(0, 1))
        elif values.ndim == 2:
            importance = np.abs(values[:, :-1]).mean(axis=0)
        else:
            raise ValueError(f"Unexpected TreeSHAP shape: {values.shape}")
        seed_importance.append(importance)
    stacked = np.vstack(seed_importance)
    result = pd.DataFrame(
        {
            "feature": sample.columns,
            "mean_abs_shap": stacked.mean(axis=0),
            "seed_std_abs_shap": stacked.std(axis=0),
            "n_explained": int(len(sample)),
        }
    )
    return result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def feature_family(feature: str) -> str:
    lower = feature.lower()
    if lower.startswith("freq_b"):
        return "frequency"
    if lower.startswith("present_b"):
        return "presence"
    if lower.startswith("rssi_rel_b"):
        return "relative_rssi"
    if lower.startswith("rssi_"):
        return "rssi_statistics"
    if lower.startswith("fp_"):
        return "fingerprint_similarity"
    if lower.startswith(("top", "rank")):
        return "beacon_rank"
    if any(marker in lower for marker in ("count", "density", "coverage", "entropy")):
        return "coverage_distribution"
    return "other"


def _model_params(config: dict[str, Any], n_classes: int, seed: int) -> dict[str, Any]:
    params = dict(config["xgboost"])
    params.update(
        {
            "num_class": n_classes,
            "random_state": seed,
            "n_jobs": 2,
            "verbosity": 0,
        }
    )
    return params


def _fit_seed_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_test: pd.DataFrame,
    config: dict[str, Any],
    n_classes: int,
    seed: int,
) -> tuple[int, XGBClassifier, np.ndarray]:
    model = XGBClassifier(**_model_params(config, n_classes, seed))
    model.fit(x_train, y_train, sample_weight=weights)
    return seed, model, model.predict_proba(x_test)


def run_fold(
    frame: pd.DataFrame,
    window: int,
    test_day: str,
    config: dict[str, Any],
    artifact_root: Path,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Train and evaluate one strict outer LODO fold."""

    window_config = config["windows"][str(window)]
    variant = str(window_config["variant"])
    fold_dir = artifact_root / f"{window}s" / test_day
    completion = fold_dir / "fold_complete.json"
    table_names = ["metrics", "class_metrics", "confusion", "shap", "selection", "audit"]
    if completion.is_file() and not force:
        return {name: pd.read_csv(fold_dir / f"{name}.csv") for name in table_names}

    fold_dir.mkdir(parents=True, exist_ok=True)
    train, test, audit = strict_fold(frame, test_day)
    train = add_visit_metadata(train, window)
    train_engineered, test_engineered, fingerprints = engineer_fold(train, test)
    selector = fit_selector(train_engineered, config["selector"])
    features = feature_set(variant, selector["features"], train_engineered)
    if not features:
        raise RuntimeError(f"No usable features selected for {window}s / {test_day}")

    augmented, augmentation_audit = augment_training(
        train_engineered,
        features,
        config["augmentation"],
    )
    encoder = LabelEncoder().fit(train_engineered["room"].astype(str))
    if set(encoder.classes_) != set(test_engineered["room"].astype(str).unique()):
        raise AssertionError("Strict fold classes changed before model fitting")
    x_train = augmented[features].replace([np.inf, -np.inf], np.nan).astype(float)
    x_test = test_engineered[features].replace([np.inf, -np.inf], np.nan).astype(float)
    y_train = encoder.transform(augmented["room"].astype(str))
    y_test = encoder.transform(test_engineered["room"].astype(str))
    weights = training_weights(variant, augmented, window, config["weighting"])

    seeds = [int(seed) for seed in config["seeds"]]
    results: list[tuple[int, XGBClassifier, np.ndarray]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as executor:
        futures = [
            executor.submit(
                _fit_seed_model,
                x_train,
                y_train,
                weights,
                x_test,
                config,
                len(encoder.classes_),
                seed,
            )
            for seed in seeds
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item[0])
    models = [item[1] for item in results]
    probabilities = normalize_rows(np.mean([item[2] for item in results], axis=0))
    raw_prediction = probabilities.argmax(axis=1)

    initial, transition = fit_markov_model(
        train_engineered,
        encoder,
        float(config["decoder"]["transition_pseudocount"]),
        window,
        float(config["decoder"]["max_gap_multiplier"]),
    )
    smoothed = causal_probability_smoothing(
        probabilities,
        test_engineered["window_ts"],
        window,
        int(window_config["smooth_steps"]),
        float(config["decoder"]["max_gap_multiplier"]),
    )
    decoded_prediction = viterbi_decode(
        smoothed,
        test_engineered["window_ts"],
        window,
        initial,
        transition,
        float(config["decoder"]["markov_lambda"]),
        float(config["decoder"]["max_gap_multiplier"]),
    )
    boundary = boundary_mask(
        test_engineered["room"].astype(str),
        test_engineered["window_ts"],
    )

    metric_rows: list[dict[str, Any]] = []
    class_tables: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, Any]] = []
    all_labels = np.arange(len(encoder.classes_))
    for decoder_name, prediction in (
        ("raw", raw_prediction),
        ("fixed_decoder", decoded_prediction),
    ):
        for region, mask in (
            ("all", np.ones(len(y_test), dtype=bool)),
            ("boundary_15s", boundary),
            ("stable", ~boundary),
        ):
            metrics = classification_metrics(
                y_test[mask],
                prediction[mask],
                all_labels if region == "all" else None,
            )
            metric_rows.append(
                {
                    "window_seconds": window,
                    "variant": variant,
                    "test_day": test_day,
                    "decoder": decoder_name,
                    "region": region,
                    **metrics,
                }
            )
        class_table = per_class_metrics(y_test, prediction, encoder, decoder_name)
        class_table.insert(0, "test_day", test_day)
        class_table.insert(0, "variant", variant)
        class_table.insert(0, "window_seconds", window)
        class_tables.append(class_table)
        matrix = confusion_matrix(y_test, prediction, labels=all_labels)
        for true_index, true_room in enumerate(encoder.classes_):
            for pred_index, pred_room in enumerate(encoder.classes_):
                confusion_rows.append(
                    {
                        "window_seconds": window,
                        "variant": variant,
                        "test_day": test_day,
                        "decoder": decoder_name,
                        "true_room": true_room,
                        "pred_room": pred_room,
                        "count": int(matrix[true_index, pred_index]),
                    }
                )

    shap_table = native_tree_shap(
        models,
        x_test,
        int(config["shap"]["max_samples_per_fold"]),
    )
    shap_table.insert(0, "test_day", test_day)
    shap_table.insert(0, "variant", variant)
    shap_table.insert(0, "window_seconds", window)
    shap_table["feature_family"] = shap_table["feature"].map(feature_family)

    selection = selector["ranking"].copy()
    selection.insert(0, "test_day", test_day)
    selection.insert(0, "variant", variant)
    selection.insert(0, "window_seconds", window)
    selection["used_by_model"] = selection["feature"].isin(features)
    audit.update(
        {
            "window_seconds": window,
            "variant": variant,
            "n_selected": len(features),
            "n_selector_candidates": int(selector["n_candidates"]),
            "n_selector_usable": int(selector["n_usable"]),
            "n_augmented": int(len(augmented) - len(train_engineered)),
            "n_train_fit": int(len(augmented)),
            "n_seeds": len(seeds),
            "boundary_samples": int(boundary.sum()),
            "stable_samples": int((~boundary).sum()),
        }
    )
    tables = {
        "metrics": pd.DataFrame(metric_rows),
        "class_metrics": pd.concat(class_tables, ignore_index=True),
        "confusion": pd.DataFrame(confusion_rows),
        "shap": shap_table,
        "selection": selection,
        "audit": pd.DataFrame([audit]),
    }

    for name, table in tables.items():
        table.to_csv(fold_dir / f"{name}.csv", index=False)
    selector["correlation_dropped"].to_csv(fold_dir / "correlation_dropped.csv", index=False)
    augmentation_audit.to_csv(fold_dir / "augmentation_audit.csv", index=False)
    fingerprints.to_csv(fold_dir / "train_fingerprints.csv")
    pd.DataFrame(transition, index=encoder.classes_, columns=encoder.classes_).to_csv(
        fold_dir / "train_transition.csv"
    )
    predictions = pd.DataFrame(
        {
            "date": test_engineered["date"].astype(str),
            "window_ts": test_engineered["window_ts"].astype(str),
            "y_true": encoder.inverse_transform(y_test),
            "y_pred_raw": encoder.inverse_transform(raw_prediction),
            "y_pred_fixed_decoder": encoder.inverse_transform(decoded_prediction),
            "confidence_raw": probabilities.max(axis=1),
            "entropy_raw": -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1),
            "is_boundary_15s": boundary.astype(int),
        }
    )
    predictions.to_csv(fold_dir / "predictions.csv", index=False)
    np.savez_compressed(
        fold_dir / "probabilities.npz",
        raw=probabilities,
        smoothed=smoothed,
        classes=encoder.classes_,
    )
    for seed, model, _ in results:
        joblib.dump(model, fold_dir / f"xgboost_seed_{seed}.joblib")
    _write_json(
        fold_dir / "feature_manifest.json",
        {"features": features, "classes": encoder.classes_.tolist(), "seeds": seeds},
    )
    _write_json(
        completion,
        {
            "status": "complete",
            "window_seconds": window,
            "test_day": test_day,
            "variant": variant,
            "completed_unix": time.time(),
        },
    )
    return tables


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_fold_f1(metrics: pd.DataFrame, path: Path) -> None:
    subset = metrics.loc[metrics["region"].eq("all")].copy()
    windows = sorted(subset["window_seconds"].unique())
    figure, axes = plt.subplots(1, len(windows), figsize=(6.4 * len(windows), 4.4), squeeze=False)
    colors = {"raw": "#3568a8", "fixed_decoder": "#d56b2d"}
    for axis, window in zip(axes[0], windows):
        window_rows = subset.loc[subset["window_seconds"].eq(window)]
        days = sorted(window_rows["test_day"].unique())
        positions = np.arange(len(days))
        for decoder in ("raw", "fixed_decoder"):
            indexed = window_rows.loc[window_rows["decoder"].eq(decoder)].set_index("test_day")
            values = indexed.reindex(days)["macro_f1"].to_numpy(dtype=float)
            axis.plot(
                positions,
                values,
                marker="o",
                linewidth=2,
                label=decoder.replace("_", " "),
                color=colors[decoder],
            )
        axis.set_title(f"{window}s strict LODO")
        axis.set_xticks(positions, [day[-2:] for day in days])
        axis.set_xlabel("Outer test day (April 2023)")
        axis.set_ylabel("Macro-F1")
        axis.set_ylim(0.0, max(0.6, axis.get_ylim()[1]))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Strong ML: raw ensemble vs fixed temporal decoder", y=1.02)
    _save_figure(path)


def _plot_region_f1(metrics: pd.DataFrame, path: Path) -> None:
    subset = metrics.loc[metrics["decoder"].eq("fixed_decoder")].copy()
    grouped = subset.groupby(["window_seconds", "region"], observed=True)["macro_f1"].agg(["mean", "std"])
    windows = sorted(subset["window_seconds"].unique())
    regions = ["all", "boundary_15s", "stable"]
    x = np.arange(len(regions))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for offset, window in enumerate(windows):
        means = [grouped.loc[(window, region), "mean"] for region in regions]
        errors = [grouped.loc[(window, region), "std"] for region in regions]
        axis.bar(
            x + (offset - (len(windows) - 1) / 2) * width,
            means,
            width,
            yerr=errors,
            capsize=4,
            label=f"{window}s",
        )
    axis.set_xticks(x, ["All", "Boundary ±15s", "Stable"])
    axis.set_ylabel("Mean outer-fold Macro-F1")
    axis.set_title("Fixed decoder performance by transition region")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    _save_figure(path)


def _plot_confusion(confusion: pd.DataFrame, path: Path) -> None:
    subset = confusion.loc[confusion["decoder"].eq("fixed_decoder")]
    windows = sorted(subset["window_seconds"].unique())
    figure, axes = plt.subplots(1, len(windows), figsize=(8 * len(windows), 7), squeeze=False)
    for axis, window in zip(axes[0], windows):
        rows = subset.loc[subset["window_seconds"].eq(window)]
        rooms = sorted(set(rows["true_room"]) | set(rows["pred_room"]))
        pivot = rows.pivot_table(
            index="true_room",
            columns="pred_room",
            values="count",
            aggfunc="sum",
            fill_value=0,
        ).reindex(index=rooms, columns=rooms, fill_value=0)
        values = pivot.to_numpy(dtype=float)
        normalized = values / np.maximum(values.sum(axis=1, keepdims=True), 1.0)
        image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_title(f"{window}s fixed decoder")
        axis.set_xlabel("Predicted room")
        axis.set_ylabel("True room")
        axis.set_xticks(np.arange(len(rooms)), rooms, rotation=90, fontsize=7)
        axis.set_yticks(np.arange(len(rooms)), rooms, fontsize=7)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized rate")
    figure.suptitle("Aggregated strict LODO confusion matrices", y=1.01)
    _save_figure(path)


def _plot_shap(shap: pd.DataFrame, path: Path, top_n: int) -> None:
    windows = sorted(shap["window_seconds"].unique())
    figure, axes = plt.subplots(1, len(windows), figsize=(8 * len(windows), 7), squeeze=False)
    for axis, window in zip(axes[0], windows):
        rows = (
            shap.loc[shap["window_seconds"].eq(window)]
            .sort_values("mean_abs_shap")
            .tail(top_n)
        )
        axis.barh(rows["feature"], rows["mean_abs_shap"], color="#3f7f93")
        axis.set_title(f"{window}s raw XGBoost ensemble")
        axis.set_xlabel("Mean |TreeSHAP value|")
        axis.grid(axis="x", alpha=0.25)
        axis.tick_params(axis="y", labelsize=8)
    figure.suptitle("Global feature attribution across strict LODO folds", y=1.01)
    _save_figure(path)


def _plot_shap_families(families: pd.DataFrame, path: Path) -> None:
    pivot = families.pivot(index="feature_family", columns="window_seconds", values="mean_abs_shap").fillna(0.0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    figure, axis = plt.subplots(figsize=(8.5, max(4.5, 0.55 * len(pivot))))
    pivot.plot.barh(ax=axis, color=["#3568a8", "#d56b2d"][: len(pivot.columns)])
    axis.set_xlabel("Summed mean |TreeSHAP value|")
    axis.set_ylabel("Feature family")
    axis.set_title("Feature-family contribution by window")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(title="Window (s)", frameon=False)
    _save_figure(path)


def _plot_feature_stability(stability: pd.DataFrame, path: Path) -> None:
    windows = sorted(stability["window_seconds"].unique())
    figure, axes = plt.subplots(1, len(windows), figsize=(7 * len(windows), 6), squeeze=False)
    for axis, window in zip(axes[0], windows):
        rows = stability.loc[stability["window_seconds"].eq(window)].nlargest(20, "fold_selection_rate")
        rows = rows.sort_values("fold_selection_rate")
        axis.barh(rows["feature"], rows["fold_selection_rate"], color="#6a8f4e")
        axis.set_xlim(0.0, 1.02)
        axis.set_xlabel("Fraction of outer folds selected")
        axis.set_title(f"{window}s top selection stability")
        axis.grid(axis="x", alpha=0.25)
        axis.tick_params(axis="y", labelsize=8)
    _save_figure(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_reports(
    all_tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    report_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Publish aggregate, reviewable artifacts without private row-level feature data."""

    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = all_tables["metrics"].sort_values(
        ["window_seconds", "test_day", "decoder", "region"]
    )
    audit = all_tables["audit"].sort_values(["window_seconds", "test_day"])
    confusion = all_tables["confusion"]
    class_metrics = all_tables["class_metrics"]
    shap_folds = all_tables["shap"]
    selection = all_tables["selection"]

    summary = (
        metrics.groupby(["window_seconds", "variant", "decoder", "region"], observed=True)
        .agg(
            n_folds=("test_day", "nunique"),
            n_samples=("n_samples", "sum"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            weighted_f1_mean=("weighted_f1", "mean"),
        )
        .reset_index()
    )
    shap_parts: list[pd.DataFrame] = []
    for (window, variant), group in shap_folds.groupby(["window_seconds", "variant"], observed=True):
        days = sorted(group["test_day"].unique())
        features = sorted(group["feature"].unique())
        dense = pd.MultiIndex.from_product(
            [days, features], names=["test_day", "feature"]
        ).to_frame(index=False)
        dense = dense.merge(
            group[["test_day", "feature", "feature_family", "mean_abs_shap", "seed_std_abs_shap"]],
            on=["test_day", "feature"],
            how="left",
        )
        dense["selected_in_fold"] = dense["mean_abs_shap"].notna()
        dense["feature_family"] = dense["feature_family"].fillna(dense["feature"].map(feature_family))
        dense[["mean_abs_shap", "seed_std_abs_shap"]] = dense[
            ["mean_abs_shap", "seed_std_abs_shap"]
        ].fillna(0.0)
        aggregated = (
            dense.groupby(["feature", "feature_family"], observed=True)
            .agg(
                mean_abs_shap=("mean_abs_shap", "mean"),
                fold_std_abs_shap=("mean_abs_shap", "std"),
                mean_seed_std_abs_shap=("seed_std_abs_shap", "mean"),
                n_folds_selected=("selected_in_fold", "sum"),
            )
            .reset_index()
        )
        aggregated.insert(0, "variant", variant)
        aggregated.insert(0, "window_seconds", window)
        aggregated["n_folds"] = len(days)
        aggregated["n_explained"] = int(
            group.groupby("test_day", observed=True)["n_explained"].max().sum()
        )
        shap_parts.append(aggregated)
    shap = pd.concat(shap_parts, ignore_index=True).sort_values(
        ["window_seconds", "mean_abs_shap"], ascending=[True, False]
    )
    families = (
        shap.groupby(["window_seconds", "variant", "feature_family"], observed=True)["mean_abs_shap"]
        .sum()
        .reset_index()
        .sort_values(["window_seconds", "mean_abs_shap"], ascending=[True, False])
    )
    n_folds = selection.groupby("window_seconds")["test_day"].nunique().to_dict()
    stability = (
        selection.loc[selection["used_by_model"]]
        .groupby(["window_seconds", "variant", "feature"], observed=True)
        .agg(n_folds_selected=("test_day", "nunique"), mean_priority_score=("priority_score", "mean"))
        .reset_index()
    )
    stability["fold_selection_rate"] = stability.apply(
        lambda row: row["n_folds_selected"] / n_folds[row["window_seconds"]], axis=1
    )
    stability = stability.sort_values(
        ["window_seconds", "fold_selection_rate", "mean_priority_score"],
        ascending=[True, False, False],
    )
    aggregate_class = (
        class_metrics.groupby(["window_seconds", "variant", "decoder", "room"], observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "support": int(group["support"].sum()),
                    "precision_weighted": float(np.average(group["precision"], weights=np.maximum(group["support"], 1))),
                    "recall_weighted": float(np.average(group["recall"], weights=np.maximum(group["support"], 1))),
                    "f1_weighted": float(np.average(group["f1"], weights=np.maximum(group["support"], 1))),
                    "n_folds": int(group["test_day"].nunique()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    public_tables = {
        "fold_metrics": metrics,
        "summary": summary,
        "protocol_audit": audit,
        "class_metrics": aggregate_class,
        "confusion_counts": confusion,
        "shap_importance": shap,
        "shap_family_importance": families,
        "feature_stability": stability,
    }
    for name, table in public_tables.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)

    _plot_fold_f1(metrics, report_dir / "fold_macro_f1.png")
    _plot_region_f1(metrics, report_dir / "boundary_stable_macro_f1.png")
    _plot_confusion(confusion, report_dir / "confusion_fixed_decoder.png")
    _plot_shap(shap, report_dir / "shap_top_features.png", int(config["shap"]["top_n_plot"]))
    _plot_shap_families(families, report_dir / "shap_feature_families.png")
    _plot_feature_stability(stability, report_dir / "feature_selection_stability.png")

    all_rows = summary.loc[summary["region"].eq("all")].copy()
    report_lines = [
        "# Strong ML 1s/3s — strict DASEL-matched LODO",
        "",
        "This report evaluates the strongest reviewed ML configurations under the strict shared-class protocol: each outer fold is filtered to `classes(train) ∩ classes(test)` before any fitting. Feature selection, drift filtering, augmentation, weighting, fingerprints, and Markov transitions use outer-training days only.",
        "",
        "## Configurations",
        "",
        "- **1s / M2:** top-100 drift-aware features, 0.75 class-balanced + 0.25 soft visit-cap weights, and a three-seed XGBoost ensemble.",
        "- **3s / M5:** 23 beacon-frequency features plus up to 77 selected non-frequency features, class-balanced weights, and a three-seed XGBoost ensemble.",
        "- **Fixed decoder:** causal five-observation probability smoothing followed by train-only Markov Viterbi decoding. At 1s this uses up to 5 seconds of past probability context; at 3s it uses up to 15 seconds.",
        "",
        "## Outer-fold results",
        "",
        "| Window | Variant | Output | Macro-F1 mean ± SD | Accuracy | Balanced accuracy |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in all_rows.sort_values(["window_seconds", "decoder"]).itertuples(index=False):
        report_lines.append(
            f"| {int(row.window_seconds)}s | {row.variant} | {row.decoder} | "
            f"{row.macro_f1_mean:.4f} ± {row.macro_f1_std:.4f} | "
            f"{row.accuracy_mean:.4f} | {row.balanced_accuracy_mean:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- TreeSHAP values are exact native XGBoost contributions for the **raw three-seed ensemble**. They do not explain the subsequent temporal decoder.",
            "- Cross-fold SHAP aggregation uses zero contribution when a feature is absent from a fold-specific selected set; `n_folds_selected` remains available for stability auditing.",
            "- The fixed Viterbi decoder uses only training-derived transition probabilities, but backtracking makes its final label sequence an **offline** output. Raw probabilities and causal smoothing remain online-compatible; a causal state filter is required for a production online decoder.",
            "- Boundary and stable metrics use a ±15-second band around ground-truth room changes. This diagnostic label is used only after prediction for evaluation.",
            "- The protocol matches the DASEL paper's per-fold shared class counts; it does not claim architectural equivalence to the DASEL neural model.",
            "- Private row-level features, probabilities, predictions, fingerprints, and serialized models remain under ignored `artifacts/strong_ml/` and are not published.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "python scripts/run_strong_ml.py --features-dir <PRIVATE_FEATURE_DIR>",
            "```",
            "",
            "The public CSV files contain only fold-level or aggregate metrics/attributions. See `manifest.json` for hashes and environment metadata.",
            "",
        ]
    )
    (report_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    files = sorted(path for path in report_dir.iterdir() if path.is_file() and path.name != "manifest.json")
    manifest = {
        "protocol": config["protocol"],
        "windows": config["windows"],
        "seeds": config["seeds"],
        "python": platform.python_version(),
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files
        ],
        "private_artifacts_published": False,
        "shap_backend": "xgboost_native_pred_contribs",
    }
    _write_json(report_dir / "manifest.json", manifest)
    return public_tables


def run_pipeline(
    features_dir: Path,
    config_path: Path,
    artifact_dir: Path,
    report_dir: Path,
    windows: Sequence[int] = (1, 3),
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    config = load_strong_config(config_path)
    collected: dict[str, list[pd.DataFrame]] = {
        name: [] for name in ("metrics", "class_metrics", "confusion", "shap", "selection", "audit")
    }
    for window in windows:
        if str(window) not in config["windows"]:
            raise KeyError(f"Window {window}s is not configured")
        frame = load_feature_frame(features_dir, window)
        days = sorted(frame["date"].unique())
        for fold_index, test_day in enumerate(days, start=1):
            print(
                f"[{window}s] fold {fold_index}/{len(days)} test={test_day} "
                f"variant={config['windows'][str(window)]['variant']}",
                flush=True,
            )
            tables = run_fold(frame, window, test_day, config, artifact_dir, force=force)
            for name, table in tables.items():
                collected[name].append(table)
            result = tables["metrics"]
            result = result.loc[(result["region"] == "all") & (result["decoder"] == "fixed_decoder")]
            print(f"[{window}s] {test_day} fixed Macro-F1={result.iloc[0]['macro_f1']:.4f}", flush=True)
    combined = {name: pd.concat(tables, ignore_index=True) for name, tables in collected.items()}
    publish_reports(combined, config, report_dir)
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True, help="Private directory containing feature CSVs")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "strong_ml_1s_3s.json",
    )
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "strong_ml")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "strong_ml_1s_3s",
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--force", action="store_true", help="Retrain completed folds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_pipeline(
        features_dir=args.features_dir,
        config_path=args.config,
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
        windows=args.windows,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

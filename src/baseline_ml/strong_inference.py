"""Final full-train inference and 1s/3s agreement analysis for unlabeled BLE data."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

from .dasel_features import BEACONS, extract_one_window
from .strong_ml import (
    PROJECT_ROOT,
    _fit_seed_model,
    _write_json,
    add_visit_metadata,
    augment_training,
    causal_probability_smoothing,
    engineer_fold,
    feature_set,
    fit_markov_model,
    fit_selector,
    load_feature_frame,
    load_strong_config,
    normalize_rows,
    split_segments,
    training_weights,
    viterbi_decode,
)


UNLABELED_ROOM = "__unlabeled__"


def load_unlabeled_detections(
    path: Path,
    timezone: str = "Asia/Seoul",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Normalize the challenge schema and remove exact unlabeled detections."""

    raw = pd.read_csv(path)
    source_columns = list(raw.columns)
    rename: dict[str, str] = {}
    for column in raw.columns:
        normalized = str(column).strip().lower().replace(" ", "_")
        if normalized in {"mac_address", "macaddress"}:
            rename[column] = "mac_address"
        elif normalized == "rssi":
            rename[column] = "RSSI"
        elif normalized == "timestamp":
            rename[column] = "timestamp"
        elif normalized == "user_id":
            rename[column] = "user_id"
        elif normalized.startswith("unnamed:"):
            rename[column] = "source_id"
    normalized = raw.rename(columns=rename).copy()
    required = {"user_id", "timestamp", "mac_address", "RSSI"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise KeyError(f"Unlabeled BLE input is missing columns: {missing}")

    normalized["source_row"] = np.arange(len(normalized), dtype=np.int64)
    if "source_id" not in normalized:
        normalized["source_id"] = normalized["source_row"]
    normalized["mac_address"] = pd.to_numeric(normalized["mac_address"], errors="raise").astype(int)
    normalized["RSSI"] = pd.to_numeric(normalized["RSSI"], errors="raise").astype(float)
    parsed = pd.to_datetime(normalized["timestamp"], errors="raise")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    else:
        parsed = parsed.dt.tz_convert(timezone)
    normalized["timestamp_local"] = parsed
    invalid = sorted(set(normalized["mac_address"]) - set(BEACONS))
    if invalid:
        raise ValueError(f"Beacon IDs outside the supported 1..25 range: {invalid}")
    if not np.isfinite(normalized["RSSI"]).all():
        raise ValueError("RSSI contains non-finite values")

    dedup_key = ["user_id", "timestamp_local", "mac_address", "RSSI"]
    detections = (
        normalized.sort_values(["timestamp_local", "source_row"], kind="stable")
        .drop_duplicates(dedup_key, keep="first")
        .reset_index(drop=True)
    )
    detections = detections.drop(columns=["timestamp"]).rename(
        columns={"timestamp_local": "timestamp"}
    )
    detections["room"] = UNLABELED_ROOM
    detections["_order"] = np.arange(len(detections), dtype=np.int64)
    audit = {
        "source_file": path.name,
        "source_rows": int(len(normalized)),
        "deduplicated_detection_rows": int(len(detections)),
        "exact_duplicate_rows_removed": int(len(normalized) - len(detections)),
        "deduplication_key": dedup_key,
        "n_users": int(normalized["user_id"].nunique()),
        "beacons": sorted(normalized["mac_address"].astype(int).unique().tolist()),
        "timestamp_min_local": str(normalized["timestamp_local"].min()),
        "timestamp_max_local": str(normalized["timestamp_local"].max()),
        "timezone_assumption": timezone,
        "source_columns": source_columns,
        "power_used": False,
        "has_ground_truth": False,
    }
    return normalized, detections, audit


def extract_unlabeled_features(
    detections: pd.DataFrame,
    window_seconds: int,
) -> pd.DataFrame:
    features = extract_one_window(detections, window_seconds)
    if not features["room"].eq(UNLABELED_ROOM).all():
        raise AssertionError("Unlabeled placeholder changed during window extraction")
    features["window_ts_local"] = pd.to_datetime(features["window_ts"])
    features["window_ts"] = pd.to_datetime(features["window_ts"], utc=True)
    return features.sort_values("window_ts", kind="stable").reset_index(drop=True)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = normalize_rows(probabilities)
    return -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)


def _margin(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(normalize_rows(probabilities), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def train_final_and_predict(
    train_frame: pd.DataFrame,
    test_features: pd.DataFrame,
    window_seconds: int,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit a full four-day final model and predict the unlabeled fifth day."""

    window_config = config["windows"][str(window_seconds)]
    variant = str(window_config["variant"])
    model_dir = output_dir / "models" / f"{window_seconds}s"
    audit_dir = output_dir / "training_audit" / f"{window_seconds}s"
    model_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    train = add_visit_metadata(train_frame, window_seconds)
    train_engineered, test_engineered, fingerprints = engineer_fold(train, test_features)
    selector = fit_selector(train_engineered, config["selector"])
    features = feature_set(variant, selector["features"], train_engineered)
    if not features:
        raise RuntimeError(f"No features selected for final {window_seconds}s model")
    augmented, augmentation_audit = augment_training(
        train_engineered,
        features,
        config["augmentation"],
    )

    encoder = LabelEncoder().fit(train_engineered["room"].astype(str))
    x_train = augmented[features].replace([np.inf, -np.inf], np.nan).astype(float)
    x_test = test_engineered[features].replace([np.inf, -np.inf], np.nan).astype(float)
    y_train = encoder.transform(augmented["room"].astype(str))
    weights = training_weights(variant, augmented, window_seconds, config["weighting"])
    seeds = [int(seed) for seed in config["seeds"]]
    fitted: list[tuple[int, Any, np.ndarray]] = []
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
            fitted.append(future.result())
    fitted.sort(key=lambda item: item[0])
    probabilities = normalize_rows(np.mean([item[2] for item in fitted], axis=0))
    smoothed = causal_probability_smoothing(
        probabilities,
        test_engineered["window_ts"],
        window_seconds,
        int(window_config["smooth_steps"]),
        float(config["decoder"]["max_gap_multiplier"]),
    )
    raw_prediction = probabilities.argmax(axis=1)
    causal_prediction = smoothed.argmax(axis=1)
    initial, transition = fit_markov_model(
        train_engineered,
        encoder,
        float(config["decoder"]["transition_pseudocount"]),
        window_seconds,
        float(config["decoder"]["max_gap_multiplier"]),
    )
    decoded_prediction = viterbi_decode(
        smoothed,
        test_engineered["window_ts"],
        window_seconds,
        initial,
        transition,
        float(config["decoder"]["markov_lambda"]),
        float(config["decoder"]["max_gap_multiplier"]),
    )

    predictions = pd.DataFrame(
        {
            "window_seconds": window_seconds,
            "window_ts_local": test_engineered["window_ts_local"].astype(str),
            "window_ts_utc": test_engineered["window_ts"].astype(str),
            "predicted_room_raw": encoder.inverse_transform(raw_prediction),
            "predicted_room_causal": encoder.inverse_transform(causal_prediction),
            "predicted_room_fixed_decoder": encoder.inverse_transform(decoded_prediction),
            "confidence_raw": probabilities.max(axis=1),
            "margin_raw": _margin(probabilities),
            "entropy_raw": _entropy(probabilities),
            "confidence_causal": smoothed.max(axis=1),
            "margin_causal": _margin(smoothed),
            "entropy_causal": _entropy(smoothed),
        }
    )
    predictions["raw_changed_by_causal"] = (
        predictions["predicted_room_raw"] != predictions["predicted_room_causal"]
    )
    predictions["raw_changed_by_fixed_decoder"] = (
        predictions["predicted_room_raw"] != predictions["predicted_room_fixed_decoder"]
    )
    probability_table = pd.DataFrame(
        probabilities,
        columns=[f"prob_raw__{room}" for room in encoder.classes_],
    )
    for index, room in enumerate(encoder.classes_):
        probability_table[f"prob_causal__{room}"] = smoothed[:, index]
    probability_table.insert(0, "window_ts_local", predictions["window_ts_local"])
    probability_table.insert(1, "window_ts_utc", predictions["window_ts_utc"])

    for seed, model, _ in fitted:
        joblib.dump(model, model_dir / f"xgboost_seed_{seed}.joblib")
    fingerprints.to_csv(audit_dir / "train_fingerprints.csv")
    selector["ranking"].to_csv(audit_dir / "feature_ranking.csv", index=False)
    selector["correlation_dropped"].to_csv(audit_dir / "correlation_dropped.csv", index=False)
    augmentation_audit.to_csv(audit_dir / "augmentation.csv", index=False)
    pd.DataFrame(transition, index=encoder.classes_, columns=encoder.classes_).to_csv(
        audit_dir / "train_transition.csv"
    )
    _write_json(
        audit_dir / "feature_manifest.json",
        {"features": features, "classes": encoder.classes_.tolist(), "seeds": seeds},
    )
    audit = {
        "window_seconds": window_seconds,
        "variant": variant,
        "training_protocol": "final_full_train_union_for_unlabeled_test",
        "n_train_windows": int(len(train_engineered)),
        "training_days": sorted(train_engineered["date"].astype(str).unique().tolist()),
        "n_augmented_windows": int(len(augmented) - len(train_engineered)),
        "n_test_windows": int(len(test_engineered)),
        "n_classes": int(len(encoder.classes_)),
        "classes": encoder.classes_.tolist(),
        "n_features": int(len(features)),
        "features": features,
        "seeds": seeds,
        "decoder_offline": True,
        "causal_output_online_compatible": True,
    }
    _write_json(audit_dir / "final_training_audit.json", audit)
    return predictions, probability_table, audit


def map_predictions_to_source_rows(
    normalized_source: pd.DataFrame,
    window_predictions: pd.DataFrame,
    window_seconds: int,
) -> pd.DataFrame:
    source = normalized_source.copy()
    source["window_ts_local"] = source["timestamp_local"].dt.floor(f"{window_seconds}s").astype(str)
    columns = [
        "window_ts_local",
        "predicted_room_raw",
        "predicted_room_causal",
        "predicted_room_fixed_decoder",
        "confidence_raw",
        "margin_raw",
        "entropy_raw",
        "confidence_causal",
        "margin_causal",
        "entropy_causal",
    ]
    mapped = source.merge(window_predictions[columns], on="window_ts_local", how="left", validate="many_to_one")
    if mapped["predicted_room_fixed_decoder"].isna().any():
        raise AssertionError("Some source rows could not be mapped to a prediction window")
    preferred = [
        "source_id",
        "source_row",
        "user_id",
        "timestamp",
        "mac_address",
        "RSSI",
        "power",
        "window_ts_local",
        "predicted_room_raw",
        "predicted_room_causal",
        "predicted_room_fixed_decoder",
        "confidence_raw",
        "margin_raw",
        "entropy_raw",
        "confidence_causal",
        "margin_causal",
        "entropy_causal",
    ]
    return mapped[[column for column in preferred if column in mapped]].sort_values("source_row")


def _js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_rows(left)
    right = normalize_rows(right)
    midpoint = 0.5 * (left + right)
    kl_left = np.sum(left * (np.log(np.maximum(left, 1e-12)) - np.log(np.maximum(midpoint, 1e-12))), axis=1)
    kl_right = np.sum(right * (np.log(np.maximum(right, 1e-12)) - np.log(np.maximum(midpoint, 1e-12))), axis=1)
    return 0.5 * (kl_left + kl_right)


def align_window_predictions(
    predictions_1s: pd.DataFrame,
    probabilities_1s: pd.DataFrame,
    predictions_3s: pd.DataFrame,
    probabilities_3s: pd.DataFrame,
) -> pd.DataFrame:
    left = predictions_1s.copy().reset_index(names="index_1s")
    right = predictions_3s.copy().reset_index(names="index_3s")
    left["context_3s"] = pd.to_datetime(left["window_ts_utc"], utc=True).dt.floor("3s")
    right["context_3s"] = pd.to_datetime(right["window_ts_utc"], utc=True)
    aligned = left.merge(right, on="context_3s", suffixes=("_1s", "_3s"), how="inner", validate="many_to_one")
    raw_columns_1 = [column for column in probabilities_1s if column.startswith("prob_raw__")]
    raw_columns_3 = [column for column in probabilities_3s if column.startswith("prob_raw__")]
    classes_1 = [column.removeprefix("prob_raw__") for column in raw_columns_1]
    classes_3 = [column.removeprefix("prob_raw__") for column in raw_columns_3]
    if classes_1 != classes_3:
        raise AssertionError("1s and 3s final model class orders do not match")
    p1 = probabilities_1s.loc[aligned["index_1s"], raw_columns_1].to_numpy(dtype=float)
    p3 = probabilities_3s.loc[aligned["index_3s"], raw_columns_3].to_numpy(dtype=float)
    aligned["raw_js_divergence"] = _js_divergence(p1, p3)
    for stage in ("raw", "causal", "fixed_decoder"):
        aligned[f"agree_{stage}"] = (
            aligned[f"predicted_room_{stage}_1s"] == aligned[f"predicted_room_{stage}_3s"]
        )
    aligned["recommended_room"] = np.where(
        aligned["agree_fixed_decoder"],
        aligned["predicted_room_fixed_decoder_1s"],
        aligned["predicted_room_fixed_decoder_3s"],
    )
    aligned["recommendation_reason"] = np.where(
        aligned["agree_fixed_decoder"],
        "1s_3s_agree",
        "3s_stronger_lodo_branch",
    )
    keep = [
        "window_ts_local_1s",
        "window_ts_local_3s",
        "predicted_room_raw_1s",
        "predicted_room_raw_3s",
        "predicted_room_causal_1s",
        "predicted_room_causal_3s",
        "predicted_room_fixed_decoder_1s",
        "predicted_room_fixed_decoder_3s",
        "confidence_raw_1s",
        "confidence_raw_3s",
        "entropy_raw_1s",
        "entropy_raw_3s",
        "raw_js_divergence",
        "agree_raw",
        "agree_causal",
        "agree_fixed_decoder",
        "recommended_room",
        "recommendation_reason",
    ]
    return aligned[keep]


def agreement_summary(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage in ("raw", "causal", "fixed_decoder"):
        left = aligned[f"predicted_room_{stage}_1s"].astype(str)
        right = aligned[f"predicted_room_{stage}_3s"].astype(str)
        rows.append(
            {
                "stage": stage,
                "n_aligned_1s_frames": int(len(aligned)),
                "agreement_rate": float(left.eq(right).mean()),
                "cohen_kappa": float(cohen_kappa_score(left, right)),
                "mean_raw_js_divergence": float(aligned["raw_js_divergence"].mean()) if stage == "raw" else np.nan,
                "confidence_1s_mean": float(aligned["confidence_raw_1s"].mean()),
                "confidence_3s_mean": float(aligned["confidence_raw_3s"].mean()),
                "agree_confidence_1s_mean": float(aligned.loc[left.eq(right), "confidence_raw_1s"].mean()),
                "disagree_confidence_1s_mean": float(aligned.loc[~left.eq(right), "confidence_raw_1s"].mean()),
                "agree_confidence_3s_mean": float(aligned.loc[left.eq(right), "confidence_raw_3s"].mean()),
                "disagree_confidence_3s_mean": float(aligned.loc[~left.eq(right), "confidence_raw_3s"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _run_lengths(labels: pd.Series, window_seconds: int) -> np.ndarray:
    groups = labels.ne(labels.shift()).cumsum()
    return labels.groupby(groups).size().to_numpy(dtype=float) * window_seconds


def prediction_diagnostics(predictions: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    timestamps = pd.to_datetime(predictions["window_ts_utc"], utc=True)
    elapsed_hours = max((timestamps.max() - timestamps.min()).total_seconds() / 3600.0, 1e-12)
    active_window_hours = max(len(predictions) * window_seconds / 3600.0, 1e-12)
    segments = split_segments(timestamps, window_seconds)
    rows: list[dict[str, Any]] = []
    for stage in ("raw", "causal", "fixed_decoder"):
        labels = predictions[f"predicted_room_{stage}"].astype(str)
        run_parts: list[np.ndarray] = []
        switches = 0
        for segment in segments:
            segment_labels = labels.iloc[segment].reset_index(drop=True)
            run_parts.append(_run_lengths(segment_labels, window_seconds))
            switches += int(segment_labels.ne(segment_labels.shift()).sum() - 1)
        runs = np.concatenate(run_parts) if run_parts else np.asarray([], dtype=float)
        rows.append(
            {
                "window_seconds": window_seconds,
                "stage": stage,
                "n_windows": int(len(predictions)),
                "n_predicted_classes": int(labels.nunique()),
                "n_contiguous_segments": int(len(segments)),
                "switches_within_segments": switches,
                "switches_per_elapsed_hour": float(switches / elapsed_hours),
                "switches_per_active_window_hour": float(switches / active_window_hours),
                "median_run_seconds": float(np.median(runs)),
                "mean_run_seconds": float(np.mean(runs)),
                "raw_confidence_mean": float(predictions["confidence_raw"].mean()),
                "raw_entropy_mean": float(predictions["entropy_raw"].mean()),
            }
        )
    return pd.DataFrame(rows)


def test_disagreement_tables(
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    disagree = aligned.loc[~aligned["agree_fixed_decoder"]]
    pairs = (
        disagree.groupby(
            ["predicted_room_fixed_decoder_1s", "predicted_room_fixed_decoder_3s"],
            observed=True,
        )
        .agg(
            count=("agree_fixed_decoder", "size"),
            confidence_1s_mean=("confidence_raw_1s", "mean"),
            confidence_3s_mean=("confidence_raw_3s", "mean"),
            raw_js_divergence_mean=("raw_js_divergence", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    pairs["share_of_all_disagreements"] = pairs["count"] / max(len(disagree), 1)

    by_class = (
        aligned.groupby("predicted_room_fixed_decoder_3s", observed=True)
        .agg(
            n_aligned=("agree_fixed_decoder", "size"),
            agreement_raw=("agree_raw", "mean"),
            agreement_causal=("agree_causal", "mean"),
            agreement_fixed_decoder=("agree_fixed_decoder", "mean"),
            confidence_1s_mean=("confidence_raw_1s", "mean"),
            confidence_3s_mean=("confidence_raw_3s", "mean"),
            raw_js_divergence_mean=("raw_js_divergence", "mean"),
        )
        .reset_index()
        .sort_values("n_aligned", ascending=False)
    )

    confidence_floor = aligned[["confidence_raw_1s", "confidence_raw_3s"]].min(axis=1)
    bins = pd.cut(
        confidence_floor,
        bins=[0.0, 0.3, 0.4, 0.5, 0.6, 1.01],
        include_lowest=True,
        right=False,
    )
    by_confidence = (
        aligned.assign(confidence_floor_bin=bins.astype(str))
        .groupby("confidence_floor_bin", observed=True)
        .agg(
            n_aligned=("agree_fixed_decoder", "size"),
            agreement_raw=("agree_raw", "mean"),
            agreement_causal=("agree_causal", "mean"),
            agreement_fixed_decoder=("agree_fixed_decoder", "mean"),
            raw_js_divergence_mean=("raw_js_divergence", "mean"),
        )
        .reset_index()
    )
    return pairs, by_class, by_confidence


def lodo_agreement(lodo_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    aligned_parts: list[pd.DataFrame] = []
    days_1s = {path.name for path in (lodo_dir / "1s").iterdir() if path.is_dir()}
    days_3s = {path.name for path in (lodo_dir / "3s").iterdir() if path.is_dir()}
    for day in sorted(days_1s & days_3s):
        p1 = pd.read_csv(lodo_dir / "1s" / day / "predictions.csv")
        p3 = pd.read_csv(lodo_dir / "3s" / day / "predictions.csv")
        p1 = p1.reset_index(names="index_1s")
        p3 = p3.reset_index(names="index_3s")
        p1["context_3s"] = pd.to_datetime(p1["window_ts"], utc=True).dt.floor("3s")
        p3["context_3s"] = pd.to_datetime(p3["window_ts"], utc=True)
        aligned = p1.merge(p3, on="context_3s", suffixes=("_1s", "_3s"), validate="many_to_one")
        aligned["same_reference"] = aligned["y_true_1s"].astype(str).eq(aligned["y_true_3s"].astype(str))
        aligned["test_day"] = day

        manifest_1 = json.loads((lodo_dir / "1s" / day / "feature_manifest.json").read_text(encoding="utf-8"))
        manifest_3 = json.loads((lodo_dir / "3s" / day / "feature_manifest.json").read_text(encoding="utf-8"))
        if manifest_1["classes"] != manifest_3["classes"]:
            raise AssertionError(f"LODO class order differs for {day}")
        raw_1 = np.load(lodo_dir / "1s" / day / "probabilities.npz")["raw"]
        raw_3 = np.load(lodo_dir / "3s" / day / "probabilities.npz")["raw"]
        aligned["raw_js_divergence"] = _js_divergence(
            raw_1[aligned["index_1s"].to_numpy(dtype=int)],
            raw_3[aligned["index_3s"].to_numpy(dtype=int)],
        )
        for stage, left_column, right_column in (
            ("raw", "y_pred_raw_1s", "y_pred_raw_3s"),
            ("fixed_decoder", "y_pred_fixed_decoder_1s", "y_pred_fixed_decoder_3s"),
        ):
            left = aligned[left_column].astype(str)
            right = aligned[right_column].astype(str)
            same_reference = aligned["same_reference"]
            truth = aligned.loc[same_reference, "y_true_1s"].astype(str)
            left_same = left[same_reference]
            right_same = right[same_reference]
            correct_1 = left_same.eq(truth)
            correct_3 = right_same.eq(truth)
            rows.append(
                {
                    "test_day": day,
                    "stage": stage,
                    "n_aligned": int(len(aligned)),
                    "same_reference_rate": float(same_reference.mean()),
                    "agreement_all": float(left.eq(right).mean()),
                    "agreement_same_reference": float(left_same.eq(right_same).mean()),
                    "cohen_kappa_same_reference": float(cohen_kappa_score(left_same, right_same)),
                    "both_correct_rate": float((correct_1 & correct_3).mean()),
                    "only_1s_correct_rate": float((correct_1 & ~correct_3).mean()),
                    "only_3s_correct_rate": float((~correct_1 & correct_3).mean()),
                    "both_wrong_rate": float((~correct_1 & ~correct_3).mean()),
                    "oracle_either_correct_rate": float((correct_1 | correct_3).mean()),
                    "mean_raw_js_divergence": float(aligned["raw_js_divergence"].mean()) if stage == "raw" else np.nan,
                }
            )
        aligned_parts.append(
            aligned[
                [
                    "test_day",
                    "window_ts_1s",
                    "window_ts_3s",
                    "y_true_1s",
                    "y_true_3s",
                    "y_pred_raw_1s",
                    "y_pred_raw_3s",
                    "y_pred_fixed_decoder_1s",
                    "y_pred_fixed_decoder_3s",
                    "confidence_raw_1s",
                    "confidence_raw_3s",
                    "same_reference",
                    "raw_js_divergence",
                ]
            ]
        )
    return pd.DataFrame(rows), pd.concat(aligned_parts, ignore_index=True)


def aggregate_lodo_agreement(by_fold: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "same_reference_rate",
        "agreement_all",
        "agreement_same_reference",
        "cohen_kappa_same_reference",
        "both_correct_rate",
        "only_1s_correct_rate",
        "only_3s_correct_rate",
        "both_wrong_rate",
        "oracle_either_correct_rate",
        "mean_raw_js_divergence",
    ]
    rows: list[dict[str, Any]] = []
    for stage, group in by_fold.groupby("stage", observed=True):
        row: dict[str, Any] = {"stage": stage, "n_folds": int(group["test_day"].nunique())}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std())
        rows.append(row)
    return pd.DataFrame(rows)


def lodo_class_complementarity(aligned: pd.DataFrame) -> pd.DataFrame:
    same = aligned.loc[aligned["same_reference"]].copy()
    truth = same["y_true_1s"].astype(str)
    correct_1 = same["y_pred_fixed_decoder_1s"].astype(str).eq(truth)
    correct_3 = same["y_pred_fixed_decoder_3s"].astype(str).eq(truth)
    same["both_correct"] = correct_1 & correct_3
    same["only_1s_correct"] = correct_1 & ~correct_3
    same["only_3s_correct"] = ~correct_1 & correct_3
    same["both_wrong"] = ~correct_1 & ~correct_3
    result = (
        same.groupby("y_true_1s", observed=True)
        .agg(
            n_aligned=("same_reference", "size"),
            both_correct_rate=("both_correct", "mean"),
            only_1s_correct_rate=("only_1s_correct", "mean"),
            only_3s_correct_rate=("only_3s_correct", "mean"),
            both_wrong_rate=("both_wrong", "mean"),
            confidence_1s_mean=("confidence_raw_1s", "mean"),
            confidence_3s_mean=("confidence_raw_3s", "mean"),
            raw_js_divergence_mean=("raw_js_divergence", "mean"),
        )
        .reset_index()
        .rename(columns={"y_true_1s": "room"})
    )
    result["window_advantage_3s_minus_1s"] = (
        result["only_3s_correct_rate"] - result["only_1s_correct_rate"]
    )
    return result.sort_values("n_aligned", ascending=False)


def lodo_performance_summary(lodo_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for window in (1, 3):
        for fold_dir in sorted(path for path in (lodo_dir / f"{window}s").iterdir() if path.is_dir()):
            metrics_path = fold_dir / "metrics.csv"
            if metrics_path.is_file():
                parts.append(pd.read_csv(metrics_path))
    metrics = pd.concat(parts, ignore_index=True)
    metrics = metrics.loc[metrics["region"].eq("all")]
    return (
        metrics.groupby(["window_seconds", "variant", "decoder"], observed=True)
        .agg(
            lodo_n_folds=("test_day", "nunique"),
            lodo_macro_f1_mean=("macro_f1", "mean"),
            lodo_macro_f1_std=("macro_f1", "std"),
            lodo_accuracy_mean=("accuracy", "mean"),
            lodo_balanced_accuracy_mean=("balanced_accuracy", "mean"),
        )
        .reset_index()
        .rename(columns={"decoder": "stage"})
    )


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """Render a compact Markdown table without the optional tabulate package."""

    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for values in frame.itertuples(index=False, name=None):
        cells: list[str] = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(value) else f"{float(value):.{float_digits}f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def create_plots(
    predictions_1s: pd.DataFrame,
    predictions_3s: pd.DataFrame,
    test_summary: pd.DataFrame,
    lodo_by_fold: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(
        set(predictions_1s["predicted_room_fixed_decoder"])
        | set(predictions_3s["predicted_room_fixed_decoder"])
    )
    label_to_y = {label: index for index, label in enumerate(labels)}
    figure, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for axis, predictions, window in (
        (axes[0], predictions_1s, 1),
        (axes[1], predictions_3s, 3),
    ):
        timestamps = pd.to_datetime(predictions["window_ts_local"])
        values = predictions["predicted_room_fixed_decoder"].map(label_to_y)
        axis.scatter(timestamps, values, s=5 if window == 1 else 9, alpha=0.75)
        axis.set_ylabel(f"{window}s")
        axis.grid(alpha=0.2)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=7)
    axes[1].set_yticks(range(len(labels)), labels, fontsize=7)
    axes[1].set_xlabel("Local timestamp")
    figure.suptitle("Unlabeled test predictions — fixed decoder")
    _save_figure(plot_dir / "test_prediction_timeline.png")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(test_summary["stage"], test_summary["agreement_rate"], color=["#3568a8", "#6a8f4e", "#d56b2d"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("1s–3s agreement rate")
    axis.set_title("Agreement on unlabeled test day (1s-frame weighted)")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(plot_dir / "test_agreement_by_stage.png")

    pivot = lodo_by_fold.pivot(index="test_day", columns="stage", values="agreement_same_reference")
    pivot.plot.bar(figsize=(8, 4.8), color=["#d56b2d", "#3568a8"])
    plt.ylim(0, 1)
    plt.ylabel("Agreement rate")
    plt.xlabel("Outer test day")
    plt.title("Strict LODO 1s–3s agreement when reference labels match")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(title="Stage", frameon=False)
    _save_figure(plot_dir / "lodo_agreement_by_fold.png")


def write_report(
    input_audit: dict[str, Any],
    training_audits: list[dict[str, Any]],
    diagnostics: pd.DataFrame,
    test_agreement: pd.DataFrame,
    lodo_summary: pd.DataFrame,
    top_disagreements: pd.DataFrame,
    agreement_by_confidence: pd.DataFrame,
    lodo_classes: pd.DataFrame,
    output_dir: Path,
) -> None:
    test_fixed = test_agreement.loc[test_agreement["stage"].eq("fixed_decoder")].iloc[0]
    test_raw = test_agreement.loc[test_agreement["stage"].eq("raw")].iloc[0]
    lodo_fixed = lodo_summary.loc[lodo_summary["stage"].eq("fixed_decoder")].iloc[0]
    lodo_raw = lodo_summary.loc[lodo_summary["stage"].eq("raw")].iloc[0]
    training_days = sorted(
        {
            str(day)
            for audit in training_audits
            for day in audit.get("training_days", [])
        }
    )
    training_day_text = ", ".join(training_days) if training_days else "not recorded"
    lines = [
        "# Strong ML inference: 1s M2 versus 3s M5",
        "",
        "## Input audit",
        "",
        f"- Source rows: {input_audit['source_rows']:,}",
        f"- Unique detections after exact deduplication: {input_audit['deduplicated_detection_rows']:,}",
        f"- Duplicate rows removed: {input_audit['exact_duplicate_rows_removed']:,}",
        f"- Local time range: {input_audit['timestamp_min_local']} to {input_audit['timestamp_max_local']}",
        f"- Timezone assumption: `{input_audit['timezone_assumption']}`",
        "- Ground truth is absent; test-day accuracy and F1 cannot be computed.",
        "",
        "## Final training",
        "",
        f"The final inference models are refit on all labeled training days ({training_day_text}). Because test labels are unavailable, the DASEL train/test class intersection cannot be applied without leakage; the final models use the training-label union. Strict shared-class filtering remains the evaluation protocol for LODO metrics.",
        "",
        "| Window | Variant | Train windows | Test windows | Features | Classes |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for audit in training_audits:
        lines.append(
            f"| {audit['window_seconds']}s | {audit['variant']} | {audit['n_train_windows']:,} | "
            f"{audit['n_test_windows']:,} | {audit['n_features']} | {audit['n_classes']} |"
        )
    lines.extend(
        [
            "",
            "## Unlabeled test comparison",
            "",
            f"Raw 1s–3s agreement is **{test_raw.agreement_rate:.1%}** (κ={test_raw.cohen_kappa:.3f}); fixed-decoder agreement is **{test_fixed.agreement_rate:.1%}** (κ={test_fixed.cohen_kappa:.3f}). These are agreement measures, not correctness.",
            f"Mean raw probability JS divergence is **{test_raw.mean_raw_js_divergence:.4f}** on the unlabeled test day versus **{lodo_raw.mean_raw_js_divergence_mean:.4f} ± {lodo_raw.mean_raw_js_divergence_std:.4f}** in strict LODO.",
            "",
            _markdown_table(diagnostics),
            "",
            "### Agreement by confidence floor",
            "",
            _markdown_table(agreement_by_confidence),
            "",
            "### Most frequent fixed-decoder disagreements",
            "",
            _markdown_table(top_disagreements.head(10)),
            "",
            "## Strict LODO agreement",
            "",
            f"Across outer folds, raw agreement on aligned frames with matching reference labels is **{lodo_raw.agreement_same_reference_mean:.1%} ± {lodo_raw.agreement_same_reference_std:.1%}**. Fixed decoding changes this to **{lodo_fixed.agreement_same_reference_mean:.1%} ± {lodo_fixed.agreement_same_reference_std:.1%}**.",
            f"The LODO oracle-either-correct rate after fixed decoding is **{lodo_fixed.oracle_either_correct_rate_mean:.1%}**, while 1s-only and 3s-only correct cases are **{lodo_fixed.only_1s_correct_rate_mean:.1%}** and **{lodo_fixed.only_3s_correct_rate_mean:.1%}**. This quantifies complementarity rather than assuming one window dominates every frame.",
            "",
            "### Class-level LODO complementarity",
            "",
            _markdown_table(
                lodo_classes.nlargest(8, "window_advantage_3s_minus_1s")[
                    [
                        "room",
                        "n_aligned",
                        "only_1s_correct_rate",
                        "only_3s_correct_rate",
                        "window_advantage_3s_minus_1s",
                        "both_wrong_rate",
                    ]
                ]
            ),
            "",
            "## Output interpretation",
            "",
            "- `predicted_room_raw` is the three-seed XGBoost output.",
            "- `predicted_room_causal` uses only current/past probabilities and is online-compatible.",
            "- `predicted_room_fixed_decoder` includes Viterbi backtracking and is the strongest offline output.",
            "- `recommended_room` uses agreement when available and otherwise selects the 3s fixed-decoder branch because it had the stronger strict-LODO mean. This is a deterministic recommendation, not a separately validated fusion model.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_inference(
    test_file: Path,
    train_features_dir: Path,
    lodo_artifact_dir: Path,
    config_path: Path,
    output_dir: Path,
    timezone: str = "Asia/Seoul",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized, detections, input_audit = load_unlabeled_detections(test_file, timezone)
    _write_json(output_dir / "input_audit.json", input_audit)

    predictions: dict[int, pd.DataFrame] = {}
    probability_tables: dict[int, pd.DataFrame] = {}
    training_audits: list[dict[str, Any]] = []
    config = load_strong_config(config_path)
    for window in (1, 3):
        prediction_path = output_dir / f"window_predictions_{window}s.csv"
        probability_path = output_dir / f"class_probabilities_{window}s.csv"
        row_path = output_dir / f"BLE_Test_predictions_{window}s.csv"
        submission_path = output_dir / f"submission_{window}s_fixed_decoder.csv"
        print(f"[{window}s] extracting unlabeled features", flush=True)
        test_features = extract_unlabeled_features(detections, window)
        feature_dir = output_dir / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        test_features.to_csv(feature_dir / f"rf_window_features_test_{window}s.csv", index=False)
        train_frame = load_feature_frame(train_features_dir, window)
        print(f"[{window}s] fitting final {config['windows'][str(window)]['variant']} ensemble", flush=True)
        window_predictions, probability_table, audit = train_final_and_predict(
            train_frame,
            test_features,
            window,
            config,
            output_dir,
        )
        predictions[window] = window_predictions
        probability_tables[window] = probability_table
        training_audits.append(audit)
        window_predictions.to_csv(prediction_path, index=False)
        probability_table.to_csv(probability_path, index=False)
        row_predictions = map_predictions_to_source_rows(normalized, window_predictions, window)
        row_predictions.to_csv(row_path, index=False)
        submission = row_predictions[["source_id", "predicted_room_fixed_decoder"]].rename(
            columns={"predicted_room_fixed_decoder": "room"}
        )
        submission.to_csv(submission_path, index=False)
        print(f"[{window}s] predicted {len(window_predictions):,} windows", flush=True)

    aligned_test = align_window_predictions(
        predictions[1], probability_tables[1], predictions[3], probability_tables[3]
    )
    test_summary = agreement_summary(aligned_test)
    diagnostics = pd.concat(
        [prediction_diagnostics(predictions[1], 1), prediction_diagnostics(predictions[3], 3)],
        ignore_index=True,
    )
    lodo_by_fold, lodo_aligned = lodo_agreement(lodo_artifact_dir)
    lodo_summary = aggregate_lodo_agreement(lodo_by_fold)
    lodo_performance = lodo_performance_summary(lodo_artifact_dir)
    top_disagreements, agreement_by_class, agreement_by_confidence = test_disagreement_tables(
        aligned_test
    )
    lodo_classes = lodo_class_complementarity(lodo_aligned)

    aligned_test.to_csv(output_dir / "test_1s_3s_aligned_agreement.csv", index=False)
    test_summary.to_csv(output_dir / "test_agreement_summary.csv", index=False)
    diagnostics.to_csv(output_dir / "test_prediction_diagnostics.csv", index=False)
    lodo_by_fold.to_csv(output_dir / "lodo_agreement_by_fold.csv", index=False)
    lodo_summary.to_csv(output_dir / "lodo_agreement_summary.csv", index=False)
    lodo_aligned.to_csv(output_dir / "lodo_aligned_predictions.csv", index=False)
    top_disagreements.to_csv(output_dir / "test_top_disagreements.csv", index=False)
    agreement_by_class.to_csv(output_dir / "test_agreement_by_3s_class.csv", index=False)
    agreement_by_confidence.to_csv(output_dir / "test_agreement_by_confidence.csv", index=False)
    lodo_classes.to_csv(output_dir / "lodo_class_complementarity.csv", index=False)
    final_comparison = diagnostics.merge(
        lodo_performance,
        on=["window_seconds", "stage"],
        how="left",
    )
    final_comparison.to_csv(output_dir / "FINAL_CONFIG_COMPARISON.csv", index=False)
    final_agreement = test_summary.merge(
        lodo_summary[
            [
                "stage",
                "agreement_same_reference_mean",
                "agreement_same_reference_std",
                "cohen_kappa_same_reference_mean",
                "cohen_kappa_same_reference_std",
                "only_1s_correct_rate_mean",
                "only_3s_correct_rate_mean",
                "oracle_either_correct_rate_mean",
            ]
        ],
        on="stage",
        how="left",
    )
    final_agreement.to_csv(output_dir / "FINAL_AGREEMENT_COMPARISON.csv", index=False)
    pd.concat(
        [
            predictions[1][["predicted_room_fixed_decoder"]].value_counts().rename("count").reset_index().assign(window_seconds=1),
            predictions[3][["predicted_room_fixed_decoder"]].value_counts().rename("count").reset_index().assign(window_seconds=3),
        ],
        ignore_index=True,
    ).to_csv(output_dir / "test_predicted_class_distribution.csv", index=False)
    create_plots(predictions[1], predictions[3], test_summary, lodo_by_fold, output_dir)
    write_report(
        input_audit,
        training_audits,
        diagnostics,
        test_summary,
        lodo_summary,
        top_disagreements,
        agreement_by_confidence,
        lodo_classes,
        output_dir,
    )
    _write_json(
        output_dir / "run_manifest.json",
        {
            "test_file": test_file.name,
            "private_source_paths_published": False,
            "windows": [1, 3],
            "config": config,
            "input_audit": input_audit,
            "training_audits": training_audits,
            "test_outputs_have_ground_truth": False,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--train-features-dir", type=Path, required=True)
    parser.add_argument(
        "--lodo-artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "strong_ml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "strong_ml_1s_3s.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timezone", default="Asia/Seoul")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_inference(
        test_file=args.test_file,
        train_features_dir=args.train_features_dir,
        lodo_artifact_dir=args.lodo_artifact_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        timezone=args.timezone,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

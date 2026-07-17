"""Executable Baseline ML pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .config import load_config, project_root, resolve_path


RESULT_COLUMNS = (
    "test_day",
    "model",
    "class_protocol",
    "closed_set",
    "n_train_raw",
    "n_test_raw",
    "n_train_classes_raw",
    "n_test_classes_raw",
    "n_common_classes",
    "n_train_dropped",
    "n_test_dropped",
    "n_train_only_classes",
    "n_test_only_classes",
    "n_train",
    "n_test",
    "n_classes",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "n_candidate_features",
    "n_constant_removed",
    "n_duplicate_removed",
    "n_final_features",
)

PREDICTION_COLUMNS = ("test_day", "y_true", "y_pred")

CLASS_AUDIT_COLUMNS = (
    "test_day",
    "class_protocol",
    "n_train_raw",
    "n_test_raw",
    "n_train_classes_raw",
    "n_test_classes_raw",
    "n_common_classes",
    "n_train_dropped",
    "n_test_dropped",
    "n_train_only_classes",
    "n_test_only_classes",
)

SUMMARY_COLUMNS = (
    "model",
    "mean_macro_f1",
    "std_macro_f1",
    "mean_accuracy",
    "mean_balanced_accuracy",
    "mean_weighted_f1",
    "total_test",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    root = project_root()
    return {
        "ble": resolve_path(config["raw"]["ble_path"], root),
        "labels": resolve_path(config["raw"]["label_path"], root),
        "aligned": resolve_path(config["processed"]["aligned_cache_path"], root),
    }


def _drop_export_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove CSV index columns without touching domain columns."""
    return df.drop(
        columns=[col for col in df.columns if str(col).strip().startswith("Unnamed")],
        errors="ignore",
    )


def _canonical_user_ids(values: pd.Series) -> pd.Series:
    """Create comparable user keys while treating values such as 1 and 1.0 alike."""
    text = values.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    numeric_mask = numeric.notna()
    text.loc[numeric_mask] = numeric.loc[numeric_mask].map(lambda value: f"{value:.15g}")
    return text


def _alignment_group_columns(
    ble: pd.DataFrame,
    labels: pd.DataFrame,
    user_col: str,
    date_col: str,
) -> list[str]:
    """Choose safe grouping keys for raw-to-label interval alignment."""
    if user_col not in ble.columns or user_col not in labels.columns:
        present = ble if user_col in ble.columns else labels
        if user_col in present.columns and present[user_col].nunique(dropna=True) > 1:
            raise ValueError("Cannot safely align multi-user raw data without user IDs on both inputs")
        return [date_col]

    ble["_alignment_user"] = _canonical_user_ids(ble[user_col])
    labels["_alignment_user"] = _canonical_user_ids(labels[user_col])
    ble_users = set(ble["_alignment_user"].dropna())
    label_users = set(labels["_alignment_user"].dropna())

    # The source dataset intentionally uses different external IDs for its one
    # BLE stream and one label stream. With one stream on each side, day/time is
    # unambiguous. Multi-user data must have matching IDs to avoid cross-user labels.
    if len(ble_users) <= 1 and len(label_users) <= 1:
        return [date_col]
    if ble_users != label_users:
        raise ValueError(
            "Cannot safely align multi-user raw data: BLE and label user IDs do not match"
        )
    return [date_col, "_alignment_user"]


def _build_aligned_from_raw(
    config: dict[str, Any],
    nrows: int | None = None,
) -> pd.DataFrame:
    """Build the canonical aligned table from private raw BLE and label CSVs."""
    paths = _paths(config)
    missing_sources = [name for name in ("ble", "labels") if not paths[name].is_file()]
    if missing_sources:
        raise FileNotFoundError(
            "Aligned cache is missing and raw fallback inputs are unavailable: "
            + ", ".join(missing_sources)
        )

    processed = config["processed"]
    processing = config.get("processing", {})
    time_col = processed["time_col"]
    date_col = processed["date_col"]
    user_col = processed["user_col"]
    label_col = processed["label_col"]
    start_col = processing.get("label_start_col", "started_at")
    end_col = processing.get("label_end_col", "finished_at")
    max_gap_minutes = int(processing.get("label_max_gap_minutes", 30))
    metadata_cols = list(processing.get("label_metadata_cols", [label_col, "floor"]))
    metadata_cols = list(dict.fromkeys([label_col, *metadata_cols]))

    ble = _drop_export_index_columns(_read_csv(paths["ble"], nrows=nrows))
    labels = _drop_export_index_columns(_read_csv(paths["labels"]))

    missing_ble = [col for col in (time_col, user_col) if col not in ble.columns]
    missing_labels = [
        col for col in (start_col, end_col, label_col) if col not in labels.columns
    ]
    if missing_ble or missing_labels:
        raise KeyError(
            f"Raw alignment schema mismatch; BLE missing {missing_ble}, "
            f"labels missing {missing_labels}"
        )

    ble[time_col] = pd.to_datetime(ble[time_col], errors="coerce")
    ble = ble.dropna(subset=[time_col, user_col]).copy()
    ble[date_col] = ble[time_col].dt.strftime("%Y-%m-%d")

    labels[start_col] = pd.to_datetime(labels[start_col], errors="coerce")
    labels[end_col] = pd.to_datetime(labels[end_col], errors="coerce")
    labels[label_col] = labels[label_col].astype("string").str.strip()
    labels = labels.dropna(subset=[start_col, label_col]).copy()
    labels[date_col] = labels[start_col].dt.strftime("%Y-%m-%d")

    group_cols = _alignment_group_columns(ble, labels, user_col, date_col)
    labels = labels.sort_values([*group_cols, start_col], kind="stable")
    next_start = labels.groupby(group_cols, dropna=False)[start_col].shift(-1)
    valid_finish = labels[end_col].notna() & (labels[end_col] > labels[start_col])
    interval_end = labels[end_col].where(valid_finish, next_start)
    maximum_end = labels[start_col] + pd.to_timedelta(max_gap_minutes, unit="m")
    labels["_interval_end"] = pd.concat(
        [interval_end.rename("candidate"), maximum_end.rename("maximum")], axis=1
    ).min(axis=1)
    labels = labels[labels["_interval_end"] > labels[start_col]].copy()

    available_metadata = [col for col in metadata_cols if col in labels.columns]
    # Raw exports should not carry labels, but drop any conflicting placeholders
    # so the aligned table always exposes the configured label column directly.
    ble = ble.drop(columns=[col for col in available_metadata if col in ble.columns])

    interval_groups = {
        key: group
        for key, group in labels.groupby(group_cols, sort=False, dropna=False)
    }
    aligned_parts: list[pd.DataFrame] = []
    for key, ble_group in ble.groupby(group_cols, sort=True, dropna=False):
        interval_group = interval_groups.get(key)
        if interval_group is None or interval_group.empty:
            continue
        left = ble_group.sort_values(time_col, kind="stable")
        right_cols = list(
            dict.fromkeys([start_col, end_col, "_interval_end", *available_metadata])
        )
        right = interval_group[right_cols].sort_values(start_col, kind="stable")
        matched = pd.merge_asof(
            left,
            right,
            left_on=time_col,
            right_on=start_col,
            direction="backward",
            allow_exact_matches=True,
        )
        matched = matched[
            matched[start_col].notna()
            & (matched[time_col] >= matched[start_col])
            & (matched[time_col] <= matched["_interval_end"])
        ]
        if not matched.empty:
            aligned_parts.append(matched)

    if not aligned_parts:
        raise ValueError("Raw BLE and label intervals produced no aligned rows")
    aligned = pd.concat(aligned_parts, ignore_index=True)
    aligned = aligned.drop(columns=["_alignment_user", "_interval_end"], errors="ignore")
    if "mac address" in aligned.columns and "mac_address" not in aligned.columns:
        aligned = aligned.rename(columns={"mac address": "mac_address"})
    return aligned


def stage_load(config: dict[str, Any], out_dir: Path, nrows: int | None = None) -> dict[str, Any]:
    paths = _paths(config)
    audit: dict[str, Any] = {"files": {}}
    sample_rows = 5 if nrows is None else nrows
    for name, path in paths.items():
        exists = path.exists()
        item: dict[str, Any] = {"path": str(path), "exists": exists}
        if exists:
            df = _read_csv(path, nrows=sample_rows)
            item["sample_shape"] = list(df.shape)
            item["columns"] = list(df.columns)
            item["size_bytes"] = path.stat().st_size
        audit["files"][name] = item
    _write_json(out_dir / "load_audit.json", audit)
    return audit


def stage_process(config: dict[str, Any], out_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    aligned_path = _paths(config)["aligned"]
    built_from_raw = False
    if aligned_path.is_file():
        df = _read_csv(aligned_path, nrows=nrows)
    elif config.get("processing", {}).get("build_aligned_if_missing", False):
        df = _build_aligned_from_raw(config, nrows=nrows)
        built_from_raw = True
        if nrows is None:
            aligned_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(aligned_path, index=False)
    else:
        raise FileNotFoundError(
            f"Aligned cache not found: {aligned_path}. "
            "Enable processing.build_aligned_if_missing to build it from raw inputs."
        )
    processed = config["processed"]
    time_col = processed["time_col"]
    date_col = processed["date_col"]
    user_col = processed["user_col"]
    label_col = processed["label_col"]

    required = [time_col, user_col, label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Required columns not found in aligned cache: {missing}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    invalid_timestamps = int(df[time_col].isna().sum())
    derived_dates = df[time_col].dt.strftime("%Y-%m-%d")
    if date_col not in df.columns:
        df[date_col] = derived_dates
    else:
        df[date_col] = df[date_col].where(df[date_col].notna(), derived_dates)

    audit = {
        "source": str(aligned_path),
        "built_from_raw": built_from_raw,
        "shape": list(df.shape),
        "invalid_timestamps": invalid_timestamps,
        "date_counts": df[date_col].dropna().astype(str).value_counts().sort_index().to_dict(),
        "n_labels": int(df[label_col].nunique(dropna=True)),
    }
    _write_json(out_dir / "process_audit.json", audit)
    return df


def _rssi_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    if not cols:
        raise KeyError(f"No RSSI columns found with prefix '{prefix}'")
    return cols


def _majority_then_latest(values: pd.Series) -> object:
    """Assign a window label by majority, breaking ties with the latest sample."""
    counts = values.value_counts(dropna=False)
    top_count = counts.max()
    tied = set(counts[counts == top_count].index)
    for value in reversed(values.tolist()):
        if value in tied:
            return value
    raise ValueError("Cannot assign a label to an empty window")


def stage_extract(
    config: dict[str, Any],
    out_dir: Path,
    nrows: int | None = None,
    processed_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    processed = config["processed"]
    features = config["features"]
    df = (
        processed_df.copy()
        if processed_df is not None
        else stage_process(config, out_dir / "process", nrows=nrows)
    )

    time_col = processed["time_col"]
    date_col = processed["date_col"]
    user_col = processed["user_col"]
    label_col = processed["label_col"]
    rssi_cols = _rssi_columns(df, features["rssi_prefix"])

    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found")
    df = df.dropna(subset=[time_col, date_col, user_col, label_col]).copy()
    # Pandas' ``last`` aggregation follows input order, so make it chronological.
    df = df.sort_values(time_col, kind="stable")

    for col in rssi_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if features.get("replace_zero_rssi", True):
            df.loc[df[col] == 0, col] = features["missing_rssi"]
        df[col] = df[col].fillna(features["missing_rssi"])

    window_seconds = int(features["window_seconds"])
    df["window_start"] = df[time_col].dt.floor(f"{window_seconds}s")

    group_cols = [date_col, user_col, "window_start"]
    aggregations = list(features["aggregations"])
    grouped = df.groupby(group_cols, dropna=False)[rssi_cols].agg(aggregations)
    grouped.columns = [f"{col}_{agg}" for col, agg in grouped.columns]
    window_labels = df.groupby(group_cols, dropna=False)[label_col].agg(
        _majority_then_latest
    )
    feature_df = grouped.join(window_labels).reset_index()
    metadata_cols = [date_col, user_col, label_col, "window_start"]
    feature_df = feature_df[
        metadata_cols + [col for col in feature_df.columns if col not in metadata_cols]
    ]
    feature_df = feature_df.fillna(0)

    mean_cols = [c for c in feature_df.columns if c.endswith("_mean")]
    if mean_cols:
        mean_values = feature_df[mean_cols]
        feature_df["signal_mean"] = mean_values.mean(axis=1)
        feature_df["signal_max"] = mean_values.max(axis=1)
        feature_df["signal_min"] = mean_values.min(axis=1)
        feature_df["visible_beacon_count"] = (mean_values > features["missing_rssi"]).sum(axis=1)

    if features.get("include_time_features", False):
        feature_df["hour"] = pd.to_datetime(feature_df["window_start"]).dt.hour
        feature_df["minute"] = pd.to_datetime(feature_df["window_start"]).dt.minute

    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / f"baseline_{window_seconds}s_features.csv"
    schema_path = out_dir / f"baseline_{window_seconds}s_feature_schema.json"
    feature_df.to_csv(feature_path, index=False)
    _write_json(
        schema_path,
        {
            "source": config["processed"]["aligned_cache_path"],
            "feature_path": str(feature_path),
            "shape": list(feature_df.shape),
            "window_seconds": window_seconds,
            "rssi_columns": rssi_cols,
            "aggregations": aggregations,
            "label_assignment": "majority_then_latest",
        },
    )
    return feature_df


def _make_model(config: dict[str, Any], n_classes: int):
    training = config["training"]
    model_name = str(training["model"]).lower()
    seed = int(training.get("random_state", 42))
    if model_name == "xgb":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise ImportError("xgboost is required for training.model='xgb'") from exc
        params = dict(training["xgb_params"])
        params.update(
            {
                "objective": "multi:softprob",
                "num_class": n_classes,
                "random_state": seed,
                "n_jobs": -1,
            }
        )
        return XGBClassifier(**params)

    if model_name == "rf":
        params = dict(training["rf_params"])
        params.update({"random_state": seed, "n_jobs": -1})
        return RandomForestClassifier(**params)

    raise ValueError(f"Unsupported model: {model_name}")


def _feature_columns(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    excluded = {
        config["processed"]["date_col"],
        config["processed"]["user_col"],
        config["processed"]["label_col"],
        "window_start",
    }
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _drop_bad_features(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    original_cols = list(train_x.columns)
    nunique = train_x.nunique(dropna=False)
    keep = [col for col in original_cols if nunique[col] > 1]
    train_x = train_x[keep]
    test_x = test_x[keep]

    duplicate_mask = train_x.T.duplicated()
    duplicate_cols = list(train_x.columns[duplicate_mask])
    if duplicate_cols:
        train_x = train_x.drop(columns=duplicate_cols)
        test_x = test_x.drop(columns=duplicate_cols)

    return train_x, test_x, {
        "n_candidate_features": len(original_cols),
        "n_constant_removed": len(original_cols) - len(keep),
        "n_duplicate_removed": len(duplicate_cols),
        "n_final_features": train_x.shape[1],
    }


def _apply_class_protocol(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    class_protocol: str,
    test_day: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Apply and audit the fold class policy before any model-side fitting.

    ``dasel_intersection`` reproduces the class-count interpretation in DASEL
    Table II: the model class set is the intersection of labels observed in
    the outer-training days and the held-out day. Both sides are filtered to
    that same set before feature cleaning, imputation, weighting, or fitting.
    """
    train_labels = set(train_df[label_col].astype(str))
    test_labels = set(test_df[label_col].astype(str))
    common = train_labels & test_labels
    audit = {
        "n_train_raw": int(len(train_df)),
        "n_test_raw": int(len(test_df)),
        "n_train_classes_raw": int(len(train_labels)),
        "n_test_classes_raw": int(len(test_labels)),
        "n_common_classes": int(len(common)),
        "n_train_only_classes": int(len(train_labels - test_labels)),
        "n_test_only_classes": int(len(test_labels - train_labels)),
    }

    if class_protocol == "dasel_intersection":
        train_df = train_df[train_df[label_col].astype(str).isin(common)].copy()
        test_df = test_df[test_df[label_col].astype(str).isin(common)].copy()
        filtered_train_labels = set(train_df[label_col].astype(str))
        filtered_test_labels = set(test_df[label_col].astype(str))
        if filtered_train_labels != common or filtered_test_labels != common:
            raise AssertionError(
                f"LODO fold '{test_day}' did not preserve the shared class set"
            )
    elif class_protocol == "train_label_space":
        unseen_labels = test_labels - train_labels
        if unseen_labels:
            raise ValueError(
                f"LODO fold '{test_day}' has {len(unseen_labels)} test label(s) "
                "absent from training while class_protocol=train_label_space"
            )
    else:
        raise ValueError(
            "Unsupported training.class_protocol: "
            f"{class_protocol!r}; expected 'dasel_intersection' or "
            "'train_label_space'"
        )

    audit["n_train_dropped"] = audit["n_train_raw"] - int(len(train_df))
    audit["n_test_dropped"] = audit["n_test_raw"] - int(len(test_df))
    return train_df, test_df, audit


def audit_lodo_classes(
    features_df: pd.DataFrame,
    date_col: str,
    label_col: str,
    class_protocol: str = "dasel_intersection",
) -> pd.DataFrame:
    """Return a row-count/class-count audit without fitting a model."""
    rows: list[dict[str, Any]] = []
    days = sorted(features_df[date_col].dropna().astype(str).unique())
    for test_day in days:
        train_df = features_df[features_df[date_col].astype(str) != test_day].copy()
        test_df = features_df[features_df[date_col].astype(str) == test_day].copy()
        _, _, audit = _apply_class_protocol(
            train_df,
            test_df,
            label_col,
            class_protocol,
            test_day,
        )
        rows.append(
            {
                "test_day": test_day,
                "class_protocol": class_protocol,
                **audit,
            }
        )
    return pd.DataFrame(rows, columns=CLASS_AUDIT_COLUMNS)


def train_lodo(config: dict[str, Any], features_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    processed = config["processed"]
    label_col = processed["label_col"]
    date_col = processed["date_col"]
    closed_set = bool(config["training"].get("closed_set", True))
    class_protocol = config["training"].get(
        "class_protocol",
        "dasel_intersection" if closed_set else "train_label_space",
    )
    feature_cols = _feature_columns(features_df, config)
    if not feature_cols:
        raise ValueError("No numeric feature columns are available for training")
    days = sorted(features_df[date_col].dropna().astype(str).unique())

    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for test_day in days:
        train_df = features_df[features_df[date_col].astype(str) != test_day].copy()
        test_df = features_df[features_df[date_col].astype(str) == test_day].copy()
        train_df, test_df, class_audit = _apply_class_protocol(
            train_df,
            test_df,
            label_col,
            str(class_protocol),
            test_day,
        )

        if train_df.empty or test_df.empty:
            continue

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(train_df[label_col].astype(str))
        y_test = encoder.transform(test_df[label_col].astype(str))

        train_x = train_df[feature_cols].copy()
        test_x = test_df[feature_cols].copy()
        train_x, test_x, feature_audit = _drop_bad_features(train_x, test_x)
        if train_x.shape[1] == 0:
            raise ValueError(f"No usable features remain for LODO fold '{test_day}'")

        model = _make_model(config, n_classes=len(encoder.classes_))
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
        pipe.fit(train_x, y_train)
        y_pred = pipe.predict(test_x)

        labels = list(range(len(encoder.classes_)))
        row = {
            "test_day": test_day,
            "model": config["training"]["model"],
            "class_protocol": class_protocol,
            "closed_set": closed_set,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_classes": int(len(encoder.classes_)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "macro_precision": float(precision_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0)),
        }
        row.update(class_audit)
        row.update(feature_audit)
        rows.append(row)

        fold_preds = pd.DataFrame(
            {
                "test_day": test_day,
                "y_true": encoder.inverse_transform(y_test),
                "y_pred": encoder.inverse_transform(y_pred.astype(int)),
            }
        )
        predictions.append(fold_preds)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_df.to_csv(out_dir / "lodo_results.csv", index=False)
    class_audit_df = audit_lodo_classes(
        features_df,
        date_col,
        label_col,
        str(class_protocol),
    )
    class_audit_df.to_csv(out_dir / "lodo_class_audit.csv", index=False)
    if predictions:
        predictions_df = pd.concat(predictions, ignore_index=True)
    else:
        predictions_df = pd.DataFrame(columns=PREDICTION_COLUMNS)
    predictions_df.to_csv(out_dir / "lodo_predictions.csv", index=False)

    if not results_df.empty:
        summary = pd.DataFrame(
            [
                {
                    "model": config["training"]["model"],
                    "mean_macro_f1": float(results_df["macro_f1"].mean()),
                    "std_macro_f1": float(results_df["macro_f1"].std()),
                    "mean_accuracy": float(results_df["accuracy"].mean()),
                    "mean_balanced_accuracy": float(results_df["balanced_accuracy"].mean()),
                    "mean_weighted_f1": float(results_df["weighted_f1"].mean()),
                    "total_test": int(results_df["n_test"].sum()),
                }
            ],
            columns=SUMMARY_COLUMNS,
        )
    else:
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    summary.to_csv(out_dir / "lodo_summary.csv", index=False)
    return results_df


def stage_train(config: dict[str, Any], out_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    features_dir = out_dir / "features"
    window_seconds = int(config["features"]["window_seconds"])
    feature_path = features_dir / f"baseline_{window_seconds}s_features.csv"
    if feature_path.exists() and nrows is None:
        features_df = pd.read_csv(feature_path)
    else:
        features_df = stage_extract(config, features_dir, nrows=nrows)
    return train_lodo(config, features_df, out_dir / "training")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ISAS26 Baseline ML pipeline.")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument(
        "--stage",
        choices=["load", "process", "extract", "train", "all"],
        default="all",
    )
    parser.add_argument("--output-dir", default="artifacts/baseline_run")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional smoke-test row limit.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_df: pd.DataFrame | None = None
    features_df: pd.DataFrame | None = None

    if args.stage in ("load", "all"):
        audit = stage_load(config, out_dir / "load", nrows=args.limit_rows)
        print(json.dumps(audit, indent=2, ensure_ascii=False)[:4000])
    if args.stage in ("process", "all"):
        processed_df = stage_process(config, out_dir / "process", nrows=args.limit_rows)
        print(f"processed shape: {processed_df.shape}")
    if args.stage in ("extract", "all"):
        features_df = stage_extract(
            config,
            out_dir / "features",
            nrows=args.limit_rows,
            processed_df=processed_df,
        )
        print(f"features shape: {features_df.shape}")
    if args.stage in ("train", "all"):
        if features_df is not None:
            results_df = train_lodo(config, features_df, out_dir / "training")
        else:
            results_df = stage_train(config, out_dir, nrows=args.limit_rows)
        print(results_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit or train the DASEL-matched 1s/3s feature matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baseline_ml.config import load_config  # noqa: E402
from baseline_ml.pipeline import audit_lodo_classes, train_lodo  # noqa: E402


def _feature_path(features_dir: Path, window: int) -> Path:
    return features_dir / f"rf_window_features_{window}s.csv"


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing private feature matrix: {path}")
    frame = pd.read_csv(path)
    required = {"date", "room", "window_ts"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Feature matrix {path.name} is missing columns: {missing}")
    frame["date"] = frame["date"].astype(str)
    frame["room"] = frame["room"].astype(str)
    frame["window_ts"] = pd.to_datetime(frame["window_ts"], errors="raise", utc=True)
    return frame.sort_values(["date", "window_ts"], kind="stable").reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run auditable DASEL-intersection LODO for 1s and 3s windows."
    )
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "dasel_1s_3s")
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 3])
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write class/row audits without fitting any model.",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_parts: list[pd.DataFrame] = []
    result_parts: list[pd.DataFrame] = []
    for window in args.windows:
        config_path = ROOT / "configs" / f"dasel_{window}s.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"No DASEL config for {window}s: {config_path}")
        config = load_config(config_path)
        frame = _load_frame(_feature_path(args.features_dir, window))
        processed = config["processed"]
        protocol = str(config["training"]["class_protocol"])
        audit = audit_lodo_classes(
            frame,
            str(processed["date_col"]),
            str(processed["label_col"]),
            protocol,
        )
        audit.insert(0, "window_seconds", window)
        audit_parts.append(audit)

        window_dir = args.output_dir / f"{window}s"
        window_dir.mkdir(parents=True, exist_ok=True)
        audit.to_csv(window_dir / "lodo_class_audit.csv", index=False)
        if not args.audit_only:
            results = train_lodo(config, frame, window_dir / "training")
            results.insert(0, "window_seconds", window)
            result_parts.append(results)

    all_audits = pd.concat(audit_parts, ignore_index=True)
    all_audits.to_csv(args.output_dir / "class_protocol_audit.csv", index=False)
    print(all_audits.to_string(index=False))

    if result_parts:
        results = pd.concat(result_parts, ignore_index=True)
        results.to_csv(args.output_dir / "lodo_results_1s_3s.csv", index=False)
        summary = (
            results.groupby("window_seconds", as_index=False)
            .agg(
                mean_macro_f1=("macro_f1", "mean"),
                std_macro_f1=("macro_f1", "std"),
                mean_accuracy=("accuracy", "mean"),
                mean_balanced_accuracy=("balanced_accuracy", "mean"),
                folds=("test_day", "nunique"),
            )
        )
        summary.to_csv(args.output_dir / "lodo_summary_1s_3s.csv", index=False)
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

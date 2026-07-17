"""Scan existing project outputs and create a compact best-result report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


INCLUDE_MARKERS = (
    "summary",
    "aggregate",
    "comparison",
    "lodo_model_comparison",
    "compact_mean_metrics",
)

EXCLUDE_MARKERS = (
    "classification_report",
    "confusion",
    "predictions",
    "lodo_results",
    "history",
    "diagnostic",
    "diagnostics",
    "feature_importance",
    "feature_filter",
    "coverage",
    "distribution",
)

META_COLUMNS = (
    "window",
    "feature_set",
    "variant",
    "model",
    "stage",
    "inference",
    "top_k",
    "config",
    "test_day",
)


def _flatten_multi_columns(columns: Iterable[tuple[object, object]]) -> list[str]:
    names: list[str] = []
    for first, second in columns:
        left = str(first).strip()
        right = str(second).strip()
        if not right or right.startswith("Unnamed"):
            names.append(left)
        else:
            names.append(f"{left}_{right}")
    return names


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read normal CSVs and two-row summary CSVs with mean/std headers."""
    try:
        single = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    lower_cols = [str(c).lower() for c in single.columns]
    has_numeric_macro = any(
        "macro_f1" in c and pd.to_numeric(single.iloc[:, i], errors="coerce").notna().any()
        for i, c in enumerate(lower_cols)
    )
    if has_numeric_macro:
        return single

    try:
        multi = pd.read_csv(path, header=[0, 1])
        multi.columns = _flatten_multi_columns(multi.columns)
        return multi
    except Exception:
        return single


def candidate_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*.csv"):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            relative_parts = path.parts
        if relative_parts and relative_parts[0].lower() == "reports":
            continue
        if (
            len(relative_parts) >= 2
            and relative_parts[0].lower() == "artifacts"
            and relative_parts[1].lower() != "best_results"
        ):
            continue
        text = str(path).lower()
        if not any(marker in text for marker in INCLUDE_MARKERS):
            continue
        if any(marker in text for marker in EXCLUDE_MARKERS):
            continue
        paths.append(path)
    return sorted(paths)


def collect_result_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in candidate_paths(root):
        df = read_csv_flexible(path)
        if df.empty:
            continue

        metric_cols = []
        for col in df.columns:
            lower = str(col).lower()
            if "macro_f1" in lower and "std" not in lower and "delta" not in lower:
                metric_cols.append(col)

        for metric_col in metric_cols:
            values = pd.to_numeric(df[metric_col], errors="coerce")
            if values.notna().sum() == 0:
                continue
            idx = values.idxmax()
            value = float(values.loc[idx])
            if value < 0 or value > 1.000001:
                continue

            row: dict[str, object] = {
                "mean_macro_f1_candidate": value,
                "metric_column": str(metric_col),
                "source_path": path.relative_to(root).as_posix(),
            }
            for meta_col in META_COLUMNS:
                if meta_col in df.columns:
                    row[meta_col] = str(df.loc[idx, meta_col])

            for extra_col in (
                "mean_macro_f1",
                "std_macro_f1",
                "mean_accuracy",
                "mean_balanced_accuracy",
                "mean_weighted_f1",
                "accuracy_mean",
                "balanced_accuracy_mean",
                "weighted_f1_mean",
                "macro_f1_mean",
            ):
                if extra_col in df.columns:
                    parsed = pd.to_numeric(pd.Series([df.loc[idx, extra_col]]), errors="coerce")
                    row[extra_col] = (
                        float(parsed.iloc[0]) if parsed.notna().iloc[0] else str(df.loc[idx, extra_col])
                    )
            rows.append(row)

    rows.sort(key=lambda item: (-float(item["mean_macro_f1_candidate"]), _source_priority(item)))
    return _dedupe_rows(rows)


def _source_priority(row: dict[str, object]) -> tuple[int, str]:
    source = str(row.get("source_path", "")).lower()
    if "summary_best" in source:
        return (0, source)
    if "overall_summary" in source or "aggregate" in source:
        return (1, source)
    if "summary" in source:
        return (2, source)
    if "reference_comparison" in source:
        return (3, source)
    return (4, source)


def _dedupe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[object, ...]] = set()
    unique: list[dict[str, object]] = []
    for row in rows:
        key = (
            row.get("source_path"),
            row.get("metric_column"),
            row.get("model"),
            row.get("stage"),
            row.get("feature_set"),
            row.get("variant"),
            row.get("config"),
            row.get("top_k"),
            row.get("window"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def write_reports(rows: list[dict[str, object]], root: Path) -> tuple[Path, Path]:
    reports_dir = root / "reports"
    docs_dir = root / "docs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "best_results_summary.csv"
    md_path = docs_dir / "RESULTS.md"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    top = rows[:10]
    strict_path = root / "reports" / "dasel_matched_1s_3s" / "summary.csv"
    strict_lines: list[str] = []
    if strict_path.is_file():
        strict = pd.read_csv(strict_path)
        required = {
            "window_seconds",
            "mean_macro_f1",
            "std_macro_f1",
            "mean_accuracy",
            "mean_balanced_accuracy",
        }
        if required.issubset(strict.columns):
            strict_lines = [
                "## DASEL-matched 1s/3s confirmatory controls",
                "",
                "Both outer-train and held-out rows are filtered to their shared room",
                "class set before model-side fitting. These are raw single-seed controls",
                "without augmentation, visit weighting, top-100 selection, or decoding.",
                "",
                "| Window | Macro-F1 mean | SD | Accuracy mean | Balanced accuracy mean |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for row in strict.sort_values("window_seconds").itertuples(index=False):
                strict_lines.append(
                    f"| {int(row.window_seconds)}s | {float(row.mean_macro_f1):.4f} | "
                    f"{float(row.std_macro_f1):.4f} | {float(row.mean_accuracy):.4f} | "
                    f"{float(row.mean_balanced_accuracy):.4f} |"
                )
            strict_lines.extend(
                [
                    "",
                    "The class-count audit matches DASEL Table II (`12, 15, 18, 13`).",
                    "Legacy M0-M8 scores are archived separately and are not class-policy",
                    "equivalent confirmatory results. See `docs/DASEL_PROTOCOL_1S_3S.md`.",
                    "",
                ]
            )

    lines = [
        "# Results",
        "",
        *strict_lines,
        "This file is generated by `python scripts/summarize_results.py`.",
        "",
        "The indexed files are historical reviewed aggregate artifacts. They are not",
        "an exact reproduction of the corrected, leakage-safe public CLI.",
        "",
    ]
    if rows:
        best = rows[0]
        lines.extend(
            [
                "## Best Recorded Result",
                "",
                f"- Macro-F1 candidate: `{float(best['mean_macro_f1_candidate']):.6f}`",
                f"- Source: `{best['source_path']}`",
                f"- Model: `{best.get('model', 'n/a')}`",
                f"- Stage/inference: `{best.get('stage', best.get('inference', 'n/a'))}`",
                f"- Feature set/variant: `{best.get('feature_set', best.get('variant', 'n/a'))}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Top Candidates",
            "",
            "| Rank | Macro-F1 | Model | Stage/Inference | Feature/Variant | Source |",
            "| ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for i, row in enumerate(top, start=1):
        stage = row.get("stage", row.get("inference", ""))
        feature = row.get("feature_set", row.get("variant", ""))
        lines.append(
            "| {rank} | {score:.6f} | `{model}` | `{stage}` | `{feature}` | `{source}` |".format(
                rank=i,
                score=float(row["mean_macro_f1_candidate"]),
                model=row.get("model", ""),
                stage=stage,
                feature=feature,
                source=row.get("source_path", ""),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation note: tuned closed-set results are diagnostic unless a separate",
            "nested validation or holdout split is used.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path

"""Build publishable aggregate plots for the BLE data and model audit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "project_analysis"
MODEL_DIR = ROOT / "reports" / "strong_ml_1s_3s"
DECODER_ORDER = ["raw", "causal_smoothing", "fixed_decoder"]
DECODER_LABELS = {
    "raw": "Raw model",
    "causal_smoothing": "Causal smoothing",
    "fixed_decoder": "Causal + Viterbi",
}
COLORS = {
    "raw": "#4c78a8",
    "causal_smoothing": "#f2a541",
    "fixed_decoder": "#59a14f",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _summary_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = _read(MODEL_DIR / "summary.csv")
    all_rows = summary.loc[
        summary["region"].eq("all"),
        [
            "window_seconds",
            "variant",
            "decoder",
            "n_folds",
            "n_samples",
            "macro_f1_mean",
            "macro_f1_std",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "weighted_f1_mean",
        ],
    ].copy()
    all_rows["decoder"] = pd.Categorical(
        all_rows["decoder"], categories=DECODER_ORDER, ordered=True
    )
    all_rows = all_rows.sort_values(["window_seconds", "decoder"])
    all_rows["decoder"] = all_rows["decoder"].astype("string")

    regions = summary.loc[
        summary["region"].isin(["stable", "boundary_15s"]),
        ["window_seconds", "decoder", "region", "n_samples", "macro_f1_mean"],
    ].copy()

    classes = _read(MODEL_DIR / "class_metrics.csv")
    classes["room"] = classes["room"].astype(str)
    key_classes = classes.loc[
        classes["room"].isin(["508", "hallway", "cafeteria", "kitchen", "nurse station"]),
        [
            "window_seconds",
            "variant",
            "decoder",
            "room",
            "support",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "n_folds",
        ],
    ].copy()
    key_classes["decoder"] = pd.Categorical(
        key_classes["decoder"], categories=DECODER_ORDER, ordered=True
    )
    key_classes = key_classes.sort_values(["window_seconds", "room", "decoder"])
    key_classes["decoder"] = key_classes["decoder"].astype("string")
    return all_rows, regions, key_classes


def plot_data_protocol() -> None:
    distortion = _read(REPORT_DIR / "window_protocol_distortion.csv")
    x = np.arange(len(distortion))
    labels = [f"{int(value)}s" for value in distortion["window_seconds"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    axes[0].bar(
        x,
        100 * distortion["median_retained"],
        color="#4c78a8",
        label="Median packets retained",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Packets retained after second-level dedup (%)")
    axes[0].set_ylim(0, 13)
    axes[0].set_title("Loss from truncating timestamps before deduplication")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(
        x,
        100 * distortion["mean_frequency_tv"],
        marker="o",
        label="Frequency total-variation",
        color="#f28e2b",
    )
    axes[1].plot(
        x,
        100 * distortion["dominant_changed_rate"],
        marker="s",
        label="Dominant beacon changed",
        color="#e15759",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Affected windows (%)")
    axes[1].set_title("Window fingerprints changed by second-level dedup")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    _save(REPORT_DIR / "data_retention_and_distortion.png")


def plot_label_mixing() -> None:
    mixing = _read(REPORT_DIR / "label_mixing_by_window.csv")
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(
        mixing["window_seconds"],
        100 * mixing["mixed_rate"],
        marker="o",
        linewidth=2,
        color="#b54a4a",
        label="Mixed-label windows",
    )
    axis.plot(
        mixing["window_seconds"],
        100 * (1 - mixing["mean_majority_share"]),
        marker="s",
        linewidth=2,
        color="#4c78a8",
        label="Mean minority-label share",
    )
    axis.set_xticks(mixing["window_seconds"])
    axis.set_xlabel("Window size (seconds)")
    axis.set_ylabel("Rate (%)")
    axis.set_title("Temporal label mixing increases with window size")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    _save(REPORT_DIR / "label_mixing_by_window.png")


def plot_postprocessing(all_rows: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    width = 0.24
    windows = sorted(all_rows["window_seconds"].unique())
    x = np.arange(len(windows))
    for offset, decoder in enumerate(DECODER_ORDER):
        rows = all_rows.loc[all_rows["decoder"].eq(decoder)].set_index("window_seconds")
        means = [rows.loc[window, "macro_f1_mean"] for window in windows]
        errors = [rows.loc[window, "macro_f1_std"] for window in windows]
        axis.bar(
            x + (offset - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=COLORS[decoder],
            label=DECODER_LABELS[decoder],
        )
    axis.set_xticks(x, [f"{window}s" for window in windows])
    axis.set_ylim(0, 0.56)
    axis.set_ylabel("Strict LODO Macro-F1")
    axis.set_title("Raw model and separately evaluated post-processing stages")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    _save(REPORT_DIR / "postprocessing_macro_f1.png")


def plot_regions(regions: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for axis, window in zip(axes, [1, 3]):
        subset = regions.loc[regions["window_seconds"].eq(window)]
        pivot = subset.pivot(index="decoder", columns="region", values="macro_f1_mean").reindex(
            DECODER_ORDER
        )
        x = np.arange(len(pivot))
        width = 0.34
        axis.bar(x - width / 2, pivot["stable"], width, label="Stable", color="#59a14f")
        axis.bar(
            x + width / 2,
            pivot["boundary_15s"],
            width,
            label="Within 15s of a transition",
            color="#e15759",
        )
        axis.set_xticks(x, [DECODER_LABELS[value] for value in pivot.index], rotation=15)
        axis.set_title(f"{window}s windows")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Macro-F1")
    axes[1].legend(frameon=False)
    figure.suptitle("Boundary windows remain substantially harder than stable periods")
    _save(REPORT_DIR / "boundary_stable_macro_f1.png")


def plot_key_classes(key_classes: pd.DataFrame) -> None:
    room_order = ["508", "hallway", "cafeteria", "kitchen", "nurse station"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, window in zip(axes, [1, 3]):
        subset = key_classes.loc[key_classes["window_seconds"].eq(window)]
        pivot = subset.pivot(index="room", columns="decoder", values="f1_weighted").reindex(room_order)
        x = np.arange(len(room_order))
        width = 0.24
        for offset, decoder in enumerate(DECODER_ORDER):
            axis.bar(
                x + (offset - 1) * width,
                pivot[decoder],
                width,
                color=COLORS[decoder],
                label=DECODER_LABELS[decoder],
            )
        axis.set_xticks(x, room_order, rotation=22)
        axis.set_title(f"{window}s windows")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Support-weighted cross-fold class F1")
    axes[1].legend(frameon=False)
    figure.suptitle("Post-processing helps common rooms but can erase rare/transition classes")
    _save(REPORT_DIR / "key_class_postprocessing_f1.png")


def _top_confusions() -> pd.DataFrame:
    confusion = _read(MODEL_DIR / "confusion_counts.csv")
    confusion["true_room"] = confusion["true_room"].astype(str)
    confusion["pred_room"] = confusion["pred_room"].astype(str)
    confusion = confusion.loc[
        confusion["decoder"].eq("fixed_decoder")
        & confusion["true_room"].ne(confusion["pred_room"])
    ]
    grouped = (
        confusion.groupby(["window_seconds", "true_room", "pred_room"], as_index=False)["count"]
        .sum()
        .sort_values(["window_seconds", "count"], ascending=[True, False])
    )
    return grouped.groupby("window_seconds", group_keys=False).head(10).reset_index(drop=True)


def plot_top_confusions(top: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis, window in zip(axes, [1, 3]):
        subset = top.loc[top["window_seconds"].eq(window)].sort_values("count")
        labels = subset["true_room"] + " -> " + subset["pred_room"]
        axis.barh(labels, subset["count"], color="#e15759")
        axis.set_xlabel("OOF windows")
        axis.set_title(f"{window}s fixed decoder")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Largest off-diagonal confusions after temporal decoding")
    _save(REPORT_DIR / "top_confusions_fixed_decoder.png")


def plot_per_class_fixed() -> None:
    classes = _read(MODEL_DIR / "class_metrics.csv")
    classes["room"] = classes["room"].astype(str)
    figure, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True)
    for axis, window in zip(axes, [1, 3]):
        subset = classes.loc[
            classes["window_seconds"].eq(window) & classes["decoder"].eq("fixed_decoder")
        ].sort_values("f1_weighted")
        colors = np.where(subset["f1_weighted"].lt(0.1), "#e15759", "#4c78a8")
        axis.barh(subset["room"], subset["f1_weighted"], color=colors)
        axis.set_xlim(0, 1)
        axis.set_xlabel("Class F1")
        axis.set_title(f"{window}s")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("All evaluated classes: support-weighted cross-fold fixed-decoder F1")
    _save(REPORT_DIR / "per_class_fixed_decoder_f1.png")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows, regions, key_classes = _summary_tables()
    top = _top_confusions()
    all_rows.to_csv(REPORT_DIR / "postprocessing_summary.csv", index=False)
    key_classes.to_csv(REPORT_DIR / "key_class_metrics.csv", index=False)
    top.to_csv(REPORT_DIR / "top_confusions_fixed_decoder.csv", index=False)

    plot_data_protocol()
    plot_label_mixing()
    plot_postprocessing(all_rows)
    plot_regions(regions)
    plot_key_classes(key_classes)
    plot_top_confusions(top)
    plot_per_class_fixed()

    inputs = [
        REPORT_DIR / "data_lineage.csv",
        REPORT_DIR / "window_protocol_distortion.csv",
        REPORT_DIR / "label_mixing_by_window.csv",
        REPORT_DIR / "data_limitations_summary.csv",
        MODEL_DIR / "summary.csv",
        MODEL_DIR / "fold_metrics.csv",
        MODEL_DIR / "class_metrics.csv",
        MODEL_DIR / "confusion_counts.csv",
    ]
    outputs = sorted(
        path.name for path in REPORT_DIR.iterdir() if path.suffix.lower() in {".csv", ".png"}
    )
    manifest = {
        "builder": Path(__file__).relative_to(ROOT).as_posix(),
        "contains_row_level_data": False,
        "inputs": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
        "outputs": outputs,
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (REPORT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {len(outputs)} aggregate files to {REPORT_DIR}")


if __name__ == "__main__":
    main()

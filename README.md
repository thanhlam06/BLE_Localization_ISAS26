# BLE Localization Baseline ML (ISAS 2026)

Reproducible classical machine-learning baseline for room-level indoor
localization from Bluetooth Low Energy (BLE) RSSI measurements. The executable
pipeline builds fixed-window signal features and evaluates XGBoost or Random
Forest models with Leave-One-Day-Out (LODO) validation.

> **Private dataset:** the raw BLE data, location labels, aligned cache,
> row-level features, and row-level predictions are not distributed with this
> repository. Access must be obtained from the project owners. Do not commit
> these files or any unreviewed derivative containing timestamps, users, or
> room labels.

## What this repository provides

```text
authorized local data
        |
        +--> input audit
        |
        +--> build or reuse aligned BLE/label cache
                    |
                    +--> timestamp/RSSI normalization
                    +--> fixed-window feature extraction
                    +--> closed-set LODO training
                    +--> fold metrics, predictions, and aggregate summary
```

The `process` stage reuses a local aligned cache when it exists. With the
default `processing.build_aligned_if_missing=true`, it can also build that cache
from authorized raw BLE and label files using bounded label intervals. The
cache and all source tables remain private.

Key implementation details:

- configurable window length and per-beacon RSSI aggregations;
- guarded raw-to-label interval alignment when the cache is absent;
- one feature row per day/user/window, with majority label and latest-sample
  tie-breaking at room transitions;
- missing/zero RSSI handling;
- fold-local removal of constant and duplicate feature columns;
- median imputation fitted on each training fold;
- XGBoost and Random Forest classifiers;
- day-based LODO evaluation with accuracy, balanced accuracy, Macro-F1, and
  Weighted-F1 outputs.

This is a research/evaluation pipeline, not a real-time localization service.

## Repository layout

```text
configs/
  baseline.json                 Default paths, feature settings, and models
data/
  README.md                     Public private-input contract (no data rows)
docs/
  PIPELINE.md                   Detailed stage contract and outputs
  PROJECT_INDEX.md              Public repository file map
  RESULTS.md                    Generated aggregate result summary
scripts/
  run_baseline.py               Main pipeline CLI
  summarize_results.py          Aggregate-result report generator
src/baseline_ml/
  config.py                     Config loading and path resolution
  pipeline.py                   Processing, extraction, and LODO training
  results.py                    Result discovery and report generation
requirements.txt                Python dependencies
tests/test_pipeline.py          Synthetic end-to-end regression tests
```

Only reviewed aggregate reports and figures should be published. Historical
notebooks and private-data-derived row-level artifacts are not required to run
the public CLI.

## Private dataset contract

Place the authorized raw inputs at the following local paths. Either provide
the aligned cache too or let the default fallback create it:

```text
data/
  raw/
    ble_train_4d.csv
    location_train_labels.csv
  processed/
    extracted_features_base.csv     # optional before the first full run
```

The default paths can be changed in `configs/baseline.json`.
The same no-data contract is kept in `data/README.md` so the expected local
directory remains discoverable while all dataset files stay ignored.

### Expected schema

No private sample rows are shown here. Column names are case-sensitive unless
the data is normalized before use.

| File | Role | Expected columns |
| --- | --- | --- |
| `ble_train_4d.csv` | Source used when the aligned cache is absent | `user_id`, `timestamp`, and one or more numeric columns beginning with `RSSI_`; long-form beacon/RSSI columns may coexist but are not the window-feature inputs |
| `location_train_labels.csv` | Label intervals used when the aligned cache is absent | `started_at`, `finished_at`, `room`; `user_id` is required for safe multi-user alignment and `floor` is optional metadata |
| `extracted_features_base.csv` | Reused pipeline cache or generated private cache | `user_id`, `timestamp`, `room`, one or more numeric columns beginning with `RSSI_`; `date` is optional and is derived from `timestamp` when absent |

For the aligned cache:

- timestamps must be parseable by `pandas.to_datetime`;
- `user_id`, `timestamp`, `room`, and the effective day must be non-null for
  rows used during extraction;
- all `RSSI_*` fields should be numeric or coercible to numeric;
- zero RSSI values are replaced by `features.missing_rssi` by default;
- all files must use a consistent timezone and day boundary because `date`
  defines the LODO folds.

When building the cache, invalid or missing label ends fall back to the next
label start when available, and every interval is capped by
`processing.label_max_gap_minutes` (30 minutes by default). Matching is
backward in time within each day. One BLE stream and one label stream may use
different external user IDs; multi-user inputs must expose matching user-ID
sets, otherwise processing stops to prevent cross-user labels.

The raw BLE and label files are inspected by `--stage load`. `process` reads
the aligned cache when present and otherwise builds it from those source files
when the fallback is enabled. A full run persists the new cache at the
configured path; a `--limit-rows` run never persists it.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/thanhlam06/BLE_Localization_ISAS26.git
cd BLE_Localization_ISAS26
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default XGBoost configuration uses the CPU-compatible `hist` tree method;
a GPU is not required. Dependencies use bounded compatibility ranges rather
than an exact lock file, so record the resolved environment when reproducing a
reported experiment.

## Run the pipeline

All relative config and output paths are resolved from the repository root.

### 1. Audit local inputs

```bash
python scripts/run_baseline.py --stage load
```

This writes `artifacts/baseline_run/load/load_audit.json` with file existence,
columns, a small sample shape, and file size. The audit reports missing files
rather than creating or downloading them; inspect its `exists` fields before
continuing. An absent aligned cache is expected when the raw fallback is used.

### 2. Smoke-test processing and extraction

```bash
python scripts/run_baseline.py --stage process --limit-rows 10000 --output-dir artifacts/smoke
python scripts/run_baseline.py --stage extract --limit-rows 10000 --output-dir artifacts/smoke
```

`--limit-rows` is useful for schema and execution checks only. A prefix of the
input table may contain too few days/classes for meaningful LODO training.
The smoke test does not write a newly built cache into `data/processed/`.

### 3. Run the complete baseline

```bash
python scripts/run_baseline.py --stage all
```

Or run only training:

```bash
python scripts/run_baseline.py --stage train
```

With a fresh output directory, `train` automatically extracts features first.
With the default directory and no row limit, it reuses
`artifacts/baseline_run/features/baseline_45s_features.csv` when that file
already exists. Remove or change the output directory when changing data or
feature configuration to avoid reusing a stale feature cache.

If the aligned cache is absent, the default full run builds it from the raw BLE
and label files and saves it at `processed.aligned_cache_path`. If it already
exists, it is treated as authoritative and reused.

Available stages:

| Stage | Behavior |
| --- | --- |
| `load` | Audits the raw BLE, label, and aligned-cache paths |
| `process` | Reads the aligned cache or builds it from bounded raw-label intervals, then normalizes timestamps/day and validates required columns |
| `extract` | Runs `process`, sorts/groups rows into configured windows, and writes aggregated features |
| `train` | Reuses or extracts features, then runs one held-out-day fold per available day |
| `all` | Runs all stages in order |

Use `--help` for the complete CLI:

```bash
python scripts/run_baseline.py --help
```

## Configuration

The default experiment is defined in `configs/baseline.json`.

| Section | Important keys | Meaning |
| --- | --- | --- |
| `raw` | `ble_path`, `label_path` | Private raw inputs audited by `load` |
| `processing` | `build_aligned_if_missing`, label time/metadata keys, `label_max_gap_minutes` | Guarded raw-to-aligned fallback behavior |
| `processed` | `aligned_cache_path`, column-name keys | Aligned-cache destination/source and its identity/time/label columns |
| `features` | `window_seconds`, `rssi_prefix`, `missing_rssi`, `aggregations` | Window feature construction |
| `training` | `model`, `closed_set`, `random_state` | Classifier and evaluation behavior |
| `training.xgb_params` | XGBoost hyperparameters | Used when `model` is `xgb` |
| `training.rf_params` | Random Forest hyperparameters | Used when `model` is `rf` |

Supported `features.aggregations` values must be valid pandas group-by
aggregations. The repository default is `mean`, `std`, `min`, `max`, and
`last`. Optional hour/minute features are controlled by
`features.include_time_features`.

To run Random Forest, change:

```json
{
  "training": {
    "model": "rf"
  }
}
```

Preserve the rest of the existing `training` object when editing the actual
config; the fragment above only illustrates the key to change.

### Closed-set interpretation

With `training.closed_set=true`, each fold is restricted to room classes found
in both its training days and held-out day. This prevents label-encoder errors
but does **not** measure unseen-room recognition. Report these scores as
closed-set LODO results, not open-world deployment performance.

With `closed_set=false`, the trainer keeps all fold rows but stops with a clear
error if the held-out day contains a class absent from training; the current
classifier path does not implement unknown-class recognition.

## Outputs

The default run writes under `artifacts/baseline_run/`:

```text
load/load_audit.json
process/process_audit.json
features/baseline_<window>s_features.csv
features/baseline_<window>s_feature_schema.json
training/lodo_results.csv
training/lodo_predictions.csv
training/lodo_summary.csv
```

A standalone `extract` or a fresh standalone `train` also writes
`features/process/process_audit.json`, because it invokes `process` internally.
The `all` stage passes its already processed table forward and does not repeat
that audit.

When the aligned cache was absent and a full raw fallback succeeded, the
pipeline also writes the private cache to the configured
`data/processed/extracted_features_base.csv` path.

Privacy classification:

| Output | Publication guidance |
| --- | --- |
| Load/process audits | Keep private unless paths, dates, counts, and columns have been reviewed |
| Window feature CSV | Private row-level derivative; never publish by default |
| LODO predictions | Private row-level derivative; never publish by default |
| Fold results | Review because held-out dates and class coverage may be sensitive |
| Aggregate summary | Publish only after owner review and de-identification |

Generated data remains private even when it lives outside `data/`. Before any
commit, inspect staged files and verify that no raw rows, timestamps, room
labels, user identifiers, local absolute paths, or model binaries are present.

## Results and report generation

The strongest historical notebook result currently retained in the reviewed
aggregate metadata is:

| Model | Window | Features | Evaluation | Mean Macro-F1 | Mean accuracy |
| --- | ---: | --- | --- | ---: | ---: |
| Optuna-tuned XGBoost | 45 s | `ble_safe_only` | closed-set LODO | 0.557071 | 0.648661 |

This is a tuned diagnostic result: its hyperparameters were selected using the
same four LODO folds. It predates the corrected, leakage-safe window grouping
in the public CLI and has not been reproduced exactly by that implementation.
It should not be presented as a current pipeline benchmark, nested-validation
result, or independent holdout estimate.

To rebuild the aggregate result index from locally available, reviewed summary
artifacts:

```bash
python scripts/summarize_results.py
```

The command rewrites:

- `reports/best_results_summary.csv`
- `docs/RESULTS.md`

Review both files before publication. The scanner intentionally ignores
prediction files and non-curated run directories, but source aggregate files
can still contain sensitive metadata.

## Validation and reproducibility

Before reporting a run:

1. Verify the input audit and aligned-cache schema.
2. Use the complete dataset; do not retain `--limit-rows` for final metrics.
3. Preserve the exact JSON config used for the run.
4. Record `python --version` and `python -m pip freeze` output privately.
5. Record a private checksum or version identifier for each input dataset.
6. Confirm the same day boundary, `closed_set` setting, and random seed.
7. Check every LODO fold and the aggregate summary for empty or missing folds.

Basic code/CLI checks:

```bash
python -m compileall src scripts
python scripts/run_baseline.py --help
python -m unittest discover -s tests -v
```

The automated tests use temporary synthetic tables and do not require the
private dataset.

The configured random seed controls supported model randomness. Exact results
can still vary across Python, pandas, scikit-learn, XGBoost, and platform
versions, which is why the resolved environment should accompany a result.

## Troubleshooting

| Symptom | Likely cause and action |
| --- | --- |
| `Aligned cache not found` | Provide the authorized cache, or enable `processing.build_aligned_if_missing` and provide both raw inputs |
| Raw alignment schema/ID error | Check configured timestamp/label columns; for multi-user data, both inputs must expose the same user-ID set |
| `No RSSI columns found` | Ensure beacon features use `features.rssi_prefix` (default `RSSI_`) or change the prefix |
| Missing/empty days after processing | Check timestamp parsing, timezone consistency, null identity/label fields, and the configured date column |
| Empty training outputs | Provide multiple days and sufficient shared room classes; a small `--limit-rows` prefix is often insufficient |
| XGBoost import/training failure | Reinstall `requirements.txt` in the active environment or switch `training.model` to `rf` |
| Memory pressure | First validate with `--limit-rows`; for the full run, use a machine sized for the private aligned cache |
| Unexpectedly reused features | Use a new `--output-dir` after changing the dataset or feature config |

## License and citation status

No `LICENSE` or `CITATION.cff` is currently included. Consequently, no
open-source reuse license or formal citation format is being asserted here.
Contact the repository maintainers before redistributing the code or derived
research artifacts.

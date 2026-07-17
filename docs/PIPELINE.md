# Baseline ML Pipeline

This document defines the executable contract implemented by
`scripts/run_baseline.py`. See the repository `README.md` for setup, the private
dataset policy, and troubleshooting.

## Data boundary

The public code operates only on locally supplied private data. It reuses the
cache configured as `processed.aligned_cache_path` when present. With
`processing.build_aligned_if_missing=true`, it can build that cache from the
configured raw BLE and label interval files.

Default local inputs:

```text
data/raw/ble_train_4d.csv
data/raw/location_train_labels.csv
data/processed/extracted_features_base.csv
```

None of these files should be committed. The raw BLE table used by the fallback
must have the configured user/time columns and wide RSSI columns matching the
configured prefix. The label table must have configured interval start/end and
label columns. Multi-user inputs also require the same user-ID set on both
sides.

## CLI

```bash
python scripts/run_baseline.py \
  --config configs/baseline.json \
  --stage all \
  --output-dir artifacts/baseline_run
```

Optional `--limit-rows N` reads only the first `N` rows at each relevant stage.
Use it for execution checks, not final metrics.

## Stage 1: `load`

The load stage visits all three configured input paths and writes:

```text
artifacts/baseline_run/load/load_audit.json
```

For each path, the audit records its resolved local path and existence. For an
existing CSV it also records columns, file size, and the shape of a small sample
(five rows by default). A missing file is reported with `exists: false`; `load`
alone does not raise an error for it.

## Stage 2: `process`

`process` reads the aligned cache when present. Otherwise, when the fallback is
enabled, it:

1. reads raw BLE rows and all label intervals;
2. removes accidental CSV export-index columns;
3. parses timestamps and aligns only within the same calendar day;
4. replaces an invalid/missing label end with the next label start when
   available, then caps every interval at the configured maximum gap;
5. uses backward as-of matching and retains only BLE rows inside the effective
   label interval;
6. aligns single-stream inputs by day even when their sole external user IDs
   differ, but rejects mismatched multi-user ID sets.

After loading or building the cache, `process`:

1. validates configured time, user, and label columns;
2. parses timestamps with `errors="coerce"`;
3. derives or fills the configured date column from the timestamp;
4. records source, whether it was built from raw inputs, shape, invalid
   timestamps, day counts, and label cardinality.

It writes:

```text
artifacts/baseline_run/process/process_audit.json
```

For a full run, a newly built cache is saved to the configured private path. If
`--limit-rows` is active, it is used in memory and never persisted. The returned
table remains row-level private data. Rows with unusable identity, time, day, or
label values are removed later by `extract`.

## Stage 3: `extract`

`extract` invokes `process`, identifies columns beginning with the configured
RSSI prefix, and then:

1. coerces RSSI values to numeric;
2. optionally replaces zero RSSI values with the missing-signal sentinel;
3. fills remaining missing RSSI values with the same sentinel;
4. sorts observations chronologically for deterministic `last` aggregation;
5. floors timestamps to `features.window_seconds`;
6. groups by day, user, and window start;
7. assigns one room label per window by majority count, breaking ties with the
   latest chronological sample;
8. applies the configured per-beacon aggregations;
9. adds signal mean/min/max and visible-beacon-count features;
10. optionally adds hour and minute.

Outputs for a standalone 45-second `extract`:

```text
artifacts/baseline_run/features/process/process_audit.json
artifacts/baseline_run/features/baseline_45s_features.csv
artifacts/baseline_run/features/baseline_45s_feature_schema.json
```

During `--stage all`, `process` has already run, so extraction reuses the table
in memory and does not write the nested `features/process/process_audit.json`.

A window crossing a labeled room transition therefore remains one feature row.

## Stage 4: `train`

`train` reuses the matching feature CSV in its output directory when it exists
and no row limit is active. Otherwise, it runs `extract` first.

For each held-out day:

1. all other days form the training fold;
2. when `class_protocol=dasel_intersection`, train and test rows are restricted
   to their shared room classes before any fitted transformation;
3. labels are encoded using the training fold;
4. constant and duplicate numeric feature columns are removed using training
   data only;
5. a median imputer and the configured classifier are fitted on the training
   fold;
6. fold predictions and metrics are recorded.

Supported classifiers are `xgb` and `rf`. The metrics are accuracy, balanced
accuracy, Macro Precision/Recall/F1, and Weighted-F1.

When `closed_set=false`, all rows are kept, but the trainer raises an explicit
error if a held-out fold contains a label absent from its training fold. The
baseline does not implement an unknown-room class.

Outputs:

```text
artifacts/baseline_run/training/lodo_results.csv
artifacts/baseline_run/training/lodo_class_audit.csv
artifacts/baseline_run/training/lodo_predictions.csv
artifacts/baseline_run/training/lodo_summary.csv
```

`lodo_class_audit.csv` records raw train/test row counts, raw class counts,
common classes, train-only/test-only class counts, and rows removed from each
side. It is the primary evidence for matching the DASEL Table II class policy.

The feature and prediction CSVs are private row-level derivatives. Fold files
also require review because they contain held-out-day metadata. Only reviewed,
de-identified aggregates should be published.

## DASEL 1s/3s research entry points

The long-form extractor and matrix runner are intentionally separate from the
wide-cache baseline:

```bash
python scripts/extract_dasel_features.py --input PRIVATE.csv --output-dir PRIVATE_FEATURES --windows 1 3
python scripts/run_dasel_windows.py --features-dir PRIVATE_FEATURES --audit-only
python scripts/run_dasel_windows.py --features-dir PRIVATE_FEATURES
```

The extractor produces tumbling windows (stride equals window length) with the
historical 262-column schema. `run_dasel_windows.py` uses
`configs/dasel_1s.json` and `configs/dasel_3s.json`, excludes calendar/time and
compatibility power fields, and writes combined audits/results under its output
directory.

## Result index

`scripts/summarize_results.py` is separate from model training. It scans
eligible summary artifacts, ranks Macro-F1 candidates, and rewrites:

```text
reports/best_results_summary.csv
docs/RESULTS.md
```

The scanner excludes reports as inputs, non-curated generated run directories,
predictions, confusion matrices, and row-level LODO result files. Its generated
outputs still require an owner privacy review before publication.

# DASEL-matched protocol for 1s and 3s windows

## Scope

This track makes the class policy in DASEL Table II explicit and auditable for
both 1-second and 3-second feature matrices. It is separate from the legacy
M0-M8 exploration so protocol corrections do not silently rewrite historical
results.

## Feature construction

`scripts/extract_dasel_features.py` consumes the authorized long-form
`DASEL_preprocessed_train.csv` schema:

```text
timestamp, mac_address, RSSI, room
```

It removes exact duplicate detections using all four fields, maps timestamps
to non-overlapping tumbling windows, and writes one 262-column feature row per
window. The 1s and 3s matrices contain the same feature schema: global RSSI
statistics, per-beacon counts/frequencies/presence, per-beacon RSSI
statistics, strongest/count-dominant beacon summaries, frequency entropy, and
cyclic time fields. Calendar/time fields and compatibility power fields are
explicitly excluded from confirmatory model input.

The published track uses stride equal to window length. It does not use 50%
overlap because that would change the sample unit, autocorrelation, effective
test size, and comparison with the current DASEL frame protocol. Overlap is a
separate ablation and must be reported as such.

## Fold class policy

For held-out day `d`:

```text
C_d = classes(training days) intersection classes(day d)
train_d = training rows whose room is in C_d
test_d  = day-d rows whose room is in C_d
```

This filtering happens before feature cleaning, imputation, weighting, model
fitting, or metric calculation. After filtering, both train and test expose
exactly the same model class set. The resulting common-class counts are:

| Held-out day | DASEL Table II | 1s audit | 3s audit |
| --- | ---: | ---: | ---: |
| 2023-04-10 (Day 1) | 12 | 12 | 12 |
| 2023-04-11 (Day 2) | 15 | 15 | 15 |
| 2023-04-12 (Day 3) | 18 | 18 | 18 |
| 2023-04-13 (Day 4) | 13 | 13 | 13 |

Day 3 and Day 4 contain one held-out-only class before filtering. Therefore,
filtering only the test labels that a train label encoder cannot transform is
not sufficient: train-only classes must also be removed to reproduce the
Table II class set.

The complete reviewed counts are in
`reports/dasel_matched_1s_3s/class_protocol_audit.csv`.

## Confirmatory control result

The first strict run is deliberately a raw control: one XGBoost seed, no
augmentation, no visit weighting, no top-100 drift-aware selection, and no
temporal decoder.

| Window | Mean Macro-F1 | Fold SD | Mean accuracy | Mean balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1s | 0.2077 | 0.0140 | 0.5522 | 0.2401 |
| 3s | 0.2542 | 0.0288 | 0.5687 | 0.2963 |

These scores establish a leakage-aware protocol baseline, not the strongest
ML result. They must not be compared with the legacy decoded M0-M8 values as
if only the window size changed. The fold metrics and exact feature counts are
in `reports/dasel_matched_1s_3s/fold_metrics.csv` and `summary.csv`.

## Reproduction

Extract private feature matrices:

```bash
python scripts/extract_dasel_features.py \
  --input /authorized/path/DASEL_preprocessed_train.csv \
  --output-dir /private/output/features \
  --windows 1 3
```

Audit class matching without training:

```bash
python scripts/run_dasel_windows.py \
  --features-dir /private/output/features \
  --output-dir artifacts/dasel_audit \
  --audit-only
```

Run the raw confirmatory XGBoost baseline:

```bash
python scripts/run_dasel_windows.py \
  --features-dir /private/output/features \
  --output-dir artifacts/dasel_confirmatory
```

The artifact directory contains private row-level predictions and stays
ignored. Only reviewed fold-level/aggregate metrics are copied into
`reports/dasel_matched_1s_3s/`.

## Interpretation boundary

Matching the class policy does not make every modeling choice identical to
the DASEL paper. The public confirmatory baseline is a raw single-seed
XGBoost control. The legacy M0-M8 results additionally used feature selection,
augmentation, three seeds, visit weighting, and temporal decoding. A strong
paper comparison requires rerunning those components after the same class
intersection is applied inside every outer and inner training split.

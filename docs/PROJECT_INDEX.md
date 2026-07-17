# Public Project Index

This index maps the publishable repository components to the executable
baseline. Private datasets, row-level derivatives, and historical notebooks
are intentionally outside the public project contract.

## Entry points

| Path | Purpose |
| --- | --- |
| `scripts/run_baseline.py` | CLI wrapper for input audit, processing, feature extraction, and LODO training |
| `scripts/extract_dasel_features.py` | Creates private 262-column tumbling-window matrices for 1s and 3s |
| `scripts/run_dasel_windows.py` | Audits or trains both DASEL-intersection LODO controls |
| `scripts/summarize_results.py` | Builds a compact index from reviewed aggregate result artifacts |
| `configs/baseline.json` | Default data paths, schema keys, feature construction, model, and seed |
| `configs/dasel_1s.json`, `configs/dasel_3s.json` | Strict 1s/3s class protocol and model controls |
| `data/README.md` | Public local-input contract; contains no dataset rows |
| `tests/test_pipeline.py` | Synthetic regression tests for alignment, extraction, and end-to-end outputs |

## Python package

| Path | Purpose |
| --- | --- |
| `src/baseline_ml/config.py` | Repository-root discovery, config loading, and relative-path resolution |
| `src/baseline_ml/pipeline.py` | Pipeline stages, feature engineering, classifiers, and fold metrics |
| `src/baseline_ml/dasel_features.py` | Historical DASEL long-form 262-column feature builder |
| `src/baseline_ml/results.py` | Aggregate result discovery and Markdown/CSV report generation |

## Documentation

| Path | Purpose |
| --- | --- |
| `README.md` | Public overview, setup, data contract, commands, privacy, and troubleshooting |
| `docs/PIPELINE.md` | Detailed executable stage contract |
| `docs/DASEL_PROTOCOL_1S_3S.md` | DASEL Table II class-match evidence and reproduction contract |
| `docs/RESULTS.md` | Generated reviewed aggregate result index |
| `notebooks/1s/`, `notebooks/3s/` | Output-free provenance notebooks for EDA, ML, ablations, H1-H6, and H7/DL |

## Local-only inputs and outputs

The following path families are operational but not automatically publishable:

| Path family | Classification |
| --- | --- |
| Private files under `data/**` | Source data and aligned cache; never commit (`data/README.md` is the public exception) |
| `artifacts/**/baseline_*_features.csv` | Private row-level features |
| `artifacts/**/lodo_predictions.csv` | Private row-level predictions |
| Other `artifacts/`, `reports/`, and `figures/` | Derived material; publish only after owner review and de-identification |

The baseline does not depend on the sanitized research notebooks. They are
retained for provenance with outputs, absolute paths, embedded samples, and
private metadata removed; generated notebook artifacts remain local-only.

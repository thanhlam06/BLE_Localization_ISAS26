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
| `scripts/run_strong_ml.py` | Trains strict 1s M2 and 3s M5 ensembles; publishes aggregate metrics, plots, and TreeSHAP |
| `scripts/predict_strong_ml.py` | Fits final 1s/3s strong configurations and predicts an authorized unlabeled BLE file; outputs remain private |
| `scripts/build_project_analysis.py` | Rebuilds de-identified data-protocol, post-processing, and error-analysis plots from aggregate CSVs |
| `scripts/summarize_results.py` | Builds a compact index from reviewed aggregate result artifacts |
| `configs/baseline.json` | Default data paths, schema keys, feature construction, model, and seed |
| `configs/dasel_1s.json`, `configs/dasel_3s.json` | Strict 1s/3s class protocol and model controls |
| `configs/strong_ml_1s_3s.json` | Strong M2/M5 selector, weighting, XGBoost, decoder, and SHAP settings |
| `data/README.md` | Public local-input contract; contains no dataset rows |
| `tests/test_pipeline.py` | Synthetic regression tests for alignment, extraction, and end-to-end outputs |
| `tests/test_strong_ml.py` | Strict-fold, feature-contract, boundary, smoothing, and decoder tests |
| `tests/test_strong_inference.py` | Full-timestamp deduplication, feature extraction, 1s/3s alignment, probability-order, and row-mapping tests |

## Python package

| Path | Purpose |
| --- | --- |
| `src/baseline_ml/config.py` | Repository-root discovery, config loading, and relative-path resolution |
| `src/baseline_ml/pipeline.py` | Pipeline stages, feature engineering, classifiers, and fold metrics |
| `src/baseline_ml/dasel_features.py` | Historical DASEL long-form 262-column feature builder |
| `src/baseline_ml/results.py` | Aggregate result discovery and Markdown/CSV report generation |
| `src/baseline_ml/strong_ml.py` | Fold-local M2/M5 training, fixed decoder, metrics, native TreeSHAP, and plots |
| `src/baseline_ml/strong_inference.py` | Audited full-train inference, causal/offline outputs, 1s/3s agreement, and private output mapping |

## Documentation

| Path | Purpose |
| --- | --- |
| `README.md` | Public overview, setup, data contract, commands, privacy, and troubleshooting |
| `docs/PIPELINE.md` | Detailed executable stage contract |
| `docs/DASEL_PROTOCOL_1S_3S.md` | DASEL Table II class-match evidence and reproduction contract |
| `docs/RESULTS.md` | Generated reviewed aggregate result index |
| `docs/DATA_AUDIT.md` | Full-timestamp deduplication, window distortion, label mixing, drift, and open-set limitations |
| `docs/MODEL_POSTPROCESSING_ANALYSIS.md` | What raw ML, causal smoothing, and Viterbi solve or leave unresolved |
| `reports/strong_ml_1s_3s/` | Reviewed strict M2/M5 fold metrics, aggregate SHAP, audit tables, and figures |
| `reports/project_analysis/` | Reproducible de-identified plots and aggregate error tables with hashed input manifest |
| `notebooks/1s/`, `notebooks/3s/` | Output-free provenance notebooks for EDA, ML, ablations, H1-H6, and H7/DL |

## Local-only inputs and outputs

The following path families are operational but not automatically publishable:

| Path family | Classification |
| --- | --- |
| Private files under `data/**` | Source data and aligned cache; never commit (`data/README.md` is the public exception) |
| `artifacts/**/baseline_*_features.csv` | Private row-level features |
| `artifacts/**/lodo_predictions.csv` | Private row-level predictions |
| `artifacts/strong_ml/` | Private models, row-level predictions/probabilities, fold selectors, and fingerprints |
| Other `artifacts/`, `reports/`, and `figures/` | Derived material; publish only after owner review and de-identification |

The baseline does not depend on the sanitized research notebooks. They are
retained for provenance with outputs, absolute paths, embedded samples, and
private metadata removed; generated notebook artifacts remain local-only.

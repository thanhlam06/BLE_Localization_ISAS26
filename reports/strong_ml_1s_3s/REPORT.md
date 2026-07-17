# Strong ML 1s/3s — strict DASEL-matched LODO

This report evaluates the strongest reviewed ML configurations under the strict shared-class protocol: each outer fold is filtered to `classes(train) ∩ classes(test)` before any fitting. Feature selection, drift filtering, augmentation, weighting, fingerprints, and Markov transitions use outer-training days only.

## Configurations

- **1s / M2:** top-100 drift-aware features, 0.75 class-balanced + 0.25 soft visit-cap weights, and a three-seed XGBoost ensemble.
- **3s / M5:** 23 beacon-frequency features plus up to 77 selected non-frequency features, class-balanced weights, and a three-seed XGBoost ensemble.
- **Fixed decoder:** causal five-observation probability smoothing followed by train-only Markov Viterbi decoding. At 1s this uses up to 5 seconds of past probability context; at 3s it uses up to 15 seconds.

## Outer-fold results

| Window | Variant | Output | Macro-F1 mean ± SD | Accuracy | Balanced accuracy |
|---:|---|---|---:|---:|---:|
| 1s | M2 | fixed_decoder | 0.4441 ± 0.0535 | 0.6602 | 0.4842 |
| 1s | M2 | raw | 0.3475 ± 0.0228 | 0.5426 | 0.4513 |
| 3s | M5 | fixed_decoder | 0.4732 ± 0.0420 | 0.6736 | 0.5558 |
| 3s | M5 | raw | 0.3881 ± 0.0326 | 0.5602 | 0.4967 |

## Interpretation boundaries

- TreeSHAP values are exact native XGBoost contributions for the **raw three-seed ensemble**. They do not explain the subsequent temporal decoder.
- Cross-fold SHAP aggregation uses zero contribution when a feature is absent from a fold-specific selected set; `n_folds_selected` remains available for stability auditing.
- The fixed Viterbi decoder uses only training-derived transition probabilities, but backtracking makes its final label sequence an **offline** output. Raw probabilities and causal smoothing remain online-compatible; a causal state filter is required for a production online decoder.
- Boundary and stable metrics use a ±15-second band around ground-truth room changes. This diagnostic label is used only after prediction for evaluation.
- The protocol matches the DASEL paper's per-fold shared class counts; it does not claim architectural equivalence to the DASEL neural model.
- Private row-level features, probabilities, predictions, fingerprints, and serialized models remain under ignored `artifacts/strong_ml/` and are not published.

## Reproduction

```powershell
python scripts/run_strong_ml.py --features-dir <PRIVATE_FEATURE_DIR>
```

The public CSV files contain only fold-level or aggregate metrics/attributions. See `manifest.json` for hashes and environment metadata.

# Strong ML 1s/3s — strict DASEL-matched LODO

This report evaluates the strongest reviewed ML configurations under the strict shared-class protocol: each outer fold is filtered to `classes(train) ∩ classes(test)` before any fitting. Feature selection, drift filtering, augmentation, weighting, fingerprints, and Markov transitions use outer-training days only.

## Configurations

- **1s / M2:** top-100 drift-aware features, 0.75 class-balanced + 0.25 soft visit-cap weights, and a three-seed XGBoost ensemble.
- **3s / M5:** 23 beacon-frequency features plus up to 77 selected non-frequency features, class-balanced weights, and a three-seed XGBoost ensemble.
- **Causal smoothing:** probabilities are averaged over the current observation and up to five historical observations within each contiguous segment. This is the online-compatible intermediate phase (up to 5 seconds at 1s, or 15 seconds at 3s).
- **Fixed decoder:** the causal-smoothed probabilities are passed to a train-only Markov Viterbi decoder. Viterbi backtracking is an offline final-label phase and is reported separately.

## Outer-fold results

| Window | Variant | Output | Macro-F1 mean ± SD | Accuracy | Balanced accuracy |
|---:|---|---|---:|---:|---:|
| 1s | M2 | causal_smoothing | 0.3918 ± 0.0325 | 0.6023 | 0.4910 |
| 1s | M2 | fixed_decoder | 0.4441 ± 0.0535 | 0.6602 | 0.4842 |
| 1s | M2 | raw | 0.3475 ± 0.0228 | 0.5426 | 0.4513 |
| 3s | M5 | causal_smoothing | 0.4526 ± 0.0441 | 0.6210 | 0.5754 |
| 3s | M5 | fixed_decoder | 0.4732 ± 0.0420 | 0.6736 | 0.5558 |
| 3s | M5 | raw | 0.3881 ± 0.0326 | 0.5602 | 0.4967 |

## Error analysis

- Fixed decoding improves stable-window Macro-F1 to 0.5408 at 1s and 0.5750
  at 3s, but boundary-window Macro-F1 remains 0.3565 and 0.4214.
- At 1s, room 508 changes from F1 0.1826 raw to 0.2610 causal and 0.0090
  fixed-decoder. Hallway changes from 0.0218 to 0.0076 and 0.0000.
- At 3s, fixed-decoder F1 is 0.2549 for room 508 and 0.0024 for hallway.
- The largest 1s fixed-decoder confusions are `cafeteria -> kitchen` (2,877),
  `cafeteria -> nurse station` (1,145), `hallway -> nurse station` (475),
  `kitchen -> cafeteria` (334), and `hallway -> 508` (235).
- These confusion counts establish spatial class overlap, but aggregate results
  are insufficient to prove that a dominant beacon caused each individual
  error.

The fixed decoder raises global Macro-F1 and accuracy while lowering balanced
accuracy relative to causal smoothing at both window sizes. Post-processing
therefore reduces stable-state flicker but does not solve rare/short visits,
hallway recognition, or transition ambiguity.

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

# Publishable project analysis

This directory contains aggregate, de-identified evidence for the data,
model, post-processing, and error analysis. It contains no row-level packet
timestamps, user IDs, model probabilities, predictions, trajectories, or
serialized estimators.

## Inputs

- `data_lineage.csv`: packet and window counts from the full-timestamp audit.
- `window_protocol_distortion.csv`: aggregate distortion caused by truncating
  timestamps before deduplication.
- `label_mixing_by_window.csv`: majority-label mixing by window size.
- `data_limitations_summary.csv`: aggregate drift, shortcut, active-beacon,
  and open-set statistics.
- The reviewed aggregate tables in `../strong_ml_1s_3s/`.

## Generated outputs

- `data_retention_and_distortion.png`
- `label_mixing_by_window.png`
- `postprocessing_macro_f1.png`
- `boundary_stable_macro_f1.png`
- `key_class_postprocessing_f1.png`
- `top_confusions_fixed_decoder.png`
- `per_class_fixed_decoder_f1.png`
- `postprocessing_summary.csv`
- `key_class_metrics.csv`
- `top_confusions_fixed_decoder.csv`
- `analysis_manifest.json`

Regenerate all outputs with:

```powershell
python scripts/build_project_analysis.py
```

`analysis_manifest.json` records SHA-256 hashes for each aggregate input. The
plots are not generated from local Colab screenshots or private row-level
prediction files.

# Sanitized 1s/3s research notebooks

The notebooks in `1s/` and `3s/` preserve the complete historical research
flow for each window:

1. `01_eda.ipynb` — EDA, PCA, correlation, mutual information, chi-square,
   and distribution diagnostics;
2. `02_ml.ipynb` — drift-aware feature filtering and XGBoost LODO;
3. `03_ablation.ipynb` — eight hallway/static/transition ablations;
4. `04_h1_h6_experiments.ipynb` — H1-H6 sequence/post-processing experiments;
5. `05_h7_dl.ipynb` — H7 sequence/deep-learning setup.

All execution counts and outputs were cleared. Local Windows paths were
replaced with:

```text
private_features/                    # authorized local feature CSVs
artifacts/notebooks/<stage>/<window> # ignored generated outputs
```

The Google Drive candidate path is
`/content/drive/MyDrive/BLE_Localization_ISAS26/private_features`.

## Protocol status

These notebooks are provenance snapshots of the earlier exploratory flow.
They do not retroactively make the archived M0-M8 scores DASEL class-matched.
For confirmatory training, first apply the explicit intersection policy in
`scripts/run_dasel_windows.py` or port the same filtering rule into every
outer/inner fold before any selector, fingerprint, weighting, model, or
decoder is fitted. See `docs/DASEL_PROTOCOL_1S_3S.md`.

No dataset rows, predictions, model binaries, or embedded images are stored in
the notebooks.

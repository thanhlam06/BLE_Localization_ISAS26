# Private data contract

The dataset used by this project is private and is intentionally excluded from
version control. Obtain access through the project owner, then place the files
at these local paths:

```text
data/
|-- raw/
|   |-- ble_train_4d.csv
|   `-- location_train_labels.csv
`-- processed/
    `-- extracted_features_base.csv
```

The DASEL-matched 1s/3s research entry point instead accepts an authorized
long-form file at any local path; a conventional ignored location is:

```text
data/private/DASEL_preprocessed_train.csv
```

Expected columns are validated by the pipeline:

- `ble_train_4d.csv`: `user_id`, `timestamp`, and RSSI feature columns whose
  names start with `RSSI_`.
- `location_train_labels.csv`: `started_at`, `finished_at`, and `room`;
  `user_id` is required for multi-user alignment and `floor` is optional.
- `extracted_features_base.csv` (optional before the first full run):
  `user_id`, `timestamp`, `room`, and RSSI columns prefixed with `RSSI_`;
  `date` is derived from `timestamp` when absent.
- `DASEL_preprocessed_train.csv`: `timestamp`, integer `mac_address` in
  `1..25`, numeric `RSSI`, and `room`. Use
  `scripts/extract_dasel_features.py` to create private 1s/3s matrices.

Do not commit raw rows, aligned rows, row-level predictions, exported samples,
or notebook outputs derived from the private dataset.

# Legacy M0-M8 result archive

`leaderboards.csv` preserves the reviewed aggregate results from the earlier
1-second and 3-second M0-M8 experiment matrix. These rows are useful for
tracking research decisions, but they are **not** DASEL class-matched
confirmatory estimates:

- outer training retained classes that were absent from the held-out day;
- M8/base/decoder promotion used the same four observed outer folds;
- the fixed decoder includes temporal smoothing/Viterbi behavior not present
  in the raw confirmatory baseline.

Do not compare these values directly with DASEL as if the protocols were
identical. The strict track is documented in `docs/DASEL_PROTOCOL_1S_3S.md`.

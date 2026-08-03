# BLE data audit and preprocessing decision

This document records only aggregate statistics from the authorized four-day
dataset. No packet timestamps, user identifiers, trajectories, or row-level
labels are published.

## Timestamp truncation and duplicate removal

The source audit contains 1,099,957 labeled packet rows with sub-second
timestamps. All 1,099,957 rows are unique under full timestamp, beacon, RSSI,
and room; therefore the measured number of true exact duplicates is zero.

Truncating timestamps to whole seconds *before* deduplicating on second,
beacon, RSSI, and room leaves 87,558 rows and removes 1,012,399 rows
(92.04%). The operation does not remove entire one-second frames: both
protocols contain 23,584 one-second frames. It removes repeated packet
detections inside those frames and therefore changes packet-count/frequency
features.

Across window sizes from 1 to 45 seconds, the median retained packet fraction
is 10.3% to 11.0%. At 1 second, the mean frequency-vector total-variation
distance is 0.0397, the mean absolute RSSI-feature shift is 0.2240 dB, and the
dominant beacon changes in 9.04% of windows. The loss is therefore not a
conventional exact-duplicate cleanup.

The reviewed pipeline keeps full timestamp precision through window
aggregation. Exact duplicates, if present in future inputs, must be removed
using the full timestamp and complete measurement identity. Timestamp
truncation is permitted only for assigning a packet to a window.

## Window-size trade-off

| Window | Frames | Mixed-label windows | Mixed rate | Mean majority-label share |
| ---: | ---: | ---: | ---: | ---: |
| 1s | 23,584 | 0 | 0.00% | 100.00% |
| 3s | 9,294 | 21 | 0.23% | 99.92% |
| 5s | 5,876 | 38 | 0.65% | 99.79% |
| 10s | 3,171 | 91 | 2.87% | 99.14% |
| 30s | 1,224 | 158 | 12.91% | 96.60% |
| 45s | 866 | 164 | 18.94% | 94.57% |

Longer windows improve temporal evidence but increasingly mix room labels at
transitions and reduce the number of training examples. The strict reviewed
experiments therefore report 1-second and 3-second results separately rather
than treating a longer window as universally better.

## Verified distribution limits

- Two beacons were inactive; the modeled active-beacon space has 23 channels.
- For total detections per window, mutual information with room is 0.1608 and
  mutual information with day is 0.4331. Packet volume contains spatial
  information, but its stronger association with day is a measured shortcut
  risk under sensor/day drift.
- The median cross-day room-frequency Jensen-Shannon divergence is 0.0602 for
  full-timestamp aggregation and 0.0570 after second-level deduplication.
- The median cross-day room RSSI shift is 0.7275 dB for full-timestamp
  aggregation and 0.7750 dB after second-level deduplication.
- The four-day audit contains two room/day cells that are absent from the
  training side of at least one open-set comparison, totaling 79 one-second
  windows. Closed-set LODO scores do not measure correct recognition of such
  unseen rooms.

These results do not support either extreme claim that all packets should be
kept because more data is always better, or that second-level packet collapse
is harmless. Full timestamps preserve measured packet-rate evidence; LODO and
drift diagnostics are required because the same evidence also carries a day
signature.

## Reproduction and traceability

The aggregate source tables and deterministic plotting command are:

```powershell
python scripts/build_project_analysis.py
```

See `reports/project_analysis/README.md` and
`reports/project_analysis/analysis_manifest.json`. The manifest hashes every
aggregate input used to build the plots.

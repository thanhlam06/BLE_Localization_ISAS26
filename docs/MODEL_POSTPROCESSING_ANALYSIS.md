# Model and post-processing error analysis

## Evaluation protocol

All reported scores use four-fold Leave-One-Day-Out evaluation. For each fold,
both outer-train and held-out rows are filtered to their room-class
intersection before model-side fitting. Feature selection, drift filtering,
weighting, augmentation, fingerprints, class probabilities, and Markov
transitions are fitted from outer-training days only.

The 1-second M2 and 3-second M5 branches use three-seed XGBoost ensembles.
`causal_smoothing` averages only the current and previous probabilities within
a contiguous sequence. `fixed_decoder` applies the train-derived Markov
Viterbi decoder after causal smoothing and uses backtracking; it is an offline
output.

## What was improved

| Window | Output | Macro-F1 | Accuracy | Balanced accuracy |
| ---: | --- | ---: | ---: | ---: |
| 1s | Raw model | 0.3475 | 0.5426 | 0.4513 |
| 1s | Causal smoothing | 0.3918 | 0.6023 | 0.4910 |
| 1s | Causal + Viterbi | 0.4441 | 0.6602 | 0.4842 |
| 3s | Raw model | 0.3881 | 0.5602 | 0.4967 |
| 3s | Causal smoothing | 0.4526 | 0.6210 | 0.5754 |
| 3s | Causal + Viterbi | 0.4732 | 0.6736 | 0.5558 |

Causal smoothing improves Macro-F1 by 0.0443 at 1 second and 0.0645 at
3 seconds relative to the raw ensembles. Adding Viterbi raises Macro-F1 by a
further 0.0523 and 0.0205. The fixed decoder also raises accuracy, but balanced
accuracy falls from 0.4910 to 0.4842 at 1 second and from 0.5754 to 0.5558 at
3 seconds. The global gain is therefore not uniform across classes.

Stable periods benefit more than transition boundaries. Fixed-decoder
Macro-F1 is 0.5408 on stable 1-second windows versus 0.3565 within 15 seconds
of a room transition. At 3 seconds the corresponding values are 0.5750 and
0.4214.

## What remains unresolved

The decoder improves common, persistent states but can propagate a confident
wrong state through rare or short visits. Support-weighted cross-fold class F1
shows this directly:

| Window | Class | Raw | Causal | Causal + Viterbi | Support |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1s | 508 | 0.1826 | 0.2610 | 0.0090 | 44 |
| 1s | hallway | 0.0218 | 0.0076 | 0.0000 | 946 |
| 1s | cafeteria | 0.2284 | 0.2058 | 0.2405 | 4,862 |
| 1s | kitchen | 0.6224 | 0.6832 | 0.7198 | 5,173 |
| 1s | nurse station | 0.7574 | 0.8065 | 0.8520 | 9,363 |
| 3s | 508 | 0.0564 | 0.0442 | 0.2549 | 24 |
| 3s | hallway | 0.0208 | 0.0233 | 0.0024 | 356 |

At 1 second, the largest fixed-decoder off-diagonal errors include
`cafeteria -> kitchen` (2,877 windows), `cafeteria -> nurse station` (1,145),
`hallway -> nurse station` (475), `kitchen -> cafeteria` (334), and
`hallway -> 508` (235). At 3 seconds they include `cafeteria -> kitchen`
(861), `kitchen -> cafeteria` (448), `cafeteria -> nurse station` (301),
`hallway -> nurse station` (162), and `hallway -> 520` (110).

These aggregate confusions are consistent with spatial fingerprint overlap and
state-duration imbalance. They do not by themselves prove that a dominant or
active beacon caused each error. Causal attribution requires matching each
out-of-fold error to its beacon feature vector, which is intentionally not
published at row level.

## Status by problem

| Problem | Result |
| --- | --- |
| Second-level timestamp collapse | Resolved in the reviewed preprocessing contract: retain full precision until window assignment |
| Random adjacent-window leakage | Resolved by day-held-out LODO evaluation |
| Post-processing evaluated as one opaque stage | Resolved: raw, causal, and Viterbi stages are scored separately |
| Stable-state flicker | Reduced; stable-window Macro-F1 improves with post-processing |
| Transition errors | Not resolved; boundary scores remain lower than stable scores |
| Hallway recognition | Not resolved; F1 remains near zero under every reviewed stage |
| Rare/short visits such as room 508 | Not robust; outcome changes sharply by window and decoder |
| Unseen room/day combinations | Not resolved by closed-set LODO; 79 one-second windows are outside that estimand |
| Online decoding | Causal averaging is online-compatible; Viterbi backtracking is not |
| Dominant-beacon causal explanation | Not established by aggregate confusion counts |

Relabeling is not applied automatically. Feature-label ambiguity is evidence
for review, not proof that the label is wrong; geometry, visit continuity, and
an independent annotation source are required before changing ground truth.

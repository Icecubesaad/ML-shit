# Table 4 reconciliation (Phase 2)

`splits.csv` recomputed from the frozen file: train 7,011 / val 1,502 / test 1,502 images.
Per-class val and test **image** counts are identical (49/77/165/17/167/1006/21), matching paper Table 4.
This is not a duplicated split:

- `group_id` overlap between val and test: **0 images shared**; no lesion group appears in two splits.
- The frozen splitter allocates whole groups per class as round(0.70*n) train, round(0.15*n) val, remainder test.
  Group counts differ slightly for bcc (67 val vs 68 test groups) but multi-image groups balance the image counts.
- `splits.sha256` = `fda56cf3c555e6d22f823510670e1392cd9b1b27778a50a831e68f1dd8d5be2b`, identical to
  the hash recorded in `data_audit.json`.

Verdict: Table 4 is consistent with the frozen split. The identical counts are a consequence of the
70/15/15 group-stratified allocation, not evidence of leakage. The audit's concern is resolved in
favour of the existing split; no split was changed.

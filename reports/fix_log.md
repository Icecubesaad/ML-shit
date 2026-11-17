# Fix log — audit-driven repair, Phases 1 and 2

Every claim below is tied to a command that ran or a file that exists. Nothing is asserted from memory. Old results are untouched: this log only adds files; no checkpoint, dataset or previous report was overwritten.

## Phase 1 — fix the notebook (code only, no training)

### 1a. Geometry functions — audit claim checked, already resolved

Audit claim: `eval_ops` and `mask_to_model_frame` undefined, so the validation transform crashes.

Evidence from the current notebook (`notebooks/05_colab_standalone_training.ipynb`):
- `eval_ops` defined: **yes** (markdown section 6e, code cell embedding `src/dm_ahr_vit/xai/geometry.py`).
- `mask_to_model_frame` defined: **yes**, same cell.
- `pass` stubs in code cells: **0** (an earlier revision had six; they were replaced with real definitions).

No change needed. The claim was true of an earlier notebook revision, not the current one.

### 1b. Geometry unit tests — PASS

New file: `tests/test_eval_geometry.py` (4 tests).

Exact command: `py -m pytest tests/test_eval_geometry.py -q -p no:cacheprovider`

Output: `4 passed in 15.94s`.

What it proves:
- `eval_resize_size(224) == 255`.
- 600×450 → resized 340×255; the 224 centre crop covers 395.3 px of the original, i.e. **57.9%** of the photo area (tolerance 0.002).
- 128×128 → resized 255×255; crop covers **77.2%** (tolerance 0.002).
- A synthetic mask with a known marked region stays aligned with its image through `to_model_frame` / `mask_to_model_frame` for both storage sizes, and the output mask is binary {0,1}.

### 1c. Stub removal and pyflakes — PASS with caveat

Command: assembled all 25 implementation cells into a temporary module and ran `py -m pyflakes`.

Output: **0 undefined names**. There are 50 warnings of the form "redefinition of unused import" and "imported but unused"; these are an expected artefact of flattening 25 separate Python modules into one notebook namespace, not undefined names. No code was changed solely to silence them.

### 1d. `train_arm` resume edge cases — FIXED

Audit claim: a finished or early-stopped run can retrain one more epoch; `epoch` undefined if the loop never runs.

Verified in `src/dm_ahr_vit/vit/train.py`: the loop used `epoch` from `for epoch in range(start_epoch, epochs)` at the final `save_checkpoint`, so a run already at `epochs` would raise `NameError`; and `last.pt` carried no "completed" flag, so an early-stopped run resumed and trained again.

Changes made (same patch applied to the notebook's embedded `train_arm` cell):
- `epoch = start_epoch - 1` is set before the loop, so it always exists.
- `last.pt` meta now records `early_stopped` and `completed`.
- On load, if either flag is set, or `start_epoch >= epochs`, the function returns the existing result with `resumed_completed_run=True` and trains nothing.
- `final.pt` records `completed=True`; the loop sets `early_stopped = bad_epochs >= patience`.

Evidence: patches applied in `src/dm_ahr_vit/vit/train.py` and in the notebook cell; no test checkpoint was overwritten.

### 1e. End-to-end smoke — PASS

New file: `experiments/real_hair_validation/modal_phase1.py`.

Exact commands (both run on Modal, A100-80GB, workspace `icysolutionscorp`):
1. `modal run experiments/real_hair_validation/modal_phase1.py` — first attempt failed, `exit_code 1`, `ModuleNotFoundError: No module named 'sklearn'`. Log: `artifacts/logs/modal_runs/phase1_smoke__A100-80GB.log`.
2. Added `scikit-learn==1.5.2` to the smoke image and re-ran the same command — passed.

Second run evidence:
- `prepare`: 30 real images copied into an isolated `/vol/phase1_smoke/data` root; splits 20 train / 5 val / 5 test. Project artifacts untouched.
- `smoke`: `exit_code 0`, `seconds 23.9`, `result_exists True`, first epoch line `A/42 ep 0: train 1.9932 val 2.3078 f1 0.0000`. Training itself took 11.8 s; the rest was weight download and startup. The 0.0 validation F1 is expected for five validation images after one epoch and is not a quality claim.

Gate: **PASS** — notebook code runs from a clean kernel to and through the first ViT epoch.

GPU spend estimate (not an invoice): 23.9 s at $0.000694/s = **$0.017** for the smoke, plus image build (not charged as GPU seconds here).

### Phase 1 blockers

None. Caveat: the smoke used the project scripts on Modal with an isolated DATA_ROOT, not the literal Colab UI. The notebook's own cells were executed in a fresh namespace up to the first epoch in a separate local check; a full Colab session was not available.

## Phase 2 — data audit

### 2a. Table 4 reconciliation — PASS

Exact command: recomputed `splits.csv` group-by-split counts with pandas.

Evidence:
- Rows: 10,015; groups spanning more than one split: **0**.
- Image overlap between val and test: **0**.
- Per-class image counts val and test identical: 49/77/165/17/167/1006/21, total 1502 each — matches paper Table 4.
- Per-class **group** counts differ slightly (bcc: 67 val groups vs 68 test groups), so val and test are not the same images.
- `splits.sha256` = `fda56cf3c555e6d22f823510670e1392cd9b1b27778a50a831e68f1dd8d5be2b`, identical to the hash in `data_audit.json`.

Cause found in `src/dm_ahr_vit/data/splits.py`, `freeze_splits`: per class it assigns `round(0.70*n)` groups to train, `round(0.15*n)` to val, and the remainder to test. With these HAM10000 class sizes, the val and test group counts often come out equal, and multi-image groups balance the image counts exactly. Verdict: Table 4 is a correct consequence of the frozen 70/15/15 group-stratified split, not a duplicate split. **No split was changed.**

File written: `reports/real_hair_validation/table4_reconciliation.md`.

### 2b. Hair-present rate — PASS, first time reported

Exact command: grouped `data/manifests/hair_screening.csv` by split and diagnosis under the frozen 1% detector.

Evidence:
- Test: **186 hair-free of 1502** images, i.e. **87.6% flagged hair-present** — the audit's inferred figure is confirmed.
- Whole cohort: 8,993 hair-present of 10,015 (**89.8%**).
- Per-split and per-class breakdown written to `reports/real_hair_validation/screening_audit.csv`; short explanation in `screening_audit.md`.

Honest caveat: this is the rate under the current detector, not a ground-truth hair rate. On 500 real annotated masks in the external validation the same detector had image-level specificity 0.0175. The high flag rate is therefore a property of the screen and is reported as such.

### Phase 2 blockers

None.

## Files created in Phases 1–2

- `tests/test_eval_geometry.py`
- `experiments/real_hair_validation/modal_phase1.py`
- `reports/real_hair_validation/screening_audit.csv`
- `reports/real_hair_validation/screening_audit.md`
- `reports/real_hair_validation/table4_reconciliation.md`
- `reports/fix_log.md` (this file)
- Modified: `src/dm_ahr_vit/vit/train.py`, `notebooks/05_colab_standalone_training.ipynb` (the same resume fix)

## NOT VERIFIED (with reason)

- Full Colab UI execution: no Colab session available; the code was executed as a fresh-namespace cell run locally and as project scripts on Modal.
- Any result of Phase 3 or later: not run. Per the plan, no detector retuning, no retraining and no re-evaluation happened in Phases 1–2.

## Status

Phases 1 and 2: **PASS**. Stopping here, as instructed, before any retraining. Phase 3 (detector retune on the 395 independent images, with its specificity ≥ 0.5 gate) has not been started.

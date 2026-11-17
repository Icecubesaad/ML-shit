# Decision rule — further DM-AHR generator training (committed BEFORE any run)

**Status: committed before step 1 and before any CPU/GPU analysis job for this task.**
This file is the pre-registration for the iteration decision. It is not edited after evidence is seen.

## Question

Would continuing DM-AHR generator training toward 660k iterations (plus fine-tuning) improve
results? The immediate GO action is **continue to 100k with checkpoints every 25k, then re-apply
this rule and ask the user before any further spend**. The 660k figure is the hypothetical end
goal stated in the request, not an authorised single run; it is recorded here so the two are not
confused. No job may jump straight to 660k.

## Data boundary (non-negotiable)

- **Validation data only for every decision.** The locked test split, its labels, masks and
  metrics are never read by any script in this task. Scripts must not call the test-split
  evaluation paths (`scripts/07_eval_vit.py`, `scripts/08_xai.py`, `scripts/58_xai_corrected.py`)
  or read `split == 'test'` rows.
- The held-out synthetic pairs used in step 2 and the real-strand pairs in steps 2–3 are drawn
  from **train/val sources only**, with source IDs verified by hash before use.
- No threshold is tuned on the test set. Null/negative results are reported as found.

## Decision rule (GO only if ALL four hold)

1. **(a) Val L1 still falling.** In `dmahr_train.json` and `dmahr_v2_train.json`, validation L1
   is still decreasing over the last 20,000 recorded iterations, with the train/val gap reported
   (a widening gap is evidence of overfitting, not grounds for more training).
2. **(b) >= 0.5 dB per doubling.** On the *same* validation synthetic pairs, the same noise seeds,
   fp32, and the committed sampler, partial-noise sampling (t0=600, 50 strided DDIM steps) gains
   at least **0.5 PSNR dB per doubling of iterations**, with a paired-bootstrap 95% CI whose lower
   bound is above zero.
3. **(c) Ceiling within 0.01.** Regenerating hair-present validation images with the RELEASED
   checkpoint through the same Cv2 sampler and training ViTs for 8 seeds with the frozen recipe
   gives validation macro-F1 **within 0.01 of arm A**. If it is more than 0.01 below A, record
   that more generator training is unlikely to beat A.
4. **(d) No recipe shortcut.** None of the step-3 variants (+10k from 50k: control unchanged,
   no grad clip, per-sample noise level, batch 16, DERMAHAIR train-source pairs added) matches
   the control's gain at equal iteration count on the step-2 metrics.

**If any of (a)–(d) fails: STOP and report exactly which condition failed and the evidence.**
A STOP is a successful outcome; it is not a reason to tune the rule.

## Cost governance

- **Hard cap: USD 10.00 for this entire task.** No exception without asking the user.
- Before **each job estimated to cost more than USD 2**, run a small benchmark (seconds/unit on
  the chosen GPU) and write the estimate to `reports/iteration_cost_log.md` *before* launching.
- Log per job: wall-clock seconds, GPU type, measured seconds/unit, estimated USD, and the source
  of the price. If Modal pricing cannot be verified, record GPU-hours only and say so; do not
  guess a price.
- When the remaining budget cannot cover a step, skip it, mark it `NOT VERIFIED — insufficient
  budget`, and still decide with the evidence that exists.

## Output

`reports/iteration_decision.md` contains every table, the curve plot (`artifacts/figures/`),
the verdict, and the full cost log. Scripts for this task are committed before the first run.

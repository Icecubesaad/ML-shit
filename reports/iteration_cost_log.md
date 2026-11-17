# Iteration-decision cost log (task cap USD 10.00)

Price source: `reports/compute_plan.md` records Modal prices read from
`https://modal.com/pricing` on 2026-11-11: A100-80GB **0.000694 USD/s**,
A100-40GB 0.000583 USD/s. These are the only prices used below. GPU type and
wall-clock are recorded per job in `artifacts/logs/run_registry.jsonl`.

## Budget

| Item | USD |
|---|---:|
| Hard cap for this task | 10.00 |
| Spent before first job | 0.00 |

## Planned jobs and estimates (written BEFORE running)

| Job | Basis for estimate | GPU | Est. seconds | Est. USD | Benchmark required (>$2)? |
|---|---|---|---:|---:|---|
| Step 1 L1 history | local file read, no GPU | none | 0 | 0.00 | no |
| Step 2 curve scoring, 3 available snapshots | archived `it*.json` on A100-40GB took 116–117 s/snapshot; fp32 and extra metrics allow 200 s/snapshot | A100-80GB | 600 | 0.42 | no |
| Step 3 five +10k trainings (if reached) | measured 20k iterations = 672.7 s (`dmahr_v2_train.json`), so 10k = 336 s each | A100-80GB | 1680 | 1.17 | no per job (0.23 each) |
| Step 4 released-checkpoint regenerate + 8 ViTs (if reached) | 50 regenerations at ~4 s + 8 x ~400 s training | A100-80GB | 3450 | 2.40 | **yes — benchmark before launching** |

Snapshots for step 2 that do NOT exist on the volume (`gen_iter025000.pth`,
`gen_iter055000.pth`) are skipped by the script; their archived aggregate JSONs are
used as evidence but carry no per-pair data, so no paired CI can be computed for them.

## Actual spend

| Job | Date | GPU | Wall seconds | USD | Evidence |
|---|---|---|---:|---:|---|
| Step 2 attempt 1 (failed at import in `prepare_val`, no GPU work) | 2026-11-17 | none | 0 | 0.00 | `iteration_curve__A100-80GB.log`: ImportError relative import |
| Step 2 attempt 2 (failed at missing archived val assets, no GPU work) | 2026-11-17 | none | 0 | 0.00 | FileNotFoundError on old-account asset path |
| Step 2 attempt 3 (sampling completed, failed at strand-metric size mismatch) | 2026-11-17 | A100-80GB | ~140 (model load + partial + 2000-step pure + grid + LPIPS) | ~0.10 | log: IndexError boolean mask 128 vs 450; pure-noise loop reached 2000/2000 in 1:45 |
| Running total | | | | ~0.10 | |

## Notes

- A separate pre-existing 8-seed extension job from a previous task may still be
  running on the same Modal account; its cost is **not** charged to this task's log.
- No job in this task exceeds USD 2, so the pre-job benchmark rule is triggered only
  for step 4, which is not launched unless all earlier GO gates pass.
- If the remaining budget cannot cover a step, it is skipped and marked
  `NOT VERIFIED — insufficient budget` in `reports/iteration_decision.md`.

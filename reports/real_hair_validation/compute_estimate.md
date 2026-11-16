# Phase 11 compute estimate (Gate 1 evidence)

## Verified measurements

- Smoke run (`artifacts/real_hair_validation/experiments/smoke_test/checkpoint_smoke_test.json`):
  `gpu_seconds = 9.75` on `NVIDIA A100 80GB PCIe` (79.3 GiB, SM 8.0).
- Work performed in that time: 2 checkpoints loaded strictly (374 state keys each), each run
  twice with the same seed on 5 validation images = **20 image-generations** of 50 strided DDIM
  steps at 128×128, FP32, `t0=600`, plus persistence/reload verification.
- Approximate throughput including checkpoint loading: ~0.5 s per generated image at batch 5.
  Real production batches (32) should be at least as efficient; this is an upper bound.

## Rate card (from `modal billing rates`, this workspace)

- A100-80GB $2.50/h ($0.000694/s), A100-40GB $2.10/h, L4 $0.80/h, A10G $1.10/h.
- CPU $0.0473/core/h, memory $0.008/GiB/h, volume $0.09/GiB/month, egress $0.04/GiB.
- `modal billing summary`: metered cost 4.37, credits −4.19, **billed cost $0.00**. No numeric
  remaining-credit balance is exposed; the user's ~$8 is the working cap.

## Projection for the central experiment (NOT an invoice)

Assume 100 independently sourced hair-positive images plus controls (say 120 images), two
checkpoints, one primary production run each, plus an exploratory `t0=1000` pass on the same 120:

| Item | Quantity | GPU seconds (upper bound) | Est. USD |
|---|---:|---:|---:|
| Benchmark 10–20 images (Phase 11 rule) | 1 run | ~10 | $0.01 |
| Primary: base checkpoint, 120 images | 120 | ~60 | $0.04 |
| Primary: fine-tuned checkpoint, 120 images | 120 | ~60 | $0.04 |
| Exploratory t0=1000, 120 images × 2 | 240 | ~120 | $0.08 |
| DullRazor + detector eval (CPU) | ~120 images | 0 GPU | <$0.02 |
| Volume storage (dedicated volume, ~a few GiB, pro-rated) | 1 month | — | <$0.50 |
| **Contingency ×3** | — | — | **~$2.00** |

Projected total with contingency: **≈ $2.00**, far below the ~$8 working budget. No parallel GPU
shards will be launched. Phase 7 will benchmark 10–20 images before any full inference run and
this estimate will be updated with the measured throughput.

## Rules applied

- Downloads/extraction use CPU workers; one GPU worker initially; resumable jobs; persistent volume.
- Do not silently reduce the evaluation sample size or change the protocol to fit the budget.
- Stop before projected cost exceeds the available budget; preserve contingency for final evaluation.

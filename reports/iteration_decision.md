# Iteration decision — further DM-AHR training

Research prototype only. Not clinically validated and not intended for diagnosis or patient care.

Validation data only. The test split was never read by this task. Rule committed before any run: `reports/iteration_rule.md` (commit 9255474).

## Step 1 — L1 history (free)

| log | field | iterations | best_val_l1 | seconds | has_history | value | n_points | first | last | min | min_iteration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dmahr_train | summary | 20000.000 | 0.018 | 1225.937 | False |  |  |  |  |  |  |
| dmahr_v2_train | summary | 20000.000 | 0.012 | 672.681 | False |  |  |  |  |  |  |
| dmahr_v2_train_log | last_20k_val_l1_slope_per_1k |  |  |  |  | -0.112 | 20.000 | 0.014 | 0.014 | 0.012 | 13000.000 |
| dmahr_v2_train_log | last_10k_val_l1_slope_per_1k |  |  |  |  | -0.036 | 10.000 | 0.013 | 0.014 |  |  |
| dmahr_v2_train_log | train_val_gap_mean_last_20k |  |  |  |  | -0.001 | 20.000 | -0.000 | 0.006 |  |  |
| dmahr_v2_train_log | full_val_l1_slope_per_1k |  |  |  |  | -0.112 | 20.000 | 0.014 | 0.014 | 0.012 | 13000.000 |

## Step 2a — archived learning curve (same val pairs, fp32 re-scoring below)

| iteration | partial_psnr | partial_ssim | partial_gain_db | pure_psnr | pure_gain_db | one_step_mae_t500 | zeroed_over_kept_hair_t200 | shuffled_over_kept_hair_t200 | val_pairs_hash | checkpoint_sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
| 25000 | 29.838 | 0.917 | 8.340 | 11.431 | -9.870 | 0.016 | 3.936 | 5.310 | 7043ff364ba475c4bae9ffd0885599957a293fb44042d0e12f5079ec7680df67 | 692163e929d3e4d3740f21decd17a9a3ea8b32ec922fd6a422364f3df04d4d4e |
| 50000 | 30.791 | 0.926 | 9.294 | 14.056 | -7.245 | 0.014 | 5.429 | 6.788 | 7043ff364ba475c4bae9ffd0885599957a293fb44042d0e12f5079ec7680df67 | 77031da6dbea73ad751adf9aaddf6c7cfb2cd762582eb1c8b2bb3956d188b845 |
| 55000 | 30.135 | 0.912 | 8.637 | 8.565 | -12.737 | 0.013 | 5.776 | 7.041 | 7043ff364ba475c4bae9ffd0885599957a293fb44042d0e12f5079ec7680df67 | e502c828658e4fd445ca0a6d79badb8185f6a2048be682d231fa6331eed4cc61 |

## Step 2b — re-scored snapshots (fp32, committed sampler, paired bootstrap over lesion groups)

| snapshot | n_pairs | partial_gain_db | ci_low | ci_high | p_raw | partial_psnr | partial_ssim | partial_lpips | strand_removal | lesion_changed | pure_psnr | val_pairs_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base50k | 50 | 9.730 | 9.110 | 10.320 | 0.00020 | 38.971 | 0.934 | 0.027 | 0.418 | 0.016 | 13.768 | 7043ff364ba475c4bae9ffd0885599957a293fb44042d0e12f5079ec7680df67 |
| ft_v2 | 50 | 10.209 | 9.589 | 10.834 | 0.00020 | 39.451 | 0.945 | 0.007 | 0.561 | 0.017 | 12.153 | 7043ff364ba475c4bae9ffd0885599957a293fb44042d0e12f5079ec7680df67 |
| released | 50 | 6.853 | 6.447 | 7.313 | 0.00020 | 36.095 | 0.911 | 0.185 | -0.316 | 0.027 | 34.889 | 7043ff364ba475c4bae9ffd0885599957a293fb44042d0e12f5079ec7680df67 |

## Step 2c — gain per doubling

| from_k | to_k | gain_db | ci_low | ci_high | per_pair_ci |
|---|---|---|---|---|---|
| 25 | 50 | 0.953 |  |  | NOT AVAILABLE (checkpoints absent) |

## Step 2d — paired snapshot comparisons (same validation pairs, 10k lesion-group bootstrap)

| comparison | iterations | mean | ci_low | ci_high | p_raw | n_groups |
|---|---|---|---|---|---|---|
| ft_v2(70k) - base50k | 50k -> 70k (1.4x) | 0.479 | 0.186 | 0.769 | 0.00120 | 50 |
| released(~725k) - base50k | 50k -> ~725k (14.5x) | -2.877 | -3.295 | -2.466 | 0.00020 | 50 |

![Learning curve](artifacts/figures/iteration_curve.png)

## Gates

- (a) val L1 still falling: **PASS** — last-10k val-L1 slope = -0.036039 per 1k iterations
  - Caveat: best val L1 0.012161 at iteration 13000; last-5k slope +0.160602 per 1k, so the most recent 5k are rising.
- (b) >=0.5 dB/doubling with CI>0: **FAIL** — NOT ESTABLISHED: no doubling pair exists with per-pair outputs; archived 25k->50k doubling = +0.953 dB but no per-pair CI (checkpoints absent); released (~725k) minus base50k = -2.877 dB (95% CI -3.295 to -2.466), so the long-horizon evidence is negative
- (c) ceiling within 0.01: **NOT RUN** — NOT RUN — decision already STOP; see cost log for estimate
- (d) no recipe shortcut: **NOT RUN** — NOT RUN — decision already STOP; see cost log for estimate

## Verdict: **STOP**

The pre-registered rule requires all four gates for GO. At least one gate failed, so the required action is STOP. Steps 3 and 4 were not launched: they cannot change a STOP verdict and would spend the capped budget without affecting the decision. Their cost estimates are recorded in `reports/iteration_cost_log.md`.

This is a successful outcome: the evidence does not support further generator training.
No checkpoint was trained past 50k/20k-fine-tune and no test metric was consulted.

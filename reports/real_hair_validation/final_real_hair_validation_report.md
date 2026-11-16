# External real-world hair-removal validation — final report

**Research prototype only. Not clinically validated; no diagnostic or patient-facing use.**

## 1. Executive summary
Both surviving DM-AHR generators loaded strictly and ran on A100-80GB with finite, reproducible
128 px outputs. On 500 independent Mendeley cases (after overlap exclusion),
fine-tuning reduced the residual-hair proxy but increased alteration of non-hair skin; lesion
preservation is NOT VERIFIED because the independent set has no lesion masks. This is a rigorous
proxy study, not proof of real-hair removal quality.

## 2. Existing model discovery
57 checkpoint files inventoried with SHA-256
(`checkpoint_inventory.csv`). The ~56.3k-iteration continuation generator is absent; the base used is
the 50k repaired generator, the fine-tuned is 50k+20k. See `discovery_report.md`.

## 3. Checkpoint verification
Smoke: finite, reproducible, strict load, 50 DDIM steps, t0=600, FP32, outputs persisted in Modal.
Contact sheet inspected (agent-reviewed). See `checkpoint_smoke_test.json`, `checkpoint_smoke_review.md`.

## 4. Datasets and licenses
Mendeley (DOI 10.17632/j5ywpd2p27.2, CC0): 500 pairs.
DermArtifactDB (DOI 10.5281/zenodo.18324962): annotation archive only; images require source access.
ISIC2019 / PH2 / Derm7pt image access requires registration or was unreachable — reported, not bypassed.

## 5. Integrity and overlap
5 documented methods. Independent A=395, exact B=7,
near-duplicate C=98, uncertain D=0. Only class A scored.

## 6. Hair-detector validation
500 real masks: pixel precision 0.249, recall
0.816, Dice 0.382, IoU
0.236; image sensitivity 0.998,
specificity 0.018. The frozen threshold over-detects on real hair.
No threshold was tuned on these masks.

## 7. DM-AHR inference methodology
395 images; 348.6 s GPU; config in `experiment_config.json`.
Outputs in the dedicated Modal volume; inputs never modified.

## 8. Base vs fine-tuned
Hair-region change difference +0.0173
(95% CI +0.0121 to
+0.0223; Holm p
0.0); residual proxy
-0.0272 (CI
-0.0363 to
-0.0182); non-hair change
+0.0158 (CI
+0.0115 to
+0.0198). Paired image-level bootstrap.

## 9. DullRazor comparison
Highest hair-region change (0.585) and
lowest residual proxy (0.186) but also
highest non-hair change (0.159).
Change is not correctness: no paired hairless reference exists for natural hair.

## 10. Real-hair results
500 images scored; per-method proxies in `method_summary.csv`,
density strata in `density_stratified_results.csv`.

## 11. Lesion preservation
NOT VERIFIED — no lesion masks in the independent set; never auto-generated.

## 12. Failure cases
Top 20 by non-hair alteration in `failure_cases.csv`.

## 13. Human review protocol
40 randomized 4-panel sheets plus a separate mapping and assessment form in
`blinded_human_review_kit/`. **No AI visual opinion is presented as a blinded human rating.**
Agent-reviewed qualitative observations are labelled as such in `checkpoint_smoke_review.md`.

## 14. Compute consumption
GPU estimate from measured seconds in `spend_summary.csv`; billing snapshot in `billing_snapshot.txt`.

## 15. Limitations
No paired natural hairless reference; no lesion masks; residual-hair proxy uses the same detector
whose real-hair specificity is poor; DermArtifactDB images not obtained; 3 excluded overlap classes.

## 16. Conclusions
The experiment is scientifically sufficient to report proxy-level natural-hair findings and the
base/fine-tuned trade-off, but NOT sufficient to support a claim of successful real-world hair
removal without paired references, lesion masks and blinded human ratings.

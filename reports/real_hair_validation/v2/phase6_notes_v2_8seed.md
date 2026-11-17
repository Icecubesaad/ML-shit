# Phase 6 notes (v2 rerun)

- Classification/XAI/quality bootstrap: 10,000 lesion-group replicates; classification/quality seed 20260101, XAI seed 20261113.
- Two-sided p is floored by the bootstrap at 2/(reps+1) = 0.00019998; one-sided floor 1/(reps+1) = 9.999e-05. A reported p can never be smaller.
- Holm correction is applied separately within `planned` and `secondary_control` families.
- Cv2 replaces C as the diffusion arm. A-A* and Cv2-A* are resolution controls, reported separately from the planned family.
- XAI includes Grad-CAM at the last encoder block `ln_1` (primary) and the patch grid (second layer, sensitivity), each in the as-is and off-centre (shift <=30%) frames.
- Image quality keeps the 128px and native geometry policies separate; they are never averaged.

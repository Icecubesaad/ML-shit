# Phase 1 smoke visual review (agent-reviewed)

Reviewed artifact: `reports/real_hair_validation/smoke_contact_sheet.png` (pulled from
`/vol/artifacts/real_hair_validation/experiments/smoke_test/contact_sheet.png`). Five naturally
hairy validation images, columns: original (ISIC id), base generator, fine-tuned generator.

**This is an AI agent's qualitative observation, not a blinded human assessment.** No clinical
claim is made.

## What I saw (agent-reviewed)

- **Row 1 (ISIC_0025394):** thick dark hairs cross a brown lesion. Both generators remove the
  visible strands. The lesion becomes flatter and its finepigment texture is smudged; the
  fine-tuned column is not obviously closer to the original than the base.
- **Row 2 (ISIC_0029241):** dense white/dark hair over a reddish lesion. Hair is largely gone in
  both outputs; global colour and lesion position are preserved, but fine texture is washed out.
  The fine-tuned output looks marginally sharper than the base here, not dramatically different.
- **Row 3 (ISIC_0032061):** hair crossing a small red lesion. Both outputs remove the strand and
  keep the lesion centred, with some blurring of the lesion edge.
- **Row 4 (ISIC_0028228):** fine hair over a pigmented network. Both outputs remove hair; the
  network pattern becomes waxy/smoothed. Differences between the two checkpoints are subtle.
- **Row 5 (ISIC_0031334):** hair over a diffuse lesion. Both remove most strands; the result is
  slightly blurry/washed relative to the original.

## Interpretation (provisional)

Hair is visibly reduced by both generators, but at a cost to fine texture and lesion-boundary
sharpness. In this 5-image smoke there is **no clear visual advantage** for the fine-tuned
checkpoint over the 50k base. This is consistent with the project's earlier finding that the
real-photo hair-removal gate was not met, but five images cannot establish that; it will be tested
in the central experiment (Phase 7+) with measured metrics and a blinded human-review kit.

## Automated checks (from checkpoint_smoke_test.json)

- Both checkpoints: strict state-dict load, no missing/unexpected parameters reported.
- Both: `finite=true`, output shape `[5, 3, 128, 128]`, 50 visited DDIM steps, `start_t=600`,
  FP32 (`bf16=false`).
- Both: `reproducible_fixed_seed=true` and second-run stats identical.
- Both: outputs saved to and reloaded from persistent Modal storage; 128×128 size; pixel minima
  and maxima within [0, 255]; no blank/black/saturated images (per-output std and extreme-pixel
  fraction recorded in the JSON).
- Original geometry preserved in the sense that the full 600×450 photograph was squashed to the
  approved 128×128 input and the output is the same 128×128 frame; no per-image crop was changed.

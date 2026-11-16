# Pre-registration: exploratory A*-fov control (client fix item 9)

**Status: design fixed before any A*-fov image was built or any model trained.** This is an
exploratory, post-hoc control requested in the client's final fix round. It does not change
the pre-registered hypotheses H1–H3 or their families; its rows are labelled exploratory and
are reported beside, not inside, the planned comparisons.

## Question

A* stores hair-present images at 128 px, which also squashes the 600 × 450 photograph into a
square and changes the field of view the ViT sees (77.2% of the photo, versus 57.9% for A).
Is any A*-versus-A difference due to that framing change rather than to resolution?

## Design (frozen)

- **Storage form (hair-present):** centre-square crop of the original image with side
  `min(w, h)` (= 450 × 450 for a 600 × 450 photo), resized to 128 × 128 with BICUBIC
  resampling. This keeps A's field of view while matching A*'s storage resolution.
- **Hair-absent images:** passthrough at the original 600 × 450, as in every arm (no re-screening).
- **Hair-present flag:** read from the frozen `data/manifests/hair_screening.csv`; no image is
  re-screened and no threshold is tuned.
- **Model view:** the unchanged evaluation transform (resize shorter side to 255, centre-crop
  to 224 × 224) then sees exactly the same window of the photograph as arm A (57.9% of the
  photo). No change to `src/dm_ahr_vit/xai/geometry.py`.
- **Training:** the same ViT-B/16 recipe, the same eight seeds (42, 123, 2026, 7, 11, 17, 23,
  31), the same split and the same evaluation code as every other arm. The only difference is
  the stored input image.
- **Exploratory comparisons** in `53_pooled_analysis.py`: A*-fov − A* and A*-fov − A, in the
  exploratory family, with the same two-level (lesion group + seed) bootstrap and Holm
  correction. The planned family (A − B) and the secondary control (A − A*) are untouched.
- **No XAI** for this arm (not requested). The test set is scored once per frozen checkpoint,
  per the project protocol.

## Endpoints and decision rule

Primary endpoint: test macro-F1 on the full test set, pooled over eight seeds. Secondary:
per-image log-loss and accuracy. A row is reported with its test name, raw p, Holm-adjusted
p, effect size (difference) and 95% CI; a difference is called established only under the
same rule as the other exploratory rows. A null or negative result is a valid outcome and will
be reported as such.

## What would falsify the framing confound

If A*-fov lands closer to A than A* does on macro-F1, that supports the reading that the
128 px squash's field-of-view change contributed to the A* result. If it stays near A*, the
framing explanation is weakened. Either way the number comes from the trained checkpoints,
not from a target.

## Reproduction

```bash
modal run --detach modal_app.py::astarfov_build
modal run --detach modal_app.py::astarfov_pipeline   # 8 seeds, train + eval, resumable
modal run --detach modal_app.py::pooled_run          # 53 + 55 regenerate pooled tables
```

Artifacts: `artifacts/logs/astarfov_build.json`, `artifacts/panels/astarfov_build.jpg`
(agent-reviewed contact sheet), per-seed predictions under `artifacts/predictions/`, and the
refreshed `artifacts/logs/pooled_per_seed.csv` / `statistical_comparisons*.csv`.

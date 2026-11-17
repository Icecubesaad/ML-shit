# Phase 5-7 notes

- DERMAHAIR: NOT VERIFIED as an independent external benchmark. The released checkpoint's training
  provenance is unknown and may overlap these public pairs. Numbers in dermahair_psnr_ssim_v2.csv
  are descriptive only.
- H1 image quality: two geometry policies (128px and native) are reported separately in
  image_quality_v2.csv; they are never averaged.
- Lesion masks are single-rater, FCN-initialised and dermatologist-corrected (Tschandl et al.),
  not expert-traced.
- Old B-arm results are SUPERSEDED by the v2 rerun under /phase4/artifacts.
- Blinded human review kit is prepared but EMPTY: no ratings were invented.
- Lost checkpoints: the ~56.3k continuation generator and the first defective generator are absent.
  Not every historical number can be regenerated from the current workspace.

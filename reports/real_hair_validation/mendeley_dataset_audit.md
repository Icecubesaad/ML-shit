# Mendeley hair-mask dataset audit

Source: https://data.mendeley.com/datasets/j5ywpd2p27/2 · DOI 10.17632/j5ywpd2p27.2 · licence CC0.
Archive: 444270971 bytes, SHA-256 `3516674e2527b7dd3c0ddc9c65b7b2018dc63eea3e062785cddcb8769da735f5`, ZIP integrity `PASS`, 1507 members.

- Images: **500**; masks: **500**; valid image–mask pairs: **500**; failures: 0.
- Filename matching: 500 matched stems, ? unmatched images, ? unmatched masks.
- Dimensions: 512 × 512 for sampled images. Masks: 500 binary, 0 non-binary under the documented ≤4 unique grey values rule.
- Duplicate images by byte SHA-256: 0 duplicated hashes (0 excess files).
- Hair-mask white-pixel coverage: min 0.0, median 0.06866, max 0.65679; 443 images at ≥1%.
- Overlay visual check (**agent-reviewed**, not expert-reviewed): sampled original/mask/overlay trios in `mendeley_contact_sheet.png` show the white mask lines following the visible strands and overlays correctly registered. Thin strands are only a few pixels wide, so small misalignments can dominate pixel-level scores.
- PSNR/SSIM were **not** computed against these masks: they are segmentation annotations, not hairless reconstruction references.

Per-pair data: `mendeley_pairs.csv` on the volume.

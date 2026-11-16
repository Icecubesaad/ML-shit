# Dataset integrity report (external sources)

## Mendeley
500/500 valid pairs, all binary masks, no duplicate image hashes; details in `mendeley_dataset_audit.md`.

## DermArtifactDB (Zenodo 10.5281/zenodo.18324962)
Archive 4139370781 bytes, SHA-256 `6b02ee2981ec075b5e3661ba7c1d67553a1a8224e20ef1f131681712c3f0b01d`, ZIP integrity `PASS`.
Extracted PNGs: 21811. Annotation CSVs:

| CSV | Rows | Columns |
|---|---:|---|
| annotations_ISIC2019.csv | 20599 | image, dark_corner, hair, gel_border, gel_bubble, ruler, ink, patches |
| annotations_PH2.csv | 200 | image, dark_corner, hair, gel_border, gel_bubble, ruler, ink, patches |
| annotations_derm7pt.csv | 1011 | image, dark_corner, hair, gel_border, gel_bubble, ruler, ink, patches |

The annotation archive does **not** contain the original dermoscopic images; image-level hair labels must be matched to source repositories. We do not treat CSV rows as image data.

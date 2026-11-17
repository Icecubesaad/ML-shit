# Code guide — standalone Colab training notebook

**Research prototype only. Not clinically validated and not intended for diagnosis, triage or patient care.**

Notebook: `notebooks/05_colab_standalone_training.ipynb` (80 cells: 40 markdown + 40 code).
It is one self-contained file: no external `.py` imports, no companion scripts, no local project
tree required. Every function used by the pipeline is defined in a code cell before it is called.

## 1. What this notebook actually does

It reproduces the **original** end-to-end pipeline in a single Colab runtime:

1. Download ISIC collection 212 (the client's 11,720-image collection) and the HAM10000
   metadata + lesion masks; quarantine the ~1,705 extra collection images.
2. Recompute the confirmatory 10,015-image cohort, verify near-duplicate links, and freeze a
   lesion-grouped 70/15/15 split (groups = `lesion_id` merged with confirmed near-duplicates).
3. Screen hair with the frozen detector; build Arm B (DullRazor), Arm A* (128 px control) and
   held-out synthetic hair pairs.
4. Harvest real hair strands from *train/val only* and build the train-only fine-tuning pairs.
5. Train the repaired DM-AHR generator (50k iterations), fine-tune it (20k iterations), and
   sample Arm Cv2 images (128 px, 50 strided DDIM steps, t0 = 600, FP32).
6. Train the matched ViT-B/16 classifiers for all arms and seeds.
7. Evaluate H1 (paired PSNR/SSIM), H2 (classification + per-image predictions), H3
   (Grad-CAM + attention rollout vs lesion masks), and the lesion-cluster bootstrap statistics.

The code cells are the project's real `src/dm_ahr_vit/*` modules pasted in dependency order,
not stubs. `py_compile` on all 40 code cells reports zero syntax errors.

## 2. Before you run

- **Runtime:** Python 3, GPU required for cells 61–73. An A100-class GPU is comfortable;
  smaller GPUs work but take longer and may need a lower batch size in `configs/vit.json`.
- **Drive:** mount Google Drive when prompted; all data, checkpoints and outputs live under
  the notebook's `DATA_ROOT` / `ARTIFACT_ROOT` paths, never in the notebook file.
- **Downloads:** ISIC API access (public collection 212) and the HAM10000 metadata/mask archives.
  The first cells check file hashes; a failed download is reported, not skipped silently.
- **Disk:** tens of GB. Check quota before Phase 1.
- **`FAST_MODE`:** cell 2 sets `FAST_MODE = False` (published setting, eight seeds). Setting it
  to `True` shortens the seed list for a smoke test. **Never compare a FAST_MODE run with the
  reported eight-seed numbers** — different seed sets are different experiments.

## 3. Cell map

| Cells | Phase | What the code does |
|---|---|---|
| 0–2 | Setup | Title, instructions, `FAST_MODE`, full seed list (42, 123, 2026, 7, 11, 17, 23, 31) |
| 3–4 | Paths | Configuration loader, directory creation, hash helpers |
| 5–6 | Data | ISIC 212 collection download with manifest verification |
| 7–8 | Data | HAM10000 metadata, cohort membership, lesion-mask discovery |
| 9–10 | Audit | File manifest, dHash near-duplicate candidates and confirmation |
| 11–12 | Split | Lesion-grouped 70/15/15 freeze; zero group overlap asserted |
| 13–14 | Masks | X1 mask manifest and model-frame mask loading |
| 15–16 | Hair | Multi-orientation black-hat/top-hat detector (Arm B screen) |
| 17–18 | Hair | DullRazor morphology + Telea inpainting (Arm B) |
| 19–20 | Hair | Seeded synthetic hair compositor for paired evaluation |
| 21–22 | Hair | Real-strand harvest (train/val sources only) |
| 23–24 | Hair | Real-strand composite dataset for diffusion fine-tuning |
| 25–27 | Diffusion | Vendored upstream U-Net, noise schedule and losses |
| 28–30 | Diffusion | Model factory and the adapter used by the trainer/sampler |
| 31–32 | Diffusion | Leakage-controlled training loop with resumable checkpoints |
| 33–34 | ViT | `ArmDataset`, transforms, seven-class head, metrics |
| 35–36 | ViT | Matched ViT-B/16 training with early stopping and resume |
| 37–38 | XAI geometry | One definition of the model's image/mask frame |
| 39–40 | XAI | Grad-CAM, attention rollout, lesion overlap, pointing game |
| 41–42 | Stats | Lesion-cluster bootstrap, Wilcoxon, Holm |
| 43–44 | Metrics | Composite image metrics with known clean targets |
| 45–52 | Run 1 | Download + audit + freeze the split + screen/build B/A*/pairs |
| 53–58 | Run 2 | Real-strand paste-back training pairs (train/val only) |
| 59–60 | Run 3 | Train the repaired base generator (50k) |
| 61–62 | Run 4 | Fine-tune on real strands (20k) |
| 63–64 | Run 5 | Sample Cv2 images (50 strided steps, t0 = 600) |
| 65–67 | Run 6 | Train the ViT classifiers (last training phase) |
| 68–71 | Run 7 | H1 image quality, H2 predictions/metrics, H3 localization |
| 72–75 | Run 8 | Lesion-cluster statistics, reproducibility notes, model inventory |

## 4. Design rules encoded in the notebook

- The split is frozen before any learned component sees data; groups never cross splits.
- The hair detector and its thresholds are tuned on train/val sources only.
- Arms A, B, A* and Cv2 share the same split, recipe, augmentation, seeds and evaluation code;
  only the input condition differs.
- The test set is evaluated once per frozen checkpoint, after the protocol and endpoints are fixed.
- Confidence intervals use lesion-group resampling, not image-level resampling; Holm correction
  is applied within declared families only.
- Every skip or bad image is counted and written to a file with a reason; nothing is silent.

## 5. Outputs to expect

Under `ARTIFACT_ROOT`: `checkpoints/dmahr`, `checkpoints/vit/*_seed*`, `predictions/*.csv`,
`logs/*.json`, `logs/xai_localization_*.csv`, and the synthetic/real composite images.
The final cells print an inventory of the produced model files with their SHA-256 hashes.

## 6. Important relation to the audit (do not mix)

This notebook implements the **original** pipeline and its original frozen detector. The later
audit found that detector's held-out specificity was 0.038 and retuned it to 0.923 on
independent labelled data; the affected arms were then reprocessed, retrained and re-evaluated.
Those v2 results are in `reports/real_hair_validation/audit_v2_integration.md` and the CSVs in
`reports/real_hair_validation/v2/`. Running this notebook reproduces the original numbers, **not**
the v2 audit numbers. Do not paste results from the two pipelines into the same table.

## 7. Troubleshooting

| Symptom | Cause / action |
|---|---|
| Cell 5 fails with a 403 | ISIC API rate limit or network; retry later, do not bypass the manifest check |
| Mask download fails | The Harvard Dataverse archive is large; retry and verify the hash |
| CUDA out of memory | Lower batch size in `configs/vit.json` or use a larger GPU; do not change the model |
| Resume trains again | Use the checkpoint logic in the training cell; a completed run is detected and skipped |
| Different numbers from the report | Check `FAST_MODE`, seed list, split hash and detector config before assuming a bug |

## 8. Verification performed for this guide

- 80 cells, 40 code / 40 markdown, nbformat 4.5, Python 3 kernel.
- All 40 code cells compile (`compile(..., 'exec')`) with zero syntax errors.
- No placeholder stubs: the only `NotImplementedError` sites are legitimate unsupported-schedule
  branches in the vendored upstream code.
- No notebook outputs are stored, so a fresh run starts from a clean state.

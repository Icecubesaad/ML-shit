# Discovery report — external real-world hair-removal validation

Generated before any external dataset download or GPU inference. Every claim below is from a
local artifact or a read-only Modal query performed for this phase. No credentials, tokens or
environment variables containing secrets are reported.

## 1. Repository and commit

- Repository: `C:\Users\saadq\Downloads\ML-shit` (branch `hoplite/dm-ahr-vit-pipeline`).
- HEAD at discovery: `d5ad3321ae1f84db6687155487d347bd3ab2318e`.
- Working tree had uncommitted regenerated reports from the preceding fix round; no source file
  in this experiment's namespace modifies the original project's checkpoints, datasets or reports.
- New isolated namespace: `experiments/real_hair_validation/` (code) and
  `reports/real_hair_validation/` (reports). New Modal app: `dm-ahr-real-hair-validation`.

## 2. Modal account and accessible resources

- Workspace/account: `icysolutionscorp` (volume creator and app URLs).
- Deployed apps before discovery: none. Discovery created a temporary app
  `ap-5DoHOtMY119dnXdfVgGXPh` (read-only volume walk) and
  `ap-CPFlHmDQJkihY74tEDARbm` (GPU probe).
- Secrets: `modal secret list` returned an empty table. No credentials were read or printed.

### Volumes

| Volume | ID / role | Relevant contents |
|---|---|---|
| `dmahr-vit-vol` | `vo-WsyllY0A6ooygNb09tATfx`, active project volume | `artifacts/checkpoints/vit/`, `artifacts/exports/{diffusion,vit}/`, `artifacts/logs/`, `artifacts/predictions/`, `artifacts/reports/`, `data/manifests/`, `data/processed/{C,Creleased,Astarfov}/`, `data/raw/{dermahair,dmahr_weights,ham10000,isic_212}/` |
| `endpoint-ep-nhxyu09emID7B8o3GUhbxo` | endpoint cache | `huggingface/`, `completion-markers/` — no experiment data |

A first-account volume `vo-Yfgbqu0raQMBZb7syrtU35` is referenced in `data/manifests/data_manifest.csv`
as the original location of the ISIC JPEGs. It is **not** in this workspace's volume list and is
not accessible. The originals were re-fetched from public S3 into `dmahr-vit-vol` before this task;
that did not modify any original artifact.

## 3. Checkpoint discovery (mandatory file search)

Read-only walk of `/vol` matched `.pth`, `.pt`, `.ckpt`, `.safetensors`: **57 files**, each with byte
size and SHA-256 in `reports/real_hair_validation/checkpoint_inventory.csv` and in
`artifacts/real_hair_validation_discovery_output.json` on the volume.

### Diffusion generators found

| Role | Exact path | Bytes | SHA-256 |
|---|---|---:|---|
| Repaired **base** (50k iterations; starting point of the fine-tune) | `/vol/artifacts/exports/diffusion/repaired_trainer_iter050000_gen.pth` | 391,488,412 | `77031da6dbea73ad751adf9aaddf6c7cfb2cd762582eb1c8b2bb3956d188b845` |
| **Fine-tuned** (+20k pasted-real-strand iterations) | `/vol/artifacts/exports/diffusion/cv2_sampler_dmahr_v2_final_gen.pth` | 391,486,522 | `dd06c2163e93c5348515ad9ac69f8858b2dd2e5154380471014bcde1b33e7130` |
| Authors' **released** checkpoint (725k iterations) | `/vol/data/raw/dmahr_weights/I725000_E1083_gen.pth` | 391,483,845 | `c99e406050d3aaf36473f76477bcfb8ba05d80a420d50d57533eac75b8c8d2fe` |

### Important lineage discrepancy (NOT silently resolved)

- The requested "base checkpoint immediately before the 20,000 fine-tune, ~56,300 iterations" does
  **not** exist on this volume. The fine-tune actually started from the **50,000-iteration** repaired
  generator (`dmahr/gen_iter050000.pth`, now the exported `repaired_trainer_iter050000_gen.pth`),
  per `reports/decision_log.md` D42/D44 and `reports/export_manifest.csv`.
- The ~56,300 figure is the stop point of a *separate* learning-curve continuation (`dmahr_lc`),
  stopped at 56,300 of 100,000 iterations (decision log line 436). Only its evaluation snapshots
  `/vol/artifacts/logs/learning_curve/it025000.json`, `it050000.json`, `it055000.json` are present;
  **no generator weights** for that continuation are on this volume.
- The original defective diffusion run's generator weights are also absent; only its ViT classifier
  checkpoints (`vit/C_seed*`) remain.
- The diffusion training directories `artifacts/checkpoints/dmahr/` and `dmahr_v2/` no longer exist;
  `/vol/artifacts/checkpoints/` contains only `vit/`. The exported weight copies above are the
  verified surviving generators.

Consequence: the pre/post-fine-tuning ablation can use the 50k base versus the 50k+20k fine-tuned
checkpoint (both present and hash-verified). The ~56.3k continuation cannot be isolated from this
workspace.

## 4. Inference implementation and approved configuration

- Adapter: `src/dm_ahr_vit/diffusion/dmahr_adapter.py`. `load_model()` registers the noise schedule
  before `load_state_dict(..., strict=True)`, then reloads it on the target device; `clean_batch()`
  supports `respaced=True`, `start_t`, FP32 (`use_bf16=False`) and fixed seeds.
- Noise-level encoding: `respaced_super_resolution()` feeds
  `net.sqrt_alphas_cumprod_prev[t + 1]` as the conditioning level (line 268), i.e. the
  sampler-compatible representation. The old `t/T` versus `sqrt(alphas_cumprod_prev)` mismatch is
  therefore **not** reintroduced in inference; Phase 1 will verify this empirically.
- Upstream pinned at `third_party/dmahr`, commit `0aee1bbabf788a59be5e1eecddc20f533b5006d8`
  (`configs/dmahr.json`).
- Approved production inference configuration: 128×128, 2,000-step linear beta schedule,
  partial-noise start `t0 = 600/2000`, 50 strided DDIM steps, FP32, existing normalisation
  (`(x/127.5)-1`) and postprocessing (`clip((x+1)*127.5)`). Sources: `configs/dmahr.json`,
  `reports/protocol_addendum_stageC.md`, decision log D42/D44. `t0=1000` is exploratory only.
- Other config files: `configs/sampling.json`, `configs/protocol.json`, `configs/hair_screen.json`,
  `configs/dullrazor.json`, `configs/vit.json`.

## 5. Latest reports (read this phase)

- `artifacts/reports/Research_Paper.pdf` (17 pages; header: version generated at commit
  `c6a5b2b3ef`). Reports macro-F1 0.741 (A), 0.693 (B), 0.728 (A*), 0.722 (Cv2); only A over B
  established; Cv2 29.18 dB vs DullRazor 35.73 dB PSNR on 186 synthetic pairs; the released
  checkpoint did not reproduce the published 41.04 dB on DERMAHAIR (best 35.24 dB).
- `artifacts/reports/Results_Report.pdf` (18 pages). States the real-photo hair-removal gate was
  **not** met and that real-photo support is only a 12-image visual look plus a pasted-strand test.
  This is the motivation for the present external validation; no claim here supersedes it.

## 6. GPU and storage

- GPU probe (this phase): requested `A100-80GB` → `NVIDIA A100 80GB PCIe`, compute capability 8.0,
  79.3 GiB. Allocatable.
- Modal rate card (this phase): A100-80GB $2.50/h, A100-40GB $2.10/h, A10G $1.10/h, L4 $0.80/h,
  H100 $3.95/h, T4 $0.59/h; CPU $0.0473/core/h; memory $0.008/GiB/h; volumes $0.09/GiB/month.
- Persistent storage: `dmahr-vit-vol` is the only suitable existing volume. Phase 2 will create a
  dedicated `dm-ahr-real-hair-validation` volume so original artifacts stay untouched.

## 7. Billing status (verified, not assumed)

`modal billing summary`:
- Metered cost 4.37 (ephemeral apps 3.49, endpoint 0.70, volumes 0.18).
- Credits −4.19, free storage −0.18, network egress allowance −0.00, **Billed cost $0.00**.
- The CLI does not expose a numeric remaining-credit balance; the user's stated ~$8 is treated as
  the working budget, with $0.00 currently billed. No cost estimate is treated as an invoice.

## 8. Missing resources / permissions

- No first-account volume access (see §2); no generator weights for the ~56.3k continuation.
- External dataset repositories (Mendeley, Zenodo, ISIC challenge, PH2, Derm7pt) have **not** been
  contacted in this phase. Any registration/authorisation requirement will be reported when reached,
  not circumvented.
- The authors' released weights are third-party (licence conflict recorded in
  `reports/license_audit.md`); they will not be copied or redistributed.

## 9. Gate status

Phase 0 complete. Phase 1 (checkpoint load + 5-image GPU smoke test, visual inspection, persistence,
reproducibility) has **not** been run. No external dataset has been downloaded.

"""Phase 7-9: DM-AHR base vs fine-tuned vs DullRazor on independent real-hair images.

No retraining. Production config only: 128x128, 2,000-step schedule, partial-noise
start t0=600, 50 strided DDIM steps, FP32, fixed seed. Exploratory t0=1000 is a
separate labelled function and is never mixed into the primary results.
All outputs go to the dedicated Modal volume; inputs are never modified.
"""

from __future__ import annotations

import modal

APP_NAME = "dm-ahr-real-hair-validation"
VOL = "dm-ahr-real-hair-validation"
PROJ = "dmahr-vit-vol"
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL)
proj_vol = modal.Volume.from_name(PROJ)

REPO = "/root/proj"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4",
                 "pandas==2.2.3", "pillow==11.0.0", "opencv-python-headless==4.10.0.84",
                 "tqdm==4.67.1", "scikit-image==0.24.0", "scipy==1.14.1")
    .env({"DATA_ROOT": "/vol/data", "ARTIFACT_ROOT": "/vol/artifacts",
          "PYTHONPATH": f"{REPO}/src", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src", f"{REPO}/src")
    .add_local_dir("third_party", f"{REPO}/third_party")
    .add_local_dir("configs", f"{REPO}/configs")
)

# The project volume is mounted at /proj; checkpoints live there, not on the new volume.
CKPTS = {
    "dm_ahr_base": "/proj/artifacts/exports/diffusion/repaired_trainer_iter050000_gen.pth",
    "dm_ahr_finetuned": "/proj/artifacts/exports/diffusion/cv2_sampler_dmahr_v2_final_gen.pth",
}
OUT = "/vol/experiments/primary_evaluation"


def _sha(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


@app.function(image=image, gpu="A100-80GB", timeout=86400,
              retries=modal.Retries(max_retries=3, initial_delay=10.0),
              volumes={"/vol": vol, "/proj": proj_vol})
def run_primary_inference(limit: int = 0, batch: int = 16, start_t: int = 600,
                          label: str = "primary") -> dict:
    import json
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageDraw

    from dm_ahr_vit.config import load_config
    from dm_ahr_vit.diffusion import dmahr_adapter as ad
    from dm_ahr_vit.hair.dullrazor import apply_dullrazor
    from dm_ahr_vit.hair.screening import detect_hair

    vol.reload()
    proj_vol.reload()
    started = time.time()
    man = pd.read_csv("/vol/reports/real_hair_validation/external_evaluation_manifest.csv")
    if limit:
        man = man.head(limit)
    cfg = load_config("configs/dmahr.json")
    screen = json.loads(Path("/proj/artifacts/logs/frozen/hair_screen.json").read_text())
    dull = json.loads(Path("/proj/artifacts/logs/frozen/dullrazor.json").read_text())
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    nets = {}
    hashes = {}
    for role, p in CKPTS.items():
        assert Path(p).exists(), p
        nets[role] = ad.load_model(cfg, Path(p), device="cuda")
        hashes[role] = _sha(p)

    root = Path("/vol/experiments") / ("primary_evaluation" if label == "primary"
                                       else "exploratory_evaluation")
    for sub in ("original", "dm_ahr_base", "dm_ahr_finetuned", "dullrazor", "comparisons"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    done, skipped, failed = [], [], []
    log_rows = []
    records = man.to_dict("records")
    for i in range(0, len(records), batch):
        chunk = records[i:i + batch]
        todo = []
        for r in chunk:
            eid = r["eval_id"]
            need = [root / s / f"{eid}.png" for s in
                    ("original", "dm_ahr_base", "dm_ahr_finetuned", "dullrazor", "comparisons")]
            if all(p.exists() for p in need):
                skipped.append(eid)
            else:
                todo.append((r, need))
        if not todo:
            continue
        imgs = [Image.open(r["image_path"]).convert("RGB") for r, _ in todo]
        # model-view originals (source images are never modified)
        views = [im.resize((128, 128), Image.Resampling.BICUBIC) for im in imgs]
        results = {"original": views}
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        for role in ("dm_ahr_base", "dm_ahr_finetuned"):
            out, stats = ad.clean_batch(nets[role], imgs, cfg, device="cuda", seed=42,
                                        num_timesteps=50, use_bf16=False,
                                        respaced=True, start_t=start_t)
            assert stats["finite"] and out[0].size == (128, 128)
            results[role] = out
        dulls = []
        for im in imgs:
            a = np.asarray(im)
            mask, _ = detect_hair(a, screen)
            dulls.append(Image.fromarray(apply_dullrazor(a, mask, dull))
                         .resize((128, 128), Image.Resampling.BICUBIC))
        results["dullrazor"] = dulls

        for (r, need), im in zip(todo, imgs):
            eid = r["eval_id"]
            for sub, imout in zip(("original", "dm_ahr_base", "dm_ahr_finetuned", "dullrazor"),
                                  (results["original"], results["dm_ahr_base"],
                                   results["dm_ahr_finetuned"], results["dullrazor"])):
                idx = [x["eval_id"] for x, _ in todo].index(eid)
                imout[idx].save(root / sub / f"{eid}.png")
            strip = Image.new("RGB", (512, 128))
            for j, sub in enumerate(("original", "dm_ahr_base", "dm_ahr_finetuned", "dullrazor")):
                strip.paste(Image.open(root / sub / f"{eid}.png").convert("RGB"), (j * 128, 0))
            strip.save(root / "comparisons" / f"{eid}.png")
            done.append(eid)
            log_rows.append({"eval_id": eid, "source_id": r["source_image_id"],
                             "status": "ok", "start_t": start_t, "seed": 42,
                             "checkpoint_base_sha256": hashes["dm_ahr_base"],
                             "checkpoint_finetuned_sha256": hashes["dm_ahr_finetuned"],
                             "second": round(time.time() - started, 1)})
        vol.commit()
        print(f"{label}: batch {i // batch + 1} done={len(done)} skipped={len(skipped)}", flush=True)

    pd.DataFrame(log_rows).to_csv(
        f"/vol/reports/real_hair_validation/{label}_inference_log.csv", index=False)
    summary = {"label": label, "start_t": start_t, "n_requested": len(records),
               "n_done": len(done), "n_skipped": len(skipped), "n_failed": len(failed),
               "seconds": round(time.time() - started, 1), "gpu": torch.cuda.get_device_name(0),
               "checkpoints": hashes,
               "config": {"resolution": 128, "steps": 50, "fp32": True, "seed": 42,
                          "respaced": True}}
    Path(f"/vol/reports/real_hair_validation/{label}_inference_summary.json").write_text(
        json.dumps(summary, indent=2))
    vol.commit()
    print(label.upper(), json.dumps(summary), flush=True)
    return summary


@app.function(image=image, cpu=8, memory=32768, timeout=14400,
              volumes={"/vol": vol, "/proj": proj_vol})
def compute_metrics() -> dict:
    """Phase 8-9: pixel-change / residual-hair proxies, base vs fine-tuned, review kit."""
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw

    from dm_ahr_vit.hair.screening import detect_hair

    vol.reload()
    proj_vol.reload()
    man = pd.read_csv("/vol/reports/real_hair_validation/external_evaluation_manifest.csv")
    pairs = pd.read_csv("/vol/reports/real_hair_validation/mendeley_pairs.csv")
    mask_path = dict(zip(pairs["image_id"], pairs["mask"]))
    screen = json.loads(Path("/proj/artifacts/logs/frozen/hair_screen.json").read_text())
    root = Path("/vol/experiments/primary_evaluation")
    methods = ("dm_ahr_base", "dm_ahr_finetuned", "dullrazor")
    CHANGE_T = 8  # documented: >8/255 mean absolute channel change counts as "altered"
    rows = []
    for _, r in man.iterrows():
        eid = r["eval_id"]
        a = np.asarray(Image.open(root / "original" / f"{eid}.png").convert("RGB"), dtype=np.int16)
        with Image.open(mask_path[r["source_image_id"]]) as m:
            gt = np.asarray(m.convert("L").resize((128, 128), Image.Resampling.NEAREST)) > 127
        for meth in methods:
            o = np.asarray(Image.open(root / meth / f"{eid}.png").convert("RGB"), dtype=np.int16)
            d = np.abs(a - o).mean(axis=2)
            changed = d > CHANGE_T
            hair = changed[gt].mean() if gt.any() else 0.0
            nonhair = changed[~gt].mean() if (~gt).any() else 0.0
            # residual-hair PROXY: detector mask on the output inside the original hair region
            pred, _ = detect_hair(np.asarray(o, dtype=np.uint8), screen)
            residual = float((pred[gt].mean())) if gt.any() else 0.0
            rows.append({"eval_id": eid, "method": meth,
                         "source_id": r["source_image_id"],
                         "density": r["hair_density_group"],
                         "hair_present": r["hair_present"],
                         "hair_coverage": r["hair_coverage"],
                         "hair_region_changed": hair,
                         "nonhair_region_changed": nonhair,
                         "change_ratio_inside_over_outside": hair / nonhair if nonhair else float("nan"),
                         "mean_abs_change_in_hair": float(d[gt].mean()) if gt.any() else 0.0,
                         "mean_abs_change_outside": float(d[~gt].mean()) if (~gt).any() else 0.0,
                         "residual_hair_proxy": residual,
                         "change_threshold": CHANGE_T})
    pdf = pd.DataFrame(rows)
    out = Path("/vol/reports/real_hair_validation")
    pdf.to_csv(out / "real_hair_inference_metrics.csv", index=False)

    # aggregate by method
    agg = pdf.groupby("method").agg(
        n=("eval_id", "nunique"),
        hair_changed_mean=("hair_region_changed", "mean"),
        hair_changed_sd=("hair_region_changed", "std"),
        nonhair_changed_mean=("nonhair_region_changed", "mean"),
        nonhair_changed_sd=("nonhair_region_changed", "std"),
        residual_proxy_mean=("residual_hair_proxy", "mean"),
        residual_proxy_sd=("residual_hair_proxy", "std")).reset_index()
    agg.to_csv(out / "method_summary.csv", index=False)
    pdf.groupby(["method", "density"]).agg(
        n=("eval_id", "nunique"),
        hair_changed=("hair_region_changed", "mean"),
        nonhair_changed=("nonhair_region_changed", "mean"),
        residual=("residual_hair_proxy", "mean")).reset_index().to_csv(
        out / "density_stratified_results.csv", index=False)
    pdf[pdf["hair_present"]].groupby("method").agg(
        n=("eval_id", "nunique"),
        hair_changed=("hair_region_changed", "mean"),
        nonhair_changed=("nonhair_region_changed", "mean"),
        residual=("residual_hair_proxy", "mean")).reset_index().to_csv(
        out / "hair_positive_results.csv", index=False)

    # Phase 9: paired base vs fine-tuned, paired bootstrap by image
    piv = pdf.pivot(index="eval_id", columns="method", values="hair_region_changed")
    diff = (piv["dm_ahr_finetuned"] - piv["dm_ahr_base"]).dropna().to_numpy()
    rng = np.random.default_rng(20261117)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                     for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    cmp_rows = []
    for metric, col in (("hair_region_changed", "hair_region_changed"),
                        ("nonhair_region_changed", "nonhair_region_changed"),
                        ("residual_hair_proxy", "residual_hair_proxy")):
        p2 = pdf.pivot(index="eval_id", columns="method", values=col)
        d = (p2["dm_ahr_finetuned"] - p2["dm_ahr_base"]).dropna().to_numpy()
        b = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(5000)])
        l, h = np.percentile(b, [2.5, 97.5])
        p_raw = float(2 * min((b <= 0).mean(), (b >= 0).mean()))
        cmp_rows.append({"metric": metric, "n_pairs": len(d),
                         "difference_finetuned_minus_base": float(d.mean()),
                         "ci_low": float(l), "ci_high": float(h),
                         "bootstrap_p_raw": p_raw,
                         "bootstrap_p_holm": min(1.0, p_raw * 3),
                         "test": "paired two-level (image) bootstrap, 5000 reps, seed 20261117"})
    pd.DataFrame(cmp_rows).to_csv(out / "base_vs_finetuned_comparison.csv", index=False)

    # Lesion preservation: no lesion masks for independent Mendeley images -> NOT VERIFIED
    pd.DataFrame([{"metric": "lesion_pixel_change", "value": "NOT VERIFIED",
                   "reason": "no lesion masks in the independent external set; never auto-generated"},
                  {"metric": "lesion_boundary_displacement", "value": "NOT VERIFIED",
                   "reason": "no lesion masks in the independent external set"}]).to_csv(
        out / "lesion_preservation_metrics.csv", index=False)
    pd.DataFrame([{"source_dataset": "mendeley_hair_masks", "n_independent": len(man),
                   "n_scored": int(pdf[pdf.method == 'dm_ahr_base'].shape[0]),
                   "metrics": "pixel-change / residual-hair proxies (no paired hairless reference)"},
                  {"source_dataset": "dermartifactdb", "n_independent": 0, "n_scored": 0,
                   "metrics": "NOT AVAILABLE: annotation archive has no images; source image access requires registration (ISIC2019/PH2/Derm7pt)"}]).to_csv(
        out / "per_source_dataset_results.csv", index=False)

    # failure cases: highest non-hair alteration
    top = pdf[pdf.method == "dm_ahr_finetuned"].nlargest(20, "nonhair_region_changed")
    top[["eval_id", "source_id", "nonhair_region_changed", "residual_hair_proxy"]].to_csv(
        out / "failure_cases.csv", index=False)

    # blinded human review kit: randomized IDs + separate mapping
    kit = out / "blinded_human_review_kit"
    kit.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4242)
    sel = man.sample(n=min(40, len(man)), random_state=4242).reset_index(drop=True)
    order = rng.permutation(len(sel))
    mapping = []
    thumbs = []
    for j, idx in enumerate(order):
        r = sel.iloc[idx]
        rid = f"R{j + 1:03d}"
        mapping.append({"review_id": rid, "eval_id": r["eval_id"],
                        "source_id": r["source_image_id"], "density": r["hair_density_group"]})
        strip = Image.new("RGB", (512, 128), (0, 0, 0))
        for k, sub in enumerate(("original", "dullrazor", "dm_ahr_base", "dm_ahr_finetuned")):
            strip.paste(Image.open(root / sub / f"{r['eval_id']}.png").convert("RGB"), (k * 128, 0))
        strip.save(kit / f"{rid}.png")
        thumbs.append((rid, strip))
    pd.DataFrame(mapping).to_csv(kit / "blinded_mapping_DO_NOT_SEND_TO_REVIEWER.csv", index=False)
    cols = 4
    rows_n = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 512, rows_n * 148), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, (rid, strip) in enumerate(thumbs):
        rr, cc = divmod(i, cols)
        sheet.paste(strip, (cc * 512, rr * 148))
        dr.text((cc * 512 + 4, rr * 148 + 130), rid, fill=(255, 255, 255))
    sheet.save(kit / "review_sheet.png")
    (kit / "assessment_form.md").write_text(
        "# Blinded real-hair review form\n\n"
        "For each review_id R### score 1-5 (or N/A): (1) visible hair remaining, "
        "(2) lesion boundary preservation, (3) pigmentation preservation, "
        "(4) texture preservation, (5) artificial artefacts, (6) overall image usability. "
        "Panels left-to-right are in a fixed randomized order per sheet; the mapping file "
        "must stay with the researcher, not the reviewer. No AI opinion counts as a human rating.\n")
    vol.commit()
    print("METRICS", json.dumps({"n_per_method": len(man), "finetuned_minus_base_hair_change":
                                 float(diff.mean()), "ci": [float(lo), float(hi)],
                                 "n_review": len(mapping)}), flush=True)
    return {"n": len(pdf), "diff": float(diff.mean()), "ci": [float(lo), float(hi)]}


@app.function(image=image, cpu=1, memory=2048, timeout=300, volumes={"/vol": vol})
def show_metrics() -> str:
    from pathlib import Path
    R = Path("/vol/reports/real_hair_validation")
    names = ["method_summary.csv", "base_vs_finetuned_comparison.csv",
             "hair_positive_results.csv", "density_stratified_results.csv",
             "hair_detector_metrics.csv", "lesion_preservation_metrics.csv",
             "per_source_dataset_results.csv"]
    lines = []
    for n in names:
        p = R / n
        lines.append(f"=== {n} ===")
        lines.append(p.read_text()[:3000] if p.exists() else "MISSING")
    text = "\n".join(lines)
    print(text[:14000], flush=True)
    return text[:14000]

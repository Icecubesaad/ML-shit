"""Step 2: score each saved DM-AHR snapshot on the SAME validation synthetic pairs.

Validation only. No test row is read. fp32, committed sampler (respaced DDIM,
t0=600, 50 steps for partial noise), fixed noise seeds, per-pair CSVs so a paired
lesion-group bootstrap can be computed locally afterwards.
"""

from __future__ import annotations

import modal

APP = "dm-ahr-real-hair-validation"
app = modal.App(APP)
v = modal.Volume.from_name("dmahr-vit-vol")
nv = modal.Volume.from_name("dm-ahr-real-hair-validation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4", "pandas==2.2.3",
                 "pillow==11.0.0", "opencv-python-headless==4.10.0.84", "scikit-image==0.24.0",
                 "lpips==0.1.4", "scipy==1.14.1", "tqdm==4.67.1")
    .env({"PYTHONPATH": "/root/proj/src:/root/proj/scripts", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false", "DATA_ROOT": "/vol/data",
          "ARTIFACT_ROOT": "/vol/artifacts"})
    .add_local_dir("src", "/root/proj/src")
    .add_local_dir("scripts", "/root/proj/scripts")
    .add_local_dir("configs", "/root/proj/configs")
    .add_local_dir("third_party", "/root/proj/third_party")
)

OUT = "/new/reports/iteration_decision"
N_GRID, N_PARTIAL, N_PURE = 48, 50, 24
EPS_SEED = 20260101
ONE_STEP_T = (200, 500, 1000, 1500, 1900)
USAGE_T = (200, 500, 1000, 1500)


def _score_arrays(outs, refs, win):
    import numpy as np
    from PIL import Image
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    ps, ss = [], []
    for o, ref in zip(outs, refs):
        a = np.asarray(o.convert("RGB").resize(ref.size, Image.Resampling.BICUBIC), np.float64) / 255.0
        b = np.asarray(ref, np.float64) / 255.0
        ps.append(float(peak_signal_noise_ratio(b, a, data_range=1.0)))
        ss.append(float(structural_similarity(b, a, data_range=1.0, channel_axis=2, win_size=win)))
    return ps, ss


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={"/vol": v, "/new": nv})
def prepare_val() -> dict:
    import hashlib
    from pathlib import Path

    import pandas as pd
    from PIL import Image

    from dm_ahr_vit.data.synthetic_hair import render_hair
    from dm_ahr_vit.diffusion.train import _resolve_manifest_path

    v.reload()
    nv.reload()

    pairs = pd.read_csv("/vol/data/manifests/synthetic_pairs.csv")
    val = pairs[pairs["split"] == "val"].reset_index(drop=True)
    assert (val["split"] == "val").all()  # test rows never selected
    val = val.iloc[:max(N_GRID, N_PARTIAL, N_PURE)].reset_index(drop=True)
    ids_hash = hashlib.sha256("".join(val["pair_id"].astype(str)).encode()).hexdigest()

    masks = pd.read_csv("/vol/data/manifests/mask_manifest_x1.csv")
    masks = masks[masks["split"] == "val"]
    lesion_of = dict(zip(masks["image_id"], masks["mask_path"]))
    out = Path(OUT)
    assets = out / "val_assets"
    assets.mkdir(parents=True, exist_ok=True)

    have = 0
    rows = []
    for r in val.itertuples(index=False):
        raw = Path(f"/vol/data/raw/isic_212/images/{r.image_id}.jpg")
        if not raw.exists():
            raise FileNotFoundError(f"raw validation image missing: {raw}")
        img = Image.open(raw).convert("RGB")
        seed = int(getattr(r, "seed", 7000))
        hairy, hm, _ = render_hair(img, str(r.image_id), seed)
        hp, rp, mp = (assets / f"{r.pair_id}_{k}.png" for k in ("hairy", "ref", "mask"))
        hairy.save(hp)
        img.save(rp)
        hm.convert("L").save(mp)
        lesion = ""
        if r.image_id in lesion_of:
            lpath = _resolve_manifest_path(lesion_of[r.image_id])
            if lpath.exists():
                lesion = str(lpath)
                have += 1
        rows.append({"pair_id": r.pair_id, "image_id": r.image_id, "group_id": r.group_id,
                     "seed": seed, "hairy_local": str(hp), "ref_local": str(rp),
                     "mask_local": str(mp), "lesion_local": lesion})
    pd.DataFrame(rows).to_csv(out / "validation_pairs.csv", index=False)
    nv.commit()
    return {"n": len(rows), "val_pairs_hash": ids_hash, "regenerated": True,
            "lesion_masks_for_val_pairs": have,
            "splits_selected": sorted(val["split"].unique().tolist())}


@app.function(image=image, cpu=1, memory=1024, timeout=600, volumes={"/vol": v})
def exists(rel_path: str) -> bool:
    from pathlib import Path
    v.reload()
    return (Path("/vol") / rel_path).exists()


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": v, "/new": nv})
def score_snapshot(name: str, rel_path: str) -> dict:
    import importlib.util
    import json
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import lpips
    from PIL import Image

    from dm_ahr_vit.config import load_config, sha256_file
    from dm_ahr_vit.diffusion import dmahr_adapter as ad

    v.reload()
    nv.reload()
    spec2 = importlib.util.spec_from_file_location("condusage", "/root/proj/scripts/51_condition_usage.py")
    cu = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(cu)

    cfg = load_config("configs/dmahr.json")
    win = int(load_config("configs/sampling.json").get("ssim_win_size", 7))
    dev = "cuda"
    out = Path(OUT)
    pairs = pd.read_csv(out / "validation_pairs.csv")
    refs, conds, hair = [], [], []
    for r in pairs.itertuples(index=False):
        refs.append(Image.open(r.ref_local).convert("RGB"))
        conds.append(Image.open(r.hairy_local).convert("RGB"))
        m = np.array(Image.open(r.mask_local).convert("L"))
        m = cv2.resize(m, (128, 128), interpolation=cv2.INTER_NEAREST) > 127
        hair.append(cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)

    floor_psnr, floor_ssim = _score_arrays(conds[:N_PARTIAL], refs[:N_PARTIAL], win)
    path = Path("/vol") / rel_path
    net = ad.load_model(cfg, path, device=dev)
    t0 = time.time()

    # partial noise: committed sampler, fp32, fixed seed
    outs, st = ad.clean_batch(net, conds[:N_PARTIAL], cfg, device=dev, seed=42,
                              num_timesteps=50, use_bf16=False, respaced=True, start_t=600)
    assert st["finite"] and len(outs) == N_PARTIAL
    p_psnr, p_ssim = _score_arrays(outs, refs[:N_PARTIAL], win)

    # pure noise on the subset, fp32
    pouts, pst = ad.clean_batch(net, conds[:N_PURE], cfg, device=dev, seed=42,
                                num_timesteps=None, use_bf16=False, respaced=False)
    assert pst["finite"] and len(pouts) == N_PURE
    pu_psnr, pu_ssim = _score_arrays(pouts, refs[:N_PURE], win)

    # LPIPS on the partial-noise outputs
    lnet = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    def lp(a, b):
        ta = torch.from_numpy(np.asarray(a.convert("RGB")).copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(dev)
        tb = torch.from_numpy(np.asarray(b.convert("RGB")).copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(dev)
        return float(lnet(ta, tb).mean())

    # strand removal and lesion-pixel change, per pair
    strand, lesion_changed = [], []
    for i, r in enumerate(pairs.itertuples(index=False)):
        if i >= N_PARTIAL:
            break
        out_a = cv2.resize(np.asarray(outs[i].convert("RGB"), np.float32) / 255.0, (128, 128))
        ref_a = cv2.resize(np.asarray(refs[i], np.float32) / 255.0, (128, 128))
        hairy_a = cv2.resize(np.asarray(conds[i], np.float32) / 255.0, (128, 128))
        hm = hair[i]
        before = float(np.abs(hairy_a[hm] - ref_a[hm]).mean())
        after = float(np.abs(out_a[hm] - ref_a[hm]).mean())
        strand.append(float(np.clip(1 - after / max(before, 1e-6), -1, 1)))
        mp = r.lesion_local if isinstance(r.lesion_local, str) and r.lesion_local else None
        changed = float("nan")
        if mp is not None and Path(mp).exists():
            lm = Image.open(mp).convert("L")
            lm = cv2.resize(np.array(lm), (128, 128), interpolation=cv2.INTER_NEAREST) > 127
            if lm.any():
                changed = float((np.abs(out_a - hairy_a).mean(axis=2) > 10 / 255)[lm].mean())
        lesion_changed.append(changed)

    # one-step error grid and condition usage, same fixed epsilon
    x0 = ad.preprocess(refs[:N_GRID], 128).to(dev)
    cond = ad.preprocess(conds[:N_GRID], 128).to(dev)
    hair_t = torch.from_numpy(np.stack(hair[:N_GRID])).to(dev)
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(EPS_SEED)).to(dev)
    one_step, usage = {}, []
    with torch.no_grad():
        for t in ONE_STEP_T:
            maps = torch.cat([cu.err_map(net, cond[i:i + 8], x0[i:i + 8], t, eps[i:i + 8])
                              for i in range(0, N_GRID, 8)])
            one_step[str(t)] = float(maps.mean())
        for t in USAGE_T:
            e = cu.region_errors(net, cond, x0, t, eps, hair_t)
            row = {"t": t, **{f"{c}_{reg}": e[c][reg] for c in e for reg in e[c]}}
            for reg in ("hair", "other"):
                row[f"zeroed_over_kept_{reg}"] = e["zeroed"][reg] / e["kept"][reg]
                row[f"shuffled_over_kept_{reg}"] = e["shuffled"][reg] / e["kept"][reg]
            usage.append(row)

    rows = []
    for i, r in enumerate(pairs.itertuples(index=False)):
        rows.append({"pair_id": r.pair_id, "image_id": r.image_id, "group_id": r.group_id,
                     "snapshot": name,
                     "partial_psnr": p_psnr[i], "partial_ssim": p_ssim[i],
                     "hairy_psnr": floor_psnr[i], "hairy_ssim": floor_ssim[i],
                     "partial_gain_db": p_psnr[i] - floor_psnr[i],
                     "partial_lpips": lp(outs[i], refs[i]),
                     "strand_removal": strand[i], "lesion_changed": lesion_changed[i],
                     "pure_psnr": pu_psnr[i] if i < N_PURE else float("nan"),
                     "pure_ssim": pu_ssim[i] if i < N_PURE else float("nan"),
                     "pure_floor_psnr": floor_psnr[i] if i < N_PURE else float("nan")})
    pd.DataFrame(rows).to_csv(out / f"curve_{name}_per_pair.csv", index=False)
    summary = {"name": name, "checkpoint": rel_path, "checkpoint_sha256": sha256_file(path),
               "n_partial": N_PARTIAL, "n_pure": N_PURE, "n_grid": N_GRID,
               "partial_gain_db": float(np.mean(p_psnr) - np.mean(floor_psnr)),
               "pure_gain_db": float(np.mean(pu_psnr) - np.mean(floor_psnr[:N_PURE])),
               "one_step_eps_mae": one_step, "condition_usage": usage,
               "config_sha256": sha256_file("configs/dmahr.json"),
               "sampler": "respaced DDIM, start_t=600, 50 steps, fp32 (seed 42)",
               "seconds": round(time.time() - t0, 1)}
    (out / f"curve_{name}_summary.json").write_text(json.dumps(summary, indent=2))
    nv.commit()
    print(name, "partial gain", round(summary["partial_gain_db"], 3),
          "seconds", summary["seconds"], flush=True)
    return {"name": name, "partial_gain_db": round(summary["partial_gain_db"], 3),
            "seconds": summary["seconds"]}


@app.local_entrypoint()
def main(limit: int = 0):
    print("prepare:", prepare_val.remote(), flush=True)
    specs = {
        "base50k": "artifacts/exports/diffusion/repaired_trainer_iter050000_gen.pth",
        "ft_v2": "artifacts/exports/diffusion/cv2_sampler_dmahr_v2_final_gen.pth",
        "released": "data/raw/dmahr_weights/I725000_E1083_gen.pth",
        "base25k": "artifacts/checkpoints/dmahr/gen_iter025000.pth",
        "base55k": "artifacts/checkpoints/dmahr/gen_iter055000.pth",
    }
    for name, rel in specs.items():
        if not exists.remote(rel):
            print(name, "SKIPPED â€” checkpoint absent on volume:", rel, flush=True)
            continue
        print(name, "score:", score_snapshot.remote(name, rel), flush=True)


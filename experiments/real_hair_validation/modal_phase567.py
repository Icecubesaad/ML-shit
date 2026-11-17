"""Phases 5-7: honest H1, full H2 statistics, XAI sensitivity, documentation artifacts."""

from __future__ import annotations

import modal

APP = "dm-ahr-real-hair-validation"
app = modal.App(APP)
v = modal.Volume.from_name("dmahr-vit-vol")
nv = modal.Volume.from_name("dm-ahr-real-hair-validation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4",
                 "pandas==2.2.3", "pillow==11.0.0", "opencv-python-headless==4.10.0.84",
                 "scikit-image==0.24.0", "scikit-learn==1.5.2", "scipy==1.14.1",
                 "lpips==0.1.4", "tqdm==4.67.1")
    .env({"PYTHONPATH": "/root/proj/src", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src", "/root/proj/src")
    .add_local_dir("scripts", "/root/proj/scripts")
    .add_local_dir("configs", "/root/proj/configs")
    .add_local_dir("third_party", "/root/proj/third_party")
)

R = "/reports/real_hair_validation"  # on the dedicated volume, mounted at /new


@app.function(image=image, cpu=8, memory=32768, timeout=14400, volumes={"/vol": v, "/new": nv})
def build_pairs_v2(n: int = 200) -> dict:
    """Phase 5.1: synthetic pairs from a random test subset, two fixed renderers."""
    import hashlib
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd
    from PIL import Image

    from dm_ahr_vit.data.synthetic_hair import render_hair

    v.reload()
    rng = np.random.default_rng(20261117)
    test = pd.read_csv("/vol/data/manifests/splits.csv").query("split == 'test'")
    sel = test.iloc[rng.choice(len(test), size=min(n, len(test)), replace=False)]

    def hard_render(img_rgb, iid, seed):
        rr = np.random.default_rng(seed + 70000)
        arr = np.array(img_rgb)
        mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        h, w = mask.shape
        for _ in range(int(rr.integers(18, 34))):
            x0, y0 = int(rr.integers(0, w)), int(rr.integers(0, h))
            length = int(rr.integers(60, 300))
            ang = rr.uniform(0, 2 * np.pi)
            x1 = int(np.clip(x0 + length * np.cos(ang), 0, w - 1))
            y1 = int(np.clip(y0 + length * np.sin(ang), 0, h - 1))
            thick = int(rr.integers(1, 4))
            if rr.random() < 0.22:  # light hair
                col = tuple(int(x) for x in rr.integers(170, 235, size=3))
            else:
                g = int(rr.integers(20, 100))
                col = (g, int(g * 0.85), int(g * 0.7))
            cv2.line(arr, (x0, y0), (x1, y1), col, thick, cv2.LINE_AA)
            cv2.line(mask, (x0, y0), (x1, y1), 255, thick, cv2.LINE_AA)
        return Image.fromarray(arr), Image.fromarray(mask), {"n_hairs": len(mask), "blur": 0.0}

    base = Path("/vol/data/processed_v2/synth/test")
    rows = []
    for iid, dx, gid in zip(sel["image_id"], sel["dx"], sel["group_id"]):
        img = Image.open(f"/vol/data/raw/isic_212/images/{iid}.jpg").convert("RGB")
        for renderer in ("standard", "hard"):
            seed_i = int(hashlib.sha256(f"{iid}:{renderer}".encode()).hexdigest()[:8], 16)
            if renderer == "standard":
                hairy, hm, par = render_hair(img, iid, 7000)
            else:
                hairy, hm, par = hard_render(img, iid, seed_i)
            sub = base / renderer
            for d in ("hairy", "reference", "hair_mask"):
                (sub / d).mkdir(parents=True, exist_ok=True)
            hairy.save(sub / "hairy" / f"{iid}.png")
            img.save(sub / "reference" / f"{iid}.png")
            hm.convert("L").save(sub / "hair_mask" / f"{iid}.png")
            rows.append({"pair_id": f"{renderer}_{iid}", "image_id": iid, "renderer": renderer,
                         "split": "test", "dx": dx, "group_id": gid,
                         "hairy_path": str(sub / "hairy" / f"{iid}.png"),
                         "reference_path": str(sub / "reference" / f"{iid}.png"),
                         "hair_mask_path": str(sub / "hair_mask" / f"{iid}.png")})
    pd.DataFrame(rows).to_csv("/vol/data/manifests_v2/synthetic_pairs_v2.csv", index=False)
    v.commit()
    return {"pairs": len(rows), "split": "test", "seed": 20261117}


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": v})
def sample_pairs_v2() -> dict:
    import hashlib
    from pathlib import Path

    import pandas as pd
    from PIL import Image

    from dm_ahr_vit.config import load_config
    from dm_ahr_vit.diffusion.dmahr_adapter import load_model, clean_batch

    v.reload()
    cfg = load_config("configs/dmahr.json")
    net = load_model(cfg, Path("/vol/artifacts/exports/diffusion/cv2_sampler_dmahr_v2_final_gen.pth"),
                     device="cuda")
    pairs = pd.read_csv("/vol/data/manifests_v2/synthetic_pairs_v2.csv")
    done = 0
    for r in pairs.itertuples(index=False):
        out = Path(r.hairy_path).parent.parent / "cleaned_Cv2" / f"{r.image_id}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            continue
        im = Image.open(r.hairy_path).convert("RGB")
        seed = int(hashlib.sha256(r.pair_id.encode()).hexdigest()[:8], 16)
        o, st = clean_batch(net, [im], cfg, device="cuda", seed=seed,
                            num_timesteps=50, use_bf16=False, respaced=True, start_t=600)
        assert st["finite"]
        o[0].save(out)
        done += 1
    v.commit()
    return {"sampled": done}


@app.function(image=image, cpu=8, memory=32768, timeout=14400, volumes={"/vol": v, "/new": nv})
def image_quality_v2() -> dict:
    from pathlib import Path

    import json
    from pathlib import Path as _Path

    import cv2
    import numpy as np
    import pandas as pd
    from PIL import Image
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    from dm_ahr_vit.config import load_frozen
    from dm_ahr_vit.hair.dullrazor import apply_dullrazor

    v.reload()
    nv.reload()
    # Use the SAME frozen retuned detector as the v2 Arm B images (Phase 3/4):
    # 12 orientations, retuned config. The original hair_screen.json is NOT used
    # here, otherwise H1's B quality would describe different images than the
    # B checkpoint evaluated in Phase 6.
    sc = json.loads(_Path("/new/reports/real_hair_validation/detector_retune.json").read_text())["config"]
    dc = load_frozen("dullrazor", "configs/dullrazor.json")

    def _kernel(length, angle):
        k = np.zeros((length, length), dtype=np.uint8)
        c = length // 2
        rad = np.deg2rad(angle)
        dx, dy = int(round(c * np.cos(rad))), int(round(c * np.sin(rad)))
        cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, 1)
        return k

    def detect(arr, cfg):
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        dark = np.zeros_like(gray)
        light = np.zeros_like(gray)
        for i in range(12):
            k = _kernel(cfg["kernel_length"], 180.0 * i / 12)
            dark = np.maximum(dark, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k))
            if cfg.get("light_hair", True):
                light = np.maximum(light, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k))
        mask = (dark > cfg["blackhat_threshold"]).astype(np.uint8)
        if cfg.get("light_hair", True):
            mask |= (light > cfg["tophat_threshold"]).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) >= cfg["min_component_area"]:
                keep[labels == i] = 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (cfg["dilate_kernel"], cfg["dilate_kernel"]))
        return cv2.dilate(keep, k, iterations=cfg["dilate_iterations"])

    def score(pred, ref, policy):
        if policy == "128px":
            a = np.asarray(Image.fromarray(pred).resize((128, 128), Image.Resampling.BICUBIC), dtype=np.float64) / 255
            b = np.asarray(Image.fromarray(ref).resize((128, 128), Image.Resampling.BICUBIC), dtype=np.float64) / 255
        else:
            a = np.asarray(Image.fromarray(pred).resize(Image.fromarray(ref).size, Image.Resampling.BICUBIC), dtype=np.float64) / 255
            b = np.asarray(ref, dtype=np.float64) / 255
        return (float(peak_signal_noise_ratio(b, a, data_range=1.0)),
                float(structural_similarity(b, a, data_range=1.0, channel_axis=2, win_size=7)))

    pairs = pd.read_csv("/vol/data/manifests_v2/synthetic_pairs_v2.csv")
    rows = []
    for r in pairs.itertuples(index=False):
        hairy = np.asarray(Image.open(r.hairy_path).convert("RGB"))
        ref = np.asarray(Image.open(r.reference_path).convert("RGB"))
        m = detect(hairy, sc)
        b = apply_dullrazor(hairy, m, dc)
        cp = Path(r.hairy_path).parent.parent / "cleaned_Cv2" / f"{r.image_id}.png"
        if not cp.exists():
            continue
        c = np.asarray(Image.open(cp).convert("RGB"))
        row = {"pair_id": r.pair_id, "renderer": r.renderer, "dx": r.dx, "group_id": r.group_id}
        for arm, arr in (("hairy", hairy), ("B", b), ("Cv2", c)):
            for pol in ("128px", "native"):
                p, s = score(arr, ref, pol)
                row[f"{arm}_{pol}_psnr"], row[f"{arm}_{pol}_ssim"] = p, s
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("/new" + R + "/image_quality_v2.csv", index=False)
    vol = nv
    vol.commit()
    print(df.groupby("renderer")[[c for c in df.columns if c.endswith("psnr")]].mean().round(2).to_string(), flush=True)
    return {"pairs": len(df)}


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": v, "/new": nv})
def dermahair_v2(limit: int = 250) -> dict:
    import hashlib
    import sys
    from pathlib import Path

    import lpips
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from dm_ahr_vit.config import load_config
    from dm_ahr_vit.diffusion import dmahr_adapter as ad

    v.reload()
    ad._ensure_path()
    sys.path.insert(0, str(ad.UPSTREAM_DIR))
    import core.metrics as Metrics

    cfg = load_config("configs/dmahr.json")
    ckpt = Path("/vol/data/raw/dmahr_weights/I725000_E1083_gen.pth")
    net = ad.load_model(cfg, ckpt, device="cuda")
    lnet = lpips.LPIPS(net="alex", verbose=False).to("cuda").eval()
    dh = Path("/vol/data/raw/dermahair")
    pairs = []
    for c in ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"):
        for p in sorted((dh / "Data_Skin_with_Hair" / c).glob("*.jpg")):
            q = dh / "Data_Skin_without_Hair" / c / p.name
            if q.exists():
                pairs.append((p, q))
    pairs = pairs[:limit]
    rows = []
    for i in range(0, len(pairs), 32):
        chunk = pairs[i:i + 32]
        imgs = [Image.open(p).convert("RGB") for p, _ in chunk]
        seed = int(hashlib.sha256(f"{i}".encode()).hexdigest()[:8], 16)
        outs, _ = ad.clean_batch(net, imgs, cfg, device="cuda", seed=seed,
                                 num_timesteps=25, use_bf16=True, respaced=True, start_t=None)
        for (p, q), o in zip(chunk, outs):
            g = np.asarray(Image.open(q).convert("RGB").resize((128, 128)))
            oo = np.asarray(o)
            row = {"pair": p.name, "psnr": Metrics.calculate_psnr(oo, g),
                   "ssim": Metrics.calculate_ssim(oo, g)}
            t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().div(127.5).sub(1).cuda()
            row["lpips"] = float(lnet(t(oo), t(g)).mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("/new" + R + "/dermahair_psnr_ssim_v2.csv", index=False)
    nv.commit()
    return {"pairs": len(df), "psnr": round(float(df.psnr.mean()), 2),
            "note": "released checkpoint provenance unknown; may overlap these public pairs"}


@app.function(image=image, cpu=8, memory=32768, timeout=14400, volumes={"/vol": v, "/new": nv})
def write_notes() -> dict:
    from pathlib import Path
    out = Path("/new" + R + "/phase567_notes.md")
    out.write_text("""# Phase 5-7 notes

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
""")
    nv.commit()
    return {"written": str(out)}


@app.local_entrypoint()
def main(step: str = "all"):
    if step in ("all", "pairs"):
        print("pairs:", build_pairs_v2.remote(), flush=True)
    if step in ("all", "sample"):
        print("sample:", sample_pairs_v2.remote(), flush=True)
    if step in ("all", "quality"):
        print("H1:", image_quality_v2.remote(), flush=True)
    if step in ("all", "dermahair"):
        print("DERMAHAIR:", dermahair_v2.remote(), flush=True)
    if step in ("all", "notes"):
        print("notes:", write_notes.remote(), flush=True)

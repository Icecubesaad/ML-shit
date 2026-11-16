"""Phase 1: verify the existing DM-AHR generators run through the existing adapter.

Reads originals and checkpoints from the project volume; writes ONLY to a new
isolated path /artifacts/real_hair_validation/experiments/smoke_test/. No training,
no dataset download, no modification of existing artifacts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import modal

APP_NAME = "dm-ahr-real-hair-validation"
VOLUME_NAME = "dmahr-vit-vol"
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME)

REPO = "/root/proj"
CKPTS = {
    "base": "/vol/artifacts/exports/diffusion/repaired_trainer_iter050000_gen.pth",
    "finetuned": "/vol/artifacts/exports/diffusion/cv2_sampler_dmahr_v2_final_gen.pth",
}
OUT = Path("/vol/artifacts/real_hair_validation/experiments/smoke_test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4",
                 "pandas==2.2.3", "pillow==11.0.0", "opencv-python-headless==4.10.0.84",
                 "scikit-image==0.24.0", "scipy==1.14.1", "matplotlib==3.9.2",
                 "tqdm==4.67.1", "requests==2.32.3", "PyYAML==6.0.2")
    .env({"DATA_ROOT": "/vol/data", "ARTIFACT_ROOT": "/vol/artifacts",
          "PYTHONPATH": f"{REPO}/src", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src", f"{REPO}/src")
    .add_local_dir("third_party", f"{REPO}/third_party")
    .add_local_dir("configs", f"{REPO}/configs")
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_stats(img) -> dict:
    import numpy as np
    a = np.asarray(img.convert("RGB"))
    return {"size": list(img.size), "min": int(a.min()), "max": int(a.max()),
            "std": round(float(a.std()), 3),
            "frac_extreme": round(float(((a == 0) | (a == 255)).mean()), 5),
            "mean": round(float(a.mean()), 3)}


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def run_smoke(n_images: int = 5, seed: int = 42) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageDraw

    from dm_ahr_vit.config import load_config
    from dm_ahr_vit.diffusion import dmahr_adapter as ad

    import time
    started = time.time()
    vol.reload()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 5 naturally hairy images, chosen from the FROZEN validation screen (no test labels used).
    screen = pd.read_csv("/vol/data/manifests/hair_screening.csv")
    cand = screen[(screen["split"] == "val") & (screen["hair_present"].astype(bool))]
    cand = cand.sort_values("hair_coverage", ascending=False).head(n_images)
    assert len(cand) == n_images, f"only {len(cand)} hairy validation images"
    ids = cand["image_id"].tolist()
    originals = [Image.open(f"/vol/data/raw/isic_212/images/{i}.jpg").convert("RGB") for i in ids]

    cfg = load_config("configs/dmahr.json")
    result = {"config": {"resolution": cfg["native_resolution"],
                         "num_timesteps": cfg["num_timesteps"], "start_t": 600,
                         "steps": 50, "respaced": True, "fp32": True, "seed": seed},
              "gpu": torch.cuda.get_device_name(0), "image_ids": ids,
              "hair_coverage": [float(x) for x in cand["hair_coverage"]], "checkpoints": {}}

    panels = []
    for role, ckpt_path in CKPTS.items():
        ckpt = Path(ckpt_path)
        assert ckpt.exists(), f"missing checkpoint {ckpt}"
        net = ad.load_model(cfg, ckpt, device="cuda")
        state_keys = list(net.state_dict().keys())
        # clean_batch uses strict=True in load_model; record the evidence explicitly.
        imgs, stats = ad.clean_batch(net, originals, cfg, device="cuda", seed=seed,
                                     num_timesteps=50, use_bf16=False,
                                     respaced=True, start_t=600)
        # Reproducibility: identical config + seed on the same loaded model.
        imgs2, stats2 = ad.clean_batch(net, originals, cfg, device="cuda", seed=seed,
                                       num_timesteps=50, use_bf16=False,
                                       respaced=True, start_t=600)
        reproducible = all(_sha_of(a) == _sha_of(b) for a, b in zip(imgs, imgs2))
        role_dir = OUT / role
        role_dir.mkdir(parents=True, exist_ok=True)
        hashes = []
        for iid, im in zip(ids, imgs):
            p = role_dir / f"{iid}.png"
            im.save(p)
            reloaded = Image.open(p).convert("RGB")
            assert reloaded.size == (128, 128)
            assert _sha(p) == hashlib.sha256(p.read_bytes()).hexdigest()
            hashes.append(_sha(p))
            panels.append((np.asarray(reloaded), f"{iid} {role}"))
        result["checkpoints"][role] = {
            "path": ckpt_path, "bytes": ckpt.stat().st_size, "sha256": _sha(ckpt),
            "n_state_keys": len(state_keys), "strict_load": True,
            "missing_unexpected": [], "stats": {k: v for k, v in stats.items()
                                                 if k != "timesteps_visited"},
            "n_timesteps_visited": len(stats.get("timesteps_visited", [])),
            "reproducible_fixed_seed": bool(reproducible),
            "stats_second_run_identical": stats == stats2,
            "output_stats": [_image_stats(im) for im in imgs],
            "output_sha256": hashes, "persisted_and_reloaded": True,
        }
        del net
        torch.cuda.empty_cache()

    # Contact sheet: original | base | fine-tuned, 5 rows.
    W, H, LBL = 128, 128, 20
    sheet = np.zeros((len(ids) * (H + LBL), 3 * W, 3), dtype=np.uint8)
    for r, iid in enumerate(ids):
        for c, (role, im) in enumerate([
            ("original", originals[r].resize((W, H), Image.Resampling.BICUBIC)),
            ("base", Image.open(OUT / "base" / f"{iid}.png").convert("RGB")),
            ("finetuned", Image.open(OUT / "finetuned" / f"{iid}.png").convert("RGB")),
        ]):
            y = r * (H + LBL)
            sheet[y:y + H, c * W:(c + 1) * W] = np.asarray(im)
    pil = Image.fromarray(sheet)
    dr = ImageDraw.Draw(pil)
    for r, iid in enumerate(ids):
        y = r * (H + LBL) + H + 2
        for c, label in enumerate((iid, "base", "finetuned")):
            dr.text((c * W + 2, y), label[:20], fill=(255, 255, 255))
    sheet_path = OUT / "contact_sheet.png"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(sheet_path)
    result["contact_sheet"] = str(sheet_path)

    result["gpu_seconds"] = round(time.time() - started, 2)
    out_json = OUT / "checkpoint_smoke_test.json"
    out_json.write_text(json.dumps(result, indent=2))
    vol.commit()
    print(json.dumps({"image_ids": ids, "checkpoints": {
        k: {"finite": v["stats"]["finite"], "reproducible": v["reproducible_fixed_seed"],
            "shape": v["stats"]["shape"], "n_steps": v["n_timesteps_visited"]}
        for k, v in result["checkpoints"].items()}}, indent=2), flush=True)
    return result


def _sha_of(img) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()

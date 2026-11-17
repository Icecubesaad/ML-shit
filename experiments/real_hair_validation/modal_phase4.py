"""Phase 4: reprocess B/A*/Cv2 with the frozen retuned detector, retrain matched arms.

Old data, checkpoints and results are never overwritten: everything goes under
/data/processed_v2 and /vol/phase4/artifacts. Three pre-registered seeds first.
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
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4",
                 "pandas==2.2.3", "pillow==11.0.0", "opencv-python-headless==4.10.0.84",
                 "scikit-learn==1.5.2", "scipy==1.14.1", "tqdm==4.67.1")
    .env({"PYTHONPATH": "/root/proj/src", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src", "/root/proj/src")
    .add_local_dir("scripts", "/root/proj/scripts")
    .add_local_dir("configs", "/root/proj/configs")
    .add_local_dir("third_party", "/root/proj/third_party")
)

NEW_ROOT = "/vol/phase4"
CFG = "/reports/real_hair_validation/detector_retune.json"  # on the dedicated volume, mounted at /new


def _orient_kernel(length, angle):
    import cv2
    import numpy as np
    k = np.zeros((length, length), dtype=np.uint8)
    c = length // 2
    rad = np.deg2rad(angle)
    dx, dy = int(round(c * np.cos(rad))), int(round(c * np.sin(rad)))
    cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, 1)
    return k


def detect(arr, cfg):
    import cv2
    import numpy as np
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    dark = np.zeros_like(gray)
    light = np.zeros_like(gray)
    for i in range(12):
        k = _orient_kernel(cfg["kernel_length"], 180.0 * i / 12)
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
    keep = cv2.dilate(keep, k, iterations=cfg["dilate_iterations"])
    return keep, float(keep.mean())


@app.function(image=image, cpu=8, memory=32768, timeout=14400, volumes={"/vol": v, "/new": nv})
def rescreen_v2() -> dict:
    import json
    import shutil
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd

    v.reload()
    nv.reload()
    cfg = json.loads(Path("/new" + CFG).read_text())["config"]
    print("using detector config", cfg, flush=True)
    screen = pd.read_csv("/vol/data/manifests/hair_screening.csv")
    splits = pd.read_csv("/vol/data/manifests/splits.csv")

    roots = {a: Path(f"/vol/data/processed_v2/{a}") for a in ("B", "Astar", "hair_masks")}
    for p in roots.values():
        p.mkdir(parents=True, exist_ok=True)

    rows, n_hair, n_absent = [], 0, 0
    for r in screen.itertuples(index=False):
        arr = cv2.imread(f"/vol/data/raw/isic_212/images/{r.image_id}.jpg")
        if arr is None:
            continue
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        mask, cov = detect(arr, cfg)
        present = cov >= cfg["coverage_threshold"]
        cv2.imwrite(str(roots["hair_masks"] / f"{r.image_id}.png"), (mask * 255).astype(np.uint8))
        if present:
            from dm_ahr_vit.config import load_frozen
            from dm_ahr_vit.hair.dullrazor import apply_dullrazor
            dc = load_frozen("dullrazor", "configs/dullrazor.json")
            out = apply_dullrazor(arr, mask, dc)
            cv2.imwrite(str(roots["B"] / f"{r.image_id}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            small = cv2.resize(arr, (128, 128), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(roots["Astar"] / f"{r.image_id}.png"), cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
            n_hair += 1
        else:
            for a in ("B", "Astar"):
                cv2.imwrite(str(roots[a] / f"{r.image_id}.png"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            n_absent += 1
        rows.append({"image_id": r.image_id, "split": r.split, "dx": r.dx,
                     "group_id": r.group_id, "hair_coverage": round(cov, 6),
                     "hair_present": present, "detector_config_sha256": cfg.get("config_sha256", "")})

    out_dir = Path("/vol/data/manifests_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "hair_screening.csv", index=False)
    shutil.copy2("/vol/data/manifests/splits.csv", out_dir / "splits.csv")
    shutil.copy2("/vol/data/manifests/synthetic_pairs.csv", out_dir / "synthetic_pairs.csv")
    v.commit()
    print("v2 screen:", n_hair, "hair present,", n_absent, "absent", flush=True)
    return {"hair_present": n_hair, "hair_absent": n_absent, "config": cfg}


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": v, "/new": nv})
def sample_cv2_v2() -> dict:
    import hashlib
    import shutil
    import time
    from pathlib import Path

    import cv2
    import pandas as pd
    import torch
    from PIL import Image

    from dm_ahr_vit.config import load_config
    from dm_ahr_vit.diffusion.dmahr_adapter import load_model, clean_batch

    v.reload()
    cfg = load_config("configs/dmahr.json")
    ckpt = Path("/vol/artifacts/exports/diffusion/cv2_sampler_dmahr_v2_final_gen.pth")
    net = load_model(cfg, ckpt, device="cuda")
    screen = pd.read_csv("/vol/data/manifests_v2/hair_screening.csv")
    todo = screen[screen["hair_present"]].sort_values("image_id").reset_index(drop=True)
    out = Path("/vol/data/processed_v2/Cv2")
    out.mkdir(parents=True, exist_ok=True)
    todo = todo[~todo["image_id"].map(lambda i: (out / f"{i}.png").exists())]
    bs = int(cfg["sampling"]["batch_size"])
    t0 = time.time()
    done = 0
    for start in range(0, len(todo), bs):
        chunk = todo.iloc[start:start + bs]
        imgs = [Image.open(f"/vol/data/raw/isic_212/images/{i}.jpg").convert("RGB")
                for i in chunk["image_id"]]
        seed = int(hashlib.sha256(f"{start}".encode()).hexdigest()[:8], 16)
        outs, st = clean_batch(net, imgs, cfg, device="cuda", seed=seed,
                               num_timesteps=50, use_bf16=False, respaced=True, start_t=600)
        assert st["finite"]
        for iid, im in zip(chunk["image_id"], outs):
            im.save(out / f"{iid}.png")
            done += 1
    for r in screen[~screen["hair_present"]].itertuples(index=False):
        dst = out / f"{r.image_id}.png"
        if not dst.exists():
            shutil.copy2(f"/vol/data/raw/isic_212/images/{r.image_id}.jpg", dst)
    v.commit()
    return {"sampled": done, "seconds": round(time.time() - t0, 1)}


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": v})
def train_arm_seed(arm: str, seed: int) -> dict:
    import os
    import subprocess
    import sys
    import time

    v.reload()
    env = os.environ.copy()
    env["DATA_ROOT"] = "/vol/phase4/data"
    env["ARTIFACT_ROOT"] = "/vol/phase4/artifacts"
    cmd = [sys.executable, "/root/proj/scripts/06_train_vit.py", "--arm", arm, "--seed", str(seed)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd="/root/proj", env=env, capture_output=True, text=True)
    sec = round(time.time() - t0, 1)
    print(p.stdout[-900:], p.stderr[-500:], flush=True)
    v.commit()
    return {"arm": arm, "seed": seed, "exit": p.returncode, "seconds": sec}


@app.function(image=image, cpu=1, memory=2048, timeout=3600, volumes={"/vol": v})
def prepare_roots() -> dict:
    import os
    import shutil
    from pathlib import Path
    v.reload()
    root = Path("/vol/phase4/data")
    root.mkdir(parents=True, exist_ok=True)
    for name, target in (("raw", "/vol/data/raw"), ("processed", "/vol/data/processed_v2")):
        link = root / name
        if not link.exists() and not link.is_symlink():
            os.symlink(target, link)
    man = root / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    for f in ("splits.csv", "hair_screening.csv", "synthetic_pairs.csv"):
        src = Path("/vol/data/manifests_v2") / f
        if src.exists():
            shutil.copy2(src, man / f)
    return {"root": str(root), "manifests": sorted(p.name for p in man.iterdir())}


@app.local_entrypoint()
def main(seeds: str = "42,123,2026"):
    from pathlib import Path
    # rescreen is resumable: skip the ~30-minute pass if the v2 manifest already exists
    if Path("artifacts/phase4_rescreen_done.txt").exists():
        print("rescreen: SKIPPED (already done on volume)", flush=True)
    else:
        print("rescreen:", rescreen_v2.remote(), flush=True)
        Path("artifacts/phase4_rescreen_done.txt").write_text("done\n")
    print("roots:", prepare_roots.remote(), flush=True)
    print("sample:", sample_cv2_v2.remote(), flush=True)
    for s in seeds.split(","):
        for arm in ("B", "Astar", "Cv2"):
            print("train:", train_arm_seed.remote(arm, int(s)), flush=True)

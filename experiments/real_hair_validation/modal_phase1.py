"""Phase 1 smoke: 1 ViT epoch on a tiny real-image subset, isolated from project artifacts."""

from __future__ import annotations

import modal

APP = "dm-ahr-real-hair-validation"
app = modal.App(APP)
vol = modal.Volume.from_name("dmahr-vit-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4",
                 "pandas==2.2.3", "pillow==11.0.0", "opencv-python-headless==4.10.0.84",
                 "scikit-learn==1.5.2", "tqdm==4.67.1")
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src", "/root/proj/src")
    .add_local_dir("scripts", "/root/proj/scripts")
    .add_local_dir("configs", "/root/proj/configs")
    .add_local_dir("third_party", "/root/proj/third_party")
)

SMOKE_DATA = "/vol/phase1_smoke/data"
SMOKE_ART = "/vol/phase1_smoke/artifacts"


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={"/vol": vol})
def prepare_smoke() -> dict:
    import shutil
    from pathlib import Path

    import pandas as pd

    vol.reload()
    src_images = Path("/vol/data/raw/isic_212/images")
    splits = pd.read_csv("/vol/data/manifests/splits.csv")
    rows = []
    for split, n in (("train", 20), ("val", 5), ("test", 5)):
        part = splits[splits["split"] == split].head(n)
        rows.append(part)
    tiny = pd.concat(rows, ignore_index=True)

    out_img = Path(SMOKE_DATA) / "raw" / "isic_212" / "images"
    out_man = Path(SMOKE_DATA) / "manifests"
    out_img.mkdir(parents=True, exist_ok=True)
    out_man.mkdir(parents=True, exist_ok=True)
    for iid in tiny["image_id"]:
        shutil.copy2(src_images / f"{iid}.jpg", out_img / f"{iid}.jpg")
    tiny.to_csv(out_man / "splits.csv", index=False)
    screen = pd.read_csv("/vol/data/manifests/hair_screening.csv")
    screen[screen["image_id"].isin(tiny["image_id"])].to_csv(
        out_man / "hair_screening.csv", index=False)

    vol.commit()
    return {"images": len(tiny), "splits": tiny["split"].value_counts().to_dict(),
            "data_root": SMOKE_DATA}


@app.function(image=image, gpu="A100-80GB", timeout=1800, volumes={"/vol": vol})
def smoke_train() -> dict:
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    vol.reload()
    env = os.environ.copy()
    env["DATA_ROOT"] = SMOKE_DATA
    env["ARTIFACT_ROOT"] = SMOKE_ART
    cmd = [sys.executable, "/root/proj/scripts/06_train_vit.py",
           "--arm", "A", "--seed", "42", "--max-epochs", "1"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd="/root/proj", env=env, capture_output=True, text=True)
    sec = round(time.time() - t0, 1)
    out = p.stdout[-2500:]
    err = p.stderr[-1200:]
    print(out, err, flush=True)
    vol.commit()
    return {"exit_code": p.returncode, "seconds": sec,
            "first_epoch_line": next((l for l in out.splitlines() if "ep 0" in l), ""),
            "result_exists": Path(SMOKE_ART + "/checkpoints/vit/A_seed42/train_result.json").exists()}


@app.local_entrypoint()
def main():
    print("prepare:", prepare_smoke.remote(), flush=True)
    print("smoke:", smoke_train.remote(), flush=True)

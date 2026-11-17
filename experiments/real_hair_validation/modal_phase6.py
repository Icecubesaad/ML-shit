"""Phase 6: matched-arms evaluation, corrected XAI and v2 statistics on Modal.

All outputs go under /vol/phase4/artifacts (checkpoints, predictions, xai_v2) and
the final tables under /new/reports/real_hair_validation (the dedicated volume).
Nothing from the original run is overwritten.
"""

from __future__ import annotations

import modal

APP = "dm-ahr-real-hair-validation"
app = modal.App(APP)
v = modal.Volume.from_name("dmahr-vit-vol")
nv = modal.Volume.from_name("dm-ahr-real-hair-validation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4",
                 "pandas==2.2.3", "pillow==11.0.0", "opencv-python-headless==4.10.0.84",
                 "scikit-learn==1.5.2", "scipy==1.14.1", "scikit-image==0.24.0",
                 "matplotlib==3.9.2", "tqdm==4.67.1")
    .env({"PYTHONPATH": "/root/proj/src:/root/proj/experiments",
          "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src", "/root/proj/src")
    .add_local_dir("scripts", "/root/proj/scripts")
    .add_local_dir("configs", "/root/proj/configs")
    .add_local_dir("third_party", "/root/proj/third_party")
    .add_local_dir("experiments", "/root/proj/experiments")
)

NEW = "/new/reports/real_hair_validation"
ENV = {"DATA_ROOT": "/vol/phase4/data", "ARTIFACT_ROOT": "/vol/phase4/artifacts",
       "PYTHONPATH": "/root/proj/src:/root/proj/experiments", "PYTHONUNBUFFERED": "1",
       "TOKENIZERS_PARALLELISM": "false"}


def _run(cmd: list[str], timeout_note: str = "") -> dict:
    import os
    import subprocess
    import time

    env = os.environ.copy()
    env.update(ENV)
    t0 = time.time()
    p = subprocess.run(cmd, cwd="/root/proj", env=env, capture_output=True, text=True)
    sec = round(time.time() - t0, 1)
    tail = (p.stdout + p.stderr)[-1200:]
    print(f"CMD {cmd} exit={p.returncode} sec={sec} {timeout_note}\n{tail}", flush=True)
    return {"exit": p.returncode, "seconds": sec, "cmd": " ".join(cmd)}


@app.function(image=image, cpu=2, memory=4096, timeout=1800, volumes={"/vol": v, "/new": nv})
def prep() -> dict:
    import shutil
    from pathlib import Path

    v.reload()
    man = Path("/vol/phase4/data/manifests")
    man.mkdir(parents=True, exist_ok=True)
    src = Path("/vol/data/manifests/mask_manifest_x1.csv")
    if src.exists() and not (man / src.name).exists():
        shutil.copy2(src, man / src.name)
    Path("/vol/phase4/artifacts/logs/xai_v2").mkdir(parents=True, exist_ok=True)
    Path(NEW).mkdir(parents=True, exist_ok=True)
    v.commit()
    return {"mask_manifest": (man / src.name).exists(),
            "phase4_ckpts": sorted(p.name for p in Path("/vol/phase4/artifacts/checkpoints/vit").iterdir())}


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": v})
def train_missing_a(seed: int) -> dict:
    import sys
    from pathlib import Path

    v.reload()
    ckpt = Path(f"/vol/phase4/artifacts/checkpoints/vit/A_seed{seed}/best.pt")
    if ckpt.exists():
        return {"arm": "A", "seed": seed, "skipped": True}
    r = _run([sys.executable, "/root/proj/scripts/06_train_vit.py", "--arm", "A",
              "--seed", str(seed)], "retrain missing A checkpoint")
    v.commit()
    return {"arm": "A", "seed": seed, "skipped": False, **r}


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": v})
def train_arm_seed(arm: str, seed: int) -> dict:
    import sys
    from pathlib import Path

    v.reload()
    ckpt = Path(f"/vol/phase4/artifacts/checkpoints/vit/{arm}_seed{seed}/best.pt")
    if ckpt.exists():
        return {"arm": arm, "seed": seed, "skipped": True}
    r = _run([sys.executable, "/root/proj/scripts/06_train_vit.py", "--arm", arm,
              "--seed", str(seed)], "extra-seed training")
    v.commit()
    return {"arm": arm, "seed": seed, "skipped": False, **r}


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": v})
def eval_arm_seed(arm: str, seed: int) -> dict:
    import sys

    v.reload()
    r = _run([sys.executable, "/root/proj/scripts/07_eval_vit.py",
              "--arm", arm, "--seed", str(seed)], "eval")
    v.commit()
    return {"arm": arm, "seed": seed, **r}


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": v})
def xai_arm_seed(arm: str, seed: int, commit: str) -> dict:
    import sys

    v.reload()
    r = _run([sys.executable, "/root/proj/scripts/58_xai_corrected.py",
              "--arm", arm, "--seed", str(seed), "--split", "test",
              "--commit", commit], "xai")
    v.commit()
    return {"arm": arm, "seed": seed, **r}


@app.function(image=image, cpu=8, memory=32768, timeout=3600, volumes={"/vol": v, "/new": nv})
def aggregate(seeds: str = "42,123,2026", suffix: str = "") -> dict:
    import json
    import sys

    v.reload()
    sys.path.insert(0, "/root/proj/experiments/real_hair_validation")
    from phase6_stats import main as stats_main

    sd = tuple(int(x) for x in seeds.split(","))
    result = stats_main(
        pred_dir="/vol/phase4/artifacts/predictions",
        log_dir="/vol/phase4/artifacts/logs",
        qcsv="/new/reports/real_hair_validation/image_quality_v2.csv",
        xai_dir="/vol/phase4/artifacts/logs/xai_v2",
        out_dir=NEW,
        seeds=sd,
        suffix=suffix,
    )
    nv.commit()
    print("aggregate:", json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main(seeds: str = "42,123,2026", step: str = "all", suffix: str = ""):
    import subprocess

    all_seeds = "42,123,2026,7,11,17,23,31"
    if step == "aggregate":
        print("aggregate:", aggregate.remote(seeds, suffix), flush=True)
        return
    if step == "extra":
        # Remaining five seeds for every arm, then re-aggregate over all eight
        # seeds into separately named files (the 3-seed results are preserved).
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip() or "unknown"
        print("prep:", prep.remote(), flush=True)
        for s in (7, 11, 17, 23, 31):
            for arm in ("A", "B", "Astar", "Cv2"):
                print("train:", train_arm_seed.remote(arm, s), flush=True)
        for s in (7, 11, 17, 23, 31):
            for arm in ("A", "B", "Astar", "Cv2"):
                print("eval:", eval_arm_seed.remote(arm, s), flush=True)
                print("xai:", xai_arm_seed.remote(arm, s, commit), flush=True)
        print("aggregate:", aggregate.remote(all_seeds, "_8seed"), flush=True)
        return
    seed_list = [int(s) for s in seeds.split(",")]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or "unknown"
    print("commit", commit, flush=True)
    print("prep:", prep.remote(), flush=True)
    if step in ("all", "train"):
        for s in seed_list:
            print("train A:", train_missing_a.remote(s), flush=True)
    if step in ("all", "eval"):
        for s in seed_list:
            for arm in ("A", "B", "Astar", "Cv2"):
                print("eval:", eval_arm_seed.remote(arm, s), flush=True)
    if step in ("all", "xai"):
        for s in seed_list:
            for arm in ("A", "B", "Astar", "Cv2"):
                print("xai:", xai_arm_seed.remote(arm, s, commit), flush=True)
    print("aggregate:", aggregate.remote(), flush=True)

"""Phase 3: retune the hair detector on real masks, then evaluate ONCE on held-out.

No model retraining here. The HELD-OUT half is only read after the config is frozen.
"""

from __future__ import annotations

import modal

APP = "dm-ahr-real-hair-validation"
app = modal.App(APP)
v = modal.Volume.from_name("dm-ahr-real-hair-validation")
p = modal.Volume.from_name("dmahr-vit-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("numpy==1.26.4", "pandas==2.2.3", "pillow==11.0.0",
                 "opencv-python-headless==4.10.0.84", "scikit-learn==1.5.2")
    .env({"PYTHONUNBUFFERED": "1"})
)

R = "/vol/reports/real_hair_validation"
SPLIT_SEED = 20261117


def _cfg_hash(cfg: dict) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _orient_kernel(length: int, angle: float):
    import cv2
    import numpy as np
    k = np.zeros((length, length), dtype=np.uint8)
    c = length // 2
    rad = np.deg2rad(angle)
    dx, dy = int(round(c * np.cos(rad))), int(round(c * np.sin(rad)))
    cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, 1)
    return k


def _response(gray, length, light_hair=True):
    """Black-hat and top-hat responses, computed once per kernel length."""
    import cv2
    import numpy as np
    dark = np.zeros_like(gray)
    light = np.zeros_like(gray)
    for i in range(12):
        k = _orient_kernel(length, 180.0 * i / 12)
        dark = np.maximum(dark, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k))
        if light_hair:
            light = np.maximum(light, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k))
    return dark, light


def _mask_from_response(dark, light, cfg):
    import cv2
    import numpy as np
    mask = (dark > cfg["blackhat_threshold"]).astype(np.uint8)
    if cfg.get("light_hair", True):
        mask |= (light > cfg["tophat_threshold"]).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < cfg["min_component_area"]:
            continue
        if cfg.get("thin", False):
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            if max(w, h) < 2 * max(min(w, h), 1):
                continue
        keep[labels == i] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (cfg["dilate_kernel"], cfg["dilate_kernel"]))
    keep = cv2.dilate(keep, k, iterations=cfg["dilate_iterations"])
    return keep


def _metrics(pred, gt):
    import numpy as np
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec,
            "recall": rec, "specificity": spec, "dice": dice, "iou": iou}


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": v, "/proj": p})
def prepare_split() -> dict:
    import hashlib
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    v.reload()
    ov = pd.read_csv(f"{R}/overlap_audit.csv")
    mp = pd.read_csv(f"{R}/mendeley_pairs.csv")
    keep = ov[ov["class"] == "A"].merge(mp, left_on="source_id", right_on="image_id")
    assert len(keep) == 395, len(keep)
    pos = keep[keep["hair_coverage"] >= 0.01].reset_index(drop=True)
    neg = keep[keep["hair_coverage"] < 0.01].reset_index(drop=True)
    print("independent:", len(keep), "pos", len(pos), "neg", len(neg), flush=True)

    rng = np.random.default_rng(SPLIT_SEED)
    parts = []
    for label, df in (("pos", pos), ("neg", neg)):
        idx = rng.permutation(len(df))
        half = len(idx) // 2
        for name, sel in (("TUNE", idx[:half]), ("HELD_OUT", idx[half:])):
            parts.append(df.iloc[sel].assign(phase=name, gt_hair_present=(label == "pos")))
    split = pd.concat(parts, ignore_index=True)
    out = Path(R) / "detector_split.csv"
    split.to_csv(out, index=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    Path(R + "/detector_split.sha256").write_text(digest + "\n")
    v.commit()
    return {"n": len(split), "tune": int((split.phase == "TUNE").sum()),
            "held_out": int((split.phase == "HELD_OUT").sum()), "sha256": digest}


@app.function(image=image, cpu=8, memory=32768, timeout=14400, volumes={"/vol": v, "/proj": p})
def tune_on_tune() -> dict:
    import hashlib
    import json
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd

    v.reload()
    split = pd.read_csv(f"{R}/detector_split.csv")
    split_hash = Path(R + "/detector_split.sha256").read_text().strip()
    tune = split[split["phase"] == "TUNE"].reset_index(drop=True)
    # HELD_OUT is never read in this function.

    imgs, gts, y = [], [], []
    for r in tune.itertuples(index=False):
        arr = cv2.imread(str(r.image))
        if arr is None:
            continue
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        m = cv2.imread(str(r.mask), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        imgs.append(gray)
        gts.append((m > 127).astype(np.uint8))
        y.append(bool(r.gt_hair_present))
    print("TUNE images loaded:", len(imgs), "positives:", sum(y), flush=True)

    grid = []
    for bh in (8, 14, 18, 24):
        for th in (12, 18, 24):
            for kl in (11, 17):
                for ma in (10, 40):
                    for dk in (3, 5):
                        for thin in (False, True):
                            grid.append({"blackhat_threshold": bh, "tophat_threshold": th,
                                         "kernel_length": kl, "min_component_area": ma,
                                         "dilate_kernel": dk, "dilate_iterations": 2,
                                         "thin": thin, "light_hair": True})
    responses = {}
    for kl in (11, 17):
        responses[kl] = [_response(g, kl) for g in imgs]

    cov_grid = (0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
    results = []
    for cfg in grid:
        masks = [_mask_from_response(d, l, cfg) for d, l in responses[cfg["kernel_length"]]]
        cov = np.array([float(m.mean()) for m in masks])
        dice = float(np.mean([_metrics(m, g)["dice"] for m, g in zip(masks, gts)]))
        yy = np.asarray(y, dtype=bool)
        best = None
        for c in cov_grid:
            pred = cov >= c
            sens = float((pred & yy).sum() / max(yy.sum(), 1))
            spec = float((~pred & ~yy).sum() / max((~yy).sum(), 1))
            bal = (sens + spec) / 2
            if best is None or bal > best["bal_acc"]:
                best = {"coverage_threshold": c, "sensitivity": sens,
                        "specificity": spec, "bal_acc": bal}
        results.append({**cfg, "tune_dice": dice, **best})
    res = pd.DataFrame(results)
    feasible = res[res["bal_acc"] >= 0.6]
    chosen = (feasible.sort_values("tune_dice", ascending=False).iloc[0]
              if len(feasible) else res.sort_values(["bal_acc", "tune_dice"], ascending=False).iloc[0])
    keys = ("blackhat_threshold", "tophat_threshold", "kernel_length",
            "min_component_area", "dilate_kernel", "dilate_iterations",
            "thin", "light_hair", "coverage_threshold")
    cfg = {}
    for k in keys:
        val = chosen[k]
        if k in ("thin", "light_hair"):
            cfg[k] = bool(val)
        elif k == "coverage_threshold":
            cfg[k] = float(val)
        else:
            cfg[k] = int(val)
    payload = {"source": "TUNE half of the 395 independent Mendeley images",
               "split_sha256": split_hash, "split_seed": SPLIT_SEED,
               "n_tune_images": len(imgs), "n_grid": len(grid),
               "constraint": "image-level balanced accuracy >= 0.6 on TUNE",
               "config": cfg, "config_sha256": _cfg_hash(cfg),
               "tune_dice": float(chosen["tune_dice"]),
               "tune_bal_acc": float(chosen["bal_acc"]),
               "tune_sensitivity": float(chosen["sensitivity"]),
               "tune_specificity": float(chosen["specificity"]),
               "feasible_configs": int(len(feasible)),
               "top5": res.sort_values("tune_dice", ascending=False).head(5).to_dict("records")}
    Path(R + "/detector_retune.json").write_text(json.dumps(payload, indent=2, default=float))
    v.commit()
    print("FROZEN CONFIG", json.dumps(cfg), "hash", payload["config_sha256"][:16], flush=True)
    return payload


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": v, "/proj": p})
def evaluate_heldout() -> dict:
    import json
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd

    v.reload()
    cfg = json.loads(Path(R + "/detector_retune.json").read_text())["config"]
    old = {"blackhat_threshold": 18, "tophat_threshold": 22, "kernel_length": 17,
           "min_component_area": 40, "dilate_kernel": 5, "dilate_iterations": 2,
           "thin": False, "light_hair": True, "coverage_threshold": 0.01}
    split = pd.read_csv(f"{R}/detector_split.csv")
    held = split[split["phase"] == "HELD_OUT"].reset_index(drop=True)  # first read after freeze

    items = []
    for r in held.itertuples(index=False):
        arr = cv2.imread(str(r.image))
        m = cv2.imread(str(r.mask), cv2.IMREAD_GRAYSCALE)
        if arr is None or m is None:
            continue
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        items.append((gray, (m > 127).astype(np.uint8), bool(r.gt_hair_present), r.image_id))

    rows = []
    for name, c in (("retuned", cfg), ("old_frozen", old)):
        resps = {}
        for kl in {cfg["kernel_length"], old["kernel_length"]}:
            resps[kl] = [_response(g, kl) for g, _, _, _ in items]
        masks = [_mask_from_response(d, l, c) for d, l in resps[c["kernel_length"]]]
        pix = [_metrics(m, gt) for m, (_, gt, _, _) in zip(masks, items)]
        cov = np.array([float(m.mean()) for m in masks])
        pred = cov >= c["coverage_threshold"]
        truth = np.array([y for _, _, y, _ in items])
        sens = float((pred & truth).sum() / max(truth.sum(), 1))
        spec = float((~pred & ~truth).sum() / max((~truth).sum(), 1))
        # image-level bootstrap CIs
        rng = np.random.default_rng(20261118)
        bs_sens, bs_spec, bs_dice = [], [], []
        idx = np.arange(len(items))
        for _ in range(1000):
            b = rng.choice(idx, size=len(idx), replace=True)
            tt, pp = truth[b], pred[b]
            if tt.sum(): bs_sens.append(float((pp[tt] & tt[tt]).sum() / tt.sum()))
            if (~tt).sum(): bs_spec.append(float((~pp[~tt] & ~tt[~tt]).sum() / (~tt).sum()))
            bs_dice.append(float(np.mean([pix[i]["dice"] for i in b])))
        rows.append({"detector": name, "n": len(items), "n_hair_positive": int(truth.sum()),
                     **{f"pixel_{k}": float(np.mean([m[k] for m in pix]))
                        for k in ("precision", "recall", "specificity", "dice", "iou")},
                     "pixel_dice_ci_low": float(np.percentile(bs_dice, 2.5)),
                     "pixel_dice_ci_high": float(np.percentile(bs_dice, 97.5)),
                     "image_sensitivity": sens, "image_specificity": spec,
                     "image_sens_ci_low": float(np.percentile(bs_sens, 2.5)),
                     "image_sens_ci_high": float(np.percentile(bs_sens, 97.5)),
                     "image_spec_ci_low": float(np.percentile(bs_spec, 2.5)),
                     "image_spec_ci_high": float(np.percentile(bs_spec, 97.5))})
    df = pd.DataFrame(rows)
    df.to_csv(R + "/hair_detector_heldout_metrics.csv", index=False)
    v.commit()
    print(df[["detector", "pixel_dice", "image_sensitivity", "image_specificity"]].to_string(index=False), flush=True)
    return rows


@app.function(image=image, cpu=8, memory=32768, timeout=14400, volumes={"/vol": v, "/proj": p})
def ham_rate() -> dict:
    import json
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd

    v.reload()
    p.reload()
    cfg = json.loads(Path(R + "/detector_retune.json").read_text())["config"]
    scr = pd.read_csv("/proj/data/manifests/hair_screening.csv")
    root = Path("/proj/data/raw/isic_212/images")
    counts = {}
    for split in ("train", "val", "test"):
        ids = scr[scr["split"] == split]["image_id"].tolist()
        cov = []
        for iid in ids:
            f = root / f"{iid}.jpg"
            arr = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                continue
            d, l = _response(arr, cfg["kernel_length"], cfg.get("light_hair", True))
            m = _mask_from_response(d, l, cfg)
            cov.append(float(m.mean()))
        pred = np.asarray(cov) >= cfg["coverage_threshold"]
        counts[split] = {"n": len(pred), "hair_present": int(pred.sum()),
                         "hair_present_pct": round(100 * float(pred.mean()), 2)}
    Path(R + "/detector_ham_rate.json").write_text(json.dumps(counts, indent=2))
    v.commit()
    print(json.dumps(counts), flush=True)
    return counts


@app.local_entrypoint()
def main():
    print("split:", prepare_split.remote(), flush=True)
    print("tune:", {k: val for k, val in tune_on_tune.remote().items()
                    if k in ("config", "config_sha256", "tune_dice", "tune_bal_acc")}, flush=True)
    r = evaluate_heldout.remote()
    retuned = next(x for x in r if x["detector"] == "retuned")
    print("HELD-OUT specificity:", retuned["image_specificity"], flush=True)
    if retuned["image_specificity"] < 0.5:
        print("GATE FAILED: image specificity < 0.5. STOPPING before Phase 4.", flush=True)
        return
    print("GATE PASSED. HAM rate:", ham_rate.remote(), flush=True)


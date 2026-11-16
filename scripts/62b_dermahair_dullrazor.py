"""Frozen classical DullRazor (Arm B) on the public DERMAHAIR pairs, CPU only.

Item 8 of the client's final fix round: report PSNR/SSIM/LPIPS next to the released
checkpoint on IDENTICAL public pairs. No diffusion model runs here and nothing is tuned:
the hair screen and the DullRazor parameters are the frozen ones used for Arm B
(``load_frozen``), and PSNR/SSIM/LPIPS use exactly the conventions of
``scripts/62_dermahair_repro.py`` (upstream core.metrics at 128 px, skimage at 128 and
256, LPIPS alex). The 3 of 707 pairs whose hairy and hairless images are identical have
undefined PSNR and are excluded from every mean, as in the released-checkpoint report.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, load_frozen, save_json, sha256_file  # noqa: E402
from dm_ahr_vit.diffusion import dmahr_adapter as ad  # noqa: E402
from dm_ahr_vit.hair.dullrazor import apply_dullrazor  # noqa: E402
from dm_ahr_vit.hair.screening import detect_hair  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def load_pairs(root: Path) -> list[dict]:
    """Byte-identical to the pair list in 62_dermahair_repro.py."""
    pairs = []
    for c in CLASSES:
        for p in sorted((root / "Data_Skin_with_Hair" / c).glob("*.jpg")):
            q = root / "Data_Skin_without_Hair" / c / p.name
            if q.exists():
                pairs.append({"cls": c, "name": p.name, "hairy": p, "clean": q})
    return pairs


def dataset_hash(pairs: list[dict]) -> str:
    h = hashlib.sha256()
    for p in pairs:
        for k in ("hairy", "clean"):
            h.update(p[k].name.encode() + str(p[k].stat().st_size).encode())
    return h.hexdigest()


def metrics_for(outs, gts, gts256, lp_net, Metrics, device):
    """Same metric conventions as 62_dermahair_repro.py; CPU-safe."""
    import torch  # noqa: PLC0415

    rows = []
    with torch.no_grad():
        for o, g, g256 in zip(outs, gts, gts256):
            o256 = np.asarray(Image.fromarray(o).resize((256, 256), Image.Resampling.BICUBIC))
            t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)  # noqa: E731
            rows.append({
                "psnr_up": Metrics.calculate_psnr(o, g), "ssim_up": Metrics.calculate_ssim(o, g),
                "psnr_sk": peak_signal_noise_ratio(g, o, data_range=255),
                "ssim_sk": structural_similarity(g, o, data_range=255, channel_axis=2, win_size=7),
                "psnr_256": peak_signal_noise_ratio(g256, o256, data_range=255),
                "ssim_256": structural_similarity(g256, o256, data_range=255, channel_axis=2, win_size=7),
                "lpips": float(lp_net(t(o), t(g)).mean()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import lpips  # noqa: PLC0415
    import torch  # noqa: PLC0415

    ad._ensure_path()
    import core.metrics as Metrics  # noqa: PLC0415

    device = "cpu"
    screen_cfg = load_frozen("hair_screen", "configs/hair_screen.json")
    dull_cfg = load_frozen("dullrazor", "configs/dullrazor.json")
    load_config("configs/dmahr.json")  # same config the released-checkpoint arm uses

    root = data_root() / "raw" / "dermahair"
    pairs = load_pairs(root)
    if args.limit:
        pairs = pairs[:args.limit]
    log.info("%d DERMAHAIR pairs (frozen DullRazor, CPU)", len(pairs))

    hairy = [Image.open(p["hairy"]).convert("RGB") for p in pairs]
    clean = [Image.open(p["clean"]).convert("RGB") for p in pairs]
    gts = [np.asarray(c.resize((128, 128), Image.Resampling.BICUBIC)) for c in clean]
    gts256 = [np.asarray(c) for c in clean]
    lp_net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    with timed("dermahair_dullrazor", log) as tmr:
        outs = []
        for h in hairy:
            arr = np.asarray(h)
            mask, _ = detect_hair(arr, screen_cfg)
            cleaned = apply_dullrazor(arr, mask, dull_cfg)
            outs.append(np.asarray(Image.fromarray(cleaned).resize((128, 128), Image.Resampling.BICUBIC)))
        df = metrics_for(outs, gts, gts256, lp_net, Metrics, device)

    # Exclude the identical pairs (PSNR undefined) from every mean, as in 62_dermahair_repro.py.
    ok = np.isfinite(df["psnr_up"]).to_numpy()
    metrics = {k: float(df[k][ok].mean()) for k in df.columns}
    metrics.update({f"{k}_sd": float(df[k][ok].std()) for k in df.columns})
    n_valid = int(ok.sum())

    out_dir = artifact_dir("logs")
    per = df.copy()
    per.insert(0, "setting", "dullrazor_frozen")
    per.insert(1, "cls", [p["cls"] for p in pairs])
    per.insert(2, "name", [p["name"] for p in pairs])
    per.insert(3, "valid", ok)
    per.to_csv(out_dir / "dermahair_dullrazor_pairs.csv", index=False)

    prov = {
        "method": "frozen classical DullRazor (Arm B procedure) on the public DERMAHAIR pairs, CPU only",
        "n_pairs": len(pairs), "n_valid": n_valid, "n_excluded_identical": int((~ok).sum()),
        "dataset_sha256_of_names_and_sizes": dataset_hash(pairs),
        "dataset_source": "kaggle riotulab/skin-cancer-hair-removal v1, anonymous public download",
        "no_tuning": "hair screen and DullRazor parameters loaded from the frozen train/val configs; DERMAHAIR pairs never influence any parameter",
        "metric_note": "psnr_up/ssim_up = upstream core.metrics at 128 px; sk = skimage at 128; 256 = output bicubic-upsampled vs the 256 px hairless image; LPIPS alex",
        "git_commit": args.commit, "seconds": tmr["seconds"], "device": device,
        "sha256": {"configs/hair_screen.json": sha256_file("configs/hair_screen.json"),
                   "configs/dullrazor.json": sha256_file("configs/dullrazor.json")},
    }
    save_json({"provenance": prov, "metrics": metrics}, out_dir / "dermahair_dullrazor.json")
    record_run("dermahair_dullrazor", gpu="CPU", seconds=tmr["seconds"], pairs=n_valid)
    print(f"DullRazor on {n_valid} DERMAHAIR pairs: PSNR(up) {metrics['psnr_up']:.2f} "
          f"SSIM(up) {metrics['ssim_up']:.4f} LPIPS {metrics['lpips']:.4f} "
          f"(excluded identical={int((~ok).sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

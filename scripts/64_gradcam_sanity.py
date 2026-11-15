"""Sanity check of the production last-block ln_1 Grad-CAM (synthetic input only).

Why: in the corrected analysis the ln_1 map's mean recall sits below the random-pixel
baseline (about 0.200) in several arms, which is suspicious for an implementation bug
(token-to-grid reshape or a row/column transposition) rather than a property of the maps.
This script tests the implementation on planted bright squares and against a reference
implementation; it never touches a dataset image, mask or label.

Checks
1. Follow check: a bright square is planted at two asymmetric locations of a 224 x 224
   smooth-noise image. The production MultiLayerGradCAM("last_ln1") map, upsampled exactly
   the way the corrected analysis upsamples it, must cover the square (top-20% recall) and
   point inside it, for the class whose logit the square most raises (largest positive
   with-minus-without logit difference; the script aborts if no class responds positively).
2. Transposed control: the same map transposed (rows and columns swapped) is scored
   against the same square. If the production reshape swapped rows and columns, the
   transposed map would score like the true map and the true map would fail.
3. Reshape unit check: a token tensor that encodes its own index must land at grid
   position (row, col) = ((i-1) // 14, (i-1) % 14) for patch token i (the class token
   at position 0 is dropped) through the production `_as_grid`.
4. Reference implementation: pytorch-grad-cam 1.5.0 with the standard reshape_transform
   and the same target layer must agree with the production map (Spearman correlation).

The checkpoint is a frozen, already-evaluated one (default A seed 42). The result is
written to artifacts/logs/gradcam_sanity.json, a contact sheet to
artifacts/logs/gradcam_sanity.png, and one line to the run registry.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm_ahr_vit.config import CLASSES, load_config, sha256_file  # noqa: E402
from dm_ahr_vit.paths import artifact_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402
from dm_ahr_vit.vit.model import build_model, load_checkpoint  # noqa: E402
from dm_ahr_vit.xai.batched import MultiLayerGradCAM, localization_scores  # noqa: E402

log = setup_logging()
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# Two asymmetric square positions, each 48 px = 3 x 3 patch cells (16 px) and aligned to the
# patch grid, so one whole interior cell lies inside the square. Without alignment the
# bilinearly upsampled argmax pixel can fall just outside the square edge (a quantization
# artifact, not an implementation fault). A transposition moves the mass to a disjoint
# location and cannot pass by accident.
PLACEMENTS = [((96, 144), (32, 80)), ((80, 128), (144, 192))]
SQUARE_LEVEL = (0.90, 1.00)
BG_RANGE = (0.42, 0.58)
BLUR_SIGMA = 6.0  # smooth background; white noise can dominate patch embeddings on its own


def base_noise(size: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth background in [BG_RANGE]: uniform noise, blurred, rescaled per channel."""
    img = rng.uniform(0.0, 1.0, size=(3, size, size)).astype(np.float32)
    for c in range(3):
        img[c] = gaussian_filter(img[c], sigma=BLUR_SIGMA)
    lo, hi = BG_RANGE
    for c in range(3):
        mn, mx = float(img[c].min()), float(img[c].max())
        img[c] = lo + (img[c] - mn) * (hi - lo) / max(mx - mn, 1e-12)
    return img


def add_square(img: np.ndarray, rows: tuple[int, int], cols: tuple[int, int],
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Overwrite the square region with bright values; return (image, square mask)."""
    img = img.copy()
    mask = np.zeros(img.shape[1:], dtype=bool)
    mask[rows[0]:rows[1], cols[0]:cols[1]] = True
    img[:, mask] = rng.uniform(*SQUARE_LEVEL, size=(3, int(mask.sum()))).astype(np.float32)
    return img, mask


def normalize(img: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(img)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD)[None]


def upsample(m: np.ndarray, size: int, device: str) -> np.ndarray:
    """The exact upsampling of the corrected analysis (scripts/58_xai_corrected.py)."""
    t = torch.from_numpy(m).float().to(device)[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def pick_class(model: torch.nn.Module, x: torch.Tensor, x_bg: torch.Tensor,
               device: str) -> tuple[int, float, list[float]]:
    """The class whose logit the square most raises (largest positive difference)."""
    with torch.no_grad():
        delta = (model(x.to(device)) - model(x_bg.to(device)))[0]
    i = int(delta.argmax())
    if float(delta[i]) <= 0.0:
        raise SystemExit(f"no class responds positively to the square; deltas={delta.tolist()}")
    return i, float(delta[i]), delta.tolist()


def ensure_reference() -> str:
    """Make pytorch-grad-cam 1.5.0 importable; return what was done (no image mutation)."""
    try:
        import pytorch_grad_cam  # noqa: F401
        return "already importable"
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               "grad-cam==1.5.0"])
        return "installed grad-cam==1.5.0 at runtime (runtime-only, not in the image)"


def reference_map(model: torch.nn.Module, x: torch.Tensor, class_idx: int, grid: int) -> np.ndarray:
    from pytorch_grad_cam import GradCAM as RefGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    def reshape_transform(tensor, height=grid, width=grid):
        t = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        return t.permute(0, 3, 1, 2)

    try:
        ref = RefGradCAM(model=model, target_layers=[model.encoder.layers[-1].ln_1],
                         reshape_transform=reshape_transform)
    except TypeError:  # newer versions dropped the `model` argument
        ref = RefGradCAM(target_layers=[model.encoder.layers[-1].ln_1],
                         reshape_transform=reshape_transform)
    x = x.to(next(model.parameters()).device)
    return np.asarray(ref(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0])


def reshape_unit_check(cam: MultiLayerGradCAM, grid: int) -> dict:
    """Patch token k (1-based after the class token) must land at ((k-1)//grid, (k-1)%grid)
    through the production reshape, which drops the class token at position 0."""
    c = 8
    tokens = torch.arange(grid * grid + 1, dtype=torch.float32).view(1, grid * grid + 1, 1).repeat(1, 1, c)
    out = cam._as_grid(tokens, "tokens")
    expect = torch.arange(1, grid * grid + 1, dtype=torch.float32).view(grid, grid)
    ok = bool(torch.equal(out[0, 0], expect))
    return {"ok": ok,
            "detail": ("class token dropped; patch token k lands at ((k-1)//14, (k-1)%14) exactly"
                       if ok else "token-to-grid reshape is not row-major after the class token")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()

    cfg = load_config("configs/vit.json")
    size = int(cfg["image_size"])
    fraction = float(load_config("configs/sampling.json")["xai_top_fraction"])
    grid = size // 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = artifact_dir("checkpoints", "vit", f"{args.arm}_seed{args.seed}") / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    model = build_model(pretrained=True).to(device)  # same construction as scripts/58
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()
    rng = np.random.default_rng(2026)
    ref_status = ensure_reference()

    with timed("gradcam_sanity", log) as tmr, MultiLayerGradCAM(model) as cam:
        unit = reshape_unit_check(cam, grid)
        placements = []
        for rows, cols in PLACEMENTS:
            img_bg = base_noise(size, rng)
            img, mask = add_square(img_bg, rows, cols, rng)
            x, x_bg = normalize(img).to(device), normalize(img_bg).to(device)
            class_idx, delta, deltas = pick_class(model, x, x_bg, device)
            grids, _ = cam(x, class_idx=torch.tensor([class_idx], device=device))
            sal = upsample(grids["last_ln1"][0], size, device)
            sal_pg = upsample(grids["patch_grid"][0], size, device)
            ref = reference_map(model, x, class_idx, grid)
            rho = float(spearmanr(sal.flatten(), ref.flatten()).statistic)
            placements.append({
                "rows": list(rows), "cols": list(cols),
                "class_idx": class_idx, "class_name": CLASSES[class_idx],
                "delta_logit": delta, "logit_deltas_by_class": dict(zip(CLASSES, deltas)),
                "follow_ln1": localization_scores(sal, mask, fraction),
                "follow_patch_grid": localization_scores(sal_pg, mask, fraction),
                "transposed_ln1": localization_scores(np.transpose(sal), mask, fraction),
                "reference": {"spearman": rho, "library": "pytorch-grad-cam 1.5.0",
                              "reshape_transform": "standard ViT reshape, same target layer"},
                "_plot": (img, sal, ref, mask),
            })

        rec_f = [p["follow_ln1"]["recall"] for p in placements]
        rec_t = [p["transposed_ln1"]["recall"] for p in placements]
        poi_f = [p["follow_ln1"]["pointing"] for p in placements]
        rho = [p["reference"]["spearman"] for p in placements]
        criteria = {
            "follow_recall_min": min(rec_f),
            "follow_pointing_min": min(poi_f),
            "transposed_recall_max": max(rec_t),
            "reference_spearman_min": min(rho),
            "chance_recall": fraction,
        }
        criteria.update({
            "follow_covers_square": min(rec_f) >= 0.60,
            "follow_points_inside": min(poi_f) >= 0.99,
            "transposed_fails": max(rec_t) <= min(rec_f) - 0.2,
            "reshape_exact": unit["ok"],
            "reference_agrees": min(rho) >= 0.90,
        })
        verdict = "passes" if all(criteria[k] for k in
                                  ("follow_covers_square", "follow_points_inside", "transposed_fails",
                                   "reshape_exact", "reference_agrees")) else "fails"

    out = {
        "what": "sanity check of the production MultiLayerGradCAM last_ln1 map on planted bright squares",
        "input": (f"synthetic {size} x {size} ImageNet-normalised smooth noise with a bright square; "
                  "no dataset image, mask or label used"),
        "checkpoint": {"path": str(ckpt), "sha256": sha256_file(ckpt)},
        "arm": args.arm, "seed": args.seed, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "size": size, "fraction": fraction,
        "reference_setup": ref_status,
        "reshape_unit_check": unit,
        "placements": [{k: v for k, v in p.items() if k != "_plot"} for p in placements],
        "criteria": criteria,
        "verdict": verdict,
        "git_commit": args.commit,
        "seconds": tmr["seconds"],
    }
    path = artifact_dir("logs") / "gradcam_sanity.json"
    path.write_text(json.dumps(out, indent=2))

    # Contact sheet for visual inspection: input, production map, reference map, transposed map.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for i, p in enumerate(placements):
        img, sal, ref, mask = p["_plot"]
        disp = np.transpose(img.clip(0, 1), (1, 2, 0))
        axes[i][0].imshow(disp)
        axes[i][0].set_title(f"input, square rows {p['rows'][0]}-{p['rows'][1]}\n"
                             f"cols {p['cols'][0]}-{p['cols'][1]} ({p['class_name']})", fontsize=8)
        axes[i][1].imshow(disp); axes[i][1].imshow(sal, cmap="inferno", alpha=0.55)
        axes[i][1].contour(mask, levels=[0.5], colors="cyan", linewidths=0.8)
        axes[i][1].set_title(f"production ln_1 (recall {p['follow_ln1']['recall']:.2f})", fontsize=8)
        axes[i][2].imshow(disp); axes[i][2].imshow(ref, cmap="inferno", alpha=0.55)
        axes[i][2].contour(mask, levels=[0.5], colors="cyan", linewidths=0.8)
        axes[i][2].set_title(f"pytorch-grad-cam (rho {p['reference']['spearman']:.3f})", fontsize=8)
        axes[i][3].imshow(sal.T, cmap="inferno", alpha=0.55)
        axes[i][3].contour(mask, levels=[0.5], colors="cyan", linewidths=0.8)
        axes[i][3].set_title(f"transposed control (recall {p['transposed_ln1']['recall']:.2f})", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Grad-CAM ln_1 sanity ({verdict}; {args.arm} seed {args.seed}; synthetic input) - agent-reviewed",
                 fontsize=10)
    fig.tight_layout()
    png = artifact_dir("logs") / "gradcam_sanity.png"
    fig.savefig(png, dpi=120)
    record_run("gradcam_sanity", gpu=out["gpu"], seconds=tmr["seconds"], status=verdict,
               checkpoint=str(ckpt), follow_recall_min=criteria["follow_recall_min"],
               reference_spearman_min=criteria["reference_spearman_min"], commit=args.commit)
    print(f"verdict {verdict}; wrote {path} and {png}")
    return 0 if verdict == "passes" else 1


if __name__ == "__main__":
    raise SystemExit(main())

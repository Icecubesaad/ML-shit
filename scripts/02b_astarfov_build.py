"""Build the exploratory A*-fov arm: A's field of view at A*'s storage resolution.

Item 9 of the client's final fix round. Hair-present images are stored as the CENTRE
SQUARE crop of the original (side = min(w, h), i.e. 450 x 450 for a 600 x 450 photo),
squashed to 128 x 128 with BICUBIC resampling. The classifier's evaluation transform
then resizes to 255 px and centre-crops to 224 px, so it sees exactly the same window
of the photograph as arm A (57.8% of the photo) while the storage resolution matches
A* (128 px). Hair-absent images pass through at the original 600 x 450, as in every arm.

No re-screening: ``hair_present`` is read from the frozen data/manifests/hair_screening.csv.
No geometry.py change: the crop/output sizes come from xai.geometry constants. This arm
is exploratory/post-hoc and gets no XAI (not requested).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import sha256_file  # noqa: E402
from dm_ahr_vit.paths import (  # noqa: E402
    artifact_dir,
    isic_images_dir,
    manifest_dir,
    processed_dir,
)
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402
from dm_ahr_vit.xai.geometry import SQUASHED_SIZE  # noqa: E402

log = setup_logging()
ARM = "Astarfov"
OUT_SIDE = int(SQUASHED_SIZE[0])  # 128


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def write_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, compress_level=3)


def centre_square_squash(img: Image.Image, out_side: int = OUT_SIDE) -> np.ndarray:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    crop = img.crop((left, top, left + s, top + s))
    return np.asarray(crop.resize((out_side, out_side), Image.Resampling.BICUBIC))


def contact_sheet(items: list[tuple[np.ndarray, str]], out: Path, cols: int = 4) -> None:
    if not items:
        return
    th = 200
    thumbs = [(np.asarray(Image.fromarray(a).resize((th, th), Image.Resampling.BICUBIC)), label)
              for a, label in items]
    rows = (len(thumbs) + cols - 1) // cols
    canvas = np.zeros((rows * (th + 22), cols * th, 3), dtype=np.uint8)
    from PIL import ImageDraw  # noqa: PLC0415

    for i, (arr, label) in enumerate(thumbs):
        r, c = divmod(i, cols)
        y, x = r * (th + 22), c * th
        canvas[y:y + th, x:x + th] = arr
    sheet = Image.fromarray(canvas)
    dr = ImageDraw.Draw(sheet)
    for i, (_, label) in enumerate(thumbs):
        r, c = divmod(i, cols)
        y, x = r * (th + 22), c * th
        dr.text((x + 2, y + th + 3), label[:34], fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote contact sheet {out} ({len(items)} panels, agent-reviewed)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    screen_path = manifest_dir() / "hair_screening.csv"
    screen = pd.read_csv(screen_path)
    if args.limit:
        screen = screen.head(args.limit)
    out_dir = processed_dir(ARM)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = failed = 0
    skipped_rows = []
    panels = []
    with timed("astarfov_build", log) as tmr:
        for _, r in screen.iterrows():
            iid = str(r["image_id"])
            dest = out_dir / f"{iid}.png"
            hair_present = bool(r["hair_present"])
            try:
                with Image.open(isic_images_dir() / f"{iid}.jpg") as im:
                    src = im.convert("RGB")
                    if hair_present:
                        arr = centre_square_squash(src)
                        if len(panels) < 8:
                            panels.append((np.asarray(src), f"{iid} original"))
                            panels.append((arr, f"{iid} A*-fov"))
                    else:
                        arr = np.asarray(src)  # passthrough at the original 600 x 450
                write_png(arr, dest)
                written += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                skipped_rows.append({"image_id": iid, "hair_present": hair_present,
                                     "reason": repr(exc)})
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(
            artifact_dir("logs") / "astarfov_build_skipped.csv", index=False)

    contact_sheet(panels, artifact_dir("panels") / "astarfov_build.jpg")

    hp = int(screen["hair_present"].astype(bool).sum())
    summary = {
        "arm": ARM, "images": int(len(screen)), "hair_present": hp,
        "hair_absent": int(len(screen)) - hp, "written": written, "failed": failed,
        "output_side": OUT_SIDE,
        "crop_rule": "centre square of side min(w, h) from the original (450 x 450 for 600 x 450)",
        "field_of_view": "identical to arm A: the 224 x 224 centre crop of the 255 px resized frame covers 57.8% of the original photo",
        "no_rescreening": "hair_present read from the frozen data/manifests/hair_screening.csv",
        "xai": False,
        "manifest_sha256": sha256_file(screen_path),
        "git_commit": args.commit, "seconds": tmr["seconds"],
    }
    from dm_ahr_vit.config import save_json  # noqa: PLC0415
    save_json(summary, artifact_dir("logs") / "astarfov_build.json")
    record_run("astarfov_build", gpu=None, images=written, failed=failed,
               seconds=tmr["seconds"])
    print(summary)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

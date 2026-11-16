"""A*-fov build logic (CPU only): centre-square crop, 128 px squash, matched field of view."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_centre_square_squash_is_centred_and_128():
    mod = _load("astarfov_build", "scripts/02b_astarfov_build.py")
    # 600 x 450 image: left 100 columns black, the rest white. The centre square starts
    # at x = (600-450)//2 = 75, so the crop keeps columns 75..524: its first 25 columns
    # are black and its last column is white.
    arr = np.ones((450, 600, 3), dtype=np.uint8) * 255
    arr[:, :100] = 0
    out = mod.centre_square_squash(Image.fromarray(arr), out_side=128)
    assert out.shape == (128, 128, 3), out.shape
    assert out[64, 0].mean() < 40, out[64, 0]
    assert out[64, -1].mean() > 215, out[64, -1]
    # uniform input stays uniform
    flat = np.full((600, 450, 3), 77, dtype=np.uint8)
    out2 = mod.centre_square_squash(Image.fromarray(flat), out_side=128)
    assert (out2 == 77).all()


def test_transform_table_matches_A_field_of_view():
    bp = _load("build_paper_fov", "scripts/80_build_paper.py")
    rows = bp.transform_table_rows(224, include_astarfov=True)
    frac = {r[0]: r[3] for r in rows}
    a = frac["A / B (hair-present)"]
    fov = frac["A*-fov centre-square squash"]
    sq = frac["A* / Cv2 / Creleased / C squash"]
    assert a == fov, (a, fov)          # A*-fov recovers A's field of view
    assert sq != a                      # A* changes the field of view
    assert float(a.rstrip("%")) < float(sq.rstrip("%"))
    assert float(a.rstrip("%")) > 55 and float(a.rstrip("%")) < 60

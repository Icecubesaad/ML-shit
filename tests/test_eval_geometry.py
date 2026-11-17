"""Phase 1 gate: eval_ops and mask_to_model_frame share one geometry.

Verifies the two documented field-of-view percentages and that a mask goes through
exactly the same resize/crop as the image it belongs to.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from dm_ahr_vit.xai.geometry import eval_resize_size, mask_to_model_frame, to_model_frame


def _resized_dims(w: int, h: int, side: int) -> tuple[int, int]:
    r = side / min(w, h)
    return int(round(w * r)), int(round(h * r))


def test_eval_resize_size():
    assert eval_resize_size(224) == 255


def test_field_of_view_fractions():
    side = eval_resize_size(224)

    rw, rh = _resized_dims(600, 450, side)
    assert (rw, rh) == (340, 255)
    window_full = 224 * 600 / rw
    frac_full = window_full * window_full / (600 * 450)
    assert abs(frac_full - 0.579) < 0.002, frac_full

    rw2, rh2 = _resized_dims(128, 128, side)
    assert (rw2, rh2) == (255, 255)
    frac_sq = (224 / rw2) ** 2
    assert abs(frac_sq - 0.772) < 0.002, frac_sq


def test_mask_and_image_align_for_full_size():
    w, h = 600, 450
    img = Image.fromarray(np.full((h, w, 3), 120, dtype=np.uint8))
    mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
    # mark a 40x40 block in the middle of the photo
    x0, y0 = w // 2 - 20, h // 2 - 20
    mask_np = np.zeros((h, w), dtype=np.uint8)
    mask_np[y0:y0 + 40, x0:x0 + 40] = 255
    mask = Image.fromarray(mask_np)

    out_img = to_model_frame(img, 224)
    out_mask = mask_to_model_frame(mask, (w, h), 224)
    assert out_img.size == (224, 224)
    assert out_mask.shape == (224, 224)
    # the marked block must still be present in the crop, near the centre
    ys, xs = np.nonzero(out_mask)
    assert len(xs) > 100, len(xs)
    assert 70 < xs.mean() < 154, xs.mean()
    assert 70 < ys.mean() < 154, ys.mean()


def test_mask_and_image_align_for_squashed_128():
    size = 128
    mask = Image.fromarray(np.zeros((size, size), dtype=np.uint8))
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[48:80, 48:80] = 255
    mask = Image.fromarray(arr)
    img = Image.fromarray(np.full((size, size, 3), 80, dtype=np.uint8))

    out_img = to_model_frame(img, 224)
    out_mask = mask_to_model_frame(mask, (size, size), 224)
    assert out_img.size == (224, 224)
    assert out_mask.shape == (224, 224)
    assert out_mask.sum() > 0
    assert out_mask.dtype == np.uint8

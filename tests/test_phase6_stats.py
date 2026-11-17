"""Local CPU tests for the Phase 6 v2 statistics module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.real_hair_validation.phase6_stats import (
    P_FLOOR_TWO_SIDED,
    quality_tables,
    safe_compare,
)


def test_p_floor_two_sided():
    # Every bootstrap difference positive: naive p would be 0.
    a = np.ones(100)
    b = np.zeros(100)
    r = safe_compare(a, b, 1.0, 0.0)
    assert r["p_raw"] >= P_FLOOR_TWO_SIDED - 1e-15
    assert r["p_raw"] == 2.0 / (100 + 1)
    assert r["n_replicates_used"] == 100


def test_p_floor_with_nan():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.zeros(4)
    r = safe_compare(a, b, 2.0, 0.0)
    assert r["n_replicates_used"] == 3
    assert np.isfinite(r["p_raw"])


def test_quality_policies_kept_separate(tmp_path):
    df = pd.DataFrame({
        "group_id": ["g1", "g1", "g2", "g2"],
        "renderer": ["standard", "hard", "standard", "hard"],
        "hairy_128px_psnr": [10.0, 11.0, 12.0, 13.0],
        "B_128px_psnr": [20.0, 21.0, 22.0, 23.0],
        "Cv2_128px_psnr": [25.0, 26.0, 27.0, 28.0],
        "hairy_128px_ssim": [0.1, 0.1, 0.2, 0.2],
        "B_128px_ssim": [0.5, 0.5, 0.6, 0.6],
        "Cv2_128px_ssim": [0.7, 0.7, 0.8, 0.8],
        "hairy_native_psnr": [9.0, 10.0, 11.0, 12.0],
        "B_native_psnr": [19.0, 20.0, 21.0, 22.0],
        "Cv2_native_psnr": [24.0, 25.0, 26.0, 27.0],
        "hairy_native_ssim": [0.05, 0.05, 0.15, 0.15],
        "B_native_ssim": [0.45, 0.45, 0.55, 0.55],
        "Cv2_native_ssim": [0.65, 0.65, 0.75, 0.75],
    })
    p = tmp_path / "image_quality_v2.csv"
    df.to_csv(p, index=False)
    out = quality_tables(p)
    assert set(out["subset"].unique()) == {"render_128px", "render_native"}
    assert set(out["endpoint"].unique()) == {"paired_psnr", "paired_ssim"}
    assert (out["p_raw"] >= P_FLOOR_TWO_SIDED - 1e-15).all() or out["p_raw"].isna().any()
    assert "p_holm" in out.columns

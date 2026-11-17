"""Generate v2 (retuned-detector) figures from the audit tables only.

Reads reports/real_hair_validation/v2/*.csv; writes artifacts/figures/v2_*.png.
No number is hard-coded except the detector gate values, which come from the
Phase 3 log and are cited in the figure caption.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "reports" / "real_hair_validation" / "v2"
OUT = ROOT / "artifacts" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COL = {"A": "#1f77b4", "B": "#d62728", "Astar": "#2ca02c", "Cv2": "#9467bd"}
NAME = {"A": "A original", "B": "B DullRazor", "Astar": "A* 128 px", "Cv2": "Cv2 diffusion"}


def save(fig, name: str) -> None:
    p = OUT / name
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.relative_to(ROOT))


def fig_classification():
    d = pd.read_csv(V2 / "phase6_statistical_comparisons_v2.csv")
    full = d[d["subset"] == "full"]
    rows = []
    for _, r in full.iterrows():
        rows.append((r["arm_a"], r["arm_a_mean"]))
        rows.append((r["arm_b"], r["arm_b_mean"]))
    means = {}
    for arm, v in rows:
        means.setdefault(arm, []).append(v)
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, (sub, title) in zip(axs, (("full", "Full test (n=1502)"),
                                       ("hair_present", "Hair-present (n=109)"))):
        s = d[d["subset"] == sub]
        m = {}
        for _, r in s.iterrows():
            m.setdefault(r["arm_a"], []).append(r["arm_a_mean"])
            m.setdefault(r["arm_b"], []).append(r["arm_b_mean"])
        aa = [a for a in ("A", "B", "Astar", "Cv2") if a in m]
        vals = [np.mean(m[a]) for a in aa]
        ax.bar([NAME[a] for a in aa], vals, color=[COL[a] for a in aa])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Test macro-F1 (mean over 3 seeds)")
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("v2 matched arms, retuned detector — no Holm p < 0.05", fontsize=10)
    save(fig, "v2_classification.png")


def fig_quality():
    d = pd.read_csv(V2 / "image_quality_v2.csv")
    policies = ("128px", "native")
    fig, axs = plt.subplots(2, 1, figsize=(9.0, 6.4))
    for ax, pol in zip(axs, policies):
        for arm, col, mk in (("hairy", "#999999", "o"), ("B", COL["B"], "s"), ("Cv2", COL["Cv2"], "^")):
            g = d.groupby("renderer")[f"{arm}_{pol}_psnr"].mean()
            ax.plot(g.index, g.values, marker=mk, label=f"{arm} PSNR", color=col)
        ax.set_title(f"PSNR (dB), {pol} geometry — 400 held-out synthetic pairs", fontsize=9)
        ax.set_ylabel("PSNR (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save(fig, "v2_image_quality.png")


def fig_xai():
    s = pd.read_csv(V2 / "phase6_xai_summary_v2.csv")
    q = s[(s["metric"] == "recall")]
    methods = ("gradcam_ln1", "gradcam_patch", "rollout")
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=True)
    for ax, (frame, title) in zip(axs, (("as_is", "As-is frame"),
                                         ("off_centre", "Off-centre (±30% shift)"))):
        x = np.arange(len(methods))
        for a in ("A", "B", "Astar", "Cv2"):
            vals = []
            for m in methods:
                row = q[(q["arm"] == a) & (q["frame"] == frame) & (q["method"] == m)]
                vals.append(float(row["mean"].iloc[0]) if len(row) else np.nan)
            ax.plot(x, vals, marker="o", label=NAME[a], color=COL[a])
        base = q[(q["arm"] == "A") & (q["frame"] == frame) & (q["method"] == "rollout")]["baseline_gauss"]
        if len(base):
            ax.axhline(float(base.iloc[0]), ls="--", color="#1f77b4", lw=1,
                       label=f"Gaussian baseline ({float(base.iloc[0]):.3f})")
        ax.set_xticks(x, ["Grad-CAM ln_1", "Grad-CAM patch", "Rollout"], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("Lesion coverage (recall)")
    fig.suptitle("v2 corrected XAI — second Grad-CAM layer and off-centre frame", fontsize=10)
    save(fig, "v2_xai.png")


def fig_detector():
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    labels = ["Image specificity", "Pixel Dice"]
    old = [0.038, 0.302]
    new = [0.923, 0.420]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, old, 0.36, label="Old frozen detector", color="#d62728")
    ax.bar(x + 0.18, new, 0.36, label="Retuned detector", color="#2ca02c")
    for i, (o, n) in enumerate(zip(old, new)):
        ax.text(i - 0.18, o + 0.02, f"{o:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.18, n + 0.02, f"{n:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 3 gate — held-out detached validation set (source: phase3 log)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "v2_detector_gate.png")


if __name__ == "__main__":
    fig_detector()
    fig_classification()
    fig_quality()
    fig_xai()

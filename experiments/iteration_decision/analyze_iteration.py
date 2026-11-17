"""Local analysis for the iteration decision: combines step-1, archived curve and step-2 tables.

Reads only validation-derived artifacts. Writes the final report, tables and curve plot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dm_ahr_vit.stats.pooled import draw_weights  # noqa: E402

LOGS = ROOT / "artifacts" / "logs"
FIG = ROOT / "artifacts" / "figures"
DEC = ROOT / "reports" / "iteration_decision"
REPORT = ROOT / "reports" / "iteration_decision.md"
REPS, BSEED = 10000, 20260101


def paired_group_ci(diff: np.ndarray, groups: np.ndarray, reps: int = REPS,
                    seed: int = BSEED) -> dict:
    g = np.asarray(groups)
    ug = sorted(set(g))
    gi = pd.Series(g).map({x: i for i, x in enumerate(ug)}).to_numpy()
    W = draw_weights(len(ug), reps, seed)
    s = np.bincount(gi, weights=np.nan_to_num(diff), minlength=len(ug))
    n = np.bincount(gi, weights=np.isfinite(diff).astype(float), minlength=len(ug))
    dist = (W @ s) / np.maximum(W @ n, 1.0)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    # p floored at 2/(reps+1), two-sided
    k = min((dist <= 0).sum(), (dist >= 0).sum())
    p = min(1.0, 2 * (k + 1) / (reps + 1))
    return {"mean": float(np.nanmean(diff)), "ci_low": float(lo), "ci_high": float(hi),
            "p_raw": float(p), "n_groups": len(ug)}


def load_archived() -> pd.DataFrame:
    rows = []
    for f in sorted((LOGS / "learning_curve").glob("it*.json")):
        j = json.loads(f.read_text())
        pp = j["partial_t0_600"]
        pn = j["pure_noise_2000"]
        rows.append({"iteration": int(f.stem[2:]), "partial_psnr": pp["psnr"],
                     "partial_ssim": pp["ssim"], "partial_gain_db": pp["gain_db"],
                     "pure_psnr": pn["psnr"], "pure_gain_db": pn["gain_db"],
                     "one_step_mae_t500": j["one_step_eps_mae"]["500"],
                     "zeroed_over_kept_hair_t200": j["condition_usage"][0]["zeroed_over_kept_hair"],
                     "shuffled_over_kept_hair_t200": j["condition_usage"][0]["shuffled_over_kept_hair"],
                     "val_pairs_hash": j["val_pairs_hash"],
                     "checkpoint_sha256": j["checkpoint_sha256"]})
    return pd.DataFrame(rows).sort_values("iteration")


def load_snapshots() -> tuple[pd.DataFrame, list]:
    rows, notes = [], []
    for f in sorted(DEC.glob("curve_*_per_pair.csv")):
        name = f.stem.replace("curve_", "").replace("_per_pair", "")
        d = pd.read_csv(f)
        diff = d["partial_gain_db"].to_numpy(dtype=float)
        ci = paired_group_ci(diff, d["group_id"].to_numpy())
        rows.append({"snapshot": name, "n_pairs": len(d), "partial_gain_db": ci["mean"],
                     "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "p_raw": ci["p_raw"],
                     "partial_psnr": float(d["partial_psnr"].mean()),
                     "partial_ssim": float(d["partial_ssim"].mean()),
                     "partial_lpips": float(d["partial_lpips"].mean()),
                     "strand_removal": float(d["strand_removal"].mean()),
                     "lesion_changed": float(d["lesion_changed"].mean()),
                     "pure_psnr": float(d["pure_psnr"].mean(skipna=True)) if d["pure_psnr"].notna().any() else float("nan"),
                     "val_pairs_hash": hashlib_hash(d)})
        s = ROOT / "reports" / "iteration_decision" / f"curve_{name}_summary.json"
        if s.exists():
            notes.append(json.loads(s.read_text()))
    return pd.DataFrame(rows), notes


def hashlib_hash(d: pd.DataFrame) -> str:
    import hashlib
    return hashlib.sha256("".join(d["pair_id"].astype(str)).encode()).hexdigest()


def md_table(df: pd.DataFrame, fmt="{:.3f}") -> str:
    out = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else fmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    DEC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    l1 = pd.read_csv(LOGS / "iteration_l1_history.csv") if (LOGS / "iteration_l1_history.csv").exists() else pd.DataFrame()
    arch = load_archived()
    snap, notes = load_snapshots()

    # gain per doubling from archived snapshots (no per-pair data -> no CI)
    doubl = []
    if len(arch) >= 2:
        a = arch.set_index("iteration")
        if 25000 in a.index and 50000 in a.index:
            g = float(a.loc[50000, "partial_gain_db"] - a.loc[25000, "partial_gain_db"])
            doubl.append({"from_k": 25, "to_k": 50, "gain_db": g,
                          "ci_low": float("nan"), "ci_high": float("nan"),
                          "per_pair_ci": "NOT AVAILABLE (checkpoints absent)"})
        if 50000 in a.index and 100000 in a.index:
            g = float(a.loc[100000, "partial_gain_db"] - a.loc[50000, "partial_gain_db"])
            doubl.append({"from_k": 50, "to_k": 100, "gain_db": g,
                          "ci_low": float("nan"), "ci_high": float("nan"),
                          "per_pair_ci": "NOT AVAILABLE"})
    dd = pd.DataFrame(doubl)

    # curve plot: archived base lineage partial gain vs iterations
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(arch["iteration"] / 1000, arch["partial_gain_db"], marker="o", label="base lineage (archived)")
    for _, r in snap.iterrows():
        ax.scatter([70], [r["partial_gain_db"]], marker="s", label=f"{r['snapshot']} (re-scored fp32)" if r["snapshot"] == "ft_v2" else None, color="#d62728", zorder=5)
    ax.set_xlabel("Generator iterations (thousands)")
    ax.set_ylabel("Partial-noise PSNR gain over hairy input (dB)")
    ax.set_title("Validation-only learning curve — t0=600, 50 DDIM steps")
    ax.grid(alpha=0.3)
    if len(snap):
        ax.legend(fontsize=8)
    p = FIG / "iteration_curve.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)

    l1.to_csv(DEC / "l1_table.csv", index=False)
    arch.to_csv(DEC / "archived_curve_table.csv", index=False)
    snap.to_csv(DEC / "snapshot_metrics_table.csv", index=False)
    dd.to_csv(DEC / "doubling_gain_table.csv", index=False)

    # ---- gates -------------------------------------------------------------
    a_ok = False
    a_evidence = "NOT VERIFIED"
    if not l1.empty:
        row = l1[l1["field"] == "last_10k_val_l1_slope_per_1k"]
        if not row.empty:
            slope = float(row["value"].iloc[0])
            a_ok = slope < 0
            a_evidence = f"last-10k val-L1 slope = {slope:+.6f} per 1k iterations"
    b_ok = False
    b_evidence = "NOT VERIFIED: no pair of saved checkpoints spans a doubling with per-pair outputs"
    if not dd.empty:
        best = dd["gain_db"].max()
        b_evidence += f"; archived 25k->50k doubling = {best:+.3f} dB but no per-pair CI (checkpoints absent)"
    c_evidence = "NOT RUN — decision already STOP; see cost log for estimate"
    d_evidence = "NOT RUN — decision already STOP; see cost log for estimate"
    verdict = "GO" if (a_ok and b_ok) else "STOP"

    L = ["# Iteration decision — further DM-AHR training", "",
         "Research prototype only. Not clinically validated and not intended for diagnosis or patient care.", "",
         "Validation data only. The test split was never read by this task. "
         "Rule committed before any run: `reports/iteration_rule.md` (commit 9255474).", "",
         "## Step 1 — L1 history (free)", "", md_table(l1) if not l1.empty else "NOT VERIFIED", "",
         "## Step 2a — archived learning curve (same val pairs, fp32 re-scoring below)", "",
         md_table(arch), "",
         "## Step 2b — re-scored snapshots (fp32, committed sampler, paired bootstrap over lesion groups)", "",
         md_table(snap) if not snap.empty else "NOT VERIFIED — no snapshots were re-scored", "",
         "## Step 2c — gain per doubling", "", md_table(dd) if not dd.empty else "NOT VERIFIED", "",
         "![Learning curve](artifacts/figures/iteration_curve.png)", "",
         "## Gates", "",
         f"- (a) val L1 still falling: **{'PASS' if a_ok else 'FAIL'}** — {a_evidence}",
         f"- (b) >=0.5 dB/doubling with CI>0: **{'PASS' if b_ok else 'FAIL'}** — {b_evidence}",
         f"- (c) ceiling within 0.01: **NOT RUN** — {c_evidence}",
         f"- (d) no recipe shortcut: **NOT RUN** — {d_evidence}", "",
         f"## Verdict: **{verdict}**", "",
         "The pre-registered rule requires all four gates for GO. At least one gate failed, "
         "so the required action is STOP. Steps 3 and 4 were not launched: they cannot change "
         "a STOP verdict and would spend the capped budget without affecting the decision. "
         "Their cost estimates are recorded in `reports/iteration_cost_log.md`.", "",
         "This is a successful outcome: the evidence does not support further generator training.",
         "No checkpoint was trained past 50k/20k-fine-tune and no test metric was consulted."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("verdict", verdict, "->", REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()

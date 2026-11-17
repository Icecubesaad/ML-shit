"""Phase 6 aggregation for the v2 (retuned-detector) rerun.

Pure CPU / numpy / pandas so it can be unit-tested locally. Reads the artifacts
written by scripts/07_eval_vit.py, scripts/58_xai_corrected.py and the Phase 5
image-quality CSV, then writes the v2 classification, statistics, image-quality
and XAI tables. All comparison tests use the archived two-level lesion-group
bootstrap in src/dm_ahr_vit/stats/pooled.py, whose two-sided p is floored at
2/(replicates+1) (one-sided 1/(replicates+1)).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from dm_ahr_vit.config import CLASSES
from dm_ahr_vit.stats.bootstrap import holm
from dm_ahr_vit.stats.pooled import (
    compare as pooled_compare,
    draw_weights,
    f1_scores,
    group_counts,
    two_level,
)

REPS = 10_000
BSEED_CLASS = 20_260_101   # configs/protocol.json bootstrap_seed
BSEED_XAI = 20_261_113     # reports/prereg_v5_client_asks.md
ARMS = ("A", "B", "Astar", "Cv2")
SEEDS = (42, 123, 2026)
SUBSETS = ("full", "hair_present")
PLANNED = (("A", "B"), ("A", "Cv2"), ("B", "Cv2"))          # A-C is Cv2 in v2
SECONDARY = (("A", "Astar"), ("Cv2", "Astar"))              # resolution controls
P_FLOOR_ONE_SIDED = 1.0 / (REPS + 1)
P_FLOOR_TWO_SIDED = 2.0 / (REPS + 1)


def _finite(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    return d[np.isfinite(d)]


def safe_compare(dist_a: np.ndarray, dist_b: np.ndarray,
                 obs_a: float, obs_b: float) -> dict:
    """pooled.compare on finite paired differences, with an explicit p floor."""
    d = _finite(np.asarray(dist_a, dtype=float) - np.asarray(dist_b, dtype=float))
    if d.size == 0:
        return {"difference": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_raw": float("nan"),
                "se_diff": float("nan"), "mdd_80pct": float("nan"),
                "p_floor_one_sided": P_FLOOR_ONE_SIDED,
                "n_replicates_used": 0}
    out = pooled_compare(dist_a, dist_b, obs_a, obs_b)
    out["p_floor_one_sided"] = P_FLOOR_ONE_SIDED
    out["n_replicates_used"] = int(d.size)
    return out


def _holm_in_families(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    df = df.copy()
    if "p_holm" not in df.columns:
        df["p_holm"] = np.nan
    for _, idx in df.groupby(keys, dropna=False).groups.items():
        p = pd.to_numeric(df.loc[idx, "p_raw"], errors="coerce").tolist()
        if all(np.isfinite(p)):
            df.loc[idx, "p_holm"] = holm(p)
    return df


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def load_prediction_frames(pred_dir: Path, seeds: tuple[int, ...] = SEEDS) -> dict:
    out = {}
    for arm in ARMS:
        for sub in SUBSETS:
            for seed in seeds:
                p = pred_dir / f"pred_{arm}_seed{seed}_{sub}.csv"
                if p.exists():
                    out[(arm, sub, seed)] = pd.read_csv(p)
    return out


def classification_tables(pred_dir: Path, log_dir: Path,
                          seeds: tuple[int, ...] = SEEDS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (tidy per-class metrics, statistical comparisons, confusion matrices)."""
    frames = load_prediction_frames(pred_dir, seeds)
    # ---- per-class metrics from the frozen eval JSONs -----------------------
    tidy_rows, cm_rows = [], []
    for arm in ARMS:
        for seed in seeds:
            # 07_eval_vit.py writes metrics JSON next to the predictions.
            p = pred_dir / f"metrics_{arm}_seed{seed}.json"
            if not p.exists():
                p = log_dir / f"metrics_{arm}_seed{seed}.json"
            if not p.exists():
                continue
            payload = json.loads(p.read_text())
            for key, m in payload["metrics"].items():
                a, s, sub = key.split("|")
                base = {"arm": a, "seed": int(s), "subset": sub, "n": m["n"],
                        "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                        "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                        "auc_macro": m.get("auc_macro", float("nan")),
                        "checkpoint_sha256": payload.get("checkpoint_sha256", ""),
                        "splits_sha256": payload.get("splits_sha256", "")}
                for c in CLASSES:
                    pc = m["per_class"][c]
                    for k in ("support", "precision", "recall", "f1", "sensitivity",
                              "specificity", "auc"):
                        base[f"{c}_{k}"] = pc[k]
                    base[f"{c}_auc_note"] = pc.get("auc_note", "")
                tidy_rows.append(base)
                cm_rows.append({"arm": a, "seed": int(s), "subset": sub,
                                "labels": ";".join(CLASSES),
                                "confusion_matrix": json.dumps(m["confusion_matrix"])})
    tidy = pd.DataFrame(tidy_rows)

    # ---- endpoint statistics: two-level lesion-group bootstrap -------------
    rows = []
    for sub in SUBSETS:
        available = [a for a in ARMS if any((a, sub, s) in frames for s in seeds)]
        if len(available) < 2:
            continue
        common = None
        for a in available:
            for seed in seeds:
                if (a, sub, seed) not in frames:
                    continue
                ids = set(frames[(a, sub, seed)]["image_id"])
                common = ids if common is None else (common & ids)
        if not common:
            continue
        common = sorted(common)
        seeds = [s for s in seeds if all((a, sub, s) in frames for a in available)]
        groups = sorted({g for a in available for s in seeds
                         for g in frames[(a, sub, s)].loc[
                             frames[(a, sub, s)]["image_id"].isin(common), "group_id"].unique()})
        W = draw_weights(len(groups), REPS, BSEED_CLASS)
        dists, obs, obs_w = {}, {}, {}
        for a in available:
            counts = []
            for s in seeds:
                sub_df = frames[(a, sub, s)]
                sub_df = sub_df[sub_df["image_id"].isin(common)]
                counts.append(group_counts(sub_df, groups))
            by_seed = f1_scores(counts, W)                  # (seeds, reps)
            dists[a] = two_level(by_seed, np.random.default_rng(BSEED_CLASS))
            obs[a] = float(np.mean([np.nanmean(by_seed[i]) for i in range(len(seeds))]))
            # Observed weighted F1 (seed-averaged) for reporting only.
            wf = []
            for s in seeds:
                d = frames[(a, sub, s)]
                d = d[d["image_id"].isin(common)]
                wf.append(f1_score(d["label"].map(CLASSES.index).to_numpy(),
                                   d["pred"].map(CLASSES.index).to_numpy(),
                                   average="weighted", zero_division=0))
            obs_w[a] = float(np.mean(wf))
        for fam, pairs in (("planned", PLANNED), ("secondary_control", SECONDARY)):
            for a, b in pairs:
                if a not in dists or b not in dists:
                    continue
                r = safe_compare(dists[a], dists[b], obs[a], obs[b])
                rows.append({"endpoint": "test_macro_f1", "subset": sub, "family": fam,
                             "comparison": f"{a}-{b}", "arm_a": a, "arm_b": b,
                             "arm_a_mean": obs[a], "arm_b_mean": obs[b],
                             "arm_a_weighted_f1": obs_w[a], "arm_b_weighted_f1": obs_w[b],
                             "n_images": len(common), "n_groups": len(groups),
                             "replicates": REPS, "seeds": ";".join(str(s) for s in seeds),
                             "test": "two-level lesion-group bootstrap (groups shared across arms; seeds resampled within arm), percentile CI, two-sided",
                             **r})
    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats = _holm_in_families(stats, ["endpoint", "subset", "family"])
    return tidy, stats, pd.DataFrame(cm_rows)


# ---------------------------------------------------------------------------
# H1 image quality (paired PSNR / SSIM, two geometry policies kept separate)
# ---------------------------------------------------------------------------

def quality_tables(qcsv: Path) -> pd.DataFrame:
    df = pd.read_csv(qcsv)
    rows = []
    policies = ("128px", "native")
    for policy in policies:
        for metric in ("psnr", "ssim"):
            cols = {"hairy": f"hairy_{policy}_{metric}", "B": f"B_{policy}_{metric}",
                    "Cv2": f"Cv2_{policy}_{metric}"}
            if any(c not in df.columns for c in cols.values()):
                continue
            sub = df.dropna(subset=[*cols.values(), "group_id"])
            groups = sorted(sub["group_id"].unique())
            W = draw_weights(len(groups), REPS, BSEED_CLASS)
            gidx = {g: i for i, g in enumerate(groups)}
            dists, obs = {}, {}
            for arm, col in cols.items():
                gi = sub["group_id"].map(gidx).to_numpy()
                v = sub[col].to_numpy(dtype=float)
                sums, cnts = np.zeros(len(groups)), np.zeros(len(groups))
                np.add.at(sums, gi, v)
                np.add.at(cnts, gi, 1.0)
                den = np.maximum(W @ cnts, 1.0)
                dists[arm] = (W @ sums) / den
                obs[arm] = float(v.mean())
            for a, b in (("B", "hairy"), ("Cv2", "hairy"), ("Cv2", "B")):
                r = safe_compare(dists[a], dists[b], obs[a], obs[b])
                rows.append({"endpoint": f"paired_{metric}", "subset": f"render_{policy}",
                             "family": "planned", "comparison": f"{a}-{b}",
                             "arm_a": a, "arm_b": b, "arm_a_mean": obs[a], "arm_b_mean": obs[b],
                             "n_images": int(len(sub)), "n_groups": len(groups),
                             "replicates": REPS, "test": "lesion-group paired bootstrap, percentile CI, two-sided",
                             **r})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = _holm_in_families(out, ["endpoint", "subset", "family"])
    return out


# ---------------------------------------------------------------------------
# XAI: second Grad-CAM layer + off-centre frame; two-level bootstrap
# ---------------------------------------------------------------------------

def xai_tables(xai_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data, seeds_of = {}, {}
    for arm in ARMS:
        for f in sorted(xai_dir.glob(f"xai_{arm}_seed*_test.csv")):
            seed = int(f.stem.split("_seed")[1].split("_")[0])
            data[(arm, seed)] = pd.read_csv(f).sort_values("image_id").reset_index(drop=True)
            seeds_of.setdefault(arm, []).append(seed)
    summary_rows, comp_rows = [], []
    if not data:
        return pd.DataFrame(summary_rows), pd.DataFrame(comp_rows)
    first = next(iter(data.values()))
    groups = sorted(first["group_id"].unique())
    gidx = {g: i for i, g in enumerate(groups)}
    gi = first["group_id"].map(gidx).to_numpy()
    W = draw_weights(len(groups), REPS, BSEED_XAI)

    def vec(arm, seed, col):
        return data[(arm, seed)][col].to_numpy(dtype=float)

    methods = ("gradcam_ln1", "gradcam_patch", "rollout")
    frames = (("", "as_is"), ("tr_", "off_centre"))
    metrics = ("recall", "pointing", "precision")
    for arm, seeds in seeds_of.items():
        for prefix, frame in frames:
            for meth in methods:
                for q in metrics:
                    col = f"{prefix}{meth}_{q}"
                    if any(col not in data[(arm, s)].columns for s in seeds):
                        continue
                    per = [np.nanmean(vec(arm, s, col)) for s in seeds]
                    gauss = np.nanmean(vec(arm, seeds[0], f"{prefix}gauss_{q}"))
                    summary_rows.append({"arm": arm, "frame": frame, "method": meth,
                                         "metric": q, "n_seeds": len(seeds),
                                         "mean": float(np.mean(per)),
                                         "seed_min": float(np.min(per)),
                                         "seed_max": float(np.max(per)),
                                         "per_seed": ";".join(f"{s}:{x:.3f}" for s, x in zip(seeds, per)),
                                         "baseline_gauss": float(gauss),
                                         "minus_gauss": float(np.mean(per) - gauss)})
    summary = pd.DataFrame(summary_rows)
    for fam, pairs in (("planned", PLANNED), ("secondary_control", SECONDARY)):
        for a, b in pairs:
            if a not in seeds_of or b not in seeds_of:
                continue
            common = sorted(set(seeds_of[a]) & set(seeds_of[b]))
            for prefix, frame in frames:
                for meth in methods:
                    for q in ("recall", "pointing"):
                        col = f"{prefix}{meth}_{q}"
                        if any(col not in data.get((a, s), pd.DataFrame()).columns for s in common):
                            continue
                        if any(col not in data.get((b, s), pd.DataFrame()).columns for s in common):
                            continue
                        va = [vec(a, s, col) for s in common]
                        vb = [vec(b, s, col) for s in common]
                        sa = np.stack([
                            (W @ np.bincount(gi, weights=np.nan_to_num(v), minlength=len(groups))) /
                            np.maximum(W @ np.bincount(gi, weights=np.isfinite(v).astype(float), minlength=len(groups)), 1.0)
                            for v in va])
                        sb = np.stack([
                            (W @ np.bincount(gi, weights=np.nan_to_num(v), minlength=len(groups))) /
                            np.maximum(W @ np.bincount(gi, weights=np.isfinite(v).astype(float), minlength=len(groups)), 1.0)
                            for v in vb])
                        da = two_level(sa, np.random.default_rng(BSEED_XAI))
                        db = two_level(sb, np.random.default_rng(BSEED_XAI))
                        obs_a = float(np.mean([np.nanmean(v) for v in va]))
                        obs_b = float(np.mean([np.nanmean(v) for v in vb]))
                        r = safe_compare(da, db, obs_a, obs_b)
                        comp_rows.append({"family": fam, "frame": frame, "method": meth,
                                          "metric": q, "comparison": f"{a}-{b}",
                                          "arm_a": a, "arm_b": b, "n_images": len(first),
                                          "n_groups": len(groups), "replicates": REPS,
                                          "test": "two-level lesion-group bootstrap, percentile CI, two-sided",
                                          **r})
    comp = pd.DataFrame(comp_rows)
    if not comp.empty:
        comp = _holm_in_families(comp, ["family", "frame", "method", "metric"])
    return summary, comp


def main(pred_dir: str, log_dir: str, qcsv: str, xai_dir: str, out_dir: str,
         seeds: tuple[int, ...] = SEEDS, suffix: str = "") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tidy, cstats, cms = classification_tables(Path(pred_dir), Path(log_dir), seeds)
    qstats = quality_tables(Path(qcsv))
    xsum, xcomp = xai_tables(Path(xai_dir))
    tidy.to_csv(out / f"phase6_classification_v2{suffix}.csv", index=False)
    cstats.to_csv(out / f"phase6_statistical_comparisons_v2{suffix}.csv", index=False)
    cms.to_csv(out / f"phase6_confusion_matrices_v2{suffix}.csv", index=False)
    qstats.to_csv(out / f"phase6_h1_quality_v2{suffix}.csv", index=False)
    xsum.to_csv(out / f"phase6_xai_summary_v2{suffix}.csv", index=False)
    xcomp.to_csv(out / f"phase6_xai_comparisons_v2{suffix}.csv", index=False)
    note = (
        "# Phase 6 notes (v2 rerun)\n\n"
        f"- Classification/XAI/quality bootstrap: {REPS:,} lesion-group replicates; "
        f"classification/quality seed {BSEED_CLASS}, XAI seed {BSEED_XAI}.\n"
        f"- Two-sided p is floored by the bootstrap at 2/(reps+1) = {P_FLOOR_TWO_SIDED:.6g}; "
        f"one-sided floor 1/(reps+1) = {P_FLOOR_ONE_SIDED:.6g}. A reported p can never be smaller.\n"
        "- Holm correction is applied separately within `planned` and `secondary_control` families.\n"
        "- Cv2 replaces C as the diffusion arm. A-A* and Cv2-A* are resolution controls, "
        "reported separately from the planned family.\n"
        "- XAI includes Grad-CAM at the last encoder block `ln_1` (primary) and the patch grid "
        "(second layer, sensitivity), each in the as-is and off-centre (shift <=30%) frames.\n"
        "- Image quality keeps the 128px and native geometry policies separate; they are never averaged.\n"
    )
    (out / f"phase6_notes_v2{suffix}.md").write_text(note)
    return {"classification_rows": int(len(tidy)), "stat_rows": int(len(cstats)),
            "quality_rows": int(len(qstats)), "xai_summary_rows": int(len(xsum)),
            "xai_comparison_rows": int(len(xcomp))}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--qcsv", required=True)
    ap.add_argument("--xai-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    sd = tuple(int(x) for x in a.seeds.split(","))
    print(json.dumps(main(a.pred_dir, a.log_dir, a.qcsv, a.xai_dir, a.out_dir, sd, a.suffix), indent=2))

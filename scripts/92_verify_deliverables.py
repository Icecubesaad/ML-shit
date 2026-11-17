"""Verify that numbers in the expanded docx and the v2 report match the source CSVs.

Writes reports/numbers_audit.md. Exits non-zero on any mismatch. This is a check of
reporting consistency, not a re-analysis of the experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "reports" / "real_hair_validation" / "v2"
DOC = ROOT / "artifacts" / "reports" / "DM-AHR-ViT_Skin_Lesion_Classification_FINAL.docx"
AUDIT = ROOT / "reports" / "real_hair_validation" / "audit_v2_integration.md"
OUT = ROOT / "reports" / "numbers_audit.md"

STATS = pd.read_csv(V2 / "phase6_statistical_comparisons_v2.csv")
QUAL = pd.read_csv(V2 / "phase6_h1_quality_v2.csv")
XSUM = pd.read_csv(V2 / "phase6_xai_summary_v2.csv")
IQ = pd.read_csv(V2 / "image_quality_v2.csv")
DH = pd.read_csv(V2 / "dermahair_psnr_ssim_v2.csv")


def doc_text():
    d = Document(str(DOC))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts.extend(c.text for c in row.cells)
    return "\n".join(parts)


def audit_text():
    return AUDIT.read_text(encoding="utf-8", errors="replace")


def mean_of(stats, subset, arm):
    vals = []
    s = stats[stats["subset"] == subset]
    for _, r in s.iterrows():
        if r["arm_a"] == arm:
            vals.append(float(r["arm_a_mean"]))
        if r["arm_b"] == arm:
            vals.append(float(r["arm_b_mean"]))
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    checks = []
    dt, at = doc_text(), audit_text()

    def check(name, needle, where="docx"):
        hay = dt if where == "docx" else at
        ok = needle in hay
        checks.append((ok, name, needle, where))

    # classification means (3 decimals) in both documents
    for sub in ("full", "hair_present"):
        for arm in ("A", "B", "Astar", "Cv2"):
            needle = f"{mean_of(STATS, sub, arm):.3f}"
            check(f"macro-F1 {sub}/{arm}", needle)
            check(f"report macro-F1 {sub}/{arm}", needle, "report")

    # key comparison values and Holm p
    for sub, comp in (("full", "A-B"), ("full", "A-Cv2"), ("full", "B-Cv2"),
                      ("hair_present", "A-Cv2")):
        r = STATS[(STATS["subset"] == sub) & (STATS["comparison"] == comp)].iloc[0]
        check(f"diff {sub}/{comp}", f"{r['difference']:.3f}")
        check(f"holm {sub}/{comp}", f"{r['p_holm']:.3f}")

    # H1 quality means
    for pol in ("128px", "native"):
        for arm in ("hairy", "B", "Cv2"):
            check(f"PSNR {pol}/{arm}", f"{IQ[f'{arm}_{pol}_psnr'].mean():.2f}")

    # XAI primary layer means
    for a in ("A", "B", "Astar", "Cv2"):
        q = XSUM[(XSUM["arm"] == a) & (XSUM["frame"] == "as_is") &
                 (XSUM["method"] == "gradcam_ln1") & (XSUM["metric"] == "recall")]
        check(f"XAI gradcam_ln1 {a}", f"{q['mean'].iloc[0]:.3f}")

    check("DERMAHAIR PSNR", f"{DH['psnr'].mean():.2f}")
    check("research-only statement", "Research prototype only")
    check("no significance claim", "no comparison reaches Holm p < 0.05")

    failed = [c for c in checks if not c[0]]
    lines = ["# Numbers audit — v2 deliverables", "",
             f"Source: v2 CSVs; targets: `artifacts/reports/DM-AHR-ViT_Skin_Lesion_Classification_FINAL.docx` "
             f"and `reports/real_hair_validation/audit_v2_integration.md`.", "",
             f"- Checks run: {len(checks)}",
             f"- Passed: {len(checks) - len(failed)}",
             f"- Failed: {len(failed)}", ""]
    if failed:
        lines += ["## Failures", ""]
        for _, name, needle, where in failed:
            lines.append(f"- {where}: {name} — expected substring `{needle}` NOT FOUND")
    lines += ["", "## Passed checks", ""]
    for _, name, needle, where in checks:
        if _:
            lines.append(f"- {where}: {name} = `{needle}`")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checks={len(checks)} failed={len(failed)} -> {OUT.relative_to(ROOT)}")
    for _, name, needle, where in failed:
        print("FAIL", where, name, needle)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

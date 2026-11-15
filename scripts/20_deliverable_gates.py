"""Verification gates V1-V14 for the final deliverables.

Every gate reports its own evidence path. The script writes
reports/deliverable_checklist.md from what it actually measured.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
ART = Path(os.environ.get("ARTIFACT_ROOT", ROOT / "artifacts"))
LOGS = ART / "logs"
MAN = DATA / "manifests"
REP = ROOT / "reports"

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
SEEDS = [42, 123, 2026]
EXPECTED_SPLIT_HASH = "fda56cf3c555e6d22f823510670e1392cd9b1b27778a50a831e68f1dd8d5be2b"

results: list[dict] = []


def gate(code: str, name: str, ok: bool, evidence: str, detail: str = "") -> None:
    results.append({"gate": code, "name": name, "status": "PASS" if ok else "FAIL",
                    "evidence": evidence, "detail": detail})
    print(f"{code}  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    out = []
    for k in range(len(CLASSES)):
        tp = float(((p == k) & (y == k)).sum())
        fp = float(((p == k) & (y != k)).sum())
        fn = float(((p != k) & (y == k)).sum())
        d = 2 * tp + fp + fn
        out.append(2 * tp / d if d > 0 else 0.0)
    return float(np.mean(out))


def v1() -> None:
    sha = (MAN / "splits.sha256").read_text().strip()
    audit = json.loads((LOGS / "data_audit.json").read_text())
    frozen = audit["frozen_splits"]["splits_sha256"]
    split = pd.read_csv(MAN / "splits.csv")
    overlap = split.groupby("group_id")["split"].nunique().max()
    ok = sha == EXPECTED_SPLIT_HASH and frozen == EXPECTED_SPLIT_HASH and overlap == 1
    gate("V1", "Split hash unchanged; zero lesion overlap across splits", ok,
         "data/manifests/splits.sha256, artifacts/logs/data_audit.json",
         f"hash={sha[:16]}… max distinct splits per group={overlap}")


def v2() -> None:
    """Recompute headline metrics from per-image predictions at 1e-6."""
    summary = pd.read_csv(LOGS / "classification_seed_summary.csv")
    worst = 0.0
    checked = 0
    for arm in ["A", "Astar", "B", "C"]:
        f1s, accs = [], []
        for seed in SEEDS:
            df = pd.read_csv(ART / "predictions"
                             / f"pred_{arm}_seed{seed}_full.csv")
            y = df["label"].map(CLASSES.index).to_numpy()
            p = df["pred"].map(CLASSES.index).to_numpy()
            f1s.append(macro_f1(y, p))
            accs.append(float((y == p).mean()))
        row = summary[(summary["arm"] == arm) & (summary["subset"] == "full")].iloc[0]
        worst = max(worst, abs(np.mean(f1s) - row["macro_f1_mean"]),
                    abs(np.mean(accs) - row["accuracy_mean"]))
        checked += 2
    gate("V2", "Headline metrics recomputed from predictions", worst < 1e-6,
         "artifacts/predictions/pred_*.csv, artifacts/logs/classification_seed_summary.csv",
         f"{checked} statistics, max abs diff {worst:.2e} (tolerance 1e-6)")


def v3() -> None:
    """Both reports must rebuild from artifacts with no manual edits."""
    before = {p.name: p.stat().st_mtime_ns for p in
              [REP / "final_research_report.md"]}
    ok = True
    detail = []
    for script in ("13_preprocessing_report.py", "14_results_report.py",
                   "15_code_guide_report.py"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                           capture_output=True, text=True, cwd=ROOT)
        ok = ok and r.returncode == 0
        detail.append(f"{script}:{'ok' if r.returncode == 0 else r.stderr[-80:]}")
    # The tracked copy must equal the generated one.
    gen = (ART / "reports" / "final_research_report.md").read_text()
    tracked = (REP / "final_research_report.md").read_text()
    gate("V3", "Reports rebuild from artifacts; tracked copy matches generated",
         ok and gen == tracked,
         "scripts/13_*, scripts/14_*, scripts/15_*",
         "; ".join(detail) + f"; report identical={gen == tracked}")


def v4() -> None:
    text = (REP / "final_research_report.md").read_text()
    docx_bits = []
    for name in ("Preprocessing_Report.docx", "Results_Report.docx"):
        p = ART / "reports" / name
        if p.exists():
            import docx
            d = docx.Document(p)
            docx_bits.append("\n".join(x.text for x in d.paragraphs))
    corpus = text + "\n" + "\n".join(docx_bits)
    stale = {
        "over-smoothing (affirmative)": corpus.count("over-smoothing"),
        "expert-traced disclaimer": corpus.count("does not state they were expert-traced"),
        "no independent hair annotations": corpus.count("No independent hair annotations"),
        "'part of that margin'": corpus.count("part of that margin"),
        "'uniform chance 0.200' as a claim": corpus.count("every arm beats uniform chance"),
    }
    ok = all(v == 0 for v in stale.values())
    gate("V4", "No stale phrases in either report", ok,
         "reports/final_research_report.md, artifacts/reports/*.docx",
         ", ".join(f"{k}={v}" for k, v in stale.items()))


def v5() -> None:
    p = LOGS / "reproduction_log.json"
    if not p.exists():
        gate("V5", "Determinism: Arm A seed 42 reproduces", False, str(p), "missing")
        return
    d = json.loads(p.read_text())
    diff = abs(d["recorded"]["best_val_macro_f1"] - d["reproduction"]["best_val_macro_f1"])
    gate("V5", "Determinism: Arm A seed 42 reproduces", diff <= d["tolerance"],
         "artifacts/logs/reproduction_log.json",
         f"diff={diff:.6f} tolerance={d['tolerance']}")


def v6() -> None:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                       capture_output=True, text=True, cwd=ROOT)
    tail = [ln for ln in r.stdout.splitlines() if "passed" in ln or "failed" in ln]
    gate("V6", "Unit tests pass", r.returncode == 0, "tests/",
         tail[-1] if tail else r.stdout[-100:])


def v7() -> None:
    docs = ["Preprocessing_Report.docx", "Results_Report.docx", "Code_Guide_Report.docx"]
    ok = all((ART / "reports" / d).exists() for d in docs)
    # PDFs are rendered with soffice and stored beside the .docx on the volume,
    # so the evidence survives a sandbox wipe instead of living in /tmp.
    pages = {}
    for d in docs:
        pdf = ART / "reports" / d.replace(".docx", ".pdf")
        if pdf.exists():
            import pymupdf
            pages[d] = pymupdf.open(pdf).page_count
    gate("V7", "All three .docx present and rendered to PDF",
         ok and len(pages) == 3, "artifacts/reports/*.docx, artifacts/reports/*.pdf",
         json.dumps(pages))


def v8() -> None:
    budget = 8.37
    reg = LOGS / "run_registry.jsonl"
    rows = [json.loads(ln) for ln in reg.read_text().splitlines() if ln.strip()]
    diag = [r for r in rows if r.get("job") == "arm_c_diagnose"]
    seconds = sum(float(r.get("seconds") or 0) for r in diag)
    # A100-40GB at the price recorded in reports/compute_plan.md.
    cost = seconds * 0.000583
    gate("V8", "Spend within budget", cost <= budget, "artifacts/logs/run_registry.jsonl",
         f"diagnosis {seconds / 60:.1f} min = ${cost:.2f}; budget ${budget:.2f}")


def v9() -> None:
    """Explanation geometry: tests pass, every corrected CSV is in the model's frame, per-arm
    mask fractions are reported."""
    import re  # noqa: PLC0415

    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "tests/test_xai_registration.py"], cwd=ROOT, capture_output=True, text=True)
    tests_ok = proc.returncode == 0
    gate("V9a", "Geometry tests (mask aligned with model input, per arm and stored size)", tests_ok,
         "tests/test_xai_registration.py", proc.stdout.strip().splitlines()[-1] if proc.stdout else "")
    files = sorted((LOGS / "xai_v2").glob("xai_*_test.csv"))
    frames = set()
    for f in files:
        frames |= set(pd.read_csv(f, usecols=["frame"])["frame"].unique())
    gate("V9b", "All corrected explanation CSVs use the model-frame geometry", bool(files) and
         frames == {"model_frame_v2"}, "artifacts/logs/xai_v2/*_test.csv", f"{len(files)} files, frames={sorted(frames)}")
    geo = LOGS / "xai_v2_analysis" / "xai_geometry_by_arm_test.csv"
    rep = REP / "xai_corrected_results.md"
    ok = geo.exists() and rep.exists()
    arms = pd.read_csv(geo)["arm"].tolist() if geo.exists() else []
    text = rep.read_text() if rep.exists() else ""
    ok = ok and all(re.search(rf"\|\s*{a}\s*\|", text) for a in arms) and len(arms) >= 4
    gate("V9c", "Mask fraction in the model's frame reported per arm", ok,
         "artifacts/logs/xai_v2_analysis/xai_geometry_by_arm_test.csv; reports/xai_corrected_results.md",
         f"arms={arms}")


def v10() -> None:
    """No stale explanation or statistics wording in the current deliverables."""
    import re  # noqa: PLC0415

    cfg = json.loads((ROOT / "configs" / "stale_phrases.json").read_text())
    texts = {}
    for rel in cfg["files"]:
        p = ROOT / rel
        if p.exists():
            texts[rel] = p.read_text()
    try:
        import docx  # noqa: PLC0415

        for p in sorted((ROOT / cfg["docx_dir"]).glob("*.docx")):
            d = docx.Document(str(p))
            body = "\n".join(x.text for x in d.paragraphs)
            for t in d.tables:
                for r in t.rows:
                    body += "\n" + " | ".join(c.text for c in r.cells)
            texts[str(p.relative_to(ROOT))] = body
    except Exception as exc:  # noqa: BLE001
        gate("V10x", "Word text readable for the stale-phrase scan", False, cfg["docx_dir"], repr(exc))
    hits = []
    paper_files = set(cfg.get("paper_files", []))
    for rel, body in texts.items():
        pats = list(cfg["patterns"]) + (list(cfg.get("paper_patterns", [])) if rel in paper_files else [])
        for pat in pats:
            for m in re.finditer(pat, body, flags=re.IGNORECASE):
                hits.append(f"{rel}: /{pat}/ -> {body[max(0, m.start() - 30):m.end() + 30]!r}")
    gate("V10", "No stale explanation or statistics phrases in current deliverables", not hits,
         "configs/stale_phrases.json", f"{len(texts)} files scanned, {len(hits)} hits" +
         ("" if not hits else ": " + " || ".join(hits[:3])))


def v11() -> None:
    """Spend for this task stays inside the owner's $60 limit."""
    p = REP / "spend_log.csv"
    ok, detail = False, "reports/spend_log.csv missing"
    if p.exists():
        df = pd.read_csv(p)
        total = float(df["est_usd"].sum())
        ok, detail = total <= 60.0, f"estimated ${total:.2f} of $60.00 over {len(df)} jobs"
    gate("V11", "Spend within the owner's $60 limit (estimate)", ok, "reports/spend_log.csv", detail)


def _iter_docx_blocks(doc):
    """Yield the document's paragraphs and tables in true document order."""
    from docx.oxml.ns import qn  # noqa: PLC0415
    from docx.table import Table  # noqa: PLC0415
    from docx.text.paragraph import Paragraph  # noqa: PLC0415

    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def v12() -> None:
    """Paper Table 8: the per-seed column lists seed:value pairs that match pooled_per_seed.csv.

    Every listed value must equal the CSV macro-F1 difference for that arm pair and seed
    (cell shows 2 dp, so tolerance 0.0051), the bracket must equal the recomputed
    '[k of N seeds favour first]', and the listed seeds must be exactly the seeds both
    arms have in the analysis — eight for every A/A*/B/Cv2 comparison shown.
    """
    import re  # noqa: PLC0415

    import docx  # noqa: PLC0415

    name = "Paper Table 8 per-seed column matches pooled_per_seed.csv"
    p = ART / "reports" / "Research_Paper.docx"
    if not p.exists():
        gate("V12", name, False, str(p), "missing")
        return
    d = docx.Document(str(p))
    t8 = [t for t in d.tables
          if len(t.rows) and len(t.rows[0].cells) >= 6
          and t.rows[0].cells[5].text.strip() == "Per-seed differences"
          and "Macro-F1 difference" in t.rows[0].cells[2].text]
    if len(t8) != 1:
        gate("V12", name, False, "artifacts/reports/Research_Paper.docx",
             f"found {len(t8)} candidate tables, expected exactly 1")
        return
    full = pd.read_csv(LOGS / "pooled_per_seed.csv")
    full = full[full["subset"] == "full"]
    problems, nrows = [], 0
    for row in t8[0].rows[1:]:
        comp = row.cells[0].text.replace("A*", "Astar").replace(" ", "")
        if "-" not in comp:
            continue
        a, b = comp.split("-", 1)
        nrows += 1
        cell = row.cells[5].text
        listed = {int(s): float(v) for s, v in re.findall(r"(\d+):([+-]\d+\.\d+)", cell)}
        m = re.search(r"\[(-?\d+) of (\d+) seeds favour first\]", cell)
        if not listed or m is None:
            problems.append(f"{comp}: unparseable per-seed cell {cell!r}")
            continue
        expected = {}
        for s in set(full[full["arm"] == a]["seed"]) & set(full[full["arm"] == b]["seed"]):
            va = full[(full["arm"] == a) & (full["seed"] == s)]["macro_f1"].iloc[0]
            vb = full[(full["arm"] == b) & (full["seed"] == s)]["macro_f1"].iloc[0]
            expected[int(s)] = float(va - vb)
        listed = {int(s): float(v) for s, v in re.findall(r"(\d+):([+-]\d+\.\d+)", cell)}
        if set(listed) != set(expected):
            problems.append(f"{comp}: listed seeds {sorted(listed)} != analysis seeds {sorted(expected)}")
        for s in sorted(set(listed) & set(expected)):
            if abs(listed[s] - expected[s]) > 0.0051:
                problems.append(f"{comp} seed {s}: cell {listed[s]:+.2f} vs csv {expected[s]:+.4f}")
        k, n = int(m.group(1)), int(m.group(2))
        if (k, n) != (sum(v > 0 for v in listed.values()), len(listed)):
            problems.append(f"{comp}: bracket [{k} of {n}] != recomputed "
                            f"[{sum(v > 0 for v in listed.values())} of {len(listed)}]")
        if len(expected) == 8 and len(listed) != 8:
            problems.append(f"{comp}: only {len(listed)} of 8 seeds listed")
    ok = nrows >= 3 and not problems
    gate("V12", name, ok,
         "artifacts/reports/Research_Paper.docx + artifacts/logs/pooled_per_seed.csv",
         f"{nrows} rows checked" + ("" if not problems else "; " + " | ".join(problems[:3])))


def v13() -> None:
    """Results Report: no legacy x1_ row outside the labelled appendix; the corrected-geometry
    comparison table appears before the appendix."""
    import docx  # noqa: PLC0415
    from docx.table import Table  # noqa: PLC0415
    from docx.text.paragraph import Paragraph  # noqa: PLC0415

    name = "Results Report keeps legacy x1_ rows only in the labelled appendix"
    p = ART / "reports" / "Results_Report.docx"
    if not p.exists():
        gate("V13", name, False, str(p), "missing")
        return
    d = docx.Document(str(p))
    appendix_seen = False
    geo_before = False
    legacy_after = 0
    leaks = []
    for block in _iter_docx_blocks(d):
        if isinstance(block, Paragraph):
            if block.text.strip().lower().startswith("appendix"):
                appendix_seen = True
            text = block.text
        else:  # type: Table
            text = "\n".join(c.text for r in block.rows for c in r.cells)
            head = " | ".join(c.text for c in block.rows[0].cells)
            if not appendix_seen and "Map" in head and "95% CI (lesions and seeds)" in head:
                geo_before = True
        if "x1_" in text:
            if appendix_seen:
                legacy_after += 1
            else:
                i = text.find("x1_")
                leaks.append(text[max(0, i - 30):i + 40])
    ok = appendix_seen and geo_before and legacy_after >= 1 and not leaks
    gate("V13", name, ok, "artifacts/reports/Results_Report.docx",
         f"appendix heading found={appendix_seen}, corrected-geometry table before it={geo_before}, "
         f"legacy rows inside appendix={legacy_after}"
         + ("" if not leaks else f"; OUTSIDE-APPENDIX x1_ at: {leaks[:2]}"))


def v14() -> None:
    """Grad-CAM ln_1 sanity: the implementation is verified (reshape exact, reference
    agreement on every planted-square input, discriminating transposition control on a
    stimulus the map follows). Per-stimulus coverage is a finding, not a gate."""
    p = LOGS / "gradcam_sanity.json"
    if not p.exists():
        gate("V14", "Grad-CAM ln_1 sanity check passes", False, str(p), "missing")
        return
    d = json.loads(p.read_text())
    c = d.get("criteria", {})
    need = ("reshape_exact", "reference_agrees", "orientation_demonstrated",
            "transposed_fails_where_following")
    missing = [k for k in need if k not in c]
    ok = (d.get("verdict") == "passes" and not missing and all(bool(c[k]) for k in need)
          and float(c.get("follow_recall_max", -1.0)) > float(c.get("chance_recall", 1.0)))
    gate("V14", "Grad-CAM ln_1 sanity check verifies the implementation", ok,
         "artifacts/logs/gradcam_sanity.json",
         f"verdict={d.get('verdict')}, follow_recall_max={c.get('follow_recall_max')} vs chance "
         f"{c.get('chance_recall')}, transposed_recall_max={c.get('transposed_recall_max')}, "
         f"reference_spearman_min={c.get('reference_spearman_min')}, reshape_exact={c.get('reshape_exact')}, "
         f"n_following={c.get('n_following')}"
         + ("" if not missing else f"; missing keys {missing}"))


def main() -> int:
    v1(); v2(); v3(); v4(); v5(); v6(); v7(); v8(); v9(); v10(); v11(); v12(); v13(); v14()
    out = ART / "reports" / "deliverable_checklist.md"
    lines = ["# Deliverable verification checklist", "",
             "Generated by `scripts/20_deliverable_gates.py`. Each gate records the evidence path",
             "it actually inspected.", "",
             "| Gate | Check | Status | Evidence | Detail |",
             "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['gate']} | {r['name']} | **{r['status']}** | "
                     f"`{r['evidence']}` | {r['detail']} |")
    passed = sum(1 for r in results if r["status"] == "PASS")
    lines += ["", f"**{passed} of {len(results)} gates pass.**"]
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out} ({passed}/{len(results)} pass)")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

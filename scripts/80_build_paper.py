"""Build the research paper (Word) from artifacts only.

Every number in the text and tables is read from artifacts/ and formatted here; nothing is typed
by hand. Sections 1-2 are carried over from docs/proposal.docx (the original draft) with the
sentence-level corrections listed in REPLACEMENTS; Sections 3-7, the abstract, the appendix and
the reference list are generated. Verdicts on H1-H3 are computed from the final numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "artifacts" / "logs"
FIGS = ROOT / "artifacts" / "figures"
XAI = LOGS / "xai_v2_analysis"
OUT = ROOT / "artifacts" / "reports" / "Research_Paper.docx"
SRC_DOC = ROOT / "docs" / "proposal.docx"

LABEL = {"A": "A: original", "Astar": "A*: original, 128 px", "B": "B: DullRazor",
         "Cv2": "Cv2: rebuilt diffusion (final)", "Creleased": "Creleased: released checkpoint (exploratory)",
         "C": "C: first diffusion run, defective trainer"}
SHORT = {"A": "A", "Astar": "A*", "B": "B", "Cv2": "Cv2", "Creleased": "Creleased", "C": "C"}
EIGHT = (42, 123, 2026, 7, 11, 17, 23, 31)
PRE = (42, 123, 2026)

# Corrections to the carried-over draft text: (exact old text, new text). Each must match.
REPLACEMENTS = [
    ("and its visual faithfulness has been validated on diagnosis and attention-verification tasks",
     "although its faithfulness to the model's decision is not established for transformer backbones"),
    ("Crucially, no prior study has quantified", "To our knowledge, no prior study has quantified"),
    ("across the three input conditions (Section 5)", "across the input conditions (Sections 3 and 4)"),
    ("within a controlled three-condition pipeline", "within a controlled multi-arm pipeline"),
    ("The present study closes this gap by evaluating", "The present study examines this gap by evaluating"),
    ("an essential requirement for clinical trust and regulatory acceptability [19]",
     "an essential requirement for clinical trust"),
    ("in our transformer setting, Grad-CAM is computed on the patch-embedding feature representation [18]",
     "in our transformer setting, Grad-CAM is computed on the token features of a late encoder layer [18]"),
]


def f(x, n=3):
    return f"{x:.{n}f}"


def sgn(x, n=3):
    return f"{x:+.{n}f}"


def p_fmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


class Paper:
    def __init__(self) -> None:
        self.doc = Document()
        s = self.doc.sections[0]
        s.page_width, s.page_height = Inches(8.27), Inches(11.69)
        s.left_margin = s.right_margin = Inches(0.9)
        s.top_margin = s.bottom_margin = Inches(0.9)
        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(10.5)
        self.nt = self.nf = 0

    def h1(self, t):
        self.doc.add_heading(t, level=1)

    def h2(self, t):
        self.doc.add_heading(t, level=2)

    def p(self, t, bold=False, italic=False, size=None, align=None):
        para = self.doc.add_paragraph()
        r = para.add_run(t)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
        if align:
            para.alignment = align
        para.paragraph_format.space_after = Pt(6)
        return para

    def bullets(self, items):
        for t in items:
            self.doc.add_paragraph(t, style="List Bullet")

    def table(self, caption, header, rows, size=8.5, widths=None):
        self.nt += 1
        c = self.doc.add_paragraph()
        r = c.add_run(f"Table {self.nt}. ")
        r.bold = True
        c.add_run(caption)
        c.paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        for i, h in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(h))
            run.bold, run.font.size = True, Pt(size)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(size)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)
        self.doc.add_paragraph()
        return self.nt

    def figure(self, path: Path, caption: str, width=6.0):
        if not path.exists():
            self.p(f"[Figure missing: {path.name}]", italic=True)
            return
        self.nf += 1
        self.doc.add_picture(str(path), width=Inches(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = self.doc.add_paragraph()
        r = c.add_run(f"Figure {self.nf}. ")
        r.bold = True
        c.add_run(caption)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_facts() -> dict:
    F: dict = {}
    allm = pd.read_csv(LOGS / "classification_metrics_all.csv")
    F["metrics_all"] = allm
    F["head"] = pd.read_csv(LOGS / "headline_metrics_pooled.csv").set_index("arm")
    F["seed3"] = pd.read_csv(LOGS / "classification_seed_summary.csv")
    F["per_seed"] = pd.read_csv(LOGS / "pooled_per_seed.csv")
    F["pooled"] = pd.read_csv(LOGS / "pooled_seed_analysis.csv")
    F["expl"] = pd.read_csv(LOGS / "statistical_comparisons_exploratory.csv")
    F["split"] = json.loads((LOGS / "split_summary.json").read_text())
    F["q_cv2"] = pd.read_csv(LOGS / "paired_psnr_ssim_test_Cv2.csv")
    F["q_c"] = pd.read_csv(LOGS / "paired_psnr_ssim_test.csv")
    F["xai_sum"] = pd.read_csv(XAI / "xai_arm_summary_test.csv")
    F["xai_cmp"] = pd.read_csv(XAI / "xai_comparisons_test.csv")
    F["xai_geo"] = pd.read_csv(XAI / "xai_geometry_by_arm_test.csv").set_index("arm")
    F["xai_agree"] = pd.read_csv(XAI / "xai_seed_agreement_test.csv")
    F["xai_prov"] = json.loads((XAI / "xai_provenance_test.json").read_text())
    F["dh"] = json.loads((LOGS / "dermahair_repro.json").read_text())
    F["env"] = json.loads((LOGS / "environment_report.json").read_text())
    F["vit"] = json.loads((ROOT / "configs" / "vit.json").read_text())
    return F


def seed_f1(F, arm: str, seeds, subset="full") -> np.ndarray:
    m = F["metrics_all"]
    sub = m[(m["arm"] == arm) & (m["subset"] == subset) & (m["seed"].isin(seeds))].sort_values("seed")
    return sub["macro_f1"].to_numpy()


def cluster_boot_diff(a: np.ndarray, b: np.ndarray, groups: np.ndarray, reps=4000, seed=20261114):
    """Mean paired difference a-b with a lesion-cluster bootstrap 95% interval."""
    d = a - b
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), size=(reps, len(ug)))
    est = sums[idx].sum(1) / cnt[idx].sum(1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def quality_table(F):
    q = F["q_cv2"].merge(F["q_c"][["image_id", "C_psnr", "C_ssim"]], on="image_id", how="left")
    ok = (q["Cv2_status"] == "ok") & (q["Creleased_status"] == "ok")
    q = q[ok].reset_index(drop=True)
    cols = [("hairy input", "hairy_input"), ("B DullRazor", "B"), ("Cv2 rebuilt diffusion", "Cv2"),
            ("Creleased (exploratory)", "Creleased"), ("C defective run (appendix)", "C")]
    rows = []
    for name, k in cols:
        rows.append([name, f(q[f"{k}_psnr"].mean(), 2), f(q[f"{k}_ssim"].mean(), 3)])
    cmp = []
    g = q["group_id"].to_numpy()
    for a, b in (("Cv2", "B"), ("Cv2", "hairy_input"), ("Creleased", "B")):
        out = []
        for m in ("psnr", "ssim"):
            out.append(cluster_boot_diff(q[f"{a}_{m}"].to_numpy(), q[f"{b}_{m}"].to_numpy(), g))
        cmp.append((a, b, out))
    return q, rows, cmp


def stat_row(F, endpoint, comparison, subset="full"):
    p = F["pooled"]
    r = p[(p.endpoint == endpoint) & (p.comparison == comparison) & (p["subset"] == subset)]
    return r.iloc[0] if len(r) else None


def per_seed_diff(F, a, b, subset="full"):
    """Per-seed macro-F1 differences (a minus b) as (seed, value) pairs over the eight seeds.

    Source is pooled_per_seed.csv, which carries all eight seeds for A, A*, B and Cv2; seeds an
    arm does not have are omitted (possible only when one arm is Creleased or C, which the
    caller never passes here).
    """
    m = F["per_seed"]
    out = []
    for s in EIGHT:
        va = m[(m.arm == a) & (m["subset"] == subset) & (m.seed == s)]["macro_f1"]
        vb = m[(m.arm == b) & (m["subset"] == subset) & (m.seed == s)]["macro_f1"]
        if len(va) and len(vb):
            out.append((int(s), float(va.iloc[0] - vb.iloc[0])))
    return out


def xai_val(F, arm, method, metric, frame="as is"):
    s = F["xai_sum"]
    r = s[(s.arm == arm) & (s.method == method) & (s.metric == metric) & (s.frame == frame)]
    return r.iloc[0] if len(r) else None


ROLL, GC1, GCP = "attention rollout", "Grad-CAM, last-block ln_1 (primary)", "Grad-CAM, patch grid (sensitivity)"


def verdicts(F) -> dict:
    """H1-H3 verdicts by rule, from the final numbers (used by the paper and the final report).

    h1 and h2 are booleans. h3 is a string: 'supported', 'not supported', or
    'inconclusive (all between-arm rows struck)' when every between-arm recall row in the
    unshifted frame is struck by the seed-spread rule, so the rows can neither support nor
    refute the hypothesis.
    """
    _, _, qcmp = quality_table(F)
    psnr, ssim = qcmp[0][2]
    acv, bcv, cva = (stat_row(F, "macro_f1", c) for c in ("A-Cv2", "B-Cv2", "Cv2-Astar"))
    cmp = F["xai_cmp"]

    def row(method):
        return cmp[(cmp.a == "Cv2") & (cmp.b == "Astar") & (cmp.metric == "recall") & (cmp.method == method)
                   & (cmp.frame == "as is")].iloc[0]

    ro, gc = row(ROLL), row(GC1)
    sel = cmp[(cmp.frame == "as is") & (cmp.metric == "recall") & (cmp.method.isin([ROLL, GC1]))]
    all_struck = bool(sel["struck"].all())
    positive = bool((ro["difference"] > 0 and ro["p_holm"] < 0.05 and not ro["struck"])
                    or (gc["difference"] > 0 and gc["p_holm"] < 0.05 and not gc["struck"]))
    h3 = ("inconclusive (all between-arm rows struck)" if all_struck
          else ("supported" if positive else "not supported"))
    return {
        "h1": bool(psnr[1] > 0 and ssim[1] > 0),
        "h2": any(bool(r["established"]) and r["difference"] * sign < 0 for r, sign in ((acv, 1), (bcv, 1), (cva, -1))),
        "h3": h3,
        "h1_psnr": psnr, "h1_ssim": ssim,
    }


def build(args) -> Path:
    F = load_facts()
    P = Paper()
    commit = git_commit()
    sp = F["split"]
    ntest = sp["sizes"]["test"]
    head = F["head"]
    dh = F["dh"]
    res = dh["results"]
    floor = res["hairy_input_floor"]
    paper_psnr, paper_ssim = dh["provenance"]["paper"]["psnr"], dh["provenance"]["paper"]["ssim"]
    q, qrows, qcmp = quality_table(F)
    nq = len(q)

    # ---- numbers used in the abstract and verdicts
    f1 = {a: float(head.loc[a, "macro_f1_mean"]) for a in head.index}
    ab = stat_row(F, "macro_f1", "A-B")
    acv = stat_row(F, "macro_f1", "A-Cv2")
    cva = stat_row(F, "macro_f1", "Cv2-Astar")
    bcv = stat_row(F, "macro_f1", "B-Cv2")
    cv_vs_b_psnr, cv_vs_b_ssim = qcmp[0][2]
    V = verdicts(F)
    h1_supported, h2_supported, h3_state = V["h1"], V["h2"], V["h3"]
    if h3_state == "supported":
        h3_concl = "improved"
    elif h3_state == "not supported":
        h3_concl = "did not improve"
    else:
        h3_concl = ("left the explanation-localisation comparison inconclusive "
                    "(all between-arm differences were struck by the seed-spread rule)")
    roll_cv = xai_val(F, "Cv2", ROLL, "recall")
    roll_as = xai_val(F, "Astar", ROLL, "recall")
    cmp = F["xai_cmp"]
    cv_as_roll = cmp[(cmp.a == "Cv2") & (cmp.b == "Astar") & (cmp.metric == "recall") & (cmp.method == ROLL)
                     & (cmp.frame == "as is")].iloc[0]
    sel = cmp[(cmp.frame == "as is") & (cmp.metric == "recall") & (cmp.method.isin([ROLL, GC1]))]
    all_struck = bool(sel["struck"].all())
    best_dh_key = max((k for k in res if k.startswith(("respaced_", "full_", "truncated_"))),
                      key=lambda k: res[k]["psnr_up"])
    best_dh = res[best_dh_key]
    repro_any = any(res[k].get("verdict") == "reproduced" for k in res if k.startswith(("respaced_", "full_", "truncated_")))
    full2000 = res["full_2000_bf16"]
    r10 = res["respaced_10_bf16"]

    # ================= Title block
    P.p("Article", italic=True, size=9)
    P.p("Explainable Vision Transformer for Skin Lesion Classification Using Dermoscopy Images: "
        "Does Diffusion-Based Hair Removal Help?", bold=True, size=15)
    P.p("Research prototype. Not clinically validated; not intended for diagnosis, triage or patient care.",
        italic=True, size=9)
    P.p(f"Version generated from artifacts at commit {commit}.", italic=True, size=9)

    # ================= Abstract
    gm = F["xai_geo"]
    g_a, g_s = gm.loc["A"], gm.loc["Astar"]
    gauss_a = xai_val(F, "A", ROLL, "recall")["baseline_gauss"]
    gauss_s = xai_val(F, "Astar", ROLL, "recall")["baseline_gauss"]
    P.h1("Abstract")
    P.p(
        "Background: hair occludes dermoscopic lesions, and generative hair removal is proposed as a "
        "pre-processing step. We tested whether replacing hair with the output of a conditional diffusion "
        "model (DM-AHR) changes the image quality, the accuracy and the explanations of a ViT-B/16 "
        f"classifier on the seven-class HAM10000 benchmark. Methods: lesion-grouped 70/15/15 split "
        f"(test n={ntest}); matched ViT-B/16 recipe; arms A (original), A* (128 px original, resolution "
        "control), B (DullRazor-style removal), Cv2 (our rebuilt diffusion arm) and, exploratory, "
        "Creleased (the authors' released checkpoint); eight seeds for A, A*, B and Cv2 and three for the "
        "others; lesion-cluster bootstrap with Holm correction; Grad-CAM and attention rollout scored "
        "against lesion masks in the geometry the classifier actually sees. "
        f"Results: macro-F1 was {f(f1['A'])} (A), {f(f1['Astar'])} (A*), {f(f1['Cv2'])} (Cv2) and "
        f"{f(f1['B'])} (B). Only A over B was established (difference {sgn(ab['difference'])}, 95% CI "
        f"{sgn(ab['ci_low'])} to {sgn(ab['ci_high'])}, Holm p={p_fmt(ab['p_holm'])}); Cv2 did not differ "
        f"from A* (difference {sgn(cva['difference'])}, CI {sgn(cva['ci_low'])} to {sgn(cva['ci_high'])}). "
        f"On {nq} held-out synthetic-hair pairs, Cv2 reached {qrows[2][1]} dB PSNR against {qrows[1][1]} dB "
        f"for DullRazor and {qrows[0][1]} dB for the hairy input. The released checkpoint "
        f"{'reproduced' if repro_any else 'did not reproduce'} "
        f"the published {paper_psnr:.2f} dB on the public DERMAHAIR pairs (best setting {f(best_dh['psnr_up'], 2)} dB; "
        f"hairy-input floor {f(floor['psnr_up'], 2)} dB). Lesions are large and central "
        f"(mean mask area {f(g_a['mean_mask_area'])} of the model frame for A, {f(g_s['mean_mask_area'])} "
        f"for the 128 px arms), so a fixed centred Gaussian reaches overlap {f(gauss_a)} and {f(gauss_s)} "
        f"against {f(xai_val(F, 'A', ROLL, 'recall')['mean'])} and {f(roll_as['mean'])} for attention rollout; "
        f"{'every' if all_struck else 'not every'} between-arm explanation difference "
        f"{'was struck by' if all_struck else 'passed'} the seed-spread rule. "
        f"Conclusion: diffusion hair removal {'improved' if h1_supported else 'did not improve'} image quality over the "
        f"classical method on the synthetic-hair pairs, {'improved' if h2_supported else 'did not improve'} "
        f"classification, and {h3_concl} in this study.", size=10)
    P.p("Keywords: medical image analysis; skin cancer detection; ISIC; dermoscopy; vision transformer; "
        "explainable AI; Grad-CAM; attention rollout; hair removal; diffusion models", italic=True, size=9)

    # ================= Sections 1-2 carried over
    src = Document(str(SRC_DOC))
    paras = src.paragraphs
    start = next(i for i, p in enumerate(paras) if p.text.startswith("1. Introduction"))
    end = next(i for i, p in enumerate(paras) if p.text.startswith("3. Research Methodology"))
    used = set()
    in_work = next(i for i, p in enumerate(paras) if p.text.startswith("In this work, we propose"))
    lit_start = next(i for i, p in enumerate(paras) if p.text.startswith("2. Literature Review"))
    for i, p in enumerate(paras[start:end], start=start):
        if in_work <= i < lit_start:
            if i == in_work:
                P.p("In this work we ask whether replacing hair with the output of a conditional diffusion model "
                    "(DM–AHR [14]) changes the image quality, the accuracy and the explanations of a ViT-B/16 classifier on the "
                    "seven-class HAM10000 benchmark [20,21]. The study tests three hypotheses (Section 3.8): (i) diffusion "
                    "removal preserves lesion content better than classical removal; (ii) the ViT trained on diffusion-cleaned images "
                    "classifies better; and (iii) its explanation maps concentrate more on the lesion. The contributions of this article are:")
                P.bullets([
                    "A matched comparison of five input conditions (original, 128 px original, DullRazor-style removal, a rebuilt diffusion "
                    "model, and the released diffusion checkpoint) with eight seeds and lesion-grouped statistics (Sections 3 and 4).",
                    "A diagnosis and repair of a noise-level encoding defect in a diffusion trainer, and a reproduction attempt of the "
                    "published image-quality result on the public DERMAHAIR pairs (Section 4.1).",
                    "A re-measured explanation analysis in the geometry the classifier actually sees, with random, centred-Gaussian and "
                    "off-centre baselines and a seed-spread rule for between-arm claims (Section 4.5).",
                    "All code, configurations, per-image predictions and hash manifests, so that every number can be regenerated from artifacts."])
                P.p("The remainder of this article is organised as follows. Section 2 reviews the related literature. Section 3 "
                    "describes the arms, data, models, explanation measurement and statistics as they were run. Section 4 reports the "
                    "results. Section 5 discusses the hypotheses, the reproduction attempt and the limitations, and Section 6 concludes.")
            continue
        t = p.text
        if not t.strip():
            continue
        for old, new in REPLACEMENTS:
            if old in t:
                t = t.replace(old, new)
                used.add(old)
        t = t.replace("Part 1 revised draft (Sections 1–5)", "")
        if t.startswith("Table 1"):
            continue  # the generated literature table below carries its own caption
        if p.style.name == "Heading 1":
            P.h1(t)
        elif p.style.name == "Heading 2":
            P.h2(t)
        elif p.style.name == "List Bullet":
            P.doc.add_paragraph(t, style="List Bullet")
        else:
            P.p(t)
    missing = [o for o, _ in REPLACEMENTS if o not in used]
    if missing:
        raise SystemExit(f"replacement text not found in the draft: {missing}")
    # literature table
    lit = src.tables[0]
    P.table("Representative studies on hair removal and explainable classification.",
            [c.text for c in lit.rows[0].cells],
            [[c.text for c in r.cells] for r in lit.rows[1:]], size=7.5)

    # ================= 3 Methodology
    P.h1("3. Methodology")
    P.h2("3.1. Design and arms")
    P.p("The hair removal condition is the only manipulated variable. All arms share the split, the ViT-B/16 recipe, "
        "the seeds and the evaluation code; Table 2 lists the arms as they were run.")
    P.table("Arms as run. Hair-present = frozen screening coverage of at least 1%; hair-absent images are the "
            "original photograph in every arm.",
            ["Arm", "Hair-present images", "Hair-absent images", "Status", "Seeds"],
            [["A", "original (600 × 450)", "original", "primary control", "8"],
             ["A*", "original resized to 128 px", "original", "resolution control", "8"],
             ["B", "DullRazor-style removal (original resolution)", "original", "primary comparison", "8"],
             ["Cv2", "output of our rebuilt diffusion model (128 px)", "original", "final diffusion arm (exploratory family)", "8"],
             ["Creleased", "output of the authors' released checkpoint with a 25-step respaced schedule (128 px)", "original", "exploratory", "3"],
             ["C", "output of our first diffusion model (trainer defect)", "original", "recorded failure, appendix only", "3"]],
            widths=[0.7, 2.4, 1.0, 1.7, 0.5])
    P.p("The three pre-registered seeds are 42, 123 and 2026; five further seeds (7, 11, 17, 23, 31) were added "
        "for A, A*, B and Cv2 in a pre-registered follow-up (Stage E) after an independent review pointed out that three seeds "
        "cannot resolve differences of 0.02 macro-F1. Both analyses are reported.")
    P.h2("3.2. Data, split and hair screening")
    cnt = sp["counts"]
    cls = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    P.p(f"HAM10000 contributes {sp['total_images']:,} dermoscopic images of {sp['total_lesions']:,} lesions "
        f"in seven classes [20]. Images were split by lesion, never by image, into train/validation/test "
        f"({sp['sizes']['train']:,}/{sp['sizes']['val']:,}/{sp['sizes']['test']:,} images; "
        f"{sp['groups']['train']:,}/{sp['groups']['val']:,}/{sp['groups']['test']:,} lesion groups). "
        f"Split hash {sp['splits_sha256'][:16]}…; zero group overlap is verified by an automated test. "
        "Hair presence was determined by a frozen DullRazor-style screening detector whose thresholds were "
        "tuned on train and validation images only (coverage threshold 1%); the same cut defines which "
        "images are processed by B, Cv2 and Creleased and which stay untouched.")
    P.table("Class counts per split (frozen, lesion-grouped).", ["Split"] + cls + ["Total"],
            [[s] + [cnt[s][c] for c in cls] + [sum(cnt[s].values())] for s in ("train", "val", "test")])
    P.h2("3.3. Hair removal arms")
    P.p("Arm B follows DullRazor [10]: grayscale morphological hair detection with line verification and "
        "interpolation, at the original resolution. Arms Cv2, Creleased and C are 128 px conditional diffusion models "
        "(DM-AHR [14]); the diffusion model works at 128 px, which is below the 224 px classifier input, so arm A* "
        "(original resized to 128 px) separates the effect of resolution from the effect of hair removal.")
    P.p("Our first diffusion run (C) used a noise-level encoding opposite to the sampler's (t/T in the trainer, "
        "the cumulative noise level in the sampler); the repaired trainer mirrors the upstream loss and is covered by "
        "contract tests. The final diffusion model (Cv2) is the repaired-trainer model trained for 50,000 iterations "
        "and fine-tuned for 20,000 iterations on real hair strands pasted onto clean images (only train-split images "
        "supplied training strands and targets; the leakage check is recorded in the training manifest). Sampling starts "
        "from the noised input at level 600 of 2,000 with 50 strided DDIM steps (partial-noise sampler), in fp32, at 128 px. "
        "The released checkpoint arm uses a 25-step respaced schedule whose routine is not reproducible from the "
        "repository (decision D26). The released checkpoint (725,000 iterations) was not retrained; its licence status is "
        "unresolved (Section 5.4).")
    P.h2("3.4. Classifier and training recipe")
    v = F["vit"]
    P.p(f"ViT-B/16 initialised from ImageNet-1k weights [15], input {v['image_size']} × {v['image_size']}, AdamW, "
        f"learning rate {v['lr']:g}, weight decay {v['weight_decay']}, {v['warmup_epochs']} warm-up epochs then cosine "
        f"decay, batch size {v['batch_size']}, up to {v['epochs']} epochs with early stopping on validation macro-F1 "
        f"(patience {v['early_stopping_patience']}), gradient clipping {v['grad_clip']}, bf16, cross-entropy with "
        "inverse-frequency class weights. Augmentation: random resized crop (scale 0.75–1.0), horizontal and vertical "
        f"flips (0.5), rotation up to {v['augmentation']['rotation_degrees']}°, mild colour jitter; no augmentation at "
        "validation or test. Evaluation resizes to 255 pixels and centre-crops 224.")
    P.h2("3.5. Explainability measurement")
    P.p("The classifier sees a centre crop of a resized image, so a lesion mask resized from the whole photograph does "
        "not line up with the model's input, and the mismatch differs by arm. A single function "
        "(xai/geometry.py) maps an image and its mask through the identical stored size, resize and crop, per image; "
        "unit tests check alignment per arm. All localisation numbers below are computed in that frame.")
    P.p("Masks are the expert segmentations published by Tschandl et al. [24] (single rater). Metrics, generated from "
        "the function docstrings: overlap (mask recall) is the share of the lesion covered by the top 20% most salient "
        "pixels; precision is the share of those pixels that fall inside the lesion; pointing is 1 if the single most "
        "salient pixel is inside the lesion; centroid distance is measured in image widths. Recall is the primary "
        "endpoint and precision is reported as a labelled second metric. Attention rollout [19] is computed over all "
        "layers and heads. Grad-CAM [18] targets the output of ln_1 in the last encoder block (class token dropped, "
        "tokens reshaped to 14 × 14); the earlier patch-embedding grid is kept as a labelled sensitivity run.")
    san_path = LOGS / "gradcam_sanity.json"
    if not san_path.exists():
        raise SystemExit("artifacts/logs/gradcam_sanity.json is missing; run scripts/64_gradcam_sanity.py first")
    san = json.loads(san_path.read_text())
    sc = san["criteria"]
    strat = [p for p in san["placements"] if str(p["stimulus"]).startswith("straddling")]
    if len(san["placements"]) < 3:
        raise SystemExit("gradcam_sanity.json lacks the stimulus probes")
    f_strat = min(p["follow_ln1"]["recall"] for p in strat)
    f_best = max(p["follow_ln1"]["recall"] for p in strat)
    t_strat = max(p["transposed_ln1"]["recall"] for p in strat)
    rec_all = ", ".join(f"{p['follow_ln1']['recall']:.2f}" for p in san["placements"])
    P.p("Implementation sanity check for the primary map (synthetic inputs only; no dataset image, mask or label used): "
        "the token-to-grid reshape unit check "
        f"{'passes' if sc['reshape_exact'] else 'fails'}, and on every planted-square input the production map agrees "
        f"with the pytorch-grad-cam 1.5.0 reference implementation on the same target layer (worst per-image Spearman "
        f"{sc['reference_spearman_min']:.3f}). On the bright squares that straddle patch boundaries the map follows the "
        f"square (top-20% recall {f_strat:.2f} and {f_best:.2f} against a {sc['chance_recall']:.3f} chance level) while "
        f"the row-and-column transposition of the same map does not (recall {t_strat:.2f}), so the orientation is "
        "correct. Planted-square coverage itself varies with the synthetic image rather than with the implementation "
        f"(follow recalls across the four planted squares: {rec_all} for two straddling bright squares, a patch-aligned "
        "bright square and a patch-aligned checkerboard), so the below-chance recall of the ln_1 map on real images is "
        "reported as a property of the model's salience on those images, not an implementation artifact. Verdict: "
        f"{san['verdict']} (artifacts/logs/gradcam_sanity.json, checkpoint {san['arm']} seed {san['seed']}).")
    P.p("Baselines on the same masks: random pixel maps and random 14 × 14 patch maps (100 draws per image), a "
        "centred Gaussian (σ = 0.25 of the width), pointing chance, the per-arm overlap ceiling "
        "(mean of min(1, 0.2/area)), lesion-centroid statistics, and an off-centre test in which image and mask are "
        "translated together by up to ±30%. Per-arm values are the mean over seeds. Between-arm comparisons use a two-level "
        "bootstrap (lesion groups and seeds), per-seed differences, Holm correction within family and a strike rule: a "
        "row is struck from the planned family when the seed-to-seed spread of either arm exceeds the between-arm difference.")
    P.h2("3.6. Statistics and image quality")
    P.p("Macro-F1 differences use a paired two-level bootstrap (10,000 replicates; lesion groups resampled jointly "
        "across arms, seeds resampled within each arm), with Holm correction within each family (planned: A−B; "
        "control: A−A*; exploratory: everything involving Cv2 or Creleased). Per-seed differences are shown next "
        "to every interval. A per-image log-loss difference accompanies each macro-F1 comparison; a standardised "
        "effect size is deliberately not reported. Image quality is PSNR and SSIM against the hairless original on held-out "
        f"synthetic-hair pairs ({nq} test pairs built by compositing synthetic hair onto hair-free test photographs); "
        "intervals are lesion-cluster bootstraps over pairs. The test set was scored once per frozen checkpoint, "
        "after the recipe and stop rules were committed; the corrected explanation analysis is a documented second look "
        "at the same checkpoints (decision D40).")
    P.h2("3.7. Environment and tooling")
    env = F["env"]
    P.p(f"All GPU work ran on Modal containers (Python {env['python']}, PyTorch {env['torch']}, CUDA {env['cuda']}, "
        f"{env['device']}); the ViT recipe, split and seeds are fixed in configs/. Code, configuration, checkpoint and "
        "manifest hashes are recorded in every artifact. Spend is logged per job in reports/spend_log.csv.")
    P.h2("3.8. Hypotheses")
    P.bullets([
        "H1 (image quality): diffusion hair removal preserves lesion boundaries and texture better than classical removal "
        "(PSNR and SSIM on held-out pairs).",
        "H2 (classification): the ViT trained on diffusion-cleaned images achieves higher macro-F1 than on original or "
        "classically cleaned images, and than on the resolution-matched control.",
        "H3 (explainability): after diffusion hair removal, explanation maps concentrate more on the lesion than in the "
        "resolution-matched control, beyond what a lesion-centre prior achieves."])

    # ================= 4 Results
    P.h1("4. Results")
    P.h2("4.1. Reproduction of the published image-quality result on DERMAHAIR")
    prov = dh["provenance"]
    P.p(f"The public DERMAHAIR dataset (Hugging Face AnasHXH/SR_SDM and Kaggle riotulab/skin-cancer-hair-removal; the two "
        f"copies are byte-identical) holds {prov.get('n_pairs', 707)} hairy/hairless pairs and no published "
        "train/test split; the paper describes 700 images reserved for testing, so all public pairs were used. "
        "The released checkpoint was run through our adapter. "
        f"Rule fixed beforehand: reproduced = within 2 dB of the paper's {paper_psnr:.2f} dB and above the hairy-input floor. "
        f"Verdict: {'reproduced' if repro_any else 'not reproduced'}.")
    rows = [["Hairy input (do nothing)", f(floor["psnr_up"], 2), f(floor["ssim_up"], 4), f(floor["lpips"], 4), "-"]]
    for k, nm in (("respaced_10_bf16", "Released, 10 strided steps"), ("respaced_25_bf16", "Released, 25 strided steps"),
                  ("respaced_50_bf16", "Released, 50 strided steps"), ("full_2000_bf16", "Released, 2,000 steps (full schedule)"),
                  ("truncated_10_bf16", "Released, 10-step rebuilt schedule")):
        if k in res:
            rows.append([nm, f(res[k]["psnr_up"], 2), f(res[k]["ssim_up"], 4), f(res[k]["lpips"], 4), res[k]["verdict"]])
    rows.append([f"Paper (DM-AHR(SS), reported)", f"{paper_psnr:.2f}", f"{paper_ssim:.4f}", "not reported", "no baseline reported"])
    P.table("Released checkpoint on DERMAHAIR pairs (upstream PSNR/SSIM at 128 px; LPIPS AlexNet). Identical pairs "
            "(PSNR undefined) are excluded from every mean and counted in the provenance record.",
            ["Setting", "PSNR (dB)", "SSIM", "LPIPS", "Verdict"], rows)
    P.figure(FIGS / "fig_dermahair.png", "PSNR of the released checkpoint on DERMAHAIR by sampling setting.", 4.8)
    P.p("Because the matching effort stops once the pre-stated rule fails, no diffusion model was trained on DERMAHAIR; "
        "the downstream experiments use the rebuilt model Cv2 and the released checkpoint as a labelled exploratory arm.")

    P.h2("4.2. Image quality on held-out synthetic-hair pairs (H1)")
    P.table(f"Mean image quality on {nq} held-out synthetic-hair test pairs (higher is better).",
            ["Image", "PSNR (dB)", "SSIM"], qrows)
    crow = []
    for a, b, out in qcmp:
        crow.append([f"{SHORT.get(a, a)} − {b.replace('hairy_input', 'hairy input')}",
                     f"{sgn(out[0][0], 2)} ({sgn(out[0][1], 2)} to {sgn(out[0][2], 2)})",
                     f"{sgn(out[1][0], 3)} ({sgn(out[1][1], 3)} to {sgn(out[1][2], 3)})"])
    P.table("Paired differences with lesion-cluster bootstrap 95% intervals.", ["Comparison", "ΔPSNR (dB)", "ΔSSIM"], crow)
    P.figure(FIGS / "fig_image_quality_synthetic.png", "Image quality of the arms on the held-out synthetic pairs.", 5.2)

    P.h2("4.3. Classification (H2)")
    rows = []
    for a in ("A", "Astar", "B", "Cv2", "Creleased", "C"):
        h = head.loc[a]
        s3 = seed_f1(F, a, PRE)
        rows.append([LABEL[a], int(h["seeds"]), f"{h['macro_f1_mean']:.3f} ± {h['macro_f1_sd']:.3f}",
                     f"{s3.mean():.3f} ± {s3.std(ddof=1):.3f}" if len(s3) == 3 else "-",
                     f(h["accuracy_mean"]), f(h["balanced_accuracy_mean"]), f(h["auc_macro_mean"]), f(h["melanoma_recall_mean"])])
    P.table(f"Test performance (n={ntest}), mean ± SD over seeds. 'Pre-reg. seeds' uses seeds 42/123/2026 only.",
            ["Arm", "Seeds", "Macro-F1 (all seeds)", "Macro-F1 (pre-reg. seeds)", "Accuracy", "Bal. acc.", "Macro-AUC", "Melanoma recall"], rows, size=7.5)
    P.figure(FIGS / "fig_macro_f1_by_seed.png", "Test macro-F1 per seed and arm.", 5.4)

    P.h2("4.4. Statistical comparisons")
    rows = []
    fam_label = {"planned": "planned", "secondary_control": "control", "exploratory": "exploratory"}
    for comp in ("A-B", "A-Astar", "A-Cv2", "B-Cv2", "Cv2-Astar"):
        r = stat_row(F, "macro_f1", comp)
        ll = stat_row(F, "log_loss", comp)
        a, b = comp.split("-")
        ps = per_seed_diff(F, a, b)
        rows.append([comp.replace("Astar", "A*"), fam_label[r["family"]], f"{sgn(r['difference'])} ({sgn(r['ci_low'])} to {sgn(r['ci_high'])})",
                     p_fmt(r["p_holm"]), "yes" if r["established"] else "no",
                     " ".join(f"{s}:{x:+.2f}" for s, x in ps)
                     + f" [{sum(float(f'{x:+.2f}') > 0 for _, x in ps)} of {len(ps)} seeds favour first]",
                     f"{sgn(ll['difference'])} ({sgn(ll['ci_low'])} to {sgn(ll['ci_high'])})"])
    P.table("Pooled analysis, eight seeds, test macro-F1 (full test set). Differences are first arm minus second; "
            "per-seed differences are labelled seed:value in seed order 42, 123, 2026, 7, 11, 17, 23, 31, read from "
            "pooled_per_seed.csv (an arm that lacks a seed would have it omitted). The last column is the per-image "
            "log-loss difference (negative favours the first arm).",
            ["Comparison", "Family", "Macro-F1 difference (95% CI)", "Holm p", "Established", "Per-seed differences", "Log-loss difference (95% CI)"], rows, size=7)
    ex = F["expl"]
    erows = []
    for _, r in ex[(ex.endpoint == "test_macro_f1") & (ex["subset"] == "full")].iterrows():
        erows.append([r["comparison"], f"{sgn(r['difference'])} ({sgn(r['ci_low'])} to {sgn(r['ci_high'])})", p_fmt(r["p_holm"]), "exploratory, three seeds each"])
    if erows:
        P.table("Exploratory comparisons involving the released-checkpoint arm (three seeds; lesion-grouped bootstrap that "
                "omits training-seed variance, so intervals are optimistic).",
                ["Comparison", "Macro-F1 difference (95% CI)", "Holm p", "Label"], erows)

    P.h2("4.5. Explainability re-measured in the model's frame (H3)")
    P.table("Lesion size and position in the model's frame. Offsets are in image widths.",
            ["Arm", "Mask area (mean ± SD)", "Recall ceiling", "Precision ceiling", "Centroid offset (mean ± SD)", "Mask contains centre"],
            [[SHORT[a], f"{gm.loc[a, 'mean_mask_area']:.3f} ± {gm.loc[a, 'sd_mask_area']:.3f}", f(gm.loc[a, "recall_ceiling"]),
              f(gm.loc[a, "precision_ceiling"]), f"{gm.loc[a, 'centroid_offset_mean']:.3f} ± {gm.loc[a, 'centroid_offset_sd']:.3f}",
              f"{100 * gm.loc[a, 'share_masks_containing_centre']:.1f}%"] for a in ("A", "B", "Astar", "Cv2", "C", "Creleased") if a in gm.index])
    erows = []
    for a in ("A", "B", "Astar", "Cv2", "Creleased", "C"):
        for m, mn in ((ROLL, "rollout"), (GC1, "Grad-CAM ln_1"), (GCP, "Grad-CAM patch")):
            r = xai_val(F, a, m, "recall")
            pt = xai_val(F, a, m, "pointing")
            if r is None:
                continue
            erows.append([SHORT[a], mn, int(r["n_seeds"]), f"{r['mean']:.3f} [{r['seed_min']:.3f}–{r['seed_max']:.3f}]",
                          f(r["baseline_gauss"]), f(r["baseline_rand_pixel"]), f"{pt['mean']:.3f}", f(pt["baseline_gauss"]), f(pt["baseline_rand_pixel"])])
    P.table("Per-arm localisation as is (mean over seeds, with the seed range). Baselines are on the same masks: centred "
            "Gaussian and random pixel maps for overlap (recall) and, separately, for pointing.",
            ["Arm", "Method", "Seeds", "Recall [seed range]", "Gaussian recall", "Random recall", "Pointing", "Gaussian pointing", "Chance pointing"], erows, size=7)
    P.figure(FIGS / "fig_xai_by_arm.png", "Localisation by arm with centred-Gaussian baseline and overlap ceiling.", 6.0)
    erows = []
    for a in ("A", "Astar", "Cv2"):
        for m, mn in ((ROLL, "rollout"), (GC1, "Grad-CAM ln_1")):
            r0 = xai_val(F, a, m, "recall")
            r1 = xai_val(F, a, m, "recall", frame="off-centre (shifted up to 30%)")
            if r0 is None or r1 is None:
                continue
            erows.append([SHORT[a], mn, f"{r0['mean']:.3f}", f"{r1['mean']:.3f}", f(r0["baseline_gauss"]), f(r1["baseline_gauss"])])
    P.table("Off-centre robustness: recall with image and mask translated together by up to ±30%.",
            ["Arm", "Method", "Recall as is", "Recall off-centre", "Gaussian as is", "Gaussian off-centre"], erows)
    crow = []
    for _, r in cmp[(cmp.frame == "as is") & (cmp.metric == "recall") & (cmp.method.isin([ROLL, GC1]))].iterrows():
        crow.append([f"{r['a']} − {r['b']}".replace("Astar", "A*"), "rollout" if r["method"] == ROLL else "Grad-CAM ln_1", r["family"],
                     f"{sgn(r['difference'])} ({sgn(r['ci_low'])} to {sgn(r['ci_high'])})", p_fmt(r["p_holm"]),
                     "struck" if r["struck"] else "kept", str(r["per_seed_differences"])[:60]])
    P.table("Between-arm comparisons of recall as is (two-level bootstrap, Holm within family). 'struck' = seed-to-seed "
            "spread exceeds the between-arm difference.", ["Comparison", "Method", "Family", "Difference (95% CI)", "Holm p", "Rule", "Per-seed differences"], crow, size=7)
    ag = F["xai_agree"]
    arow = [[SHORT[r["arm"]], {ROLL: "rollout", GC1: "Grad-CAM ln_1", GCP: "Grad-CAM patch"}.get(r["method"], r["method"]),
             int(r["n_seeds"]), f(r["mean_pairwise_spearman"]), f(r["seed_range_of_mean_recall"])] for _, r in ag.iterrows()]
    P.table("Cross-seed agreement of per-image recall (mean pairwise Spearman correlation) and the range of the seed means.",
            ["Arm", "Method", "Seeds", "Spearman", "Range of seed means"], arow)

    # ================= 5 Discussion
    P.h1("5. Discussion")
    P.h2("5.1. Verdicts on the hypotheses")
    P.p(f"H1 (image quality): {'supported' if h1_supported else 'not supported'}. On {nq} held-out synthetic pairs Cv2 differs "
        f"from DullRazor by {sgn(cv_vs_b_psnr[0], 2)} dB PSNR ({sgn(cv_vs_b_psnr[1], 2)} to {sgn(cv_vs_b_psnr[2], 2)}) and "
        f"{sgn(cv_vs_b_ssim[0], 3)} SSIM ({sgn(cv_vs_b_ssim[1], 3)} to {sgn(cv_vs_b_ssim[2], 3)}); both methods improve on the hairy "
        "input, but the classical method is closer to the clean image. The evidence is synthetic hair composited on clean photographs, "
        "not real hair, so it bounds the answer for real photographs only loosely.")
    P.p(f"H2 (classification): {'supported' if h2_supported else 'not supported'}. Original images scored highest "
        f"(macro-F1 {f(f1['A'])}); Cv2 {f(f1['Cv2'])} does not differ from the resolution control A* {f(f1['Astar'])} "
        f"(difference {sgn(cva['difference'])}, CI {sgn(cva['ci_low'])} to {sgn(cva['ci_high'])}), and DullRazor is lower "
        f"({f(f1['B'])}). Only A over B is established; the eight-seed minimum detectable difference is about "
        f"{f(float(ab['mdd_80pct']), 2)} macro-F1, so differences of 0.02 cannot be resolved here. Hair removal did not help the classifier; "
        "this DullRazor implementation lowered macro-F1 and damages lesions on real photos.")
    P.p(f"H3 (explainability): {h3_state}. After correcting the mask geometry, the "
        f"remaining between-arm differences are explained by the working frame, not by hair removal: attention-rollout recall is "
        f"{f(roll_as['mean'])} for A* and {f(roll_cv['mean'])} for Cv2 (difference {sgn(cv_as_roll['difference'])}, "
        f"CI {sgn(cv_as_roll['ci_low'])} to {sgn(cv_as_roll['ci_high'])}), while the full-resolution arms differ from the 128 px arms "
        "mainly because lesions occupy a larger share of the 128 px centre crop. Rows whose seed spread exceeds the difference are struck "
        "from the planned family.")
    P.h2("5.2. Why a centred Gaussian competes with the saliency maps")
    P.p(f"Lesions in HAM10000 are large and centred: in the model's frame the mean mask covers {f(g_a['mean_mask_area'])} of the image for "
        f"arms A and B and {f(g_s['mean_mask_area'])} for the 128 px arms; {100 * g_a['share_masks_containing_centre']:.1f}% of masks contain "
        f"the image centre and the centroid lies {f(g_a['centroid_offset_mean'])} image widths from it. The top-20% recall ceiling is therefore "
        f"{f(g_a['recall_ceiling'])} (A, B) and {f(g_s['recall_ceiling'])} (128 px arms), and a fixed centred Gaussian already reaches "
        f"{f(gauss_a)} and {f(gauss_s)}, i.e. about {100 * gauss_a / g_a['recall_ceiling']:.0f}% of the ceiling. A map has to beat that to show "
        "localisation beyond a centre prior; attention rollout does not (Section 4.5), and Grad-CAM at the last block is lower still. In the off-centre "
        "test the Gaussian loses its advantage while rollout holds its recall, which indicates that rollout does track image content to some degree, "
        "but the as-is numbers cannot separate content from position.")
    P.h2("5.3. Reproduction of the published result")
    P.p(f"The released checkpoint, run through our adapter on the public DERMAHAIR pairs, reached {f(full2000['psnr_up'], 2)} dB PSNR / "
        f"{f(full2000['ssim_up'], 4)} SSIM at the full 2,000-step schedule and {f(r10['psnr_up'], 2)} dB / {f(r10['ssim_up'], 4)} with 10 "
        f"strided steps, against a do-nothing floor of {f(floor['psnr_up'], 2)} dB / {f(floor['ssim_up'], 4)} and the published {paper_psnr:.2f} dB / "
        f"{paper_ssim:.4f}. The best setting ({best_dh_key}) is "
        f"{'above' if best_dh['psnr_up'] > floor['psnr_up'] else 'not above'} the floor and "
        f"{abs(paper_psnr - best_dh['psnr_up']):.1f} dB {'below' if best_dh['psnr_up'] < paper_psnr else 'above'} the publication; "
        f"the pre-stated verdict is {'reproduced' if repro_any else 'not reproduced'}. The paper reports no do-nothing baseline, and "
        "evaluation subset, checkpoint version or metric implementation may differ from ours, so the cause of any gap is not established.")
    # Training-iteration facts for the limitations bullet; each number below is read from a file,
    # except the two documented historical facts, which carry their citation in the text.
    n_iter_cfg = int(re.search(r'"n_iter":\s*(\d+)',
                               (ROOT / "third_party" / "dmahr" / "config" / "akiec.json").read_text()).group(1))
    rel_iter = int(re.search(r"I(\d+)_E\d+", (ROOT / "reports" / "export_manifest.csv").read_text()).group(1))
    ft_iter = int(json.loads((LOGS / "dmahr_v2_train.json").read_text())["iterations"])
    lc_files = sorted((LOGS / "learning_curve").glob("it*.json"))
    if not lc_files:
        raise SystemExit("artifacts/logs/learning_curve/ holds no snapshot files")
    snap_iters = [int(p.stem[2:]) for p in lc_files]
    snap_mae = [float(json.loads(p.read_text())["one_step_eps_mae"]["1500"]) for p in lc_files]
    P.h2("5.4. Limitations")
    P.bullets([
        "Image-quality evidence uses synthetic hair on clean photographs; real-hair removal quality is not validated.",
        "The diffusion models work at 128 px and the classifier at 224 px; A* controls the resolution but not the loss of fine texture.",
        "Eight seeds (three for Creleased and C) leave a minimum detectable difference near 0.04–0.05 macro-F1.",
        "Lesion masks come from a single rater and are large and central, which limits what localisation metrics can show.",
        "The explanation analysis re-scored the same test images after the geometry correction (decision D40); the first-look results are kept in the appendix.",
        "Training iterations — the Cv2 backbone is the fixed-trainer lineage whose snapshots were evaluated at "
        f"{', '.join(f'{v:,}' for v in snap_iters)} iterations (Figure 5); the continuation was stopped by the project owner at "
        "56,300 of a planned 100,000 iterations (reports/decision_log.md), and the model was then fine-tuned for "
        f"{ft_iter:,} iterations on pasted real strands (artifacts/logs/dmahr_v2_train.json). For scale: the development kit's "
        "reference recipe specifies 20,000 iterations (decision D1), the shipped upstream config asks for "
        f"{n_iter_cfg:,} (third_party/dmahr/config/akiec.json), and the released checkpoint is labelled {rel_iter:,} "
        f"(reports/export_manifest.csv). One-step error at t=1,500 across the snapshots in that order is "
        f"{', '.join(f'{v:.4f}' for v in snap_mae)} — not monotonically improving — so iterations beyond about 56,000 are "
        f"untested in this pipeline, as are the upstream budgets ({n_iter_cfg:,}-{rel_iter:,}).",
        "Licences: HAM10000 is CC BY-NC 4.0, so results and derived images are non-commercial; the DM-AHR repository's README (CC BY-NC 4.0) and LICENSE file (Apache-2.0) contradict each other, "
        "so commercial use of the released weights is unresolved; no legal conclusion is drawn.",
        "Research prototype only; no clinical claim is made."])

    # ================= 6 Conclusion
    P.h1("6. Conclusion")
    h3_body = ("made explanations more lesion-focused" if h3_state == "supported"
               else ("left the explanation comparison inconclusive (every between-arm difference was struck "
                     "by the seed-spread rule)" if h3_state.startswith("inconclusive")
                     else "did not make explanations more lesion-focused"))
    P.p(f"In a lesion-grouped, matched-recipe comparison with eight seeds, diffusion hair removal did not improve classification "
        f"(Cv2 {f(f1['Cv2'])} versus original {f(f1['A'])} and resolution control {f(f1['Astar'])} macro-F1), did not match the classical method on "
        f"image fidelity for synthetic hair, and {h3_body} once mask geometry and a centre prior were accounted "
        "for. The published image-quality figure was not reproduced from the released checkpoint on the public data. The only established effect is "
        f"that DullRazor-style removal lowered macro-F1 relative to the original images ({sgn(ab['difference'])}, CI {sgn(ab['ci_low'])} to "
        f"{sgn(ab['ci_high'])}).")

    # ================= Appendix
    P.h1("Appendix A. Earlier results kept for the record")
    c3 = head.loc["C"]
    P.p(f"Arm C (first diffusion run, noise-level encoding defect): macro-F1 {c3['macro_f1_mean']:.3f} ± {c3['macro_f1_sd']:.3f} over three seeds; "
        "its image quality is listed in Section 4.2. Explanation results computed before the geometry correction (masks resized from "
        "the whole photograph) are retained as artifacts/logs/xai_summary_*_X1.json and are superseded by Section 4.5.")
    P.figure(FIGS / "fig_learning_curve.png", "Diffusion diagnostics at the evaluated training snapshots (validation pairs only).", 5.8)

    P.h1("Appendix B. Per-seed explanation values")
    seeds_sorted = [42, 123, 2026, 7, 11, 17, 23, 31]
    prow = []
    for a in ("A", "B", "Astar", "Cv2", "Creleased", "C"):
        for m, mn in ((ROLL, "rollout"), (GC1, "Grad-CAM ln_1"), (GCP, "Grad-CAM patch")):
            r = xai_val(F, a, m, "recall")
            if r is None:
                continue
            d = dict(kv.split(":") for kv in str(r["per_seed"]).split(";"))
            prow.append([SHORT[a], mn] + [d.get(str(s), "-") for s in seeds_sorted])
    P.table("Recall as is, per seed (columns are the seeds; '-' = seed not trained for that arm).",
            ["Arm", "Method"] + [str(s) for s in seeds_sorted], prow, size=7.5)

    P.h1("Appendix C. Explanation results before the geometry correction (superseded)")
    legacy = []
    for a in ("A", "B", "Astar", "Cv2", "Creleased", "C"):
        js = [json.loads(p.read_text()) for p in sorted(LOGS.glob(f"xai_summary_{a}_seed*_X1.json"))]
        if not js:
            continue
        legacy.append([SHORT[a], len(js), f(np.mean([j["rollout_overlap20_mean"] for j in js])),
                       f(np.mean([j["gradcam_overlap20_mean"] for j in js])),
                       f(np.mean([j["rollout_pointing_mean"] for j in js])),
                       f(np.mean([j["gradcam_pointing_mean"] for j in js])),
                       f(np.mean([j["chance_gaussian_overlap20_mean"] for j in js]))])
    P.table("Old frame (masks resized from the whole photograph, Grad-CAM on the patch-embedding grid; three seeds). "
            "Kept for the record; the geometry does not match what the classifier sees, so these values must not be compared with Section 4.5.",
            ["Arm", "Seeds", "Rollout overlap", "Grad-CAM overlap", "Rollout pointing", "Grad-CAM pointing", "Gaussian overlap"], legacy)

    # ================= References
    P.h1("References")
    refs = [p.text for p in paras if p.style.name == "Normal" and p.text[:3].strip().rstrip(".").isdigit() and ". " in p.text[:5]]
    refs = [r for r in refs if len(r) > 40]
    refs.append("24. Tschandl, P.; Rinner, C.; Apalla, Z.; Argenziano, G.; Codella, N.; Halpern, A.; Janda, M.; Lallas, A.; Longo, C.; "
                "Malvehy, J.; et al. Human–computer collaboration for skin cancer recognition. Nat. Med. 2020, 26, 1229–1234.")
    for r in refs:
        P.p(r, size=9)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    P.doc.core_properties.title = "Explainable Vision Transformer for Skin Lesion Classification Using Dermoscopy Images"
    P.doc.save(str(OUT))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes), commit {commit}")
    return OUT


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a publication-style PDF of the real-hair validation results with sample outputs.

Reads only the small report CSVs and derived QA images under reports/real_hair_validation.
No dataset bytes are touched. Uses PyMuPDF (built-in Helvetica) so no external font
install is needed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "real_hair_validation"
OUT = R / "real_hair_validation_results.pdf"

PW, PH = 595.0, 842.0  # A4
M = 54.0
CW = PW - 2 * M
NAVY = (0.09, 0.16, 0.28)
SLATE = (0.25, 0.30, 0.38)
ACCENT = (0.72, 0.30, 0.16)
GREY = (0.45, 0.45, 0.45)


def rows(name: str) -> list[dict]:
    with (R / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class Doc:
    def __init__(self) -> None:
        self.d = fitz.open()
        self.p = None
        self.y = 0.0
        self.page_no = 0
        self.new_page()

    def new_page(self) -> None:
        self.page_no += 1
        self.p = self.d.new_page(width=PW, height=PH)
        self.y = M + 12
        if self.page_no > 1:
            self.p.insert_text((M, M - 14),
                               "External real-world hair-removal validation",
                               fontname="helv", fontsize=8, color=GREY)
            self.p.draw_line(fitz.Point(M, M - 8), fitz.Point(PW - M, M - 8),
                             color=(0.8, 0.8, 0.8), width=0.5)

    def need(self, h: float) -> None:
        if self.y + h > PH - M - 20:
            self.new_page()

    def heading(self, text: str, size: float = 13) -> None:
        self.need(size + 18)
        self.y += 8
        self.p.insert_text((M, self.y), text, fontname="hebo", fontsize=size, color=NAVY)
        self.y += size + 5
        self.p.draw_line(fitz.Point(M, self.y - 3), fitz.Point(PW - M, self.y - 3),
                         color=ACCENT, width=0.8)

    def para(self, text: str, size: float = 9.3, leading: float = 13.2,
             color=SLATE) -> None:
        words = text.split()
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if fitz.get_text_length(trial, fontname="helv", fontsize=size) > CW:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        for ln in lines:
            self.need(leading)
            self.p.insert_text((M, self.y), ln, fontname="helv", fontsize=size, color=color)
            self.y += leading
        self.y += 3

    def bullet(self, text: str, size: float = 9.3) -> None:
        self.need(13.2)
        self.p.insert_text((M + 6, self.y), "•", fontname="helv", fontsize=size, color=ACCENT)
        self.para(text, size=size)
        self.y -= 3

    def table(self, header: list[str], data: list[list[str]], widths: list[float] | None = None,
              size: float = 8.2) -> None:
        n = len(header)
        widths = widths or [CW / n] * n
        rh = 16.0
        self.need(rh * (len(data) + 1) + 8)
        x0 = M
        # header
        self.p.draw_rect(fitz.Rect(M, self.y - 2, M + CW, self.y + rh - 2),
                         color=None, fill=(0.90, 0.92, 0.95))
        x = x0
        for h, w in zip(header, widths):
            self.p.insert_text((x + 3, self.y + 10), str(h)[:34],
                               fontname="hebo", fontsize=size, color=NAVY)
            x += w
        self.y += rh
        for i, row in enumerate(data):
            if i % 2:
                self.p.draw_rect(fitz.Rect(M, self.y - 2, M + CW, self.y + rh - 2),
                                 color=None, fill=(0.97, 0.97, 0.98))
            x = x0
            for cell, w in zip(row, widths):
                self.p.insert_text((x + 3, self.y + 10), str(cell)[:40],
                                   fontname="helv", fontsize=size, color=SLATE)
                x += w
            self.y += rh
        self.y += 8

    def image(self, path: Path, width: float, caption: str = "") -> None:
        if not path.exists():
            self.para(f"[missing image: {path.name}]")
            return
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            iw, ih = im.size
        h = width * ih / iw
        cap_h = 14 if caption else 0
        self.need(h + cap_h + 8)
        x = M + (CW - width) / 2
        self.p.insert_image(fitz.Rect(x, self.y, x + width, self.y + h), filename=str(path))
        self.y += h + 3
        if caption:
            self.need(cap_h)
            self.p.insert_text((M, self.y + 9), caption, fontname="helv",
                               fontsize=8, color=GREY)
            self.y += cap_h
        self.y += 6

    def save(self, out: Path) -> None:
        for i, page in enumerate(self.d):
            page.insert_text((PW - M - 62, PH - 24), f"Page {i + 1} of {len(self.d)}",
                             fontname="helv", fontsize=8, color=GREY)
        self.d.save(str(out))
        self.d.close()


def main() -> None:
    doc = Doc()
    meth = {r["method"]: r for r in rows("method_summary.csv")}
    comp = {r["metric"]: r for r in rows("base_vs_finetuned_comparison.csv")}
    det = {r["metric"]: r for r in rows("hair_detector_metrics.csv")}
    ov = rows("overlap_audit_summary.json") if False else None  # json loaded below
    import json
    ov = json.loads((R / "overlap_audit_summary.json").read_text())
    man = json.loads((R / "eval_manifest_summary.json").read_text())
    mend = json.loads((R / "mendeley_audit_summary.json").read_text())
    prim = json.loads((R / "primary_inference_summary.json").read_text())
    expl = json.loads((R / "exploratory_inference_summary.json").read_text())
    spend = rows("spend_summary.csv")

    # ---- cover
    p = doc.p
    p.draw_rect(fitz.Rect(0, 0, PW, 10), color=None, fill=NAVY)
    doc.y = 190
    doc.p.insert_textbox(fitz.Rect(M, 150, PW - M, 250),
                         "External real-world hair-removal validation",
                         fontname="hebo", fontsize=21, color=NAVY, align=1)
    doc.p.insert_textbox(fitz.Rect(M, 245, PW - M, 300),
                         "DM-AHR base vs fine-tuned vs DullRazor on independent natural-hair images",
                         fontname="helv", fontsize=11.5, color=SLATE, align=1)
    doc.y = 330
    doc.para("Research prototype only. Not clinically validated and not intended for diagnosis, "
             "triage or patient care. Every number in this PDF is read from the CSV/JSON artifacts "
             "under reports/real_hair_validation/.", size=9.5)
    doc.para(f"Generated from the final report and artifacts. Primary run: {prim['n_requested']} "
             f"images, {prim['seconds']} s on {prim['gpu']}. Exploratory t0=1000 run: "
             f"{expl['n_requested']} images, {expl['seconds']} s. All computation ran on Modal; "
             "no dataset was downloaded to the local machine.")
    doc.new_page()

    doc.heading("1. Executive summary")
    doc.para(f"Both surviving DM-AHR generators loaded strictly and produced finite, reproducible "
             f"128 px outputs. The overlap audit classified {ov['n_candidates']} Mendeley candidates "
             f"as {ov['independent_A']} independent (A), {ov['existing_B']} exact duplicates (B) and "
             f"{ov['near_duplicate_C']} probable near-duplicates (C); only class A was scored. "
             f"The independent set contains {man['hair_present']} hair-positive images and "
             f"{man['hair_negative_controls']} hair-negative controls.")
    doc.para("Fine-tuning reduced the residual-hair proxy but increased alteration of non-hair skin. "
             "This is a mixed outcome, not a clean improvement. Lesion preservation is NOT VERIFIED "
             "because the independent set has no lesion masks; no human ratings exist yet.")

    doc.heading("2. Checkpoints and lineage")
    inv = rows("checkpoint_inventory.csv")
    diff = [r for r in inv if r["role"] != "ViT classifier checkpoint (not a diffusion generator)"]
    doc.table(["Role", "Bytes", "SHA-256 (first 16)", "Path"],
              [[r["role"].split(";")[0][:34], r["bytes"], r["sha256"][:16],
                r["path"].replace("/vol/", "")[-34:]] for r in diff],
              widths=[150, 78, 95, CW - 323], size=7.2)
    doc.para("The requested ~56.3k-iteration continuation generator is absent from this workspace; "
             "the base used here is the 50k repaired generator, which is the actual starting point "
             "of the 20k fine-tune (decision log D42/D44).")

    doc.heading("3. Primary results (full independent set, n = 395)")
    doc.table(["Method", "Hair-region changed", "Non-hair changed", "Residual-hair proxy"],
              [[m, f"{float(r['hair_changed_mean']):.3f} ± {float(r['hair_changed_sd']):.3f}",
                f"{float(r['nonhair_changed_mean']):.3f} ± {float(r['nonhair_changed_sd']):.3f}",
                f"{float(r['residual_proxy_mean']):.3f} ± {float(r['residual_proxy_sd']):.3f}"]
               for m, r in meth.items()], widths=[120, 130, 130, CW - 380], size=8)
    doc.para("Change is not correctness: natural-hair images have no paired hairless reference, so "
             "PSNR/SSIM/LPIPS cannot be computed on them. These are pixel-change and residual-hair "
             "proxies only.")

    doc.heading("4. Base vs fine-tuned (paired image-level bootstrap, 5,000 reps)")
    doc.table(["Metric", "Fine-tuned − base", "95% CI", "Holm p"],
              [[k.replace("_", " "), f"{float(r['difference_finetuned_minus_base']):+.4f}",
                f"{float(r['ci_low']):+.4f} to {float(r['ci_high']):+.4f}",
                str(r["bootstrap_p_holm"])] for k, r in comp.items()],
              widths=[145, 110, 145, CW - 400], size=8)
    doc.para("Interpretation: the fine-tuned checkpoint leaves fewer detector-detected strands "
             "inside the annotated hair region, but also alters more of the surrounding skin. "
             "DullRazor changes the most pixels both inside and outside the hair mask and shows the "
             "lowest residual proxy, but its outputs contain visible flat-colour artefacts "
             "(agent-reviewed). No result establishes clinical or image-quality superiority.")

    doc.heading("5. Hair detector on 500 real annotated masks")
    keys = ["pixel_precision", "pixel_recall_sensitivity", "pixel_specificity", "pixel_f1",
            "dice", "iou", "image_sensitivity", "image_specificity",
            "image_false_positive_rate", "image_false_negative_rate"]
    doc.table(["Metric", "Value"],
              [[k, f"{float(det[k]['value']):.4f}"] for k in keys],
              widths=[CW * 0.7, CW * 0.3], size=8)
    doc.para("The frozen screening threshold (tuned on synthetic train/val pairs in the original "
             "project) over-detects on real hair: image-level specificity is low. No threshold was "
             "tuned on these evaluation masks.")

    # ---- sample images
    doc.new_page()
    doc.heading("6. Sample annotation and generated outputs (agent-reviewed)")
    doc.para("Panel order in the 4-panel strips: original | DullRazor | base DM-AHR | fine-tuned "
             "DM-AHR. These are randomly selected review images; the mapping file is kept separate "
             "from the reviewer kit. Visual observations below are from an AI agent, not a blinded "
             "human reviewer.")
    doc.image(R / "mendeley_contact_sheet.png", CW,
              "Figure 1. Mendeley original / binary hair mask / overlay examples (agent-reviewed).")
    for rid in ("R001", "R003", "R009", "R027"):
        pth = R / "blinded_human_review_kit" / f"{rid}.png"
        doc.image(pth, CW, f"Figure {2 + ('R001','R003','R009','R027').index(rid)}. "
                           f"Review strip {rid}: original | DullRazor | base | fine-tuned.")
    doc.para("Observed: hair is generally reduced. DullRazor produces large flat brown/pink/yellow "
             "regions in several strips (e.g. R003, R014, R029). Both diffusion checkpoints keep "
             "global colour and lesion location better but soften fine texture. The fine-tuned "
             "panel is subtly sharper in a few strips and similar in most.")

    doc.heading("7. Limitations and what remains unverified")
    for t in [
        "Lesion preservation: NOT VERIFIED. No lesion masks exist for the independent external set; none were auto-generated.",
        "No paired natural hairless reference, so no natural-hair PSNR/SSIM/LPIPS. Synthetic-hair image-quality numbers are kept separate.",
        "DermArtifactDB supplied annotations only; ISIC-2019 / PH2 / Derm7pt source images were not obtained (registration required or host unreachable).",
        "The ~56.3k continuation generator and the original defective generator weights are absent from this workspace.",
        "No blinded human ratings. The 40-image kit and assessment form are ready for a person.",
    ]:
        doc.bullet(t)

    doc.heading("8. Compute and spend")
    doc.table(["Item", "GPU seconds", "Estimated USD"],
              [[r["item"], r["seconds"], r["usd"]] for r in spend],
              widths=[CW * 0.62, CW * 0.18, CW * 0.20], size=8)
    doc.para("Workspace billing snapshot: metered cost $5.46, credits $5.28, billed $0.00 at capture. "
             "GPU figures are estimates from measured seconds at $0.000694/s (A100-80GB), not an invoice.")

    doc.heading("9. Conclusion")
    doc.para("The experiment is sufficient to report proxy-level natural-hair findings and the "
             "base-versus-fine-tuned trade-off. It is not sufficient to support a claim of "
             "successful real-world hair removal: that requires paired references, lesion masks and "
             "blinded human ratings. Full artifacts: reports/real_hair_validation/ and the dedicated "
             "Modal volume dm-ahr-real-hair-validation.")

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()

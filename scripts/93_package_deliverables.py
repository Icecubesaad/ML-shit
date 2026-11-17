"""Assemble one ZIP with the edited client document, code guide, reports and paper.

Reproducible: copies tracked/generated files into a staging tree and zips it.
No generated artifact is edited here.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path(r"C:\Users\saadq\AppData\Local\Temp\opencode\deliverables_stage")
ZIP = Path(r"C:\Users\saadq\Downloads\DM-AHR-ViT_Full_Deliverables.zip")
REPO_ZIP = ROOT / "artifacts" / "reports" / "DM-AHR-ViT_Full_Deliverables.zip"


def copy(src: Path, dest: Path, required: bool = True):
    if not src.exists():
        if required:
            raise FileNotFoundError(src)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    # 1. Edited client document
    copy(ROOT / "artifacts/reports/DM-AHR-ViT_Skin_Lesion_Classification_FINAL.docx",
         STAGE / "document/DM-AHR-ViT_Skin_Lesion_Classification_(2)_EDITED.docx")
    copy(ROOT / "reports/numbers_audit.md", STAGE / "document/numbers_audit.md")

    # 2. Code guide + standalone notebook
    copy(ROOT / "reports/colab_notebook_code_guide.md", STAGE / "code_guide/colab_notebook_code_guide.md")
    copy(ROOT / "notebooks/05_colab_standalone_training.ipynb",
         STAGE / "code_guide/05_colab_standalone_training.ipynb")
    copy(ROOT / "docs/CODE_GUIDE.md", STAGE / "code_guide/CODE_GUIDE.md")
    copy(ROOT / "artifacts/reports/Code_Guide_Report.docx", STAGE / "code_guide/Code_Guide_Report.docx")
    copy(ROOT / "artifacts/reports/Code_Guide_Report.pdf", STAGE / "code_guide/Code_Guide_Report.pdf")

    # 3. Reports
    report_files = [
        "final_research_report.md", "audit_v2_integration.md", "numbers_audit.md",
        "decision_log.md", "fix_log.md", "PROGRESS.md", "experiment_protocol.md",
        "hypotheses.md", "license_audit.md", "deliverable_checklist.md",
        "xai_corrected_results.md", "stageE_results.md", "stageC_results.md",
        "client_summary.md", "CHANGELOG_final.md", "client_plan_and_timeline.md",
    ]
    for name in report_files:
        copy(ROOT / "reports" / name, STAGE / "reports" / name, required=False)
    copy(ROOT / "reports/artifact_inventory.csv", STAGE / "reports/artifact_inventory.csv", required=False)
    for p in sorted((ROOT / "reports/real_hair_validation").glob("*.md")):
        copy(p, STAGE / "reports/real_hair_validation" / p.name)
    for p in sorted((ROOT / "reports/real_hair_validation/v2").iterdir()):
        if p.is_file():
            copy(p, STAGE / "reports/v2_tables" / p.name)
    for name in ("final_research_report.md", "data_audit.md", "deliverable_checklist.md"):
        copy(ROOT / "artifacts/reports" / name, STAGE / "reports" / f"artifact_{name}", required=False)
    for name in ("Results_Report.docx", "Results_Report.pdf", "Preprocessing_Report.docx",
                 "Preprocessing_Report.pdf"):
        copy(ROOT / "artifacts/reports" / name, STAGE / "reports" / name, required=False)

    # 4. Research paper
    copy(ROOT / "artifacts/reports/Research_Paper.docx", STAGE / "research_paper/Research_Paper.docx")
    copy(ROOT / "artifacts/reports/Research_Paper.pdf", STAGE / "research_paper/Research_Paper.pdf")

    # 5. Figures (v2 first, archived ones clearly named)
    for p in sorted((ROOT / "artifacts/figures").glob("v2_*.png")):
        copy(p, STAGE / "figures/v2" / p.name)
    for p in sorted((ROOT / "artifacts/figures").glob("fig_*.png")):
        copy(p, STAGE / "figures/archived_original_run" / p.name)

    readme = """DM-AHR-ViT — complete deliverable set
=======================================

Research prototype only. Not clinically validated and not intended for diagnosis,
triage or patient care.

Contents
--------
document/       The client's proposal-derived document, edited with the full audit
                findings, verification table and 11 embedded figures. The untouched
                original is docs/proposal.docx in the repository.
code_guide/     Standalone Colab notebook (80 cells), its dedicated code guide,
                and the project code guide/report.
reports/        All markdown reports, the v2 CSV tables, and the Word/PDF reports.
research_paper/ Research_Paper.docx and Research_Paper.pdf.
figures/v2/     Figures generated from the audited v2 tables.
figures/archived_original_run/  Figures from the original run, retained for provenance
                and SUPERSEDED where they describe hair conditions.

Verification
------------
reports/numbers_audit.md records 41 checks comparing every reported v2 metric with
its source CSV; all 41 passed at packaging time. The 8-seed extension (7/11/17/23/31)
is complete; its tables are phase6*_v2_8seed.csv alongside the original 3-seed tables.
Both agree: no Holm-significant advantage for either hair-removal arm on the full test.

Licence: HAM10000 CC BY-NC 4.0; DM-AHR README/LICENSE conflict (see license_audit.md).
Commercial use unresolved. No legal conclusion is given.
"""
    (STAGE / "README.txt").write_text(readme, encoding="utf-8")

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    REPO_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(STAGE).as_posix())
    shutil.copy2(ZIP, REPO_ZIP)
    n = len(list(STAGE.rglob("*")))
    print("wrote", ZIP, ZIP.stat().st_size, "bytes;", n, "entries")
    print("copy", REPO_ZIP)


if __name__ == "__main__":
    main()

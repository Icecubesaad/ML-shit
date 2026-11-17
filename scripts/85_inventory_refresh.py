"""Refresh reports/artifact_inventory.csv: repair the header, update hashes of files regenerated this round,
and add the new deliverables (paper, corrected explanation outputs, DERMAHAIR result, export manifest and the
exported checkpoints with the hashes recorded in reports/export_manifest.csv).

Rows for volume-only large files keep sha256 = skipped_large. The file is copied to artifacts/reports/.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm_ahr_vit.config import sha256_file  # noqa: E402

INV = ROOT / "reports" / "artifact_inventory.csv"
PREFIX = "/__modal/volumes/vo-TbNx5IITKUKx9iVQZHZpNF/"
GLOBS = ["artifacts/reports/*.docx", "artifacts/reports/*.pdf", "artifacts/reports/*.md", "artifacts/figures/*.png",
         "artifacts/logs/xai_v2_analysis/*", "artifacts/logs/dermahair_repro*", "artifacts/logs/split_summary.json",
         "artifacts/logs/run_registry.jsonl", "artifacts/logs/modal_runs/*", "artifacts/logs/acceptance_report.json",
         "reports/*.md", "reports/*.csv", "reports/archive/*", "reports/real_hair_validation/audit_v2_integration.md",
         "reports/real_hair_validation/v2/*", "configs/*.json", "scripts/8*.py",
         "experiments/real_hair_validation/*.py", "tests/test_paper_build.py", "tests/test_phase6_stats.py"]


def kind_of(rel: str) -> str:
    return "_".join(rel.split("/")[:2]).replace("artifacts_", "stage_")


def main() -> int:
    rows = []
    for line in INV.read_text().replace("\r", "").splitlines():
        parts = next(csv.reader([line]))
        if len(parts) == 4 and parts[0] != "kind" and parts[1]:
            rows.append(dict(zip(("kind", "path", "bytes", "sha256"), parts)))
    refreshed = 0
    for r in rows:
        rel = r["path"][len(PREFIX):] if r["path"].startswith(PREFIX) else r["path"]
        p = ROOT / rel
        if p.is_file() and r["sha256"] != "skipped_large":
            h = sha256_file(p)
            if h != r["sha256"]:
                r["bytes"], r["sha256"] = p.stat().st_size, h
                refreshed += 1
    have = {r["path"] for r in rows} | {r["path"][len(PREFIX):] for r in rows if r["path"].startswith(PREFIX)}
    new = []
    for pat in GLOBS:
        for p in sorted(ROOT.glob(pat)):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and rel not in have:
                new.append({"kind": kind_of(rel), "path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
                have.add(rel)
    man = ROOT / "reports" / "export_manifest.csv"
    for r in csv.DictReader(man.open()):
        if r["kind"] in ("vit", "diffusion"):
            rel = "exports/" + r["file"]
            if rel not in have:
                new.append({"kind": "exports_" + r["kind"], "path": rel, "bytes": r["bytes"], "sha256": r["sha256"]})
                have.add(rel)
    with INV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "path", "bytes", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows + new)
    shutil.copy2(INV, ROOT / "artifacts" / "reports" / "artifact_inventory.csv")
    print(f"{len(rows)} existing rows ({refreshed} hashes refreshed), {len(new)} added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

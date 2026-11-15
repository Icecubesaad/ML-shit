"""The paper builds from artifacts and carries none of the superseded wording."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
needed = [ROOT / "artifacts" / "logs" / n for n in
          ("dermahair_repro.json", "split_summary.json", "classification_metrics_all.csv",
           "gradcam_sanity.json")]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in needed), reason="artifacts not present")


def _load():
    spec = importlib.util.spec_from_file_location("build_paper", ROOT / "scripts" / "80_build_paper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_paper"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_paper_builds_and_is_clean(tmp_path):
    import docx

    mod = _load()
    mod.OUT = tmp_path / "paper.docx"
    mod.build(None)
    d = docx.Document(str(mod.OUT))
    text = "\n".join(p.text for p in d.paragraphs)
    for t in d.tables:
        for r in t.rows:
            text += "\n" + " | ".join(c.text for c in r.cells)
    cfg = json.loads((ROOT / "configs" / "stale_phrases.json").read_text())
    for pat in cfg["patterns"] + cfg["paper_patterns"]:
        assert not re.search(pat, text, flags=re.IGNORECASE), pat
    assert "Tschandl, P.; Rinner" in text
    assert "rotation up to 15°" in text
    assert "To our knowledge, no prior study" in text
    assert "Verdict:" in text and "H1 (image quality):" in text
    assert "ln_1" in text and "Abstract" in text


def test_per_seed_diff_covers_all_analysis_seeds():
    """The per-seed helper behind Table 8 returns one labelled (seed, value) per seed the
    analysis has — all eight for the A/A*/B/Cv2 comparisons — with values equal to the
    pooled_per_seed.csv differences."""
    mod = _load()
    F = mod.load_facts()
    full = F["per_seed"]
    full = full[full["subset"] == "full"]
    for a, b in (("A", "B"), ("A", "Astar"), ("A", "Cv2"), ("B", "Cv2"), ("Cv2", "Astar")):
        ps = mod.per_seed_diff(F, a, b)
        assert [s for s, _ in ps] == list(mod.EIGHT), (a, b, [s for s, _ in ps])
        for s, v in ps:
            va = float(full[(full["arm"] == a) & (full["seed"] == s)]["macro_f1"].iloc[0])
            vb = float(full[(full["arm"] == b) & (full["seed"] == s)]["macro_f1"].iloc[0])
            assert v == va - vb, (a, b, s, v, va - vb)


def test_table8_cells_label_every_seed(tmp_path):
    """In the built paper, every Table 8 per-seed cell lists seed:value pairs for exactly the
    seeds the analysis provides, with the matching '[k of N seeds favour first]' bracket."""
    import docx

    mod = _load()
    F = mod.load_facts()
    mod.OUT = tmp_path / "paper.docx"
    mod.build(None)
    d = docx.Document(str(mod.OUT))
    t8 = [t for t in d.tables
          if len(t.rows) and len(t.rows[0].cells) >= 6
          and t.rows[0].cells[5].text.strip() == "Per-seed differences"
          and "Macro-F1 difference" in t.rows[0].cells[2].text]
    assert len(t8) == 1, "Table 8 (pooled per-seed) not found in the built paper"
    for row in t8[0].rows[1:]:
        comp = row.cells[0].text.replace("A*", "Astar").replace(" ", "")
        if "-" not in comp:
            continue
        a, b = comp.split("-", 1)
        cell = row.cells[5].text
        listed = [(int(s), float(v)) for s, v in re.findall(r"(\d+):([+-]\d+\.\d+)", cell)]
        expect = mod.per_seed_diff(F, a, b)
        assert listed and [s for s, _ in listed] == [s for s, _ in expect], (comp, cell)
        assert all(abs(v - x) <= 0.0051 for (_, v), (_, x) in zip(listed, expect)), (comp, cell)
        assert len(expect) == 8, (comp, len(expect))
        n_fav = sum(float(f"{x:+.2f}") > 0 for _, x in expect)
        assert re.search(rf"\[{n_fav} of {len(expect)} seeds favour first\]", cell), cell

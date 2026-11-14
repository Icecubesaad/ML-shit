"""The paper builds from artifacts and carries none of the superseded wording."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
needed = [ROOT / "artifacts" / "logs" / n for n in
          ("dermahair_repro.json", "split_summary.json", "classification_metrics_all.csv")]
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

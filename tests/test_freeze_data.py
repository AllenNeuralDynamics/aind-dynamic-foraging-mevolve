"""Gate 0 on the real freeze (``pytest -m data``; needs MEVOLVE_FREEZE and MEVOLVE_STUDIES)."""
import json
import os
from pathlib import Path

import pytest

from mevolve.data import Freeze
from mevolve.gate0 import check

pytestmark = pytest.mark.data
FREEZE, STUDIES = os.environ.get("MEVOLVE_FREEZE"), os.environ.get("MEVOLVE_STUDIES")


@pytest.mark.skipif(not (FREEZE and STUDIES), reason="set MEVOLVE_FREEZE and MEVOLVE_STUDIES")
def test_gate0_on_the_freeze():
    manifest = json.loads((Path(FREEZE) / "manifest.json").read_text())
    result = check(manifest, Path(STUDIES))
    assert result["ok"], [r for r in result["checks"] if not r["ok"]]


@pytest.mark.skipif(not FREEZE, reason="set MEVOLVE_FREEZE")
def test_the_reader_sees_every_scored_trial():
    freeze = Freeze(FREEZE)
    held = freeze.manifest["pools"]["heldout"]
    n = sum(len(s.choices) for sid in freeze.subjects("heldout") for s in freeze.subject_sessions(sid, "score"))
    assert n == held["n_choice_trials"]["score"]

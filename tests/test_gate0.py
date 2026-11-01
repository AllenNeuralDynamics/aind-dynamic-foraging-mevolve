import json

from mevolve.gate0 import check, gru_extra_test_subjects


def test_gate0_passes_on_matching_lists(freeze_dir, studies_dir):
    result = check(json.loads((freeze_dir / "manifest.json").read_text()), studies_dir)
    assert result["ok"], [r for r in result["checks"] if not r["ok"]]


def test_gate0_fails_on_a_wrong_count(freeze_dir, studies_dir):
    manifest = json.loads((freeze_dir / "manifest.json").read_text())
    manifest["pools"]["heldout"]["n_choice_trials"]["score"] += 1
    result = check(manifest, studies_dir)
    assert not result["ok"]
    assert {r["check"] for r in result["checks"] if not r["ok"]} == {
        "held-out scored trials equal HB-CTT's (study 13 r5)",
        "held-out scored trials equal the MLE baselines' (study 01 r8)"}


def test_gru_extra_test_subjects(freeze_dir, studies_dir):
    assert gru_extra_test_subjects(json.loads((freeze_dir / "manifest.json").read_text()), studies_dir) == ["t3"]

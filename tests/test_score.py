"""The reported-score path: max_em reaches the 2-level fit, and the score CLI's records."""
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from mevolve import evaluate, program, score

from test_fitting import TRUE, simulate


@dataclass
class Sess:
    session_id: str
    choices: np.ndarray
    rewards: np.ndarray
    task: str = "Coupled Baiting"
    stage_group: str = "final"
    index: int = 0


def _subject(seed):
    sims = simulate(program.literature("hattori2019"), TRUE, np.random.default_rng(seed), n_sessions=4,
                    n_trials=120)
    sessions = [Sess(f"s{i}", s.choices, s.rewards, index=i) for i, s in enumerate(sims)]
    return evaluate.SubjectData(f"sub{seed}", sessions[::2], sessions[1::2])


def test_max_em_reaches_the_two_level_fit():
    path = program.LITERATURE_DIR / "hattori2019.py"
    rec, = evaluate.evaluate(path, [_subject(0)], regime="2level", n_draws=8, n_starts=2, workers=1, max_em=2)
    assert 1 <= rec["em_iterations"] <= 2 and isinstance(rec["converged"], bool)
    for key in ("log_lik", "n_trials", "fit_log_lik", "session_ids", "session_groups", "session_log_lik",
                "session_n_trials", "typical_params", "log_sigma", "ess", "fingerprints"):
        assert key in rec


def test_score_cli_records_pools_provenance_and_the_reported_em_cap(tmp_path, monkeypatch):
    freeze_dir = tmp_path / "freeze"
    freeze_dir.mkdir()
    (freeze_dir / "manifest.json").write_text("{}")
    fake = SimpleNamespace(manifest={"pools": {"train": {"subjects": ["a", "b"]}, "heldout": {"subjects": ["c"]}}},
                           subjects=lambda p: {"train": ["a", "b"], "heldout": ["c"]}[p])
    seen = {}

    def fake_evaluate(path, data, regime, seed, n_draws, workers, max_em):
        seen.update(regime=regime, n_draws=n_draws, max_em=max_em)
        return [{"subject_id": d, "converged": d != "b"} for d in data]

    monkeypatch.setattr(score, "Freeze", lambda path: fake)
    monkeypatch.setattr(score, "_git_sha", lambda repo: "f" * 40)
    monkeypatch.setattr(score.evaluate, "subject_data", lambda freeze, s: s)
    monkeypatch.setattr(score.evaluate, "evaluate", fake_evaluate)
    score.main(["--model", "Hattori2019", "--freeze", str(freeze_dir), "--out", str(tmp_path / "out")])
    out = json.loads((tmp_path / "out" / "2level_Hattori2019.json").read_text())
    assert seen == {"regime": "2level", "n_draws": 2048, "max_em": 100}
    assert [(r["subject_id"], r["pool"]) for r in out["subjects"]] == [("a", "train"), ("b", "train"), ("c", "heldout")]
    assert out["mevolve_git_sha"] == "f" * 40 and out["n_not_converged"] == 1 and len(out["program_sha256"]) == 64


def test_score_cli_rejects_a_nonpositive_em_cap_or_draw_count(tmp_path):
    import pytest
    for flag in ("--max-em", "--n-draws"):
        for value in ("0", "-3"):
            with pytest.raises(SystemExit):
                score.main(["--freeze", str(tmp_path), "--out", str(tmp_path / "out"), flag, value])
    assert not (tmp_path / "out").exists()

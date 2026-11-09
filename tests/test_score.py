"""The reported-score path: max_em reaches the 2-level fit, and the score CLI's records."""
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_two_level_records_carry_the_per_subject_held_out_score():
    path = program.LITERATURE_DIR / "hattori2019.py"
    d = _subject(0)
    rec, = evaluate.evaluate(path, [d], regime="2level", n_draws=8, n_starts=2, workers=1, max_em=2)
    sub, = evaluate.evaluate(path, [d], regime="subject", n_starts=2, workers=1)
    assert rec["per_subject_log_lik"] == sub["log_lik"]       # the same per-subject fit, scored the same way


def test_program_score_falls_back_per_mouse_beyond_the_tolerance():
    # mouse a: a failed 2-level fit (0.05 per trial below its per-mouse fit); mouse b: a Monte Carlo miss
    # (0.001 per trial below, inside the tolerance); mouse c: a normal fit
    a = {"subject_id": "a", "log_lik": -70.0, "per_subject_log_lik": -65.0, "n_trials": 100}
    b = {"subject_id": "b", "log_lik": -65.1, "per_subject_log_lik": -65.0, "n_trials": 100}
    c = {"subject_id": "c", "log_lik": -55.0, "per_subject_log_lik": -64.0, "n_trials": 100}
    s = evaluate.program_score([a, b, c])
    assert s["score"] == pytest.approx(np.exp((-65.0 - 65.1 - 55.0) / 300))      # a falls back, b and c keep 2-level
    assert s["n_fallback"] == 1 and s["fallback_subjects"] == ["a"]
    assert s["two_level"] == pytest.approx(np.exp((-70.0 - 65.1 - 55.0) / 300))  # the raw pools are unchanged
    assert s["per_subject"] == pytest.approx(np.exp((-65.0 - 65.0 - 64.0) / 300))
    assert not s["two_level_failed"]                    # the pooled flag keeps its meaning


def test_program_score_falls_back_for_a_non_finite_mouse():
    a = {"subject_id": "a", "log_lik": float("nan"), "per_subject_log_lik": -65.0, "n_trials": 100}
    c = {"subject_id": "c", "log_lik": -60.0, "per_subject_log_lik": -64.0, "n_trials": 100}
    s = evaluate.program_score([a, c])
    assert s["score"] == pytest.approx(np.exp((-65.0 - 60.0) / 200)) and s["fallback_subjects"] == ["a"]
    assert not s["two_level_failed"] and not np.isfinite(s["two_level"])   # one NaN mouse does not fail the program


def test_program_score_keeps_the_per_subject_score_when_the_2_level_fit_is_lower_everywhere():
    # every mouse a little below its per-mouse fit (inside the tolerance): a systematic failure, not noise
    recs = [{"subject_id": k, "log_lik": -65.3, "per_subject_log_lik": -65.0, "n_trials": 100} for k in "abc"]
    s = evaluate.program_score(recs)
    assert s["two_level_failed"] and s["n_fallback"] == 0 and s["score"] == s["per_subject"]


def test_program_score_falls_back_when_the_two_level_fit_scores_lower():
    good = [{"log_lik": -60.0, "per_subject_log_lik": -65.0, "n_trials": 100}]
    bad = [{"log_lik": -80.0, "per_subject_log_lik": -65.0, "n_trials": 100}]
    g, b = evaluate.program_score(good), evaluate.program_score(bad)
    assert g["score"] == g["two_level"] and not g["two_level_failed"]
    assert b["score"] == b["per_subject"] == pytest.approx(np.exp(-0.65)) and b["two_level_failed"]
    nan = evaluate.program_score([{"log_lik": float("nan"), "per_subject_log_lik": -65.0, "n_trials": 100}])
    assert nan["two_level_failed"] and nan["score"] == pytest.approx(np.exp(-0.65))

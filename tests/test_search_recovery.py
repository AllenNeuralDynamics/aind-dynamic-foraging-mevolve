"""Gate 4's second half: simulated subjects reach the search, the library can leave programs out,
and the pre-check's rule is the one pre-registered."""
import gzip
import json

import numpy as np
import pytest

from mevolve import gate4
from mevolve.search import precheck, prompt


def saved(path, n_subjects=2):
    subjects = [{"subject_id": f"sim-{i}", "mu": [0.0], "log_sigma": [0.0],
                 "sessions": [{"session_id": f"sim-s{i}-{k}", "index": k, "role": "fit" if k % 2 == 0 else "score",
                               "stage": {3: None, 1: "None"}.get(k, "STAGE_FINAL"), "params": {"a": 0.5},
                               "choices": [0, 1, 1], "rewards": [1.0, 0.0, 1.0]} for k in (3, 0, 2, 1)]}
                for i in range(n_subjects)]
    with gzip.open(path, "wt") as f:
        json.dump({"model": "Hattori2019", "params": ["a"], "seed": 0, "subjects": subjects}, f)
    return path


def test_simulated_subjects_load_as_the_evaluator_takes_them(tmp_path):
    data = gate4.load_simulated(saved(tmp_path / "s.json.gz"))
    assert [d.subject_id for d in data] == ["sim-0", "sim-1"]
    fit, score = data[0].fit, data[0].score
    assert [s.index for s in fit] == [0, 2] and [s.index for s in score] == [1, 3]
    assert all(s.task == gate4.SIM_TASK for s in fit + score)
    assert score[0].stage is None and score[1].stage is None          # the string "None" and null alike
    assert [s.stage_group for s in score] == ["unknown", "unknown"] and fit[0].stage_group == "final"
    assert fit[0].choices.dtype == np.int32 and fit[0].rewards.dtype == np.float64


@pytest.mark.parametrize("value, stage", [(None, None), (float("nan"), None), ("None", None), ("", None),
                                          ("STAGE_2", "STAGE_2"), ("GRADUATED", "GRADUATED")])
def test_a_missing_stage_is_none_however_the_freeze_spells_it(value, stage):
    from mevolve.data import stage_or_none
    assert stage_or_none(value) == stage


def test_other_cohorts_get_their_own_random_stream_and_generation_keeps_its_own():
    first_half = np.random.default_rng([0, gate4.MODELS.index("Hattori2019")]).random(3)
    assert np.array_equal(gate4.stream(0, "Hattori2019", "generation", 0).random(3), first_half)
    validation = gate4.stream(0, "Hattori2019", "validation", 0).random(3)
    held_out = gate4.stream(0, "Hattori2019", "validation", 24).random(3)
    assert not np.array_equal(validation, first_half) and not np.array_equal(validation, held_out)


def test_the_library_can_leave_out_the_generating_program():
    text = prompt.task_message(exclude=("Hattori2019",))
    assert "## Hattori2019" not in text and "## Bari2019" in text
    assert "## Hattori2019" in prompt.task_message()
    with pytest.raises(ValueError, match="not literature models"):
        prompt.library(("Hattori2020",))


def scores(**kw):
    base = {"template": 70.0, "Hattori2019": 73.0, "Miller-RHG": 72.0, "Bari2019": 72.5}
    base.update(kw)
    return {k: {"score": v, "two_level_failed": False} for k, v in base.items()}


SEEDS = ["Miller-RHG"]


def test_precheck_leaves_out_the_target_and_programs_within_twice_the_tolerance():
    out = precheck.decide(scores(Bari2019=72.7), "Hattori2019", SEEDS)
    assert out["library_exclude"] == ["Bari2019", "Hattori2019"] and out["ok"]
    out = precheck.decide(scores(), "Hattori2019", SEEDS)       # Bari 0.5 below: kept
    assert out["library_exclude"] == ["Hattori2019"] and out["ok"] and out["margins"]["Bari2019"] == 0.5


@pytest.mark.parametrize("change, reason", [
    ({"Miller-RHG": 72.9}, "seeds"),
    ({"template": 72.8}, "template"),
])
def test_precheck_stops_when_the_rule_would_change_the_design(change, reason):
    out = precheck.decide(scores(**change), "Hattori2019", SEEDS)
    assert not out["ok"] and any(reason in r for r in out["stop"])


def test_precheck_stops_when_the_target_has_no_2_level_score():
    s = scores()
    s["Hattori2019"]["two_level_failed"] = True
    assert not precheck.decide(s, "Hattori2019", SEEDS)["ok"]


def test_precheck_stops_when_too_many_programs_are_close():
    close = {f"P{i}": 72.9 for i in range(precheck.MAX_EXCLUDED + 1)}
    out = precheck.decide(scores(**close), "Hattori2019", SEEDS)
    assert not out["ok"] and any("leave out" in r for r in out["stop"])


def test_a_simulated_run_needs_no_freeze_and_passes_its_files_to_the_eval_script(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    from mevolve.search import run

    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({"results_dir": str(tmp_path / "r"), "out_dir": str(tmp_path / "o"),
                               "simulated": {"generation": "$SIM/g.json.gz", "validation": "$SIM/v.json.gz"},
                               "library_exclude": ["Hattori2019"], "workers": 1, "max_evaluation_jobs": 1}))
    monkeypatch.setenv("SIM", str(tmp_path))
    config = run.load_config(cfg)
    (tmp_path / "r").mkdir()
    assert config["simulated"]["validation"] == str(tmp_path / "v.json.gz") and config["freeze"] is None

    import shinka.launch
    seen = {}

    class Stop(Exception):
        pass

    def record(**kw):
        seen.update(kw)
        raise Stop
    monkeypatch.setattr(shinka.launch, "LocalJobConfig", record)
    with pytest.raises(Stop):
        run.build_runner(config, "TASK")
    args = seen["extra_cmd_args"]
    assert args["simulated_validation"] == str(tmp_path / "v.json.gz") and "freeze" not in args


@pytest.mark.parametrize("extra", [[], ["--simulated_generation", "g.json.gz"],
                                   ["--simulated_validation", "v.json.gz", "--freeze", "f", "--cohorts", "c",
                                    "--generation", "g", "--validation", "v"],
                                   ["--simulated_generation", "g", "--simulated_validation", "v", "--freeze", "f"]])
def test_the_eval_script_needs_the_freeze_or_both_simulated_files(tmp_path, extra):
    from mevolve.search import shinka_eval

    with pytest.raises(SystemExit):
        shinka_eval.main(["--program_path", str(tmp_path / "m.py"), "--results_dir", str(tmp_path / "r"),
                          "--workers", "1", *extra])


def test_precheck_stops_when_the_target_is_a_seed():
    out = precheck.decide(scores(), "Hattori2019", SEEDS + ["Hattori2019"])
    assert not out["ok"] and any("seeds" in r and "Hattori2019" in r for r in out["stop"])


def test_precheck_compares_unrounded_margins():
    out = precheck.decide(scores(Bari2019=72.60004), "Hattori2019", SEEDS)     # 0.39996 below: within 0.4
    assert "Bari2019" in out["library_exclude"]


def test_a_target_outside_the_library_is_never_in_library_exclude():
    s = scores(Bari2019=72.7)
    s["NewProgram"] = s.pop("Hattori2019")
    out = precheck.decide(s, "NewProgram", SEEDS, target_in_library=False)
    assert out["library_exclude"] == ["Bari2019"] and out["ok"]


def test_a_new_generating_program_gets_its_own_stream():
    a = gate4.stream(0, "hattori_satiety", "generation", 0).random(3)
    assert np.array_equal(a, gate4.stream(0, "hattori_satiety", "generation", 0).random(3))
    assert not np.array_equal(a, gate4.stream(0, "Hattori2019", "generation", 0).random(3))


def test_truths_come_from_the_programs_own_fit(tmp_path):
    from mevolve import program
    from mevolve.search import seeds

    path = tmp_path / "gen.py"
    path.write_text(seeds.TEMPLATE)
    prog = program.load(path)
    fit = tmp_path / "truths.json"
    fit.write_text(json.dumps({"subjects": {"m1": {"typical_params": {"side_bias": 0.0, "stickiness": 1.0},
                                                   "log_sigma": [-1.0, -2.0], "converged": True}}}))
    true = gate4.truths_from_fit(prog, fit, ["m1"])
    import hashlib
    saved = json.loads(fit.read_text())
    saved["program_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    fit.write_text(json.dumps(saved))
    gate4.truths_from_fit(prog, fit, ["m1"], path)                  # the same program: accepted
    path.write_text(seeds.TEMPLATE.replace("side_bias", "bias_side"))
    with pytest.raises(ValueError, match="different program"):
        gate4.truths_from_fit(prog, fit, ["m1"], path)
    assert true["m1"][1].tolist() == [-1.0, -2.0] and true["m1"][0].shape == (2,)
    with pytest.raises(ValueError, match="no fit"):
        gate4.truths_from_fit(prog, fit, ["m1", "m2"])


@pytest.mark.parametrize("extra", [["--program-path", "g.py"], ["--program-path", "g.py", "--model", "Hattori2019",
                                                                 "--truths", "t.json"],
                                   ["--truths", "t.json", "--simulate-only"], ["--fit-truths"]])
def test_gate4_refuses_a_program_path_without_truths_or_with_a_model(tmp_path, extra):
    with pytest.raises(SystemExit):
        gate4.main(["--freeze", str(tmp_path), "--cohorts", str(tmp_path / "c.json"), "--studies", str(tmp_path),
                    "--out", str(tmp_path / "o"), *extra])


def test_precheck_refuses_a_target_file_named_like_the_template_or_a_library_program(tmp_path, monkeypatch):
    for name in ("template", "Bari2019"):
        target = tmp_path / f"{name}.py"
        target.write_text("x = 1\n")
        cfg = tmp_path / "run.json"
        cfg.write_text(json.dumps({"results_dir": str(tmp_path / "r"), "out_dir": str(tmp_path / "o"),
                                   "simulated": {"generation": "g", "validation": "v"},
                                   "workers": 1, "max_evaluation_jobs": 1}))
        with pytest.raises(SystemExit):
            precheck.main(["--config", str(cfg), "--target", str(target), "--out", str(tmp_path / "p.json")])


def test_precheck_stops_when_the_new_mechanism_adds_too_little():
    s = scores(Bari2019=70.0)
    s["Sat"] = {"score": 73.1, "two_level_failed": False}      # the target: Hattori2019 plus a mechanism
    out = precheck.decide(s, "Sat", SEEDS, target_in_library=False, reference="Hattori2019")
    assert not out["ok"] and any("without its mechanism" in r for r in out["stop"])
    s["Sat"]["score"] = 73.5
    assert precheck.decide(s, "Sat", SEEDS, target_in_library=False, reference="Hattori2019")["ok"]

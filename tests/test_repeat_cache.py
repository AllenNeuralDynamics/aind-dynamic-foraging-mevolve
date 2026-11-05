"""The repeat cache (#52): a program whose code repeats an earlier correct one reuses its result."""
import json

import pytest

from mevolve.search import shinka_eval
from mevolve.search.seeds import TEMPLATE

BASE = '''"""Q-learning with a side bias."""
import jax.numpy as jnp

PARAMS = {"learn_rate": (0.0, 1.0), "beta": (0.0, 20.0)}   # name: (low, high)
N_LATENT = 2
LATENT_NAMES = ("q_left", "q_right")


def update(latents, choice, reward, p):
    return latents + p["learn_rate"] * (reward - latents)
'''


@pytest.mark.parametrize("edit", [
    lambda s: s.replace("# name: (low, high)", "# bounds"),                          # a comment
    lambda s: s.replace('"""Q-learning with a side bias."""', '"""Another description."""'),   # the docstring
    lambda s: s.replace("(0.0, 1.0)", "( 0.0 ,1.0 )").replace("\n\n\n", "\n\n"),    # formatting
])
def test_comments_docstrings_and_formatting_do_not_change_the_key(edit):
    assert shinka_eval.code_key(edit(BASE)) == shinka_eval.code_key(BASE)


@pytest.mark.parametrize("edit", [
    lambda s: s.replace("(0.0, 1.0)", "(0.0, 0.5)"),                                  # a bound
    lambda s: s.replace("learn_rate", "alpha"),                                       # a name
    lambda s: s.replace("(reward - latents)", "(reward - 0.5 * latents)"),            # the mechanism
])
def test_any_change_to_the_code_changes_the_key(edit):
    assert shinka_eval.code_key(edit(BASE)) != shinka_eval.code_key(BASE)


def test_code_that_does_not_parse_has_no_key():
    assert shinka_eval.code_key("def f(:\n") is None


def run(tmp_path, name, source, monkeypatch, calls, fail=False):
    program = tmp_path / name / "main.py"
    program.parent.mkdir()
    program.write_text(source)

    def fake(*args, **kw):
        calls.append(name)
        if fail:
            raise RuntimeError("fit diverged")
        return {"combined_score": 72.5, "public": {"validation_score": 72.5}, "private": {"seconds": 270.0},
                "text_feedback": "feedback"}
    monkeypatch.setattr(shinka_eval, "evaluate_program", fake)
    for f in ("g.json.gz", "v.json.gz"):
        (tmp_path / f).write_bytes(b"subjects")
    shinka_eval.main(["--program_path", str(program), "--results_dir", str(program.parent / "results"),
                      "--simulated_generation", str(tmp_path / "g.json.gz"),
                      "--simulated_validation", str(tmp_path / "v.json.gz"),
                      "--workers", "1", "--repeat_cache", str(tmp_path / "cache")])
    return json.loads((program.parent / "results" / "metrics.json").read_text())


def test_a_repeat_reuses_the_result_without_a_fit(tmp_path, monkeypatch):
    calls = []
    first = run(tmp_path, "gen_1", TEMPLATE, monkeypatch, calls)
    again = run(tmp_path, "gen_2", TEMPLATE + "# a new comment\n", monkeypatch, calls)
    assert calls == ["gen_1"]                                   # the repeat was not fitted
    assert again["public"] == first["public"] and again["text_feedback"] == first["text_feedback"]
    assert again["private"]["repeat_of"].endswith("gen_1/main.py") and "repeat_of" not in first["private"]
    run(tmp_path, "gen_3", TEMPLATE.replace("(-5.0, 5.0)", "(-4.0, 4.0)", 1), monkeypatch, calls)
    assert calls == ["gen_1", "gen_3"]                          # a changed bound is fitted


def test_a_failure_is_not_stored(tmp_path, monkeypatch):
    calls = []
    run(tmp_path, "gen_1", TEMPLATE, monkeypatch, calls, fail=True)
    assert not list(tmp_path.glob("cache/*.json"))
    run(tmp_path, "gen_2", TEMPLATE, monkeypatch, calls)
    assert calls == ["gen_1", "gen_2"]                          # evaluated again


def test_a_repeat_still_passes_the_static_checks(tmp_path, monkeypatch):
    calls = []
    run(tmp_path, "gen_1", TEMPLATE, monkeypatch, calls)
    monkeypatch.setattr("mevolve.search.checks.check_source", lambda s: (_ for _ in ()).throw(
        __import__("mevolve.search.checks", fromlist=["ProgramRejected"]).ProgramRejected("forbidden")))
    program = tmp_path / "gen_2" / "main.py"
    out = run(tmp_path, "gen_2", TEMPLATE, monkeypatch, calls)
    assert calls == ["gen_1"] and out["combined_score"] == 0.0
    assert "static checks" in json.loads((program.parent / "results" / "correct.json").read_text())["error"]


def test_an_unreadable_cache_entry_is_a_miss(tmp_path, monkeypatch):
    calls = []
    run(tmp_path, "gen_1", TEMPLATE, monkeypatch, calls)
    for entry in (tmp_path / "cache").glob("*.json"):
        entry.write_text("{not json")
    out = run(tmp_path, "gen_2", TEMPLATE, monkeypatch, calls)
    assert calls == ["gen_1", "gen_2"] and out["combined_score"] == 72.5


def test_a_missing_data_file_still_writes_the_result_files(tmp_path, monkeypatch):
    program = tmp_path / "gen_1" / "main.py"
    program.parent.mkdir()
    program.write_text(TEMPLATE)
    shinka_eval.main(["--program_path", str(program), "--results_dir", str(program.parent / "results"),
                      "--simulated_generation", str(tmp_path / "missing.json.gz"),
                      "--simulated_validation", str(tmp_path / "missing.json.gz"),
                      "--workers", "1", "--repeat_cache", str(tmp_path / "cache")])
    assert not json.loads((program.parent / "results" / "correct.json").read_text())["correct"]


def test_the_run_passes_the_cache_to_the_evaluation(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    import shinka.launch

    from mevolve.search import run as search_run

    assert search_run.DEFAULTS["repeat_cache"] is True
    seen = {}

    class Stop(Exception):
        pass

    def record(**kw):
        seen.update(kw)
        raise Stop
    monkeypatch.setattr(shinka.launch, "LocalJobConfig", record)
    config = {**search_run.DEFAULTS, "results_dir": str(tmp_path), "max_usd": 1.0, "simulated": {"generation": "g", "validation": "v"}}
    with pytest.raises(Stop):
        search_run.build_runner(config, "TASK")
    assert seen["extra_cmd_args"]["repeat_cache"] == str(tmp_path / "repeat_cache")
    seen.clear()
    with pytest.raises(Stop):
        search_run.build_runner({**config, "repeat_cache": False}, "TASK")
    assert "repeat_cache" not in seen["extra_cmd_args"]

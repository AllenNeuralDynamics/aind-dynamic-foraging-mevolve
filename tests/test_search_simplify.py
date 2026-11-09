"""Simplification mode (ADR 0006): Halstead effort, the hard floor, its feedback, config and stopping rule."""
import json
import math

import pytest

from mevolve.search import simplify, stopping

PROGRAM = '''"""A side bias plus the last choice; no learning."""
import jax.numpy as jnp

PARAMS = {"bias": (-5.0, 5.0)}      # a comment
N_LATENT = 1
LATENT_NAMES = ("last",)


def init_latents(p, session):
    """Start at zero."""
    return jnp.zeros(1)


def predict(latents, p):
    return jnp.array([0.0, p["bias"] + latents[0]])


def update(latents, choice, reward, p):
    return jnp.array([2.0 * choice - 1.0])
'''


def bare(source: str) -> str:
    """``source`` without its docstrings and comments, by hand."""
    lines = [line.split("#")[0].rstrip() for line in source.splitlines()]
    return "\n".join(line for line in lines if '"""' not in line) + "\n"


def raw_metrics(score=72.3456, n_params=1, n_latent=1):
    return {"combined_score": score,
            "public": {"validation_score": round(score, 3), "generation_score": 71.0, "two_level_failed": False,
                       "validation_fallback_mice": 0, "n_params": n_params, "n_latent": n_latent},
            "private": {"max_em": 20}, "text_feedback": "Validation score 72.346\nGeneration mice: score 71.000"}


# Halstead effort

def test_effort_is_radons_total_effort_of_the_module():
    from radon.metrics import h_visit

    report = h_visit(PROGRAM).total
    assert (report.h1, report.N1) == (4, 4)          # unary -, +, *, binary -
    assert simplify.halstead_effort(PROGRAM) == pytest.approx(report.effort)
    assert simplify.halstead_effort(PROGRAM) == pytest.approx(simplify.halstead_effort(PROGRAM))


def test_docstrings_comments_and_the_evolve_block_do_not_count():
    from mevolve.search.seeds import wrap

    effort = simplify.halstead_effort(PROGRAM)
    assert simplify.halstead_effort(bare(PROGRAM)) == pytest.approx(effort)
    assert simplify.halstead_effort(wrap(PROGRAM)) == pytest.approx(effort)
    noisy = PROGRAM.replace("    return jnp.zeros(1)", '    "a string statement"\n    return jnp.zeros(1)  # zero')
    assert simplify.halstead_effort(noisy) == pytest.approx(effort)


def test_more_operations_cost_more_effort():
    more = PROGRAM.replace('p["bias"] + latents[0]', 'p["bias"] + 0.5 * latents[0] - 0.1')
    assert simplify.halstead_effort(more) > simplify.halstead_effort(PROGRAM)


def test_the_template_is_cheaper_than_a_literature_seed():
    from mevolve.search import seeds

    assert 0 < simplify.halstead_effort(seeds.TEMPLATE) < simplify.halstead_effort(seeds.literature_source("Miller-RHG"))


# The floor

def test_at_or_above_the_floor_is_correct_and_the_score_is_minus_effort():
    effort = simplify.halstead_effort(PROGRAM)
    for floor in (72.3456, 72.0):
        metrics, below = simplify.apply_floor(raw_metrics(72.3456), PROGRAM, floor, seed_effort=2 * effort)
        assert below is None
        assert metrics["combined_score"] == pytest.approx(-effort) and metrics["combined_score"] < 0
        assert metrics["public"]["validation_score"] == 72.346          # the score is still reported
        assert metrics["public"]["halstead_effort"] == pytest.approx(effort, abs=0.05)
        assert metrics["public"]["floor"] == floor


def test_below_the_floor_is_incorrect_but_keeps_its_metrics():
    metrics, below = simplify.apply_floor(raw_metrics(72.0), PROGRAM, 72.1, seed_effort=500.0)
    assert below and "below the floor" in below and "72.100" in below
    assert metrics["combined_score"] == pytest.approx(-simplify.halstead_effort(PROGRAM))
    assert metrics["public"]["slack"] == pytest.approx(-0.1)


def test_a_score_that_is_not_finite_is_left_to_the_failure_path():
    metrics, below = simplify.apply_floor(raw_metrics(float("nan")), PROGRAM, 72.0, seed_effort=500.0)
    assert below is None and math.isnan(metrics["combined_score"])


def test_feedback_holds_score_floor_slack_efforts_and_counts():
    effort = simplify.halstead_effort(PROGRAM)
    metrics, _ = simplify.apply_floor(raw_metrics(72.3456, n_params=1, n_latent=1), PROGRAM, 72.1, seed_effort=1234.5)
    text = metrics["text_feedback"]
    assert "validation score 72.346" in text and "floor 72.100" in text and "slack +0.246" in text
    assert f"Halstead effort {effort:.1f}" in text and "seed program 1234.5" in text
    assert "1 parameter" in text and "1 latent variable" in text
    assert text.endswith(raw_metrics()["text_feedback"])                # then the usual diagnostics
    metrics, _ = simplify.apply_floor(raw_metrics(72.0), PROGRAM, 72.1, seed_effort=1234.5)
    assert "slack -0.100" in metrics["text_feedback"] and "BELOW the floor" in metrics["text_feedback"]


def test_a_program_below_the_floor_is_written_incorrect_with_its_metrics(tmp_path):
    from mevolve.search import shinka_eval

    metrics, below = simplify.apply_floor(raw_metrics(72.0), PROGRAM, 72.1, seed_effort=500.0)
    assert not shinka_eval.write_results(tmp_path, metrics, below, keep_metrics=True)
    assert json.loads((tmp_path / "correct.json").read_text()) == {"correct": False, "error": below}
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["combined_score"] == pytest.approx(metrics["combined_score"]) and "BELOW" in saved["text_feedback"]


def test_the_eval_script_applies_the_floor_and_caches_the_search_metrics(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    from mevolve.search import shinka_eval

    for key in shinka_eval.SECRETS:          # main removes them from the environment: restore after
        monkeypatch.delenv(key, raising=False)
    program = tmp_path / "main.py"
    program.write_text(PROGRAM)
    calls = []

    def fake_evaluate(*a, **k):
        calls.append(a)
        return raw_metrics(72.0)
    monkeypatch.setattr(shinka_eval, "evaluate_program", fake_evaluate)
    monkeypatch.setattr(shinka_eval, "repeat_key", lambda *a, **k: "k")
    sim = []
    for name in ("g", "v"):
        (tmp_path / f"{name}.json.gz").write_bytes(b"x")
        sim += [f"--simulated_{'generation' if name == 'g' else 'validation'}", str(tmp_path / f"{name}.json.gz")]
    base = ["--program_path", str(program), "--workers", "1", "--repeat_cache", str(tmp_path / "cache"), *sim]

    shinka_eval.main([*base, "--results_dir", str(tmp_path / "a"), "--floor", "72.1", "--seed_effort", "500"])
    assert json.loads((tmp_path / "a" / "correct.json").read_text())["correct"] is False     # below 72.1
    cached = json.loads((tmp_path / "cache" / "k.json").read_text())["metrics"]
    assert cached["combined_score"] == 72.0                     # the search's metrics, whatever the floor

    shinka_eval.main([*base, "--results_dir", str(tmp_path / "b"), "--floor", "71.9", "--seed_effort", "500"])
    assert len(calls) == 1                                      # a repeat: not refitted
    assert json.loads((tmp_path / "b" / "correct.json").read_text())["correct"] is True
    saved = json.loads((tmp_path / "b" / "metrics.json").read_text())
    assert saved["combined_score"] == pytest.approx(-simplify.halstead_effort(PROGRAM))

    shinka_eval.main([*base, "--results_dir", str(tmp_path / "c")])          # search mode reads the same entry
    assert json.loads((tmp_path / "c" / "metrics.json").read_text())["combined_score"] == 72.0


def test_the_eval_script_needs_a_seed_effort_with_a_floor(tmp_path, monkeypatch):
    from mevolve.search import shinka_eval

    for key in shinka_eval.SECRETS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit):
        shinka_eval.main(["--program_path", "p.py", "--results_dir", str(tmp_path), "--workers", "1",
                          "--simulated_generation", "g", "--simulated_validation", "v", "--floor", "72"])


# The stopping rule: the search's rule on -effort, with se = 0

def test_a_simplify_run_stops_after_the_window_without_a_lower_effort():
    anchor = -1000.0                                    # the seed program's effort is 1000
    finished = [-1000.0, None, -1200.0, -999.5, None, -999.5, -999.6]
    state = stopping.plateau(anchor, finished, 0.0, 4)  # ties, failures, higher efforts: no rise
    assert state["rises"] == [{"proposal": 4, "score": -999.5}]
    assert not state["met"] and state["since_rise"] == 3
    state = stopping.plateau(anchor, [*finished, -999.5], 0.0, 4)
    assert state["met"] and state["anchor"] == -999.5


def test_the_simplify_rule_reads_correct_programs_only(tmp_path):
    from test_search_stopping import make_db

    db = tmp_path / "programs.sqlite"
    make_db(db, [-1000.0], [(1, 1, None), (2, 2, -900.0), (3, 3, -950.0)], [(4, 4)])   # 1: below floor or failed
    state = stopping.from_database(db, 0.0, 2)
    assert state["met"] and state["anchor"] == -900.0 and state["since_rise"] == 2


# Config

BASE = {"results_dir": "/r", "out_dir": "/o", "freeze": "/f", "cohorts": "/c", "generation": "g", "validation": "v",
        "workers": 1, "max_evaluation_jobs": 1}


def load(tmp_path, **extra):
    from mevolve.search import run

    path = tmp_path / "run.json"
    path.write_text(json.dumps({**BASE, **extra}))
    return run.load_config(path)


def test_search_mode_is_the_default(tmp_path):
    pytest.importorskip("shinka")
    config = load(tmp_path)
    assert config["mode"] == "search" and config["floor"] is None and config["seed_programs"] == []
    assert config["library_in_prompt"] is True and config["stop_window_simplify"] == 300


@pytest.mark.parametrize("extra, message", [
    ({"mode": "simplfy"}, "mode"),
    ({"mode": "simplify", "seed_programs": ["/a.py"]}, "floor"),
    ({"mode": "simplify", "floor": 72.1}, "seed_programs"),
    ({"mode": "simplify", "floor": 72.1, "seed_programs": []}, "seed_programs"),
    ({"mode": "simplify", "floor": float("inf"), "seed_programs": ["/a.py"]}, "floor"),
    ({"mode": "simplify", "floor": True, "seed_programs": ["/a.py"]}, "floor"),
    ({"mode": "simplify", "floor": 72.1, "seed_programs": ["Miller-RHG"]}, "program file"),
    ({"mode": "simplify", "floor": 72.1, "seed_programs": ["/a/p.py", "/b/p.py"]}, "distinct"),
    ({"mode": "simplify", "floor": 72.1, "seed_programs": ["/a.py"], "seeds": ["Miller-RHG"]}, "seed_programs"),
    ({"mode": "simplify", "floor": 72.1, "seed_programs": ["/a.py"], "stop_se": 0.05, "stop_window": 9}, "stop_window_simplify"),
    ({"mode": "simplify", "floor": 72.1, "seed_programs": ["/a.py"], "stop_window_simplify": 0}, "stop_window_simplify"),
    ({"floor": 72.1}, "simplify mode"),
    ({"seed_programs": ["/a.py"]}, "simplify mode"),
])
def test_config_refuses_a_simplify_run_without_its_settings(tmp_path, extra, message):
    pytest.importorskip("shinka")
    with pytest.raises(ValueError, match=message):
        load(tmp_path, **extra)


def test_a_simplify_config_expands_its_programs_and_leaves_the_library_out(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    monkeypatch.setenv("BEST", str(tmp_path))
    config = load(tmp_path, mode="simplify", floor=72.1, seed_programs=["$BEST/best.py"])
    assert config["seed_programs"] == [str(tmp_path / "best.py")] and config["seeds"] == []
    assert config["library_in_prompt"] is False
    assert load(tmp_path, mode="simplify", floor=72.1, seed_programs=["/a.py"], library_in_prompt=True)["library_in_prompt"]


# The prompt

def test_the_simplify_task_names_the_floor_and_keeps_the_interface_and_limits():
    from mevolve.search import prompt

    text = prompt.task_message(with_library=False, max_params=12, max_latent=6, floor=72.123)
    assert "simpler" in text and "72.123" in text and "Halstead effort" in text
    assert "At most 12 parameters and 6 latent variables" in text and "; at most 12" in text
    assert "# The interface" in text and "predicts better" not in text
    assert prompt.task_message(with_library=False) == prompt.TASK                # the search's is unchanged


# The runner's wiring

def write_programs(tmp_path, n):
    from mevolve.search import seeds

    paths = []
    for i in range(n):
        path = tmp_path / f"best{i}.py"
        source = seeds.TEMPLATE if i == 0 else PROGRAM
        path.write_text(source)
        paths.append(str(path))
    return paths


def test_the_runner_starts_from_the_first_program_and_passes_the_floor(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    import shinka.core
    import shinka.database
    import shinka.launch

    from mevolve.search import run, seeds

    seen = {}

    def record(name):
        def f(*a, **kw):
            seen[name] = kw
            return kw
        return f
    monkeypatch.setattr(shinka.core, "EvolutionConfig", record("evo"))
    monkeypatch.setattr(shinka.launch, "LocalJobConfig", record("job"))
    monkeypatch.setattr(shinka.database, "DatabaseConfig", record("db"))

    class Runner:
        def __init__(self, *a, **kw):
            pass
    monkeypatch.setattr(shinka.core, "ShinkaEvolveRunner", Runner)
    programs = write_programs(tmp_path, 3)
    config = {**run.DEFAULTS, "results_dir": str(tmp_path), "freeze": "/f", "cohorts": "/c", "generation": "g",
              "validation": "v", "mode": "simplify", "floor": 72.1, "seed_programs": programs, "seeds": [],
              "stop_window_simplify": 50}
    runner, _ = run.build_runner(config, "TASK")
    assert (tmp_path / "initial.py").read_text() == seeds.wrap(seeds.TEMPLATE)        # the first program, island 0
    assert seen["db"]["num_islands"] == 3
    efforts = [simplify.halstead_effort(seeds.TEMPLATE), simplify.halstead_effort(PROGRAM)]
    assert seen["job"]["extra_cmd_args"]["floor"] == 72.1
    assert seen["job"]["extra_cmd_args"]["seed_effort"] == pytest.approx(min(efforts))
    assert runner.stop_rule == (tmp_path, 0.0, 50)
    assert [s for s in run.island_seeds(config)] == programs[1:]


def test_a_search_run_keeps_its_template_and_passes_no_floor(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    import shinka.launch

    from mevolve.search import run, seeds

    seen = {}

    class Stop(Exception):
        pass

    def record(**kw):
        seen.update(kw)
        raise Stop
    monkeypatch.setattr(shinka.launch, "LocalJobConfig", record)
    with pytest.raises(Stop):
        run.build_runner({**run.DEFAULTS, "results_dir": str(tmp_path), "freeze": "/f", "cohorts": "/c",
                          "generation": "g", "validation": "v"}, "TASK")
    assert (tmp_path / "initial.py").read_text() == seeds.TEMPLATE and "floor" not in seen["extra_cmd_args"]


def test_a_dry_simplify_run_scores_every_program_and_refuses_one_below_the_floor(tmp_path, monkeypatch):
    """``run.main --dry`` in simplify mode, with the evaluator replaced: no Claude call, no fit, no Slurm."""
    pytest.importorskip("shinka")
    from mevolve.search import claude, run, seeds

    programs = write_programs(tmp_path, 2)
    scores = {"best0": 72.5, "best1": 72.3}
    monkeypatch.setattr(run, "seed_metrics", lambda name, code, config: raw_metrics(scores[name]))
    monkeypatch.setattr(claude, "install", lambda settings: None)       # nothing to call, nothing patched
    monkeypatch.setattr(run.signal, "signal", lambda *a: None)           # pytest keeps its own handlers
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({**BASE, "results_dir": str(tmp_path / "r"), "out_dir": str(tmp_path / "o"),
                               "mode": "simplify", "floor": 72.2, "seed_programs": programs}))
    assert run.main(["--config", str(cfg), "--dry"]) == 0
    islands = seeds.islands_summary(tmp_path / "r")
    assert [p["seed"] for p in islands[1]] == ["best1"]                  # island 0 waits for Shinka's own scoring
    placed = islands[1][0]
    assert placed["correct"] and placed["score"] == pytest.approx(-simplify.halstead_effort(seeds.wrap(PROGRAM)))
    task = json.loads((tmp_path / "r" / "run_config.json").read_text())
    assert task["mode"] == "simplify" and task["library_in_prompt"] is False

    scores["best0"] = 72.1                                                # the first program is below the floor
    cfg.write_text(json.dumps({**BASE, "results_dir": str(tmp_path / "r2"), "out_dir": str(tmp_path / "o2"),
                               "mode": "simplify", "floor": 72.2, "seed_programs": programs}))
    with pytest.raises(SystemExit, match="best0.*below the floor"):
        run.main(["--config", str(cfg), "--dry"])


def test_the_fake_llm_rewrite_keeps_the_effort(tmp_path):
    """The $0 rehearsal (``--fake-llm``) widens bounds: constants change, operations do not, so its proposals
    tie with their parent and never reset the simplify window."""
    import re

    widened = re.sub(r"\((-?[0-9.]+), (-?[0-9.]+)\)", lambda m: f"({float(m.group(1)) - 0.5:.4g}, {float(m.group(2)) + 0.5:.4g})",
                     PROGRAM)
    assert widened != PROGRAM
    assert simplify.halstead_effort(widened) == pytest.approx(simplify.halstead_effort(PROGRAM))

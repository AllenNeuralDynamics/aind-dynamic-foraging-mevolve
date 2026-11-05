"""Seeds land on their islands; the template, added after them, lands on island 0 with no copies."""
import json

import pytest

pytest.importorskip("shinka")

from mevolve.search import checks, prompt, seeds, shinka_eval  # noqa: E402


def metrics(score):
    return {"combined_score": score, "public": {"validation_score": score}, "private": {}, "text_feedback": "f"}


def test_template_and_wrapped_seeds_pass_the_checks():
    assert checks.check_source(seeds.TEMPLATE) == {"n_params": 2, "n_latent": 1}
    code = seeds.literature_source("Miller-RHG")
    assert code.startswith(seeds.START) and code.rstrip().endswith(seeds.END)
    assert seeds.wrap(code) == code
    checks.check_source(code)


def test_seeds_on_their_islands_and_template_on_island_zero(tmp_path):
    from shinka.database import DatabaseConfig, Program, ProgramDatabase

    names = ["Miller-RHG", "Lopez-DoubleTrace", "Lebedeva-PR", "Eckstein-BI"]
    placed = [{"name": n, "code": seeds.literature_source(n), "island": i, "metrics": metrics(72.0 + i)}
              for i, n in enumerate(names, start=1)]
    seeds.place(tmp_path, DatabaseConfig(num_islands=5), placed)
    db = ProgramDatabase(DatabaseConfig(num_islands=5, db_path=str(tmp_path / "programs.sqlite")), embedding_model=None)
    db.add(Program(id="template", code=seeds.TEMPLATE, generation=0, combined_score=70.0, correct=True))
    db.add(Program(id="child", code=seeds.TEMPLATE, parent_id="template", generation=1,
                   combined_score=70.5, correct=True))
    db.close()
    islands = seeds.islands_summary(tmp_path)
    assert sorted(islands) == [0, 1, 2, 3, 4]
    for i, name in enumerate(names, start=1):
        assert [p["seed"] for p in islands[i] if p["generation"] == 0] == [name]
    gen0_island0 = [p for p in islands[0] if p["generation"] == 0]
    assert [p["id"] for p in gen0_island0] == ["template"]          # no copies anywhere
    assert sum(len(v) for v in islands.values()) == 6


def test_place_refuses_island_zero_duplicates_and_reuse(tmp_path):
    from shinka.database import DatabaseConfig

    seed = {"name": "Miller-RHG", "code": seeds.literature_source("Miller-RHG"), "metrics": metrics(72.0)}
    with pytest.raises(ValueError, match="island 0"):
        seeds.place(tmp_path / "a", DatabaseConfig(num_islands=3), [{**seed, "island": 0}])
    with pytest.raises(ValueError, match="distinct"):
        seeds.place(tmp_path / "b", DatabaseConfig(num_islands=3), [{**seed, "island": 1}, {**seed, "island": 1}])
    seeds.place(tmp_path / "c", DatabaseConfig(num_islands=3), [{**seed, "island": 1}])
    with pytest.raises(FileExistsError):
        seeds.place(tmp_path / "c", DatabaseConfig(num_islands=3), [{**seed, "island": 2}])


def test_task_message_holds_the_library_and_shinkas_answer_format():
    text = prompt.task_message()
    assert "DESCRIPTION" in text and "at most 8" in text
    for name in ("Miller-RHG", "Eckstein-BI", "Rescorla-Wagner"):
        assert f"## {name}" in text
    assert prompt.task_message(with_library=False) == prompt.TASK


def test_failed_or_nonfinite_candidates_are_incorrect(tmp_path):
    assert shinka_eval.write_results(tmp_path / "ok", metrics(72.5), None)
    assert not shinka_eval.write_results(tmp_path / "nan", metrics(float("nan")), None)
    assert not shinka_eval.write_results(tmp_path / "err", None, "rejected by the static checks: import of os")
    saved = json.loads((tmp_path / "err" / "metrics.json").read_text())
    assert saved["combined_score"] == 0.0 and "import of os" in saved["text_feedback"]
    assert json.loads((tmp_path / "nan" / "correct.json").read_text())["correct"] is False


def test_eval_script_removes_secrets_before_rejecting(tmp_path, monkeypatch):
    bad = tmp_path / "main.py"
    bad.write_text(seeds.TEMPLATE.replace("import jax.numpy as jnp", "import os"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    shinka_eval.main(["--program_path", str(bad), "--results_dir", str(tmp_path / "r"), "--freeze", str(tmp_path),
                      "--cohorts", str(tmp_path / "c.json"), "--generation", "g", "--validation", "v",
                      "--workers", "1"])
    import os
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "only jax" in json.loads((tmp_path / "r" / "correct.json").read_text())["error"]


def test_a_parameter_added_after_the_literals_is_refused(tmp_path):
    sneaky = seeds.TEMPLATE.replace('N_LATENT = 1', 'PARAMS.update({"extra": (0.0, 1.0)})\nN_LATENT = 1')
    path = tmp_path / "main.py"
    path.write_text(sneaky)
    shinka_eval.main(["--program_path", str(path), "--results_dir", str(tmp_path / "r"), "--freeze", str(tmp_path),
                      "--cohorts", str(tmp_path / "c.json"), "--generation", "g", "--validation", "v",
                      "--workers", "1"])
    error = json.loads((tmp_path / "r" / "correct.json").read_text())["error"]
    assert "3 parameters" in error and "the 2" in error


def test_a_program_with_more_latents_than_declared_is_refused(tmp_path):
    hidden = seeds.TEMPLATE.replace("jnp.zeros(1)", "jnp.zeros(7)")
    assert hidden != seeds.TEMPLATE
    path = tmp_path / "main.py"
    path.write_text(hidden)
    shinka_eval.main(["--program_path", str(path), "--results_dir", str(tmp_path / "r"), "--freeze", str(tmp_path),
                      "--cohorts", str(tmp_path / "c.json"), "--generation", "g", "--validation", "v",
                      "--workers", "1"])
    error = json.loads((tmp_path / "r" / "correct.json").read_text())["error"]
    assert "init_latents returns shape (7,)" in error and "N_LATENT is 1" in error


def test_the_template_and_the_seeds_pass_the_shape_check(tmp_path):
    from mevolve import program
    for name, code in [("template", seeds.TEMPLATE)] + [(n, seeds.literature_source(n)) for n in
                                                       ("Miller-RHG", "Lopez-DoubleTrace", "Lebedeva-PR", "Eckstein-BI")]:
        path = tmp_path / f"{name}.py"
        path.write_text(code)
        shinka_eval.check_shapes(program.load(path))


def test_feedback_after_a_failed_2_level_fit_omits_its_details():
    failed = {"score": 0.73, "two_level_failed": True}
    text = shinka_eval.feedback({"score": 0.73, "two_level_failed": False}, failed, [], {"bias": (-5.0, 5.0)})
    assert "omitted" in text and "by stage and task (score" not in text and "Fitted parameters" not in text


def test_the_prompt_states_the_runs_limits():
    from mevolve.search import prompt

    assert prompt.task_message(with_library=False, max_params=8, max_latent=6) == prompt.TASK
    text = prompt.task_message(with_library=False, max_params=12, max_latent=8)
    assert "At most 12 parameters and 8 latent variables" in text and "at most 8 parameters" not in text.lower()
    assert "; at most 12" in text and "# at most 8" in text

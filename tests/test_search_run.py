"""The launch config: variables from the job expand, unknown keys and unset variables fail."""
import json

import pytest

pytest.importorskip("shinka")

from mevolve.search import run  # noqa: E402

BASE = {"results_dir": "$RUN_TMP/medium", "out_dir": "$RUN_OUT/medium", "freeze": "/f", "cohorts": "$RUN_VARIANT/c.json",
        "generation": "smoke_generation", "validation": "smoke_validation", "workers": 1, "max_evaluation_jobs": 1,
        "wandb": {"group": "engine-smoke@$RUN_ID", "name": "medium"}}


def write(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_job_variables_expand(tmp_path, monkeypatch):
    for k, v in {"RUN_TMP": "/scratch/fast/1", "RUN_OUT": "/out", "RUN_VARIANT": "/v", "RUN_ID": "20261103-010203"}.items():
        monkeypatch.setenv(k, v)
    config = run.load_config(write(tmp_path, BASE))
    assert config["results_dir"] == "/scratch/fast/1/medium" and config["cohorts"] == "/v/c.json"
    assert config["wandb"]["group"] == "engine-smoke@20261103-010203"
    assert config["model"] == "claude-sonnet-5-5" and config["n_proposals"] == 10


def test_unset_variable_and_unknown_key_fail(tmp_path, monkeypatch):
    monkeypatch.delenv("RUN_TMP", raising=False)
    with pytest.raises(ValueError, match="unset variable"):
        run.load_config(write(tmp_path, BASE))
    with pytest.raises(ValueError, match="unknown config keys"):
        run.load_config(write(tmp_path, {**BASE, "results_dir": "/r", "out_dir": "/o", "cohorts": "/c", "wandb": None,
                                         "efort": "high"}))


def test_pricing_snapshot_makes_the_claude_models_known(tmp_path):
    run.write_pricing_snapshot(tmp_path)
    from shinka.pricing.catalog import load_run_pricing_snapshot

    snapshot = load_run_pricing_snapshot(tmp_path)
    assert snapshot is not None
    entry = snapshot.catalog.get_by_name("claude-sonnet-5-5")
    assert entry.input_price == pytest.approx(2e-6) and entry.output_price == pytest.approx(1e-5)


def test_a_lost_copy_fails_a_good_run_but_not_a_failed_one(tmp_path, monkeypatch, capsys):
    import shutil

    from mevolve.search import run

    def full(*a, **k):
        raise OSError("No space left on device")
    monkeypatch.setattr(shutil, "copytree", full)
    with pytest.raises(OSError, match="No space"):
        run.keep_outputs(tmp_path / "results", tmp_path / "out", run_failed=False)
    run.keep_outputs(tmp_path / "results", tmp_path / "out", run_failed=True)     # the run's error wins
    assert "copy to" in capsys.readouterr().err


def test_generations_that_never_ran_count_as_proposals(tmp_path):
    from mevolve.search import report

    for g in (1, 5):
        (tmp_path / f"gen_{g}").mkdir()
        (tmp_path / f"gen_{g}" / "failure.json").write_text("{}")
    (tmp_path / "gen_2").mkdir()            # a generation with a program row has no failure.json
    assert report.failed_generations(tmp_path) == {1, 5}

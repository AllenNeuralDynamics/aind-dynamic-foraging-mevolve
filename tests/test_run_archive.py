"""A search's run directory is archived as a W&B artifact (#54)."""
import json
import sys
import types

import pytest

from mevolve.search import archive


class FakeArtifact:
    def __init__(self, name, type, description, metadata):
        self.name, self.type, self.metadata, self.files = f"{name}:v0", type, metadata, {}
        self.digest = "d1g3st"

    def add_file(self, path, name):
        self.files[name] = path

    def wait(self):
        pass


@pytest.fixture
def fake_wandb(monkeypatch):
    seen = {"logged": []}

    class Run:
        url = "https://wandb.example/run"

        def log_artifact(self, art):
            seen["logged"].append(art)

        def finish(self):
            seen["finished"] = True

    def init(**kw):
        seen["init"] = kw
        return Run()
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init, Artifact=FakeArtifact))
    return seen


def run_dir(tmp_path):
    d = tmp_path / "search"
    for f in ("programs.sqlite", "programs.sqlite-wal", "programs.sqlite-shm", "ledger.jsonl", "gen_1/main.py",
              "wandb/run-x/files/output.log"):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text(f)
    return d


def test_the_run_directory_becomes_one_artifact_with_a_record(tmp_path, fake_wandb):
    d = run_dir(tmp_path)
    record = archive.archive(d, "gate4-recovery@20261104-132600", "NeuralDynamics", "dfbfm-mevolve")
    art = fake_wandb["logged"][0]
    assert art.type == "mevolve-run" and art.name == "run-gate4-recovery-20261104-132600:v0"
    assert sorted(art.files) == ["gen_1/main.py", "ledger.jsonl", "programs.sqlite", "programs.sqlite-wal"]
    assert fake_wandb["init"]["group"] == "gate4-recovery@20261104-132600" and fake_wandb["init"]["job_type"] == "archive"
    assert fake_wandb["init"]["mode"] == "online"           # not an inherited WANDB_MODE
    assert fake_wandb["finished"]
    saved = json.loads((d / "archive.json").read_text())
    assert saved == record and record["digest"] == "d1g3st" and record["n_files"] == 4
    assert set(record["sha256"]) == {"programs.sqlite", "programs.sqlite-wal", "ledger.jsonl"}
    archive.archive(d, "g@1", "e", "p")                         # the record itself is never archived
    assert "archive.json" not in fake_wandb["logged"][1].files


@pytest.mark.parametrize("wandb_cfg, dry", [(None, False), ({"group": "g@1", "mode": "offline"}, False),
                                            ({"group": "g@1", "mode": "disabled"}, False), ({"group": "g@1"}, True),
                                            ({"name": "no group", "mode": "online"}, False)])
def test_no_archive_without_online_wandb_or_on_a_dry_run(tmp_path, fake_wandb, wandb_cfg, dry):
    assert archive.archive_run({"wandb": wandb_cfg}, run_dir(tmp_path), dry) is None
    assert not fake_wandb["logged"]


def test_a_failed_upload_warns_and_keeps_the_run(tmp_path, monkeypatch, capsys):
    def broken(**kw):
        raise ConnectionError("no network")
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=broken, Artifact=FakeArtifact))
    config = {"wandb": {"group": "g@1", "mode": "online", "entity": "E", "project": "my project"}}
    assert archive.archive_run(config, run_dir(tmp_path), False) is None
    err = capsys.readouterr().err       # the retry reproduces the upload: same destination, quoted
    assert f"python -m mevolve.search.archive {tmp_path}/search --group g@1 --entity E --project 'my project'" in err


def test_an_online_run_is_archived_under_its_group(tmp_path, fake_wandb):
    record = archive.archive_run({"wandb": {"group": "g@1", "mode": "online", "project": "p"}}, run_dir(tmp_path), False)
    assert record["artifact"] == "NeuralDynamics/p/run-g-1:v0"


def test_a_failed_search_is_kept_and_archived(tmp_path, fake_wandb):
    pytest.importorskip("shinka")
    from mevolve.search import run

    results = run_dir(tmp_path)
    config = {"out_dir": str(tmp_path / "out"), "wandb": {"group": "g@1", "mode": "online"}}
    run.end_of_run(config, results, run_failed=True, dry=False)
    assert (tmp_path / "out" / "ledger.jsonl").exists() and fake_wandb["logged"]
    assert (tmp_path / "out" / "archive.json").exists()


def test_a_failed_upload_still_finishes_its_wandb_run(tmp_path, fake_wandb, monkeypatch):
    def broken(*a, **k):
        raise OSError("disk")
    monkeypatch.setattr(FakeArtifact, "add_file", broken)
    assert archive.archive_run({"wandb": {"group": "g@1", "mode": "online"}}, run_dir(tmp_path), False) is None
    assert fake_wandb["finished"]


def test_an_inherited_offline_mode_skips_the_archive(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert archive.archive_run({"wandb": {"group": "g@1"}}, run_dir(tmp_path), False) is None
    assert archive.archive_run({"wandb": {"group": "g@1", "mode": "online"}}, run_dir(tmp_path / "b"), False)
    assert len(fake_wandb["logged"]) == 1                  # an explicit online wins


def test_a_missing_directory_is_never_uploaded(tmp_path, fake_wandb):
    with pytest.raises(FileNotFoundError):
        archive.archive(tmp_path / "typo", "g@1", "e", "p")
    assert "init" not in fake_wandb


@pytest.mark.parametrize("run_failed", [True, False])
def test_a_failed_copy_archives_the_complete_original(tmp_path, fake_wandb, monkeypatch, run_failed):
    pytest.importorskip("shinka")
    import shutil

    from mevolve.search import run

    results = run_dir(tmp_path)

    def partial(src, dst, **k):
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "ledger.jsonl").write_text("x")
        raise OSError("No space left on device")
    monkeypatch.setattr(shutil, "copytree", partial)
    config = {"out_dir": str(tmp_path / "out"), "wandb": {"group": "g@1", "mode": "online"}}
    if run_failed:
        run.end_of_run(config, results, run_failed=True, dry=False)       # the run's own error is the one reported
    else:
        with pytest.raises(OSError, match="No space"):                     # a good run fails on its lost copy
            run.end_of_run(config, results, run_failed=False, dry=False)
    assert "programs.sqlite" in fake_wandb["logged"][0].files and (results / "archive.json").exists()

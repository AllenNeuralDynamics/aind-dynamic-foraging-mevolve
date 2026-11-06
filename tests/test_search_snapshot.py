"""Snapshots and resume (#65): a consistent database while the search writes, settled generations only."""
import json
import sqlite3
import threading

import pytest

from mevolve.search import snapshot


def make_run(results_dir, generations=(0, 1, 2), failed=(3,)):
    results_dir.mkdir()
    con = sqlite3.connect(results_dir / "programs.sqlite")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE programs (id INTEGER PRIMARY KEY, generation INTEGER, code TEXT)")
    con.execute("CREATE TABLE attempt_log (generation INTEGER, status TEXT)")
    con.executemany("INSERT INTO programs (generation, code) VALUES (?, 'x')", [(g,) for g in generations])
    con.executemany("INSERT INTO attempt_log VALUES (?, 'failed')", [(g,) for g in failed])
    con.commit()
    for g in (*generations, *failed, 4):                       # generation 4 is still running
        (results_dir / f"gen_{g}" / "results").mkdir(parents=True)
        (results_dir / f"gen_{g}" / "main.py").write_text(f"# {g}\n")
    (results_dir / "ledger.jsonl").write_text('{"cost": 0.5}\n')
    (results_dir / "bandit_state.pkl").write_bytes(b"\x80")
    return con


def test_a_snapshot_holds_the_database_files_and_settled_generations(tmp_path):
    con = make_run(tmp_path / "run")
    record = snapshot.take(tmp_path / "run", tmp_path / "snap")
    con.close()
    snap = tmp_path / "snap"
    assert record["programs"] == 3 and record["generations"] == 4 and record["gen_dirs_copied"] == 4
    assert sorted(p.name for p in snap.glob("gen_*")) == ["gen_0", "gen_1", "gen_2", "gen_3"]   # not the running 4
    assert not list(snap.glob("programs.sqlite-*"))            # one file, no WAL beside it
    assert (snap / "ledger.jsonl").read_text() == '{"cost": 0.5}\n' and (snap / "bandit_state.pkl").exists()
    assert json.loads((snap / "snapshot.json").read_text())["programs"] == 3


def test_a_snapshot_is_consistent_while_the_search_writes(tmp_path):
    con = make_run(tmp_path / "run")
    stop = threading.Event()

    def writer():
        w = sqlite3.connect(tmp_path / "run" / "programs.sqlite", check_same_thread=False)
        while not stop.is_set():
            w.execute("INSERT INTO programs (generation, code) VALUES (5, ?)", ("y" * 10_000,))
            w.commit()
        w.close()

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(5):
            snapshot.take(tmp_path / "run", tmp_path / "snap")
            check = sqlite3.connect(tmp_path / "snap" / "programs.sqlite")
            assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            check.close()
    finally:
        stop.set()
        t.join()
        con.close()


def test_a_later_snapshot_adds_new_generations(tmp_path):
    con = make_run(tmp_path / "run")
    snapshot.take(tmp_path / "run", tmp_path / "snap")
    con.execute("INSERT INTO programs (generation, code) VALUES (4, 'z')")
    con.commit()
    con.close()
    record = snapshot.take(tmp_path / "run", tmp_path / "snap")
    assert record["programs"] == 4 and record["gen_dirs_copied"] == 1 and (tmp_path / "snap" / "gen_4").exists()


def test_restore_fills_a_fresh_results_dir(tmp_path):
    make_run(tmp_path / "run").close()
    snapshot.take(tmp_path / "run", tmp_path / "snap")
    restored = snapshot.restore(tmp_path / "snap", tmp_path / "new")
    assert restored["programs"] == 3 and restored["max_generation"] == 2    # 3 only failed
    con = sqlite3.connect(tmp_path / "new" / "programs.sqlite")
    assert con.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 3
    con.close()
    assert (tmp_path / "new" / "gen_2" / "main.py").exists() and (tmp_path / "new" / "ledger.jsonl").exists()
    assert not (tmp_path / "new" / "gen_3").exists()        # above the last program: Shinka makes it again
    assert not (tmp_path / "new" / "snapshot.json").exists()


def test_restore_refuses_an_incomplete_snapshot(tmp_path):
    (tmp_path / "snap").mkdir()
    with pytest.raises(FileNotFoundError, match="no complete snapshot"):
        snapshot.restore(tmp_path / "snap", tmp_path / "new")


def test_the_snapshotter_takes_a_last_snapshot(tmp_path):
    make_run(tmp_path / "run").close()
    snapshot.Snapshotter(tmp_path / "run", tmp_path / "snap", minutes=60).start().final()
    assert (tmp_path / "snap" / "snapshot.json").exists()


def test_config_checks_the_snapshot_settings(tmp_path):
    pytest.importorskip("shinka")
    from mevolve.search import run

    base = {"results_dir": "/r", "out_dir": str(tmp_path / "out"), "freeze": "/f", "cohorts": "/c",
            "generation": "g", "validation": "v", "workers": 1, "max_evaluation_jobs": 1}
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**base, "snapshot_minutes": 10}))
    with pytest.raises(ValueError, match="both snapshot_minutes and snapshot_dir"):
        run.load_config(path)
    path.write_text(json.dumps({**base, "snapshot_minutes": 10, "snapshot_dir": str(tmp_path / "out" / "snap")}))
    with pytest.raises(ValueError, match="inside out_dir"):
        run.load_config(path)
    for bad in (0, -5):
        path.write_text(json.dumps({**base, "snapshot_minutes": bad, "snapshot_dir": str(tmp_path / "snap")}))
        with pytest.raises(ValueError, match="finite number above 0"):
            run.load_config(path)
    path.write_text(json.dumps({**base, "snapshot_minutes": 10, "snapshot_dir": str(tmp_path / "snap")}))
    assert run.load_config(path)["snapshot_minutes"] == 10


def test_sigterm_takes_the_snapshot_before_the_search_unwinds(monkeypatch):
    pytest.importorskip("shinka")
    from mevolve.search import run

    done = []
    monkeypatch.setattr(run, "_AT_TERM", [lambda: done.append("snapshot")])
    monkeypatch.setattr(run, "_TERMINATING", [])
    with pytest.raises(KeyboardInterrupt):
        run._terminate(15, None)
    run._terminate(15, None)                    # a second signal: ignored, no second snapshot
    assert done == ["snapshot"]


def test_take_now_does_not_wait_on_a_snapshot_under_way(tmp_path):
    make_run(tmp_path / "run").close()
    s = snapshot.Snapshotter(tmp_path / "run", tmp_path / "snap", minutes=60)
    with s._lock:                               # e.g. final() interrupted by the signal in the same thread
        s.take_now()                            # returns at once instead of deadlocking
    assert not (tmp_path / "snap" / "snapshot.json").exists()
    s.take_now()
    assert (tmp_path / "snap" / "snapshot.json").exists()


def test_the_database_is_replaced_last(tmp_path, monkeypatch):
    con = make_run(tmp_path / "run")
    snapshot.take(tmp_path / "run", tmp_path / "snap")
    con.execute("INSERT INTO programs (generation, code) VALUES (4, 'z')")
    con.commit()
    con.close()
    (tmp_path / "run" / "ledger.jsonl").write_text('{"cost": 0.5}\n{"cost": 0.7}\n')
    real = snapshot.shutil.copytree

    def fail(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(snapshot.shutil, "copytree", fail)
    with pytest.raises(OSError):
        snapshot.take(tmp_path / "run", tmp_path / "snap")
    monkeypatch.setattr(snapshot.shutil, "copytree", real)
    check = sqlite3.connect(tmp_path / "snap" / "programs.sqlite")
    assert check.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 3    # the old database
    check.close()
    assert (tmp_path / "snap" / "ledger.jsonl").read_text().count("cost") == 2    # a newer ledger


def test_restore_drops_generation_directories_its_database_does_not_record(tmp_path):
    make_run(tmp_path / "run").close()
    snapshot.take(tmp_path / "run", tmp_path / "snap")
    (tmp_path / "snap" / "gen_7" / "results").mkdir(parents=True)     # copied by a snapshot cut short
    snapshot.restore(tmp_path / "snap", tmp_path / "new")
    assert not (tmp_path / "new" / "gen_7").exists() and not (tmp_path / "snap" / "gen_7").exists()
    assert (tmp_path / "new" / "gen_2").exists()


def test_a_leftover_partial_copy_is_replaced_not_merged(tmp_path):
    make_run(tmp_path / "run").close()
    (tmp_path / "snap" / ".gen_1.part").mkdir(parents=True)
    (tmp_path / "snap" / ".gen_1.part" / "failure.json").write_text("{}")
    snapshot.take(tmp_path / "run", tmp_path / "snap")
    assert not (tmp_path / "snap" / "gen_1" / "failure.json").exists() and (tmp_path / "snap" / "gen_1" / "main.py").exists()

"""The stopping rule (#63): anchored rises above one SE reset the count; every finished proposal counts."""
import json
import sqlite3
import types

import pytest

from mevolve.search import stopping

SE = 0.05


def test_a_rise_must_be_strictly_above_one_se():
    assert stopping.plateau(72.0, [72.05], SE, 1)["met"]             # equal to anchor + SE: no rise
    state = stopping.plateau(72.0, [72.0501], SE, 1)
    assert not state["met"] and state["anchor"] == 72.0501 and state["since_rise"] == 0


def test_small_rises_count_once_they_add_up_past_one_se():
    state = stopping.plateau(72.0, [72.02, 72.04, 72.06], SE, 10)
    assert state["anchor"] == 72.06 and state["rises"] == [{"proposal": 3, "score": 72.06}]
    assert state["since_rise"] == 0


def test_every_finished_proposal_counts_and_the_rule_is_met_at_the_window():
    finished = [72.1, None, 71.5, 72.12, None]          # a rise, then four without (two left no score)
    assert not stopping.plateau(72.0, finished, SE, 5)["met"]
    state = stopping.plateau(72.0, finished, SE, 4)
    assert state["met"] and state["since_rise"] == 4 and state["proposals"] == 5


def make_db(path, gen0, programs, failed):
    """gen0: scores of generation 0; programs: (generation, finished_at, score or None); failed: (generation, at)."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE programs (generation INTEGER, timestamp REAL, combined_score REAL, correct INTEGER)")
    con.execute("CREATE TABLE attempt_log (generation INTEGER, status TEXT, details TEXT, created_at REAL)")
    con.executemany("INSERT INTO programs VALUES (0, 0, ?, 1)", [(s,) for s in gen0])
    con.executemany("INSERT INTO programs VALUES (?, ?, ?, ?)",
                    [(g, t, s if s is not None else 0.0, int(s is not None)) for g, t, s in programs])
    con.executemany("INSERT INTO attempt_log VALUES (?, 'failed', ?, ?)",
                    [(g, json.dumps({"node_kind": "failed_proposal"}), t) for g, t in failed])
    con.execute("INSERT INTO attempt_log VALUES (9, 'failed', ?, 1)", (json.dumps({"node_kind": "other"}),))
    con.commit()
    con.close()


def test_database_order_is_finishing_order_and_each_generation_counts_once(tmp_path):
    db = tmp_path / "programs.sqlite"
    # generation 2 finishes before 1; 3 failed twice; 4 failed once, then left a program; 5 was rejected
    make_db(db, [70.0, 72.0], [(2, 10, 72.2), (1, 20, 72.21), (4, 40, 72.0), (5, 50, None), (5, 55, 72.0)],
            [(3, 30), (3, 31), (4, 35)])                   # generation 5 has two rows: it counts once
    state = stopping.from_database(db, SE, 4)
    assert state["anchor"] == 72.2 and state["proposals"] == 5          # 2, 1, 3, 4, 5
    assert state["since_rise"] == 4 and state["met"]                    # 72.21 is within SE of 72.2


def test_no_score_in_generation_0_yet_is_not_met(tmp_path):
    assert not stopping.from_database(tmp_path / "missing.sqlite", SE, 1)["met"]
    db = tmp_path / "programs.sqlite"
    make_db(db, [], [(1, 1, 72.0)], [])
    assert not stopping.from_database(db, SE, 1)["met"]


def test_the_runner_reports_the_cap_once_the_rule_is_met(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    from mevolve.search import run

    class Base:
        def __init__(self):
            self.evo_config = types.SimpleNamespace(max_api_costs=450.0)

        def _get_committed_cost(self):
            return 3.0

        def _get_completed_job_work_count(self):
            return self.pending

    monkeypatch.setattr(run, "STOP_CHECK_SECONDS", 0)
    make_db(tmp_path / "programs.sqlite", [72.0], [(1, 1, 72.01), (2, 2, None)], [])
    runner = run.stopping_runner(Base)()
    runner.stop_rule, runner.pending = (tmp_path, SE, 3), 0
    assert runner._get_committed_cost() == 3.0                     # two proposals without a rise: not yet
    con = sqlite3.connect(tmp_path / "programs.sqlite")
    con.execute("INSERT INTO programs VALUES (3, 3, 72.02, 1)")
    con.commit()
    con.close()
    runner.pending = 1
    assert runner._get_committed_cost() == 3.0                     # finished work not yet recorded: wait
    runner.pending = 0
    assert runner._get_committed_cost() == 450.0                   # three: the cap, so Shinka drains and stops
    record = json.loads((tmp_path / "stop_rule.json").read_text())
    assert record["met"] and record["since_rise"] == 3 and record["anchor"] == 72.0


def test_config_needs_both_stop_settings(tmp_path):
    pytest.importorskip("shinka")
    from mevolve.search import run

    base = {"results_dir": "/r", "out_dir": "/o", "freeze": "/f", "cohorts": "/c", "generation": "g",
            "validation": "v", "workers": 1, "max_evaluation_jobs": 1}
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**base, "stop_se": 0.05}))
    with pytest.raises(ValueError, match="both stop_se and stop_window"):
        run.load_config(path)
    path.write_text(json.dumps({**base, "stop_se": 0.05, "stop_window": 1.9}))
    with pytest.raises(ValueError, match="whole number"):
        run.load_config(path)
    path.write_text(json.dumps({**base, "stop_se": 0.05, "stop_window": 2000}))
    assert run.load_config(path)["stop_window"] == 2000


def test_the_first_time_the_rule_is_met_stands():
    state = stopping.plateau(72.0, [72.0, 72.0, 72.0, 72.2], SE, 3)      # a rise after the window
    assert state["met"] and state["met_at_proposal"] == 3 and state["anchor"] == 72.0


def test_a_run_the_dollar_cap_ended_is_not_put_down_to_the_rule(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    from mevolve.search import run

    class Base:
        cost_limit_reached = True

        def __init__(self):
            self.evo_config = types.SimpleNamespace(max_api_costs=450.0)

        def _get_committed_cost(self):
            return 450.0

        def _get_completed_job_work_count(self):
            return 0

    monkeypatch.setattr(run, "STOP_CHECK_SECONDS", 0)
    make_db(tmp_path / "programs.sqlite", [72.0], [(1, 1, 72.0), (2, 2, 72.0), (3, 3, 72.0)], [])
    runner = run.stopping_runner(Base)()
    runner.stop_rule = (tmp_path, SE, 3)
    assert runner._get_committed_cost() == 450.0 and not (tmp_path / "stop_rule.json").exists()

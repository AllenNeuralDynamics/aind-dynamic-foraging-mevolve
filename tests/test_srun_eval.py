"""Evaluations on any node of the allocation (#59): staging, no keys, results back, exit code."""
import json
import os
import stat
import sys

import pytest

from mevolve.search import srun_eval

FAKE_SRUN = """#!/bin/sh
# drops srun's own options, records its environment and arguments, then runs the command here
env > "$FAKE_SRUN_LOG.env"
echo "$@" > "$FAKE_SRUN_LOG.args"
while [ "${1#--}" != "$1" ]; do shift; done
exec "$@"
"""

FAKE_EVAL = """import json, sys
from pathlib import Path
a = sys.argv[1:]
program, results = Path(a[a.index("--program_path") + 1]), Path(a[a.index("--results_dir") + 1])
results.mkdir(parents=True, exist_ok=True)
(results / "metrics.json").write_text(json.dumps({"combined_score": len(program.read_text()), "program": str(program)}))
(results / "correct.json").write_text(json.dumps({"correct": True, "error": None}))
sys.exit(int(a[a.index("--exit") + 1]))
"""


def executable(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(bin_dir / "srun", FAKE_SRUN)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SRUN_LOG", str(tmp_path / "srun"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("WANDB_API_KEY", "wb-secret")
    gen = tmp_path / "local" / "gen_7"
    gen.mkdir(parents=True)
    (gen / "main.py").write_text("PARAMS = {}\n")
    (tmp_path / "eval.py").write_text(FAKE_EVAL)
    return tmp_path, gen


def run(tmp_path, gen, exit_code=0):
    cmd = [sys.executable, str(tmp_path / "eval.py"), "--program_path", str(gen / "main.py"),
           "--results_dir", str(gen / "results"), "--workers", "28", "--exit", str(exit_code)]
    return srun_eval.main(["--stage-dir", str(tmp_path / "stage"), "--cpus", "56", "--", *cmd])


def test_the_staged_program_runs_through_srun_and_its_results_come_back(cluster):
    tmp_path, gen = cluster
    assert run(tmp_path, gen) == 0
    metrics = json.loads((gen / "results" / "metrics.json").read_text())
    assert metrics["combined_score"] == len("PARAMS = {}\n")
    assert metrics["program"].startswith(str(tmp_path / "stage" / "gen_7_"))      # the staged copy ran
    assert json.loads((gen / "results" / "correct.json").read_text())["correct"]
    args = (tmp_path / "srun.args").read_text()
    assert args.startswith("--nodes=1 --ntasks=1 --exact --job-name=mevolve-eval --cpus-per-task=56 ")
    assert not any((tmp_path / "stage").iterdir())                                 # the stage is cleaned up


def test_the_evaluation_never_sees_the_api_keys(cluster):
    tmp_path, gen = cluster
    run(tmp_path, gen)
    env = (tmp_path / "srun.env").read_text()
    assert "sk-secret" not in env and "wb-secret" not in env and "FAKE_SRUN_LOG" in env


def test_the_exit_code_passes_through_and_missing_results_are_not_invented(cluster):
    tmp_path, gen = cluster
    assert run(tmp_path, gen, exit_code=3) == 3
    (gen / "results" / "metrics.json").unlink()
    srun_eval.copy_back(tmp_path / "nowhere", gen / "results", wait_s=0)           # missing files: nothing copied
    assert not (gen / "results" / "metrics.json").exists()


def test_the_runner_starts_evaluations_through_the_wrapper(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    import shinka.launch

    from mevolve.search import run as search_run

    seen = {}

    class Stop(Exception):
        pass

    def record(**kw):
        seen.update(kw)
        raise Stop
    monkeypatch.setattr(shinka.launch, "LocalJobConfig", record)
    config = {**search_run.DEFAULTS, "results_dir": str(tmp_path / "r"), "freeze": "/f", "cohorts": "/c",
              "generation": "g", "validation": "v", "workers": 28, "eval_launch": "srun", "stage_dir": str(tmp_path / "s")}
    (tmp_path / "r").mkdir()
    with pytest.raises(Stop):
        search_run.build_runner(config, "TASK")
    wrapper = (tmp_path / "r" / "srun_python").read_text()
    assert seen["python_executable"] == str(tmp_path / "r" / "srun_python") and os.access(seen["python_executable"], os.X_OK)
    assert "-m mevolve.search.srun_eval" in wrapper and "--cpus 56" in wrapper and str(tmp_path / "s") in wrapper
    assert seen["extra_cmd_args"]["repeat_cache"] == str(tmp_path / "s" / "repeat_cache")   # every node reads it


def test_srun_needs_a_stage_dir_and_counts_the_allocations_nodes(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    from mevolve.search import run as search_run

    monkeypatch.setattr(os, "cpu_count", lambda: 112)          # an aind node, whatever runs the test

    base = {"results_dir": "/r", "out_dir": "/o", "freeze": "/f", "cohorts": "/c", "generation": "g",
            "validation": "v", "workers": 28, "eval_launch": "srun"}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="stage_dir"):
        search_run.load_config(path)
    path.write_text(json.dumps({**base, "stage_dir": "/s", "max_evaluation_jobs": 5}))   # 2 steps of 56 per node
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    with pytest.raises(ValueError, match="5 evaluation steps of 56 CPUs exceed 2 node"):
        search_run.load_config(path)
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "4")
    assert search_run.load_config(path)["eval_launch"] == "srun"
    path.write_text(json.dumps({**base, "stage_dir": "/s", "workers": 80, "max_evaluation_jobs": 1}))
    with pytest.raises(ValueError, match="needs 160 CPUs"):                  # no node holds one such step
        search_run.load_config(path)


def test_a_step_gets_one_memory_setting_and_no_keys():
    env = srun_eval.step_env({"SLURM_MEM_PER_CPU": "3500", "SLURM_MEM_PER_NODE": "200000", "ANTHROPIC_API_KEY": "k",
                              "PATH": "/bin"})
    assert env == {"SLURM_MEM_PER_CPU": "3500", "PATH": "/bin"}
    assert srun_eval.step_env({"SLURM_MEM_PER_NODE": "200000"}) == {"SLURM_MEM_PER_NODE": "200000"}


def test_results_are_copied_without_their_mode(tmp_path):
    # files on /allen carry mode 000 and grant access through ACLs: copying the mode makes the local copy unreadable
    src = tmp_path / "stage"
    src.mkdir()
    for name in srun_eval.RESULT_FILES:
        (src / name).write_text("{}")
        (src / name).chmod(0o400)
    srun_eval.copy_back(src, tmp_path / "results", wait_s=0)
    for name in srun_eval.RESULT_FILES:
        assert (tmp_path / "results" / name).stat().st_mode & 0o600 == 0o600


def test_a_killed_wrapper_stops_its_step(cluster):
    # Shinka kills an evaluation past its time with SIGKILL; the srun child must not keep the node busy
    import signal
    import subprocess
    import time

    tmp_path, gen = cluster
    executable(tmp_path / "bin" / "srun", '#!/bin/sh\necho $$ > "$FAKE_SRUN_LOG.pid"\nexec sleep 120\n')
    cmd = [sys.executable, "-m", "mevolve.search.srun_eval", "--stage-dir", str(tmp_path / "stage"), "--cpus", "56", "--",
           sys.executable, str(tmp_path / "eval.py"), "--program_path", str(gen / "main.py"),
           "--results_dir", str(gen / "results"), "--exit", "0"]
    wrapper = subprocess.Popen(cmd, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    pid_file = tmp_path / "srun.pid"
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        time.sleep(0.1)
    step = int(pid_file.read_text())
    wrapper.send_signal(signal.SIGKILL)
    wrapper.wait()
    for _ in range(50):
        try:
            with open(f"/proc/{step}/stat") as f:      # a zombie waiting to be reaped has stopped
                if f.read().split()[2] == "Z":
                    return
        except (FileNotFoundError, ProcessLookupError):
            return
        time.sleep(0.1)
    os.kill(step, signal.SIGKILL)
    pytest.fail("the srun child outlived its killed wrapper")


def test_the_step_is_told_the_programs_own_path(cluster):
    tmp_path, gen = cluster
    run(tmp_path, gen)
    assert f"--program_origin {gen / 'main.py'}" in (tmp_path / "srun.args").read_text()


def test_an_own_job_drops_the_search_jobs_slurm_variables_and_the_keys():
    env = srun_eval.job_env({"SLURM_JOB_ID": "1", "SLURM_NTASKS": "32", "SLURM_MEM_PER_CPU": "1500",
                             "SLURM_CONF": "/etc/slurm.conf", "ANTHROPIC_API_KEY": "k", "WANDB_API_KEY": "w",
                             "PATH": "/bin"})
    assert env == {"SLURM_CONF": "/etc/slurm.conf", "PATH": "/bin"}


def test_an_own_job_asks_srun_for_a_new_allocation_with_its_options(cluster, monkeypatch):
    # eval_launch "job" (#70): no step of the search's allocation, but a job of its own
    tmp_path, gen = cluster
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    cmd = [sys.executable, str(tmp_path / "eval.py"), "--program_path", str(gen / "main.py"),
           "--results_dir", str(gen / "results"), "--exit", "0"]
    code = srun_eval.main(["--stage-dir", str(tmp_path / "stage"), "--cpus", "24",
                           "--job-args", "--partition=aind --mem-per-cpu=1500M --time=02:00:00", "--", *cmd])
    assert code == 0 and json.loads((gen / "results" / "metrics.json").read_text())["combined_score"] > 0
    args = (tmp_path / "srun.args").read_text().split()
    assert args[:7] == ["--nodes=1", "--ntasks=1", "--job-name=mevolve-eval", "--cpus-per-task=24",
                        "--partition=aind", "--mem-per-cpu=1500M", "--time=02:00:00"]
    assert "--exact" not in args
    env = (tmp_path / "srun.env").read_text()
    assert "SLURM_JOB_ID=" not in env and "SLURM_NTASKS=" not in env and "sk-secret" not in env


def test_the_runner_starts_own_job_evaluations_and_lets_them_queue(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    import shinka.launch

    from mevolve.search import run as search_run

    seen = {}

    class Stop(Exception):
        pass

    def record(**kw):
        seen.update(kw)
        raise Stop
    monkeypatch.setattr(shinka.launch, "LocalJobConfig", record)
    config = {**search_run.DEFAULTS, "results_dir": str(tmp_path / "r"), "freeze": "/f", "cohorts": "/c",
              "generation": "g", "validation": "v", "workers": 12, "eval_launch": "job", "stage_dir": str(tmp_path / "s"),
              "eval_job_args": ["--partition=aind", "--mem-per-cpu=1500M"], "eval_time": "02:00:00",
              "eval_queue_time": "12:30:00"}
    (tmp_path / "r").mkdir()
    with pytest.raises(Stop):
        search_run.build_runner(config, "TASK")
    wrapper = (tmp_path / "r" / "srun_python").read_text()
    assert "--cpus 24 --job-args '--partition=aind --mem-per-cpu=1500M --time=02:00:00' --" in wrapper
    assert seen["time"] == "14:30:00"             # Shinka's clock: the run's limit plus the queue


def test_job_mode_needs_a_stage_dir_owns_time_and_cpus_and_ignores_the_allocation(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    from mevolve.search import run as search_run

    monkeypatch.setenv("SLURM_NTASKS", "1")            # the small driver job: never a limit in job mode
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    base = {"results_dir": "/r", "out_dir": "/o", "freeze": "/f", "cohorts": "/c", "generation": "g",
            "validation": "v", "workers": 12, "eval_launch": "job", "max_evaluation_jobs": 48}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="stage_dir"):
        search_run.load_config(path)
    for bad in (["--time=01:00:00"], ["-c", "8"], ["--cpus-per-task=8"]):
        path.write_text(json.dumps({**base, "stage_dir": "/s", "eval_job_args": bad}))
        with pytest.raises(ValueError, match="eval_job_args"):
            search_run.load_config(path)
    path.write_text(json.dumps({**base, "stage_dir": "/s", "eval_job_args": ["--partition=aind"]}))
    assert search_run.load_config(path)["max_evaluation_jobs"] == 48


def test_times_add_past_a_day():
    from mevolve.search.run import add_times
    assert add_times("02:00:00", "23:30:45") == "25:30:45"

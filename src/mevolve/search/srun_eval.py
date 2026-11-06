"""Run one evaluation on any node of the search's Slurm allocation (#59).

Shinka starts every evaluation as ``[python_executable, shinka_eval.py, --program_path P,
--results_dir R, ...]`` and reads ``R/metrics.json`` and ``R/correct.json`` when the process exits.
With ``eval_launch: "srun"`` the run's ``python_executable`` is a script that calls this module on
the launch node. The database, every ``gen_N/`` and the logs stay on the launch node's local disk
(SQLite's write-ahead log is not safe on NFS); only the one program and its results pass through
``stage_dir``, a directory every node can read:

1. copy ``P`` to ``stage_dir/<gen_N>/main.py``;
2. ``srun`` one task on a free node of the allocation, with ``2 x workers`` CPUs (one worker per core),
   running the same evaluation on the staged program, without the API keys in its environment (the
   job sets ``--mem-per-cpu``, so two steps fit on a node);
3. copy ``metrics.json`` and ``correct.json`` back to ``R``, waiting briefly for NFS to show them;
4. exit with the evaluation's exit code.

With ``eval_launch: "job"`` (#70) the wrapper passes ``--job-args``: the evaluation is then its own Slurm
job, not a step of the search's allocation. The wrapper drops the search job's ``SLURM_*`` variables, so
``srun`` asks for a new allocation with these options (partition, memory, time limit), waits in the queue
and runs it. Killing ``srun`` revokes that job, queued or running.

    python -m mevolve.search.srun_eval --stage-dir D --cpus N [--job-args "..."] -- <python> shinka_eval.py ...
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shlex
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

RESULT_FILES = ("metrics.json", "correct.json")
SECRET_ENV = ("ANTHROPIC_API_KEY", "WANDB_API_KEY")
MEM_ENV = ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_GPU")   # srun refuses more than one
SRUN = ["srun", "--nodes=1", "--ntasks=1", "--exact", "--job-name=mevolve-eval"]   # --exact: only its CPUs
JOB = ["srun", "--nodes=1", "--ntasks=1", "--job-name=mevolve-eval"]     # its own job (eval_launch "job")


def rewrite(cmd: list[str], stage: Path) -> tuple[list[str], Path, Path]:
    """The evaluation command on the staged program; returns it, the program and the results dir."""
    out = list(cmd)
    i, j = out.index("--program_path") + 1, out.index("--results_dir") + 1
    program, results = Path(out[i]), Path(out[j])
    out[i], out[j] = str(stage / "main.py"), str(stage / "results")
    return out, program, results


def stage_name(program: Path) -> str:
    return f"{program.parent.name}_{os.getpid()}"     # gen_N, unique if Shinka evaluates it again


def copy_back(stage_results: Path, results: Path, wait_s: float = 30.0) -> None:
    """Copy the result files by name; NFS may show a file the remote node closed a little later."""
    results.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + wait_s
    for name in RESULT_FILES:
        src = stage_results / name
        while not src.exists() and time.time() < deadline:
            time.sleep(1.0)
        if src.exists():
            shutil.copyfile(src, results / name)     # contents only: /allen files carry mode 000 + ACLs


def step_env(environ) -> dict:
    """The evaluation step's environment: no API keys, and one memory setting. A job submitted from
    inside another allocation inherits that allocation's ``SLURM_MEM_PER_NODE`` beside its own
    ``SLURM_MEM_PER_CPU``; the job's per-CPU setting is the one that lets two steps share a node."""
    env = {k: v for k, v in environ.items() if k not in SECRET_ENV}
    if "SLURM_MEM_PER_CPU" in env:
        for k in MEM_ENV[1:]:
            env.pop(k, None)
    return env


def job_env(environ) -> dict:
    """The environment of an evaluation that is its own job: no API keys, and none of the search job's
    ``SLURM_*`` variables (``srun`` would read them as a step of that job), but ``SLURM_CONF``."""
    return {k: v for k, v in environ.items()
            if k not in SECRET_ENV and (not k.startswith("SLURM_") or k == "SLURM_CONF")}


def _die_with_parent() -> None:
    """In the ``srun`` child: SIGTERM when this wrapper dies, even by SIGKILL (Shinka kills an evaluation
    past its time); ``srun`` then cancels the step, so a killed evaluation frees its node."""
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)     # PR_SET_PDEATHSIG


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--cpus", type=int, required=True, help="CPUs of the evaluation's step")
    parser.add_argument("--job-args", default=None,
                        help="srun options of the evaluation's own job (eval_launch 'job'); else a step")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the evaluation command")
    args = parser.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    stage = args.stage_dir / stage_name(Path(cmd[cmd.index("--program_path") + 1]))
    staged, program, results = rewrite(cmd, stage)
    staged += ["--program_origin", str(program)]       # the repeat cache names the kept program, not the stage
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(program, stage / "main.py")
    if args.job_args is None:
        launch, env = [*SRUN, f"--cpus-per-task={args.cpus}"], step_env(os.environ)
    else:
        launch = [*JOB, f"--cpus-per-task={args.cpus}", *shlex.split(args.job_args)]
        env = job_env(os.environ)
    code = subprocess.run([*launch, *staged], env=env, preexec_fn=_die_with_parent).returncode
    copy_back(stage / "results", results)
    if all((results / name).exists() for name in RESULT_FILES):
        shutil.rmtree(stage, ignore_errors=True)
    return code


if __name__ == "__main__":
    sys.exit(main())

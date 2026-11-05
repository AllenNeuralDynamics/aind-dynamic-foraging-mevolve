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

    python -m mevolve.search.srun_eval --stage-dir D --cpus N -- <python> shinka_eval.py --program_path P ...
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RESULT_FILES = ("metrics.json", "correct.json")
SECRET_ENV = ("ANTHROPIC_API_KEY", "WANDB_API_KEY")
MEM_ENV = ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_GPU")   # srun refuses more than one
SRUN = ["srun", "--nodes=1", "--ntasks=1", "--exact", "--job-name=mevolve-eval"]   # --exact: only its CPUs


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
            shutil.copy2(src, results / name)


def step_env(environ) -> dict:
    """The evaluation step's environment: no API keys, and one memory setting. A job submitted from
    inside another allocation inherits that allocation's ``SLURM_MEM_PER_NODE`` beside its own
    ``SLURM_MEM_PER_CPU``; the job's per-CPU setting is the one that lets two steps share a node."""
    env = {k: v for k, v in environ.items() if k not in SECRET_ENV}
    if "SLURM_MEM_PER_CPU" in env:
        for k in MEM_ENV[1:]:
            env.pop(k, None)
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--cpus", type=int, required=True, help="CPUs of the evaluation's step")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the evaluation command")
    args = parser.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    stage = args.stage_dir / stage_name(Path(cmd[cmd.index("--program_path") + 1]))
    staged, program, results = rewrite(cmd, stage)
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(program, stage / "main.py")
    env = step_env(os.environ)
    code = subprocess.run([*SRUN, f"--cpus-per-task={args.cpus}", *staged], env=env).returncode
    copy_back(stage / "results", results)
    if all((results / name).exists() for name in RESULT_FILES):
        shutil.rmtree(stage, ignore_errors=True)
    return code


if __name__ == "__main__":
    sys.exit(main())

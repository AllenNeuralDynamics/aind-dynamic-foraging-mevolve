"""Evaluation workers exit when their parent dies (issue #33)."""
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

CHILD = textwrap.dedent("""
    import multiprocessing as mp, os, sys, time
    from mevolve.evaluate import _die_with_parent

    def init(d):
        _die_with_parent()
        open(os.path.join(d, str(os.getpid())), "w").close()

    if __name__ == "__main__":
        with mp.get_context("spawn").Pool(2, initializer=init, initargs=(sys.argv[1],)) as pool:
            pool.map(time.sleep, [600, 600])
""")


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:      # a zombie is gone in every way that matters
            return f.read().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):      # gone before the open, or during the read
        return False


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG and /proc are Linux only")
def test_workers_exit_when_their_parent_is_killed(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    (tmp_path / "pids").mkdir()
    # its own process group, so cleanup reaches every worker, started or not
    parent = subprocess.Popen([sys.executable, str(script), str(tmp_path / "pids")], env=os.environ.copy(),
                              start_new_session=True)
    try:
        deadline = time.time() + 120
        while len(os.listdir(tmp_path / "pids")) < 2 and time.time() < deadline:
            time.sleep(0.2)
        pids = [int(p) for p in os.listdir(tmp_path / "pids")]
        assert len(pids) == 2, "workers did not start"
        parent.send_signal(signal.SIGKILL)
        parent.wait()
        deadline = time.time() + 20
        while any(_alive(p) for p in pids) and time.time() < deadline:
            time.sleep(0.2)
        leftover = [p for p in pids if _alive(p)]
        assert not leftover, f"workers {leftover} outlived their parent"
    finally:
        try:
            os.killpg(parent.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        parent.wait()

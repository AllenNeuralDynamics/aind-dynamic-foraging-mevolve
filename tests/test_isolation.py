"""Evaluation isolation (#40): in a pool worker, a program cannot read the key files, write, or reach the network."""
import json
import subprocess
import sys
import textwrap

import pytest

from mevolve import evaluate, program

from test_score import _subject

PROBE = textwrap.dedent("""
    import json, socket, subprocess, sys
    from mevolve import isolation
    isolation.install(sys.argv[1])
    out = {}
    def attempt(name, fn):
        try:
            fn()
            out[name] = "ok"
        except PermissionError as e:
            out[name] = "denied"
    attempt("read_secret", lambda: open(sys.argv[2]).read())
    attempt("read_program_dir", lambda: open(sys.argv[3]).read())
    attempt("write_program_dir", lambda: open(sys.argv[3] + ".new", "w"))
    attempt("socket", lambda: socket.create_connection(("127.0.0.1", 9), timeout=1))
    attempt("subprocess", lambda: subprocess.run(["true"]))
    attempt("lazy_import", lambda: __import__("tomllib"))
    import os
    attempt("parent_environ", lambda: open(f"/proc/{os.getppid()}/environ").read())
    print(json.dumps(out))
""")


def test_the_hook_denies_files_writes_network_and_processes(tmp_path):
    secret = tmp_path / "home" / ".config" / "mevolve" / "anthropic_api_key"
    secret.parent.mkdir(parents=True)
    secret.write_text("not-a-key")
    prog_dir = tmp_path / "prog"
    prog_dir.mkdir()
    (prog_dir / "p.py").write_text("# a program\n")
    (prog_dir / "notes.txt").write_text("beside the program\n")
    r = subprocess.run([sys.executable, "-c", PROBE, str(prog_dir / "p.py"), str(secret), str(prog_dir / "notes.txt")],
                       capture_output=True, text=True, check=True)
    assert json.loads(r.stdout) == {"read_secret": "denied", "read_program_dir": "ok", "write_program_dir": "denied",
                                    "socket": "denied", "subprocess": "denied", "lazy_import": "ok",
                                    "parent_environ": "denied"}


def test_a_program_that_escapes_the_static_checks_cannot_read_a_key_in_a_worker(tmp_path):
    secret = tmp_path / "home" / ".config" / "mevolve" / "anthropic_api_key"
    secret.parent.mkdir(parents=True)
    secret.write_text("not-a-key")
    source = (program.LITERATURE_DIR / "hattori2019.py").read_text()
    (tmp_path / "prog").mkdir()
    path = tmp_path / "prog" / "escape.py"             # the program's directory is readable; the key's is not
    path.write_text(source + f"\n_stolen = open({str(secret)!r}).read()\n")      # the checks would reject this
    with pytest.raises(PermissionError, match="evaluation isolation"):
        evaluate.evaluate(path, [_subject(0), _subject(1)], regime="subject", n_starts=2, workers=2)


def test_isolated_workers_score_as_before():
    path = program.LITERATURE_DIR / "hattori2019.py"
    subjects = [_subject(0), _subject(1)]
    pooled = evaluate.evaluate(path, subjects, regime="subject", n_starts=2, workers=2)
    alone = evaluate.evaluate(path, subjects, regime="subject", n_starts=2, workers=1)
    assert [r["log_lik"] for r in pooled] == [r["log_lik"] for r in alone]


def test_the_search_loads_a_candidate_only_in_an_isolated_worker(tmp_path):
    pytest.importorskip("shinka")
    from mevolve.search import shinka_eval
    from mevolve.search.checks import check_source

    secret = tmp_path / "home" / ".config" / "mevolve" / "anthropic_api_key"
    secret.parent.mkdir(parents=True)
    secret.write_text("not-a-key")
    (tmp_path / "prog").mkdir()
    good = tmp_path / "prog" / "good.py"
    good.write_text((program.LITERATURE_DIR / "hattori2019.py").read_text())
    shape = check_source(good.read_text())
    assert set(shinka_eval.inspect_isolated(good, shape)) == set(program.load(good).params)
    bad = tmp_path / "prog" / "bad.py"
    bad.write_text(good.read_text() + f"\n_stolen = open({str(secret)!r}).read()\n")
    with pytest.raises(PermissionError, match="evaluation isolation"):
        shinka_eval.inspect_isolated(bad, shape)

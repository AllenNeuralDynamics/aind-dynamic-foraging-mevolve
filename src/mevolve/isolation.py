"""Evaluation isolation (#40): a candidate program in an evaluation worker cannot reach files or the network.

The static checks (``mevolve.search.checks``) parse a program before it runs, but they are not a
sandbox. ``install`` adds a Python audit hook to the worker process, after its imports and before the
program loads, in pool workers only (``mevolve.evaluate``; with ``workers=1`` the evaluation runs in the
calling process, which must still write its results, and is not isolated). The search's evaluation
script (``search.shinka_eval``) also loads and traces each candidate once in an isolated worker before
fitting it, so a candidate never runs in the process that writes the results. The hook stays for the life
of the process, so every program the worker runs later is covered. From then on, in that process:

- ``open`` for writing fails anywhere, and ``open`` for reading fails outside Python's import paths
  (``sys.prefix``, ``sys.path``: the lazy imports a fit does), the program's own directory, ``/dev``
  and ``/sys``: the key files in the home directory, and other processes' start-up environments under
  ``/proc``, are out of reach;
- sockets, subprocesses, ``exec``-family calls, ``fork``, ``ctypes`` and file system changes fail.

Measured on 2026-11-05: fitting a program raised only reads under the environment's library
directory (lazy imports of scipy, numpy and others), no writes, no sockets, no ``ctypes``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DENIED = frozenset({
    "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto",
    "socket.sendmsg", "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
    "os.fork", "os.forkpty", "pty.spawn", "ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function",
    "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.truncate", "os.link",
    "os.symlink", "shutil.rmtree", "shutil.copyfile", "shutil.move", "os.putenv", "os.unsetenv",
    "urllib.Request", "webbrowser.open",
})
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_installed = False


def _roots(program_path: Path) -> tuple[str, ...]:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, "/dev",
             "/sys",        # hardware facts; JAX reads PCI vendor ids when it looks for accelerators
             str(Path(program_path).resolve().parent)}   # not /proc: other processes' start-up environments
    roots |= {str(Path(p).resolve()) for p in sys.path if p}
    return tuple(sorted(r.rstrip("/") + "/" for r in roots))


def install(program_path: Path | str) -> None:
    """Install the hook in this process (once; later calls do nothing)."""
    global _installed
    if _installed:
        return
    sys.dont_write_bytecode = True
    roots = _roots(Path(program_path))

    def hook(event, args):
        if event == "open":
            path, mode, flags = args
            if isinstance(path, int):           # an inherited descriptor (the pool's pipes), not a file by name
                return
            writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool((flags or 0) & WRITE_FLAGS)
            if writes:
                raise PermissionError(f"evaluation isolation: no file writes ({path})")
            if isinstance(path, (str, bytes, os.PathLike)):
                full = os.path.realpath(os.fsdecode(path)) + "/"
                if not full.startswith(roots):
                    raise PermissionError(f"evaluation isolation: {path} is outside the allowed paths")
        elif event in DENIED:
            raise PermissionError(f"evaluation isolation: {event} is not allowed")

    sys.addaudithook(hook)
    _installed = True

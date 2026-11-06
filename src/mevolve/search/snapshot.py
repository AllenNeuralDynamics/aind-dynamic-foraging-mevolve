"""Snapshots of a running search on shared disk, and the resume from one (#65).

A search keeps Shinka's database on its first node's local disk (WAL is unsafe on NFS) and copies it
to ``out_dir`` only at the end. A snapshot copies, to ``snapshot_dir``: the database through SQLite's
backup API (one consistent file, even while the search writes); the run's small top-level files (the
ledger, Shinka's bandit state, the configs, ``stop_rule.json``); and every settled ``gen_*`` directory,
that is one whose generation is in the copied database or left no program. Each piece is replaced
whole (the ledger after the database, so it never undercounts the database's calls); ``snapshot.json``
is written last, and a directory without it was never completed.

``restore`` fills a fresh ``results_dir`` from a snapshot; Shinka then resumes the database (it restores
its generation counter and its spent dollars from it), and MEvolve's Claude wrapper reads the restored
ledger before every call, so the dollar cap counts what was spent before.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

DB = "programs.sqlite"


def _copy_file(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.part")
    shutil.copyfile(src, tmp)          # copyfile, never copy2: files on /allen are mode 000 with ACLs
    os.replace(tmp, dst)


def take(results_dir: Path, dest: Path) -> dict:
    """One snapshot of ``results_dir`` into ``dest``; returns what ``snapshot.json`` records."""
    results_dir, dest = Path(results_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)       # each piece is replaced whole, so a snapshot stays usable
    part = dest / f".{DB}.part"
    src = sqlite3.connect(f"file:{results_dir / DB}?mode=ro", uri=True, timeout=60)
    try:
        out = sqlite3.connect(part)
        with out:
            src.backup(out)
        settled = {g for (g,) in out.execute("SELECT DISTINCT generation FROM programs")}
        settled |= {g for (g,) in out.execute("SELECT DISTINCT generation FROM attempt_log WHERE status = 'failed'")}
        n_programs = out.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
        out.close()
    finally:
        src.close()
    os.replace(part, dest / DB)
    for f in results_dir.iterdir():
        if f.is_file() and not f.name.startswith(DB):
            _copy_file(f, dest / f.name)
    copied = 0
    for d in results_dir.glob("gen_*"):
        if d.is_dir() and d.name[4:].isdigit() and int(d.name[4:]) in settled and not (dest / d.name).exists():
            shutil.copytree(d, dest / f".{d.name}.part", copy_function=shutil.copyfile, dirs_exist_ok=True)
            os.replace(dest / f".{d.name}.part", dest / d.name)
            copied += 1
    record = {"time": time.time(), "programs": n_programs, "generations": len(settled), "gen_dirs_copied": copied}
    (dest / "snapshot.json").write_text(json.dumps(record) + "\n")
    return record


def restore(snapshot_dir: Path, results_dir: Path) -> dict:
    """Fill an empty ``results_dir`` from a complete snapshot; returns its ``snapshot.json``."""
    snapshot_dir, results_dir = Path(snapshot_dir), Path(results_dir)
    marker = snapshot_dir / "snapshot.json"
    if not marker.exists():
        raise FileNotFoundError(f"{snapshot_dir} holds no complete snapshot (no snapshot.json)")
    results_dir.mkdir(parents=True, exist_ok=True)
    for f in snapshot_dir.iterdir():
        if f.name.startswith(".") or f.name == "snapshot.json":
            continue
        if f.is_dir():
            shutil.copytree(f, results_dir / f.name, copy_function=shutil.copyfile)
        else:
            shutil.copyfile(f, results_dir / f.name)
    return json.loads(marker.read_text())


class Snapshotter:
    """Takes a snapshot every ``minutes`` in a background thread; ``final()`` takes the last one."""

    def __init__(self, results_dir: Path, dest: Path, minutes: float):
        self.results_dir, self.dest, self.seconds = Path(results_dir), Path(dest), 60 * minutes
        self._stop, self._lock = threading.Event(), threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="mevolve-snapshot", daemon=True)

    def start(self) -> "Snapshotter":
        self._thread.start()
        return self

    def _take(self) -> None:
        with self._lock:
            try:
                record = take(self.results_dir, self.dest)
                print(f"snapshot: {record['programs']} programs to {self.dest}", flush=True)
            except Exception as e:      # a failed snapshot is reported, never stops the search
                print(f"snapshot failed: {type(e).__name__}: {e}", flush=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.seconds):
            self._take()

    def take_now(self) -> None:
        """A snapshot now, in the calling thread (the SIGTERM handler: before the search unwinds)."""
        print("snapshot at SIGTERM", flush=True)
        self._take()

    def final(self) -> None:
        self._stop.set()
        self._thread.join(timeout=600)
        self._take()

"""Archive a search's run directory as a W&B artifact (#54).

Run directories live on scratch, which is not durable, and a search's database is too big for a
study to commit (Han, 2026-11-04: archives go to W&B artifacts; the study's freeze keeps minimal
rows and checks them against the archive). ``archive`` uploads every file under a directory,
without ``wandb/`` and SQLite's ``-shm``, as one ``mevolve-run`` artifact named after the W&B group,
logged from an ``archive`` run in that group, and writes ``archive.json`` beside the files: the
artifact's name and digest (the immutable key) and the sha256 of the files a freeze reads.

``run.main`` archives the search's ``out_dir`` at the end of every run with W&B online. A study
that adds steps after the search (a held-out check) archives the whole launch directory again,
which logs a new version of the same artifact:

    python -m mevolve.search.archive <dir> --group <variant@launch_id> [--entity E --project P]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ARTIFACT_TYPE = "mevolve-run"
RECORD = "archive.json"
KEY_FILES = ("programs.sqlite", "programs.sqlite-wal", "ledger.jsonl", "run_config.json", "run_summary.json",
             "heldout.json")


def artifact_name(group: str) -> str:
    """``run-`` and the W&B group, with characters W&B does not allow in a name replaced by ``-``."""
    return "run-" + re.sub(r"[^A-Za-z0-9._-]", "-", group)


def files_to_archive(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and "wandb" not in p.relative_to(root).parts
                  and not p.name.endswith("-shm") and p.name != RECORD)


def archive(root: Path, group: str, entity: str, project: str) -> dict:
    """Upload ``root`` as one artifact and write ``root/archive.json``; returns the record."""
    import wandb

    root = Path(root)
    files = files_to_archive(root)
    sha256 = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name in KEY_FILES}
    run = wandb.init(entity=entity, project=project, group=group, job_type="archive", name=f"archive-{group}",
                     notes=f"Archive of {group}'s run directory {root} (scratch is not durable).", reinit="create_new")
    art = wandb.Artifact(artifact_name(group), type=ARTIFACT_TYPE, description=f"{group}: {root}",
                         metadata={"wandb_group": group, "sha256": sha256})
    for p in files:
        art.add_file(str(p), name=str(p.relative_to(root)))
    run.log_artifact(art)
    art.wait()
    record = {"artifact": f"{entity}/{project}/{art.name}", "digest": art.digest, "wandb_group": group,
              "n_files": len(files), "bytes": sum(p.stat().st_size for p in files), "archive_run": run.url,
              "sha256": sha256}
    run.finish()
    (root / RECORD).write_text(json.dumps(record, indent=1) + "\n")
    return record


def archive_run(config: dict, out_dir: Path | None, dry: bool) -> dict | None:
    """The end of ``run.main``: archive ``out_dir`` when W&B is on and online. A failed upload warns
    and returns None; the outputs are already in ``out_dir``, and this module's CLI can retry."""
    from .run import WANDB_ENTITY, WANDB_PROJECT

    wandb_cfg = config["wandb"]
    if dry or not out_dir or not wandb_cfg or wandb_cfg.get("mode") in ("offline", "disabled"):
        return None
    try:
        return archive(Path(out_dir), wandb_cfg["group"], wandb_cfg.get("entity", WANDB_ENTITY),
                       wandb_cfg.get("project", WANDB_PROJECT))
    except Exception as e:  # noqa: BLE001 - the archive must never cost the run its outputs
        print(f"archive of {out_dir} failed ({type(e).__name__}: {e}); retry with "
              f"python -m mevolve.search.archive {out_dir} --group {wandb_cfg['group']}", file=sys.stderr, flush=True)
        return None


def main(argv: list[str] | None = None) -> int:
    from .run import WANDB_ENTITY, WANDB_PROJECT

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--group", required=True, help="the launch's W&B group, variant@launch_id")
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    args = parser.parse_args(argv)
    record = archive(args.root, args.group, args.entity, args.project)
    print(f"{record['artifact']} ({record['n_files']} files) digest {record['digest']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

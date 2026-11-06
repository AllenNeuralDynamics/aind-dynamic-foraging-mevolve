"""The search's stopping rule (#63): stop when ``window`` proposals finish without the best validation
score rising more than ``se`` above the anchor.

The anchor starts at the best score of generation 0 (the template and the seeds). Proposals are taken
in the order they finished, each generation once: a program of generation > 0 (scored or not), or a
proposal that left no program (``attempt_log``, ``failed_proposal``). A proposal scoring strictly above
anchor + ``se`` becomes the anchor and resets the count; any other proposal adds one. Small rises are
measured against the anchor, so they count once their total passes ``se``. The rule is met when the
count reaches ``window``.

It is computed from the database each time, so a resumed run gets the same answer.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def plateau(anchor: float, finished: list[float | None], se: float, window: int) -> dict:
    """The rule over ``finished``: each finished proposal's score, or None if it has none, in finishing order."""
    since, rises = 0, []
    for i, score in enumerate(finished, start=1):
        if score is not None and score > anchor + se:
            anchor, since = score, 0
            rises.append({"proposal": i, "score": score})
        else:
            since += 1
    return {"met": since >= window, "anchor": anchor, "since_rise": since, "proposals": len(finished),
            "rises": rises, "se": se, "window": window}


def from_database(db_path: Path, se: float, window: int) -> dict:
    """The rule on a Shinka database, read-only (safe beside the running search's writer). Not met while
    generation 0 has no score yet (a run without seeds, before its template is scored)."""
    if not Path(db_path).exists():
        return {"met": False, "reason": "no database yet"}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        start = con.execute("SELECT MAX(combined_score) FROM programs WHERE generation = 0 AND correct").fetchone()[0]
        rows = con.execute("SELECT generation, timestamp, CASE WHEN correct THEN combined_score END "
                           "FROM programs WHERE generation > 0").fetchall()
        failed = con.execute("SELECT generation, created_at FROM attempt_log WHERE status = 'failed' "
                             "AND json_extract(details, '$.node_kind') = 'failed_proposal'").fetchall()
    finally:
        con.close()
    if start is None:
        return {"met": False, "reason": "generation 0 has no score yet"}
    seen, no_program = {g for g, _, _ in rows}, {}
    for g, t in sorted(failed, key=lambda r: r[1]):        # a generation counts once, at its first failure
        if g not in seen:
            no_program.setdefault(g, t)
    events = [(t, s) for _, t, s in rows] + [(t, None) for t in no_program.values()]
    return plateau(float(start), [s for _, s in sorted(events, key=lambda e: e[0])], se, window)

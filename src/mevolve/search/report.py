"""What a search run did, readable at a glance: one row per program and a short report.

    python -m mevolve.search.report <run_dir> [<run_dir> ...] [--out comparison.md]

Per run it writes ``proposals.csv`` (generation, island, parent, score and its change from the
parent, correct, patch type, cost, LLM and evaluation seconds, Claude's name and description of the
change) and ``run_report.md``; with several runs it also prints a side-by-side comparison (for
gate 3: medium vs high effort, per dollar).
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path


def proposals(run_dir: Path) -> list[dict]:
    con = sqlite3.connect(f"file:{Path(run_dir) / 'programs.sqlite'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = list(con.execute("SELECT id, generation, island_idx, parent_id, combined_score, correct, metadata, "
                            "text_feedback FROM programs ORDER BY generation, timestamp"))
    con.close()
    score = {r["id"]: r["combined_score"] for r in rows}
    out = []
    for r in rows:
        m = json.loads(r["metadata"] or "{}")
        parent = score.get(r["parent_id"])
        out.append({
            "generation": r["generation"], "island": r["island_idx"], "seed": m.get("mevolve_seed", ""),
            "id": r["id"], "parent_id": r["parent_id"] or "", "correct": bool(r["correct"]),
            "score": round(r["combined_score"], 4),
            "delta_vs_parent": round(r["combined_score"] - parent, 4) if parent is not None and r["correct"] else "",
            "patch_type": m.get("patch_type", ""), "cost_usd": round(float(m.get("api_costs") or 0.0), 5),
            "llm_seconds": round(float(m.get("sampling_seconds") or 0.0), 1),
            "eval_seconds": round(float(m.get("evaluation_seconds") or 0.0), 1),
            "name": m.get("patch_name") or "", "description": (m.get("patch_description") or "").replace("\n", " "),
        })
    return out


def summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    rows = proposals(run_dir)
    ledger = run_dir / "ledger.jsonl"
    calls = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    config = json.loads((run_dir / "run_config.json").read_text()) if (run_dir / "run_config.json").exists() else {}
    children = [r for r in rows if r["generation"] > 0]
    valid = [r for r in children if r["correct"]]
    improving = [r for r in valid if isinstance(r["delta_vs_parent"], float) and r["delta_vs_parent"] > 0.05]
    cost = sum(c["cost"] for c in calls)
    seeds = [r for r in rows if r["generation"] == 0]
    return {
        "run": run_dir.name, "effort": config.get("effort"), "model": config.get("model"),
        "proposals": len(children), "valid": len(valid), "improving_by_0.05": len(improving),
        "api_calls": len(calls), "errors": sum("error" in c for c in calls),
        "refusals": sum(c.get("stop_reason") == "refusal" for c in calls),
        "cut_off": sum(c.get("stop_reason") == "max_tokens" for c in calls),
        "cost_usd": round(cost, 4),
        "usd_per_valid": round(cost / len(valid), 4) if valid else None,
        "cache_read_tokens": sum(c.get("cache_read_input_tokens", 0) for c in calls),
        "cache_write_tokens": sum(c.get("cache_creation_input_tokens", 0) for c in calls),
        "output_tokens": sum(c.get("output_tokens", 0) for c in calls),
        "mean_llm_seconds": round(sum(c.get("seconds", 0) for c in calls) / len(calls), 1) if calls else None,
        "best_seed": max((r["score"] for r in seeds if r["correct"]), default=None),
        "best": max((r["score"] for r in rows if r["correct"]), default=None),
        "best_by_island": {i: max((r["score"] for r in rows if r["island"] == i and r["correct"]), default=None)
                           for i in sorted({r["island"] for r in rows})},
    }


def write_run_files(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    rows = proposals(run_dir)
    with (run_dir / "proposals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["generation"])
        writer.writeheader()
        writer.writerows(rows)
    s = summary(run_dir)
    lines = [f"# Run {s['run']}: {s['model']}, effort {s['effort']}", "",
             "| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s.items() if k not in ("run", "best_by_island")]
    lines += ["", "Best score by island: " + ", ".join(f"{i}: {v}" for i, v in s["best_by_island"].items()), "",
              "| gen | island | score | vs parent | ok | patch | $ | name | description |",
              "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['generation']} | {r['island']} | {r['score']:.3f} | {r['delta_vs_parent']} | {'yes' if r['correct'] else 'no'} "
              f"| {r['patch_type']} | {r['cost_usd']} | {r['name'] or r['seed']} | {r['description'][:160]} |" for r in rows]
    (run_dir / "run_report.md").write_text("\n".join(lines) + "\n")
    return s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize one or more search runs")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="write the side-by-side comparison here")
    args = parser.parse_args(argv)
    summaries = [write_run_files(r) for r in args.runs]
    keys = [k for k in summaries[0] if k not in ("best_by_island",)]
    table = ["| | " + " | ".join(str(s["run"]) for s in summaries) + " |", "|---|" + "---|" * len(summaries)]
    table += [f"| {k} | " + " | ".join(str(s[k]) for s in summaries) + " |" for k in keys if k != "run"]
    text = "\n".join(table) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

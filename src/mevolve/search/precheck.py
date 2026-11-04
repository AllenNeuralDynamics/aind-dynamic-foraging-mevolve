"""Gate 4's pre-check, before any Claude call: is the target out of reach of every program Claude sees?

The second half of gate 4 asks whether a short search on subjects simulated from a known program
(the target) reaches that program's score. Claude sees the seeds, the template and the literature
library, and can copy any of them in one proposal. So every one is scored first on the same
simulated subjects, with the search's settings, and the rule fixed before the search (study 40,
variant ``gate4-recovery``) is applied:

- the target is left out of the library, always;
- any other library program within ``2 x TOLERANCE`` of the target is left out too;
- stop and ask (Han) if the target's own 2-level fit fails, if the rule would leave out a seed or
  more than ``MAX_EXCLUDED`` programs, or if the template is already that close.

Scores are cached like the search's seeds (``seed_metrics``), so the search reuses them.

    python -m mevolve.search.precheck --config run.json --target Hattori2019 --out precheck.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOLERANCE = 0.2         # the pass tolerance on the x100 scale (study 40 PLAN: 0.002 in pooled likelihood)
EXCLUDE_WITHIN = 2 * TOLERANCE
MAX_EXCLUDED = 3        # besides the target; more would narrow what the search is shown too far


def decide(scores: dict[str, dict], target: str, seeds: list[str]) -> dict:
    """The rule, from each program's ``{"score", "two_level_failed"}`` (x100). ``template`` is the
    near-empty program on island 0; every other key is a literature program."""
    goal = scores[target]["score"]
    margins = {name: round(goal - s["score"], 4) for name, s in scores.items()}
    close = sorted(n for n, m in margins.items() if n not in (target, "template") and m < EXCLUDE_WITHIN)
    stop = []
    if scores[target]["two_level_failed"]:
        stop.append(f"{target}'s own 2-level fit failed, so the target is not a 2-level score")
    if margins["template"] < EXCLUDE_WITHIN:
        stop.append(f"the template is within {EXCLUDE_WITHIN} of the target")
    if set(close) & set(seeds):
        stop.append(f"the rule would leave out seeds: {sorted(set(close) & set(seeds))}")
    if len(close) > MAX_EXCLUDED:
        stop.append(f"the rule would leave out {len(close)} programs besides {target}")
    return {"target": target, "target_score": goal, "tolerance": TOLERANCE, "exclude_within": EXCLUDE_WITHIN,
            "margins": dict(sorted(margins.items(), key=lambda kv: kv[1])),
            "library_exclude": sorted({target, *close}), "stop": stop, "ok": not stop}


def main(argv: list[str] | None = None) -> int:
    from mevolve.freeze import _git_sha
    from mevolve.literature import LITERATURE

    from . import seeds
    from .run import load_config, seed_metrics

    parser = argparse.ArgumentParser(description="Gate 4: score everything Claude is shown before the search")
    parser.add_argument("--config", type=Path, required=True, help="the search's run config")
    parser.add_argument("--target", required=True, choices=sorted(LITERATURE))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if not config["simulated"]:
        parser.error("the pre-check is for a search on simulated subjects (config simulated)")
    Path(config["results_dir"]).mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, code in [("template", seeds.TEMPLATE)] + [(n, seeds.literature_source(n)) for n in sorted(LITERATURE)]:
        m = seed_metrics(name, code, config)
        rows[name] = {"score": m["combined_score"], "two_level_failed": m["public"]["two_level_failed"],
                      "generation_score": m["public"]["generation_score"], "seconds": m["private"]["seconds"]}
        print(f"{name:32s} {m['combined_score']:8.3f}{'  (2-level failed)' if rows[name]['two_level_failed'] else ''}",
              flush=True)
    result = decide({k: {"score": v["score"], "two_level_failed": v["two_level_failed"]} for k, v in rows.items()},
                    args.target, config["seeds"])
    result.update({"mevolve_git_sha": _git_sha(Path(__file__).resolve().parents[3]), "config": str(args.config),
                   "simulated": config["simulated"], "max_em": config["max_em"], "n_draws": config["n_draws"],
                   "programs": rows})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"target {args.target} {result['target_score']:.3f}; leave out {result['library_exclude']}", flush=True)
    for reason in result["stop"]:
        print(f"STOP: {reason}", flush=True)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())

"""Gate 4's pre-check, before any Claude call: is the target out of reach of every program Claude sees?

The second half of gate 4 asks whether a short search on subjects simulated from a known program
(the target) reaches that program's score. Claude sees the seeds (literature programs or program
files), the template and, unless ``library_in_prompt`` is false, the literature library, and can copy
any of them in one proposal. So every one is scored first on the same
simulated subjects, with the search's settings, and the rule fixed before the search (study 40,
variant ``gate4-recovery``) is applied:

- the target is left out of the library, always (when it is a library program; a target from a
  file, outside the library, is never in it);
- any other library program within ``2 x TOLERANCE`` of the target is left out too;
- stop and ask (Han) if the target's own 2-level fit fails, if the rule would leave out a seed or
  more than ``MAX_EXCLUDED`` programs, or if the template is already that close;
- with ``--reference`` (repeatable: the target without one of its mechanisms, e.g. Hattori2019 for
  Hattori2019 plus satiety; a literature name or a program file, scored but not shown), also stop
  if the target beats it by less than ``2 x TOLERANCE``: a pass would not need the mechanism.

Scores are cached like the search's seeds (``seed_metrics``), so the search reuses them.

    python -m mevolve.search.precheck --config run.json --target Hattori2019 --out precheck.json
    python -m mevolve.search.precheck --config run.json --target path/to/generator.py --out precheck.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

TOLERANCE = 0.2         # the pass tolerance on the x100 scale (study 40 PLAN: 0.002 in pooled likelihood)
EXCLUDE_WITHIN = 2 * TOLERANCE
MAX_EXCLUDED = 3        # besides the target; more would narrow what the search is shown too far


def decide(scores: dict[str, dict], target: str, seeds: list[str], target_in_library: bool = True,
           references: tuple[str, ...] = ()) -> dict:
    """The rule, from each program's ``{"score", "two_level_failed"}`` (x100). ``template`` is the
    near-empty program on island 0; every other key is a literature program, or the target."""
    goal = scores[target]["score"]
    margins = {name: goal - s["score"] for name, s in scores.items()}      # unrounded: the rule uses these
    close = sorted(n for n, m in margins.items() if n not in (target, "template") and m < EXCLUDE_WITHIN)
    stop = []
    missing = sorted(n for n, s in scores.items() if not math.isfinite(s["score"]))
    if missing:                 # NaN compares False with everything, so no rule below would fire
        stop.append(f"no finite score for {missing}")
    if scores[target]["two_level_failed"]:
        stop.append(f"{target}'s own 2-level fit failed, so the target is not a 2-level score")
    if margins["template"] < EXCLUDE_WITHIN:
        stop.append(f"the template is within {EXCLUDE_WITHIN} of the target")
    if {target, *close} & set(seeds):        # a seed is on an island whatever the library holds
        stop.append(f"the rule would leave out seeds: {sorted({target, *close} & set(seeds))}")
    for ref in references:
        if margins[ref] < EXCLUDE_WITHIN:
            stop.append(f"the target beats {ref}, the same program without its mechanism, by "
                        f"{margins[ref]:.3f}, less than {EXCLUDE_WITHIN}")
    if len(close) > MAX_EXCLUDED:
        stop.append(f"the rule would leave out {len(close)} programs besides {target}")
    return {"target": target, "target_score": goal, "tolerance": TOLERANCE, "exclude_within": EXCLUDE_WITHIN,
            "margins": {n: round(m, 4) for n, m in sorted(margins.items(), key=lambda kv: kv[1])},
            "library_exclude": sorted(({target} if target_in_library else set()) | set(close)),
            "stop": stop, "ok": not stop}


def main(argv: list[str] | None = None) -> int:
    from mevolve.freeze import _git_sha
    from mevolve.literature import LITERATURE

    from . import seeds
    from .run import load_config, seed_metrics

    parser = argparse.ArgumentParser(description="Gate 4: score everything Claude is shown before the search")
    parser.add_argument("--config", type=Path, required=True, help="the search's run config")
    parser.add_argument("--target", required=True, help="a literature program's name, or a program file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", action="append", default=[],
                        help="repeatable: the target without one of its mechanisms (a literature program's name or a "
                             "program file, scored but not shown); the target must beat it by 2 x TOLERANCE")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if not config["simulated"]:
        parser.error("the pre-check is for a search on simulated subjects (config simulated)")
    # its own working directory: the search needs an empty results_dir; the seed cache is shared
    config = {**config, "results_dir": tempfile.mkdtemp(prefix="precheck-")}
    in_library = args.target in LITERATURE
    if not in_library and not Path(args.target).is_file():
        parser.error(f"--target {args.target} is neither a literature program nor a file")
    target = args.target if in_library else Path(args.target).stem
    if not in_library and (target in LITERATURE or target == "template"):
        parser.error(f"the target file's name {target} is taken (a literature program or the template)")
    # everything Claude is shown: the template, the seeds, and the library when it is in the prompt
    shown = [("template", seeds.TEMPLATE)] + [seeds.seed_name_and_source(s) for s in config["seeds"]]
    if config["library_in_prompt"]:
        shown += [(n, seeds.literature_source(n)) for n in sorted(LITERATURE)]
    references = [seeds.seed_name_and_source(r) for r in args.reference]
    target_code = seeds.literature_source(target) if in_library else seeds.wrap(Path(args.target).read_text())
    programs = {}                       # each program once; one name never stands for two programs
    for name, code in shown + references + [(target, target_code)]:
        if programs.setdefault(name, code) != code:
            parser.error(f"two different programs are named {name}")
    rows = {}
    for name, code in programs.items():
        m = seed_metrics(name, code, config)
        rows[name] = {"score": m["combined_score"], "two_level_failed": m["public"]["two_level_failed"],
                      "generation_score": m["public"]["generation_score"], "seconds": m["private"]["seconds"]}
        print(f"{name:32s} {m['combined_score']:8.3f}{'  (2-level failed)' if rows[name]['two_level_failed'] else ''}",
              flush=True)
    result = decide({k: {"score": v["score"], "two_level_failed": v["two_level_failed"]} for k, v in rows.items()},
                    target, [seeds.seed_name_and_source(s)[0] for s in config["seeds"]], in_library,
                    tuple(name for name, _ in references))
    result.update({"mevolve_git_sha": _git_sha(Path(__file__).resolve().parents[3]), "config": str(args.config),
                   "target_source": args.target,
                   "simulated": config["simulated"], "max_em": config["max_em"], "n_draws": config["n_draws"],
                   "programs": rows})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"target {target} {result['target_score']:.3f}; leave out {result['library_exclude']}", flush=True)
    for reason in result["stop"]:
        print(f"STOP: {reason}", flush=True)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())

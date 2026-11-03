"""Launch one search: ShinkaEvolve in-process, with MEvolve's shim, seeds and evaluator.

    python -m mevolve.search.run --config run.json

The config (JSON) names the data (freeze, cohorts file, generation and validation cohort names),
the evaluator settings (workers per evaluation, max_em, n_draws), the model and its effort, the
dollar cap, the number of proposals, the seeds (literature names, one per island after the
template's island 0), Shinka's concurrency and migration, W&B, a node-local ``results_dir`` and an
``out_dir`` the results are copied to at the end. Every key is listed in ``DEFAULTS``.

In order: environment checks; a pricing snapshot with the Claude models, offline (Shinka's bundled
prices stop before Claude 5.x); the literature seeds evaluated (or read from ``seed_cache``) and
placed on their islands; the Claude shim installed; Shinka's runner built and run; then a summary
(islands, cost from the ledger) and the copy to ``out_dir``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

DEFAULTS = {
    "results_dir": None, "out_dir": None, "freeze": None, "cohorts": None,
    "generation": None, "validation": None, "workers": 16, "max_em": 20, "n_draws": 2048,
    "model": "claude-sonnet-5-5", "effort": "medium", "max_usd": 3.5, "max_tokens": 32000,
    "n_proposals": 10, "seeds": ["Miller-RHG", "Lopez-DoubleTrace", "Lebedeva-PR", "Eckstein-BI"],
    "seed_cache": None, "library_in_prompt": True,
    "max_evaluation_jobs": 3, "max_proposal_jobs": 3, "migration_interval": 10, "migration_rate": 0.1,
    "archive_size": 40, "eval_time": "02:00:00", "patch_types": None,
    "wandb": None,      # {"entity", "project", "group", "name", "mode", "notes"} or None for no W&B
    "launch": {},       # free-form provenance copied into W&B and the summary (label, note, SHAs)
}


def load_config(path: Path) -> dict:
    config = {**DEFAULTS, **json.loads(Path(path).read_text())}
    def expand(key, value):         # e.g. "$TMPDIR/run" for node-local disk, set by the job
        out = os.path.expandvars(value)
        if "$" in out:
            raise ValueError(f"{key} has an unset variable: {value}")
        return out

    for key in ("results_dir", "out_dir", "seed_cache", "freeze", "cohorts"):
        if config[key]:
            config[key] = expand(key, config[key])
    for section in ("wandb", "launch"):
        if config[section]:
            config[section] = {k: expand(f"{section}.{k}", v) if isinstance(v, str) else v
                               for k, v in config[section].items()}
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    missing = [k for k in ("results_dir", "out_dir", "freeze", "cohorts", "generation", "validation")
               if not config[k]]
    if missing:
        raise ValueError(f"config needs {missing}")
    if config["max_evaluation_jobs"] * config["workers"] > (os.cpu_count() or 1):
        raise ValueError("max_evaluation_jobs x workers exceeds this node's CPUs")
    return config


def write_pricing_snapshot(results_dir: Path) -> Path:
    """Shinka's bundled prices plus the Claude models in ``claude.PRICES`` (per token)."""
    from shinka.pricing.catalog import CatalogSnapshot, ModelPrice, PricingCatalog, write_run_pricing_snapshot
    from shinka.pricing.normalization import load_bundled_entries

    from .claude import PRICES

    ours = tuple(ModelPrice(model_name=m, api_model_name=m, provider="anthropic", kind="llm",
                            input_price=p[0] / 1e6, output_price=p[1] / 1e6) for m, p in PRICES.items())
    entries = tuple(e for e in load_bundled_entries()
                    if not (e.provider == "anthropic" and e.api_model_name in PRICES)) + ours
    digest = hashlib.sha256(json.dumps([m for m in PRICES], sort_keys=True).encode()).hexdigest()
    snapshot = CatalogSnapshot(PricingCatalog(entries), "bundled", None, None, digest, False)
    return write_run_pricing_snapshot(snapshot, results_dir)


def seed_metrics(name: str, code: str, config: dict) -> dict:
    """A literature seed's evaluation, cached by code and evaluator settings."""
    from .shinka_eval import evaluate_program

    key = hashlib.sha256(json.dumps([code, Path(config["cohorts"]).read_text(), config["generation"],
                                     config["validation"], config["max_em"], config["n_draws"]]).encode()).hexdigest()[:16]
    cache = Path(config["seed_cache"]) / f"{name}_{key}.json" if config["seed_cache"] else None
    if cache and cache.exists():
        return json.loads(cache.read_text())
    path = Path(config["results_dir"]) / "seeds" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    metrics = evaluate_program(path, Path(config["freeze"]), Path(config["cohorts"]), config["generation"],
                               config["validation"], config["max_evaluation_jobs"] * config["workers"],
                               config["max_em"], config["n_draws"])
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(metrics, default=float) + "\n")
    return metrics


def build_runner(config: dict, task_message: str):
    from shinka.core import EvolutionConfig, ShinkaEvolveRunner
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig

    from . import shinka_eval
    from .seeds import TEMPLATE

    results_dir = Path(config["results_dir"])
    init_path = results_dir / "initial.py"
    init_path.write_text(TEMPLATE)
    wandb = config["wandb"] or {}
    evo = EvolutionConfig(
        task_sys_msg=task_message, num_generations=config["n_proposals"] + 1, language="python",
        llm_models=[config["model"]], llm_dynamic_selection=None,
        llm_kwargs={"temperatures": [1.0], "max_tokens": config["max_tokens"]},
        meta_llm_models=None, novelty_llm_models=None, embedding_model=None,
        init_program_path=str(init_path), results_dir=str(results_dir), use_text_feedback=True,
        **({"patch_types": config["patch_types"], "patch_type_probs": [1 / len(config["patch_types"])] * len(config["patch_types"])}
           if config["patch_types"] else {}),
        max_api_costs=config["max_usd"],
        enable_wandb_logging=bool(wandb), wandb_project=wandb.get("project", "shinka-evolve"),
        wandb_entity=wandb.get("entity"), wandb_group=wandb.get("group"), wandb_name=wandb.get("name"),
        wandb_mode=wandb.get("mode"), wandb_notes=wandb.get("notes"), wandb_tags=["mevolve"],
        wandb_config={"mevolve": {k: v for k, v in config.items() if k != "wandb"}},
    )
    job = LocalJobConfig(
        eval_program_path=str(Path(shinka_eval.__file__).resolve()), python_executable=sys.executable,
        numeric_threads_per_job=1, time=config["eval_time"],
        extra_cmd_args={"freeze": str(config["freeze"]), "cohorts": str(config["cohorts"]),
                        "generation": config["generation"], "validation": config["validation"],
                        "workers": config["workers"], "max_em": config["max_em"], "n_draws": config["n_draws"]})
    db = DatabaseConfig(num_islands=1 + len(config["seeds"]), archive_size=config["archive_size"],
                        migration_interval=config["migration_interval"], migration_rate=config["migration_rate"])
    runner = ShinkaEvolveRunner(evo, job, db, max_evaluation_jobs=config["max_evaluation_jobs"],
                                max_proposal_jobs=config["max_proposal_jobs"], max_db_workers=2)
    return runner, db


def summarize(config: dict, started: float) -> dict:
    from .seeds import islands_summary

    results_dir = Path(config["results_dir"])
    ledger = results_dir / "ledger.jsonl"
    calls = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    islands = islands_summary(results_dir)
    summary = {
        "config": config, "seconds": time.time() - started,
        "calls": len(calls), "cost_usd": sum(c["cost"] for c in calls),
        "stop_reasons": {r: sum(c.get("stop_reason") == r for c in calls)
                         for r in sorted({str(c.get("stop_reason")) for c in calls})},
        "errors": [c["error"] for c in calls if "error" in c],
        "cache_read_tokens": sum(c.get("cache_read_input_tokens", 0) for c in calls),
        "cache_write_tokens": sum(c.get("cache_creation_input_tokens", 0) for c in calls),
        "islands": {i: {"programs": len(p), "seeds": [x["seed"] for x in p if x["seed"]],
                        "best": max((x["score"] for x in p if x["correct"]), default=None)}
                    for i, p in sorted(islands.items())},
    }
    (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


def install_fake_llm(ledger: Path) -> None:
    """Replace Claude with a canned reply: the parent program with every bound widened by 10%,
    as a full rewrite. Exercises Shinka's loop, the evaluator and the islands at no cost."""
    import re

    import shinka.llm.query as shinka_query
    from shinka.llm.providers.result import QueryResult

    from .claude import _record

    def widen(match):
        lo, hi = float(match.group(1)), float(match.group(2))
        pad = 0.05 * (hi - lo)
        return f"({lo - pad if lo != 0 else lo:.4g}, {hi + pad:.4g})"

    async def fake(client, model, msg, system_msg, msg_history, output_model, model_posteriors=None, **kwargs):
        blocks = re.findall(r"```python\n(.*?)```", msg, re.S)
        code = re.sub(r"\((-?[0-9.]+), (-?[0-9.]+)\)", widen, blocks[0]) if blocks else ""
        content = f"<NAME>widened_bounds</NAME>\n<DESCRIPTION>Fake: bounds widened.</DESCRIPTION>\n<CODE>\n```python\n{code}```\n</CODE>"
        _record(ledger, {"time": time.time(), "model": "fake", "stop_reason": "end_turn", "cost": 0.0})
        return QueryResult(content=content, msg=msg, system_msg=system_msg,
                           new_msg_history=msg_history + [{"role": "assistant", "content": content}],
                           model_name=model, kwargs={}, input_tokens=0, output_tokens=0, cost=0.0)

    shinka_query.query_anthropic_async = fake


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one MEvolve search on ShinkaEvolve")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry", action="store_true", help="evaluate and place the seeds, build the runner, stop")
    parser.add_argument("--fake-llm", action="store_true",
                        help="a $0 rehearsal: canned full rewrites instead of Claude (use patch_types [\"full\"])")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.fake_llm:
        os.environ.setdefault("ANTHROPIC_API_KEY", "fake")     # Shinka checks the variable exists
    if not args.dry and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (export it from ~/.config/mevolve/ in the job)")
    for env in (Path.cwd() / ".env",):
        if env.exists():
            sys.exit(f"{env} exists; Shinka would load it over the job's keys")
    os.environ["SHINKA_PRICING_MODE"] = "offline"
    os.environ["SHINKA_LLM_BACKOFF_MAX_TRIES"] = "2"
    started = time.time()
    results_dir = Path(config["results_dir"])
    if results_dir.exists() and any(results_dir.iterdir()):
        sys.exit(f"{results_dir} is not empty; a run starts in a fresh directory")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "run_config.json").write_text(json.dumps(config, indent=1) + "\n")

    from . import claude, prompt, seeds

    write_pricing_snapshot(results_dir)
    task_message = prompt.task_message(config["library_in_prompt"])
    placed = []
    for island, name in enumerate(config["seeds"], start=1):
        code = seeds.literature_source(name)
        placed.append({"name": name, "code": code, "island": island, "metrics": seed_metrics(name, code, config)})
        print(f"seed {name} on island {island}: {placed[-1]['metrics']['combined_score']:.3f}", flush=True)
    claude.install(claude.Settings(ledger=results_dir / "ledger.jsonl", effort=config["effort"],
                                   max_usd=config["max_usd"], max_tokens=config["max_tokens"],
                                   cache_prefix=task_message))
    if args.fake_llm:
        install_fake_llm(results_dir / "ledger.jsonl")
    seeds.install_island_patch()
    from shinka.database import DatabaseConfig
    seeds.place(results_dir, DatabaseConfig(num_islands=1 + len(config["seeds"])), placed)
    runner, _ = build_runner(config, task_message)
    if not args.dry:
        runner.run()
    summary = summarize(config, started)
    print(json.dumps({k: summary[k] for k in ("calls", "cost_usd", "stop_reasons", "islands")}, default=str), flush=True)
    if config["out_dir"]:
        shutil.copytree(results_dir, Path(config["out_dir"]), dirs_exist_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

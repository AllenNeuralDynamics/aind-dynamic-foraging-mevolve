"""Gate 4, first half: the 2-level EM fitter recovers simulated typical values and spreads (#36).

For each model, 24 generation subjects are simulated from known 2-level parameters: study 12's
2-level EM fit of the real subject (Rescorla-Wagner's diverged, so its per-subject fit with spread
0.3). Each simulated subject has the real subject's fit and score sessions, with the same lengths,
stages and indices; each session's parameters are drawn around the typical values, and its choices
come from the program playing ``aind_behavior_gym``'s coupled block task with baiting. The
evaluator then fits them as reported scores are fitted (2-level EM, ``max_em`` 100), and the
recovered typical values, spreads and score are checked against the pass criteria fixed in #36.
The simulated subjects are saved with their generating parameters for gate 4's second half.

    python -m mevolve.gate4 --freeze <dir> --cohorts cohorts.json --studies <studies checkout> \\
        --out <dir> --workers 56
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from . import evaluate, fitting, literature, program
from .data import Freeze
from .freeze import _git_sha

MODELS = ("Hattori2019", "ActorCritic", "Rescorla-Wagner")
FIXED_SPREAD = {"Rescorla-Wagner": 0.3}    # its real 2-level fits diverged: per-subject fit, this spread
N_SUBJECTS = 24
MAX_EM = 100
TWO_LEVEL = "studies/12-cognitive-literature-leaderboard/analysis/two_level_em_fits.csv"
PER_SUBJECT = "studies/12-cognitive-literature-leaderboard/analysis/per_subject_fits.csv"
OFF_BOUND = 3.0      # |mu| on the probit scale up to which a typical value is off its bounds
# off a bound, the median |recovered - true mu| (probit scale) must not exceed the per-subject fit's on
# the same data (#36, amended 2026-11-03: a fixed 0.1 was tighter than ~11 sessions allow)
BOUND_TOL = 0.05     # median |recovered - true| / range, natural scale, for a typical value on a bound
SIGMA_MIN = 0.2      # spreads checked only where the true spread is at least this
SIGMA_RATIO = 2.0    # median recovered / true spread within [1/2, 2]


def truths(model: str, subjects: list[str], studies: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each subject's generating typical values and log spreads on the probit scale."""
    prog = program.literature(literature.LITERATURE[model][0])
    rows = pd.read_csv(studies / (PER_SUBJECT if model in FIXED_SPREAD else TWO_LEVEL))
    rows = rows[rows["literature_model"] == model].set_index("subject_id")
    rows.index = rows.index.astype(str)
    out = {}
    for s in subjects:
        if model in FIXED_SPREAD:
            p = json.loads(rows.loc[s, "fitted_params"])
            log_sigma = np.full(len(prog.params), np.log(FIXED_SPREAD[model]))
        else:
            p = json.loads(rows.loc[s, "typical_params"])
            log_sigma = np.asarray(json.loads(rows.loc[s, "log_sigma"]))   # in prog.params order
        out[s] = (np.asarray(fitting.from_params(prog, p)), log_sigma)
    return out


def _stepper(prog):
    """Choice probabilities after an update, jitted: one call per simulated trial."""
    def step(latents, choice, reward, p):
        latents = prog.update(latents, choice, reward, p)
        return latents, jax.nn.softmax(prog.predict(latents, p))
    return jax.jit(step)


def simulate(prog, mu, log_sigma, real: list, rng, step) -> list:
    """One subject: each real session replaced by a simulated one of the same length, stage and
    index, at parameters drawn around ``mu``; returns the sessions and each one's parameters."""
    from aind_behavior_gym.dynamic_foraging.task.coupled_block_task import CoupledBlockTask   # via the models library
    sessions, params = [], []
    for r in real:
        p = fitting.to_params(prog, mu + np.exp(log_sigma) * rng.normal(size=len(mu)))
        s = dataclasses.replace(r, subject_id=f"sim-{r.subject_id}", session_id=f"sim-{r.session_id}",
                                task="Coupled Baiting")
        task = CoupledBlockTask(reward_baiting=True, num_trials=len(r.choices), seed=int(rng.integers(2**31)))
        task.reset()
        latents = prog.init_latents(p, fitting.session_facts(s))
        probs = jax.nn.softmax(prog.predict(latents, p))
        choices, rewards = np.zeros(len(r.choices), np.int32), np.zeros(len(r.choices))
        for t in range(len(r.choices)):
            c = int(rng.random() < float(probs[1]))
            _, reward, _, _, _ = task.step(c)
            choices[t], rewards[t] = c, reward
            latents, probs = step(latents, c, reward, p)
        sessions.append(dataclasses.replace(s, choices=choices, rewards=rewards))
        params.append({k: float(v) for k, v in p.items()})
    return sessions, params


def check(prog, true: dict, records: list[dict]) -> dict:
    """The pass criteria of #36, per parameter, from the recovered records."""
    lo, hi = (np.asarray(a) for a in fitting.bounds_arrays(prog))
    mu_true = np.stack([true[r["subject_id"].removeprefix("sim-")][0] for r in records])
    sig_true = np.exp(np.stack([true[r["subject_id"].removeprefix("sim-")][1] for r in records]))
    mu_rec = np.stack([np.asarray(fitting.from_params(prog, r["typical_params"])) for r in records])
    mu_per = np.stack([np.asarray(fitting.from_params(prog, r["params"])) for r in records])
    sig_rec = np.exp(np.asarray([r["log_sigma"] for r in records]))
    nat = lambda x: lo + (hi - lo) * jax.scipy.stats.norm.cdf(x)      # noqa: E731
    params = {}
    for j, k in enumerate(prog.params):
        off = np.abs(mu_true[:, j]) <= OFF_BOUND
        row = {"n_off_bound": int(off.sum()), "n_on_bound": int((~off).sum())}
        if off.any():
            row["mu_error_median"] = float(np.median(np.abs(mu_rec[off, j] - mu_true[off, j])))
            row["per_subject_error_median"] = float(np.median(np.abs(mu_per[off, j] - mu_true[off, j])))
            row["mu_ok"] = row["mu_error_median"] <= row["per_subject_error_median"]
        if (~off).any():
            err = np.abs(np.asarray(nat(mu_rec[~off]))[:, j] - np.asarray(nat(mu_true[~off]))[:, j])
            row["bound_error_median"] = float(np.median(err / (hi[j] - lo[j])))
            row["bound_ok"] = row["bound_error_median"] <= BOUND_TOL
        spread = off & (sig_true[:, j] >= SIGMA_MIN)
        row["n_spread_checked"] = int(spread.sum())
        if spread.any():
            ratio = sig_rec[spread, j] / sig_true[spread, j]
            row["sigma_ratio_median"] = float(np.median(ratio))
            row["sigma_ratio_q10_q90"] = np.percentile(ratio, [10, 90]).tolist()
            row["sigma_ok"] = 1 / SIGMA_RATIO <= row["sigma_ratio_median"] <= SIGMA_RATIO
        params[k] = row
    score = evaluate.program_score(records)
    ok = all(v for row in params.values() for name, v in row.items() if name.endswith("_ok")) \
        and not score["two_level_failed"]
    return {"params": params, "score": score, "converged": float(np.mean([r["converged"] for r in records])),
            "ok": bool(ok)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gate 4, first half: recovery of simulated 2-level subjects")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True, help="cohort lists (mevolve.cohorts output)")
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=MODELS, help="repeatable; default all three")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    sha = _git_sha(Path(__file__).resolve().parents[2])
    studies_sha = _git_sha(args.studies.resolve())
    freeze = Freeze(args.freeze)
    subjects = json.loads(args.cohorts.read_text())["cohorts"]["generation"][:N_SUBJECTS]
    if len(subjects) < N_SUBJECTS:
        parser.error(f"the generation cohort has {len(subjects)} subjects; gate 4 needs {N_SUBJECTS} (#36)")
    args.out.mkdir(parents=True, exist_ok=True)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()     # noqa: E731
    result = {"mevolve_git_sha": sha, "studies_git_sha": studies_sha, "seed": args.seed, "max_em": MAX_EM,
              "freeze_manifest_sha256": digest(args.freeze / "manifest.json"),
              "cohorts_sha256": digest(args.cohorts),
              "sources": [{"path": p, "sha256": digest(args.studies / p)} for p in (TWO_LEVEL, PER_SUBJECT)],
              "criteria": {"off_bound": OFF_BOUND, "mu": "<= per-subject fit", "bound_tol": BOUND_TOL,
                           "sigma_min": SIGMA_MIN, "sigma_ratio": SIGMA_RATIO},
              "subjects": subjects, "models": {}}
    for model in args.model or MODELS:
        path = program.LITERATURE_DIR / f"{literature.LITERATURE[model][0]}.py"
        prog = program.load(path)
        true = truths(model, subjects, args.studies)
        rng = np.random.default_rng([args.seed, MODELS.index(model)])
        step, data, saved = _stepper(prog), [], []
        t0 = time.time()
        for s in subjects:
            fit_sessions, fit_p = simulate(prog, *true[s], freeze.subject_sessions(s, "fit"), rng, step)
            score_sessions, score_p = simulate(prog, *true[s], freeze.subject_sessions(s, "score"), rng, step)
            data.append(evaluate.SubjectData(f"sim-{s}", fit_sessions, score_sessions))
            saved.append({"subject_id": f"sim-{s}", "mu": true[s][0].tolist(), "log_sigma": true[s][1].tolist(),
                          "sessions": [{"session_id": x.session_id, "index": x.index, "role": x.role,
                                        "stage": x.stage, "params": p, "choices": x.choices.tolist(),
                                        "rewards": x.rewards.tolist()}
                                       for x, p in zip(fit_sessions + score_sessions, fit_p + score_p)]})
        t_sim = time.time() - t0
        with gzip.open(args.out / f"simulated_{model}.json.gz", "wt") as f:
            json.dump({"model": model, "params": list(prog.params), "seed": args.seed, "subjects": saved}, f)
        records = evaluate.evaluate(path, data, regime="2level", seed=args.seed, workers=args.workers,
                                    max_em=MAX_EM)
        c = check(prog, true, records)
        result["models"][model] = {"program_sha256": digest(path), "simulate_seconds": t_sim,
                                   "wall_seconds": time.time() - t0, **c, "records": records}
        (args.out / "gate4.json").write_text(json.dumps(result, indent=1) + "\n")
        print(f"{model}: {'PASS' if c['ok'] else 'FAIL'}, two-level {c['score']['two_level']:.5f} vs per-subject "
              f"{c['score']['per_subject']:.5f}, converged {c['converged']:.2f}, wall {time.time() - t0:.0f} s",
              flush=True)
        for k, row in c["params"].items():
            print(f"  {k}: {row}", flush=True)
    failed = [m for m, c in result["models"].items() if not c["ok"]]
    if failed:
        sys.exit(f"gate 4 failed for {', '.join(failed)}")


if __name__ == "__main__":
    main()

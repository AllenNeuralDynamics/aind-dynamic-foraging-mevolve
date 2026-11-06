# MEvolve

LLM-driven evolutionary discovery of interpretable cognitive models of mouse dynamic foraging.
The M stands for mice and for models; the repo pairs with
[`aind-dynamic-foraging-models`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models).

**Status: building.** Every settled decision is recorded in [`docs/adr/`](docs/adr/) with its
vocabulary in [`CONTEXT.md`](CONTEXT.md); the build order is study 40's `PLAN.md`.

## Use

```bash
mamba env create -f environment.yml && conda activate mevolve && pip install -e ".[test]"
pytest -m "not data"                     # unit tests, no data needed
```

**Data freeze (build step 1, gate 0).** Run once, on a compute node, in the shared
`dynamic-foraging-bfm-cpu` env, which has the wrapper's dependencies; the wrapper checkout must be
clean and at the commit the comparators ran:

```bash
conda activate dynamic-foraging-bfm-cpu      # the freeze only
PYTHONPATH=src python -m mevolve.freeze --wrapper <wrapper checkout> --out <scratch>/freeze/20260603
conda activate mevolve                       # everything else
python -m mevolve.gate0 --freeze <scratch>/freeze/20260603 --studies <studies checkout>
python -m mevolve.cohorts --freeze <scratch>/freeze/20260603 --studies <studies checkout> \
    --seed <seed> --size generation=48 --size validation=48 --size selection=100 --out cohorts.json
```

The freeze holds every trial of every session in the wrapper's split (ignored ones as response 2;
sessions with only ignored trials are not in the split), each session's pool, fit or score
role, date-order index, task and stage, and a manifest with the settings, the wrapper commit and
file digests. It lives on scratch, never in this repo; study 40 records its digests and its cohort
lists.

**Programs, the fitter and the evaluator (build steps 3 and 2).** A program is one Python file
defining `PARAMS` (at least one fitted parameter), `N_LATENT`, `LATENT_NAMES`, `init_latents`,
`predict` and `update` (`src/mevolve/program.py`); the literature models are in
`src/mevolve/programs/literature/`. The fitter (`mevolve.fitter.fit`) fits a program to a subject's
sessions per subject (differential evolution, then L-BFGS and Nelder-Mead) or as a 2-level model
(session -> subject) fitted by Laplace-EM ([ADR 0004](docs/adr/0004-two-level-by-laplace-em.md)),
and returns its parameters and, on request, its latent variables on every trial. It moves to
`aind-dynamic-foraging-models` once gate 4 passes ([ADR 0005](docs/adr/0005-fitter-moves-to-the-models-library.md)).
The evaluator (`mevolve.evaluate`) is the search's use of it: it fits each subject's fit sessions,
scores the held-out ones and returns the score and feedback, one single-threaded process per
subject. The gates run as SLURM jobs on a compute node:

```bash
python -m mevolve.gate1a --freeze <freeze> --out gate1a.json --workers 64        # programs vs foragers
python -m mevolve.gate1 --regime subject --model Hattori2019 --freeze <freeze> --studies <studies> --out <dir>
python -m mevolve.gate1 --regime 2level --model CompareToThreshold --model Hattori2019 ...       # vs study 13
python -m mevolve.gate2 --freeze <freeze> --cohorts cohorts.json --model Hattori2019 --model CompareToThreshold --out gate2.json
python -m mevolve.gate4 --freeze <freeze> --cohorts cohorts.json --studies <studies> --out <dir> --workers 56   # recovery
```

Reported 2-level scores of the literature models (study 12's regime 2) run Laplace-EM to
convergence, on any pool of the freeze, from a clean checkout:

```bash
python -m mevolve.score --model Hattori2019 --pool all --freeze <freeze> --out <dir> --workers 112
```

**A search on many nodes** (`eval_launch: "srun"`, `stage_dir` on /allen; #59). One Slurm job holds
`--ntasks=N --cpus-per-task=C --mem-per-cpu=1500M`. Each evaluation is an `srun` step of `C` CPUs with
`workers = C / 2` (one per core), and `max_evaluation_jobs = N`. Choose `C` from the free-CPU landscape
so that the job **ends** earliest (Han, 2026-11-05), not by the speed of one evaluation:

- **No GPU nodes:** the cluster's submit filter gives every CPU-only job `Features=cpu`, so only free
  CPUs on `cpu`-feature nodes count; on 2026-11-05 most free CPUs were on GPU nodes.
- **Count the slots:** for each candidate `C`, count the slots that fit on `cpu` nodes, in both CPUs and
  memory (`scontrol show node -o`: `CPUTot - CPUAlloc` and `RealMemory - AllocMem`). Read Slurm's
  planned start with `sbatch --test-only`.
- **Estimate the run:** one 48 + 48 evaluation is about 9,000 CPU-seconds, so it takes about
  9,000 / workers × 1.2 s, and never less than its slowest mouse (about 250 s).

On 2026-11-05, slots of 56 CPUs were planned for the next morning, while 17 slots of 24 CPUs fit at once.

**Stopping rule** (`stop_se`, `stop_window`; #63). Besides `max_usd` and `n_proposals`, a search can stop
when `stop_window` proposals finish without the best validation score rising more than `stop_se` above
an anchor. The anchor starts at the best seed (or template) and moves only on such a rise, so small
rises count once they add up past `stop_se`. Every finished proposal counts, scored or not. When the
rule is met, no new proposals start; running evaluations and Claude calls finish and are recorded, as
under the dollar cap, and `stop_rule.json` (and `run_summary.json`) say the rule ended the run. The rule
is read from the database each time (`mevolve.search.stopping`), so a resumed run gets the same answer.

## What it does

- An LLM (Claude) proposes **programs**: short JAX modules that predict each choice of a mouse
  from the session's earlier choices and rewards, with a few named parameters.
- An **evaluator** fits each program's parameters per subject and per session, scores it on
  held-out sessions, and returns diagnostics that guide the next proposals.
- An evolutionary search, run by [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve), keeps
  the best programs and breeds new ones from them. A simplification stage then trades fit for
  readability.

The scientific question, the comparisons and the results live in study 40 of
[`aind-dynamic-foraging-bfm-studies`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies)
([issue 73](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/issues/73)).
This repo holds the tool.

## Where it sits

| repo | role for MEvolve |
|---|---|
| this repo | data freeze, evaluator, seed library, Shinka adapter, launch wrapper |
| [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) | the search engine, pinned to one commit |
| [`aind-dynamic-foraging-models`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models) | literature models as reference implementations; the task simulator through `aind-behavior-gym` |
| [`aind-dynamic-foraging-bfm-wrapper`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper) and [`-dispatcher`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher) | used once, to build the same subject and session split as the GRU and the cognitive baselines |
| [`aind-dynamic-foraging-bfm-studies`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies) | study 40: manifests, launch records, frozen results, reports |

Runs log to W&B entity `NeuralDynamics`, project `dfbfm-mevolve`. Runs logged before 2026-11-03
stay in `aind-dynamic-foraging-model` ([ADR 0003](docs/adr/0003-wandb-neuraldynamics-account.md)).

## License

MIT, see [LICENSE](LICENSE).

# MEvolve

LLM-driven evolutionary discovery of interpretable cognitive models of mouse dynamic foraging.
The M stands for mice and for models; the repo pairs with
[`aind-dynamic-foraging-models`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models).

**Status: round 1 done; scientist mode proposed.** Search and simplification modes have run study 40's
first round (results below). A third mode, scientist mode, is proposed in #87. Every settled decision
is recorded in [`docs/adr/`](docs/adr/) with its vocabulary in [`CONTEXT.md`](CONTEXT.md).

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

**Simplification runs** (`"mode": "simplify"`, `floor`, `seed_programs`, `stop_window_simplify`;
[ADR 0006](docs/adr/0006-simplification-mode.md), #83). The same `python -m mevolve.search.run --config
run.json` rewrites `seed_programs` (program files; the first on island 0) to be simpler. A candidate is
correct only if its validation score (×100, the search's evaluator) is at or above `floor`, and it is
ranked by minus its Halstead effort. Every seed program must reach the floor, or the run stops before
any Claude call; set `max_params` and `max_latent` to the search's limits, and keep the program files
on shared disk (`--resume` reads them again). The run stops after `stop_window_simplify` proposals
(default 300) without a correct program of lower effort, or at `max_usd`; set `n_proposals` well above
the window. Each floor is its own run; the floors are study 40's.

**Snapshots and resume** (`snapshot_minutes`, `snapshot_dir`, `--resume`; #65). The database lives on the
first node's local disk until the end of the run. With `snapshot_minutes`, a consistent copy of it
(SQLite's backup API), the run's small files (ledger, bandit state, configs) and every settled `gen_*`
directory go to `snapshot_dir` on shared disk (not inside `out_dir`). At SIGTERM the snapshot is taken
first; give the job `--signal=B:TERM@120` and forward it from the batch script to the search. A new
job with the same config and `--resume` fills a fresh `results_dir` from the snapshot and continues:
Shinka restores its generation counter and spent dollars from the database, and the dollar cap reads
the restored ledger. Programs that finished after the last snapshot are proposed again.

## What it does

- An LLM (Claude) proposes **programs**: short JAX modules that predict each choice of a mouse
  from the session's earlier choices and rewards, with a few named parameters.
- An **evaluator** fits each program's parameters per subject and per session, scores it on
  held-out sessions, and returns diagnostics that guide the next proposals.
- An evolutionary search, run by [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve), keeps
  the best programs and breeds new ones from them. A simplification stage then trades fit for
  readability.

![Study 40's procedure: (a) four cohorts of mice; (b) the search loop, with the score from the validation mice and Claude's feedback from the generation mice; (c) simplification, pre-registration on the selection mice and one score on the test mice, as the gap closed from HB-CTT to the GRU](https://raw.githubusercontent.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/ce2aab2fcb4e6d9d138da486b3e5c920f6076c82/studies/40-cognitive-llm-discovery/analysis/fig_procedure.svg)

*The procedure, from study 40 ([`fig_procedure.svg`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/ce2aab2fcb4e6d9d138da486b3e5c920f6076c82/studies/40-cognitive-llm-discovery/analysis/fig_procedure.svg)).*

The scientific question, the comparisons and the results live in
[study 40](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/tree/main/studies/40-cognitive-llm-discovery) of
[`aind-dynamic-foraging-bfm-studies`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies)
([issue 73](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/issues/73)).
This repo holds the tool.

## Preliminary results (study 40, round 1)

The source is study 40's reports at studies commit `3f724cb`; numbers here are quoted from them. r5
is a draft.

- **The search** ([r2](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/3f724cb821a92b8d1aee3e827b5435cd724ff45e/studies/40-cognitive-llm-discovery/analysis/reports/r2-search-curve.md)) ran 2,562 proposals in 42 hours for $209
  of Claude API (Opus 5.5).
  - It climbed +0.167 (likelihood ×100) above the best literature model, Miller-RHG, on the
    validation mice, nearly all of it in the first 500 proposals.
  - The top programs share a core: counterfactual update, forgetting of the unchosen side, and fast
    and slow habits. They differ in add-ons (satiety, engagement, experience, reward rate, bait).
  - On 100 mice the search never used, the early gain holds; the late gain does not.
- **On the test mice** ([r3](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/3f724cb821a92b8d1aee3e827b5435cd724ff45e/studies/40-cognitive-llm-discovery/analysis/reports/r3-search-best.md)), the program chosen on the selection
  mice, `satiety_engagement`, closes **0.59 of the gap from HB-CTT to the GRU** (95% CI 0.53 to 0.64,
  on the 146 test mice shared with the GRU).
  - About half of that comes from the fitting regime: Miller-RHG fitted the same way (2-level EM)
    already closes 0.30.
  - Its mouse-average behaviour curves beat Miller-RHG's, but the GRU stays best on 8 of the 9.
- **Simplification** ([r4](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/3f724cb821a92b8d1aee3e827b5435cd724ff45e/studies/40-cognitive-llm-discovery/analysis/reports/r4-simplification.md)) cut Halstead effort from 2,634 to
  403 and 107 while keeping 0.55 and 0.53 of the gap. No program met the 90% floor.
- **Individual differences** ([r5](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/3f724cb821a92b8d1aee3e827b5435cd724ff45e/studies/40-cognitive-llm-discovery/analysis/reports/r5-individual-differences.md), draft): the top
  programs fit some mice better than others, stably across session halves (r 0.53).
  - Each add-on helps about half of the mice and hurts the rest, so pooled scores tie.
  - Choosing a program per mouse gains little on unseen sessions (+0.19 mnats per trial, CI −0.14 to
    +0.51).

![Test subjects: gap closed from HB-CTT to the GRU, with paired standard errors](https://raw.githubusercontent.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/3f724cb821a92b8d1aee3e827b5435cd724ff45e/studies/40-cognitive-llm-discovery/analysis/fig_r3_test.svg)

*Gap closed on the test subjects ([r3](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/3f724cb821a92b8d1aee3e827b5435cd724ff45e/studies/40-cognitive-llm-discovery/analysis/reports/r3-search-best.md), [`fig_r3_test.svg`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/3f724cb821a92b8d1aee3e827b5435cd724ff45e/studies/40-cognitive-llm-discovery/analysis/fig_r3_test.svg)).*

![Simplification: fit against Halstead effort](https://raw.githubusercontent.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/3f724cb821a92b8d1aee3e827b5435cd724ff45e/studies/40-cognitive-llm-discovery/analysis/fig_r4_frontier.svg)

*Fit against complexity at three floors ([r4](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/3f724cb821a92b8d1aee3e827b5435cd724ff45e/studies/40-cognitive-llm-discovery/analysis/reports/r4-simplification.md), [`fig_r4_frontier.svg`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/3f724cb821a92b8d1aee3e827b5435cd724ff45e/studies/40-cognitive-llm-discovery/analysis/fig_r4_frontier.svg)).*

**What round 1 taught the tool.**
1. A pooled score saturates early.
2. Near-ties between programs reflect real but small differences between mice (r5).
3. Choosing the winner on fixed validation mice does not carry over well to other mice.

The next step is therefore not a longer search but a different mode: scientist mode (#87; ADR 0007
proposed). A Claude agent forms claims about the mice, tests them on discovery mice, has them
attacked by a reviewer, and confirms them on mice it never used. Its first user is study 41, mouse
strategy profiles ([studies #149](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/issues/149)).
Its residuals will come from study 17's cross-validated GRU map
([studies #148](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/issues/148)).

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

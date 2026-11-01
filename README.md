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

Runs log to W&B entity `NeuralDynamics`, project `aind-dynamic-foraging-model`.

## License

MIT, see [LICENSE](LICENSE).

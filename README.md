# MEvolve

LLM-driven evolutionary discovery of interpretable cognitive models of mouse dynamic foraging.
The M stands for mice and for models; the repo pairs with
[`aind-dynamic-foraging-models`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models).

**Status: design.** There is no code yet. The design is being sharpened before the first line of
code, and every settled decision is recorded in [`docs/adr/`](docs/adr/) with its vocabulary in
[`CONTEXT.md`](CONTEXT.md).

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
| [`aind-dynamic-foraging-models`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models) | published models as reference implementations; the task simulator through `aind-behavior-gym` |
| [`aind-dynamic-foraging-bfm-wrapper`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper) and [`-dispatcher`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher) | used once, to build the same subject and session split as the GRU and the cognitive baselines |
| [`aind-dynamic-foraging-bfm-studies`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies) | study 40: manifests, launch records, frozen results, reports |

Runs log to W&B entity `NeuralDynamics`, project `aind-dynamic-foraging-model`.

## License

MIT, see [LICENSE](LICENSE).

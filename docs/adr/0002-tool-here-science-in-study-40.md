# The tool lives here; the science lives in study 40

This repo holds code only: the data freeze, the evaluator, the seed library, the Shinka adapter and
the launch wrapper. Configurations, launch records, frozen results and reports live in study 40 of
`aind-dynamic-foraging-bfm-studies`, the same split the lab uses between the training wrapper and the
studies. There, comparator numbers (the per-subject MLE baselines, the hierarchical Bayesian
baselines, the GRU) are read from the other studies' committed JSON, and a correction to any of them
marks study 40's reports stale; a separate repo would have to copy them by hand (decided with Han,
2026-10-31).

## Considered options

- **Everything in this repo.** One repo and one PR stream, but comparators copied by hand with no
  staleness check, provenance rebuilt from scratch, and no place in the studies atlas.

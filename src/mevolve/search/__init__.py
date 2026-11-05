"""The search: ShinkaEvolve, pinned, run in-process with MEvolve's shim (ADR 0001).

    claude       the Claude request Shinka cannot make itself, a ledger of every call, the dollar cap
    checks       static checks on a candidate program before any of it runs
    prompt       the task message Claude reads with every proposal
    shinka_eval  the evaluation script Shinka runs for each candidate
    seeds        seed programs, and their placement on chosen islands
    run          the launch: configs, pricing snapshot, seeds, then Shinka's runner
    archive      the run directory as a W&B artifact, at the end of every run
    embed_server code embeddings from a local model, for the viewer; they never steer the search
"""

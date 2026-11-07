"""Seed programs and their islands (study 40 ADR 0006).

Shinka starts every island from one initial program, copied to all islands. MEvolve instead gives
each island its own seed: the template (Shinka's initial program) and one literature model per
other island. ``place`` writes the literature seeds, already evaluated, into the run's database
before the runner starts, each on its island; ``install_island_patch`` makes Shinka's island
assignment honour a seed's ``mevolve_island``. The runner then evaluates the template and, since
islands 1 to N-1 already hold a correct program, Shinka's own rule puts it on island 0, with no
copies. Children stay on their parent's island; migration moves programs between islands later.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from ..literature import LITERATURE
from ..program import LITERATURE_DIR

START, END = "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"

TEMPLATE = f'''"""Template: a side bias and a tendency to repeat the last choice; no learning from reward."""
{START}
import jax.numpy as jnp

PARAMS = {{
    "side_bias": (-5.0, 5.0),        # a fixed preference for the right side, in logits
    "stickiness": (-5.0, 5.0),       # a pull towards repeating the last choice, in logits
}}
N_LATENT = 1
LATENT_NAMES = ("last_choice_signed",)    # +1 after a right choice, -1 after a left one, 0 at the start


def init_latents(p, session):
    return jnp.zeros(1)


def predict(latents, p):
    return jnp.array([0.0, p["side_bias"] + p["stickiness"] * latents[0]])


def update(latents, choice, reward, p):
    return jnp.array([2.0 * choice - 1.0])
{END}
'''


def wrap(source: str) -> str:
    """A program as Shinka evolves it: the whole file inside one evolve block."""
    if START in source:
        return source
    return f"{START}\n{source.rstrip()}\n{END}\n"


def literature_source(name: str) -> str:
    return wrap((LITERATURE_DIR / f"{LITERATURE[name][0]}.py").read_text())


def seed_name_and_source(seed: str) -> tuple[str, str]:
    """A seed is a literature program's name or a program file (``.py``), named by its stem."""
    if seed.endswith(".py"):
        name = Path(seed).stem
        if name in LITERATURE or name == "template":
            raise ValueError(f"seed file {seed}: the name {name} is taken (a literature program or the template)")
        return name, wrap(Path(seed).read_text())
    return seed, literature_source(seed)


def install_island_patch() -> None:
    """Make Shinka's island assignment place a program with ``metadata["mevolve_island"]`` on
    that island, with no copies. Patched on the class, because Shinka builds a fresh database
    object for every asynchronous insert."""
    from shinka.database.islands import CopyInitialProgramIslandStrategy as Strategy

    if getattr(Strategy.assign_island, "_mevolve", False):
        return
    original = Strategy.assign_island

    def assign_island(self, program):
        island = (program.metadata or {}).get("mevolve_island")
        if island is None:
            return original(self, program)
        if not 0 <= int(island) < getattr(self.config, "num_islands", 0):
            raise ValueError(f"seed island {island} is outside 0..{self.config.num_islands - 1}")
        program.island_idx = int(island)

    assign_island._mevolve = True
    Strategy.assign_island = assign_island


def place(results_dir: Path, db_config, seeds: list[dict], embedding_model: str | None = None) -> list[str]:
    """Write evaluated seeds into ``<results_dir>/programs.sqlite`` before the runner starts.

    Each seed is ``{"name", "code", "island", "metrics"}``, where ``metrics`` is what
    ``shinka_eval`` wrote. All are generation 0, so the runner does not treat the database as a
    resumed run. With ``embedding_model`` each seed gets its code embedding, as the runner gives
    every other program (the first 10,000 characters). Returns the program ids.
    """
    from shinka.database.dbase import Program, ProgramDatabase
    from shinka.embed import EmbeddingClient

    install_island_patch()
    islands = [s["island"] for s in seeds]
    if len(set(islands)) != len(islands) or 0 in islands:
        raise ValueError(f"seed islands must be distinct and leave island 0 for the template: {islands}")
    db_path = Path(results_dir) / "programs.sqlite"
    if db_path.exists():
        raise FileExistsError(f"{db_path} exists; seeds go into a fresh run")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    db_config.db_path = str(db_path)
    db = ProgramDatabase(db_config, embedding_model=None)
    embedder = EmbeddingClient(embedding_model) if embedding_model else None
    ids = []
    try:
        for s in seeds:
            m = s["metrics"]
            embedding = embedder.get_embedding(s["code"][:10000])[0] if embedder else []
            if embedder and not embedding:          # Shinka's client returns [] on any error
                raise RuntimeError(f"the embedding server returned nothing for seed {s['name']}")
            program = Program(
                id=f"seed-{s['name']}-{uuid.uuid4().hex[:8]}", code=s["code"], generation=0,
                combined_score=float(m["combined_score"]), public_metrics=m.get("public", {}),
                private_metrics=m.get("private", {}), text_feedback=m.get("text_feedback", ""),
                correct=True, embedding=embedding, metadata={"mevolve_island": s["island"], "mevolve_seed": s["name"],
                          "patch_name": s["name"], "patch_description": f"literature seed {s['name']}"})   # the portal's label
            db.add(program)
            ids.append(program.id)
    finally:
        db.close()
    return ids


def islands_summary(results_dir: Path) -> dict[int, list[dict]]:
    """``{island: [{id, generation, parent_id, score, seed}]}`` from a run's database, for checks."""
    import sqlite3

    con = sqlite3.connect(f"file:{Path(results_dir) / 'programs.sqlite'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out: dict[int, list[dict]] = {}
    for row in con.execute("SELECT id, island_idx, generation, parent_id, combined_score, correct, metadata "
                           "FROM programs ORDER BY generation, timestamp"):
        meta = json.loads(row["metadata"] or "{}")
        out.setdefault(row["island_idx"], []).append(
            {"id": row["id"], "generation": row["generation"], "parent_id": row["parent_id"],
             "score": row["combined_score"], "correct": bool(row["correct"]), "seed": meta.get("mevolve_seed")})
    con.close()
    return out

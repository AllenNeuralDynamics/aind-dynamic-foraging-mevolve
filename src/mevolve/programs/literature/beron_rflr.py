"""Beron-RFLR: recursively formulated logistic regression. Rewarded choices add to an
exponentially decaying evidence trace; the last choice adds a fixed stay term (ForagerBeronRFLR)."""
import jax.numpy as jnp

PARAMS = {
    "choice_history_weight": (-20.0, 20.0),
    "reward_evidence_weight": (-20.0, 20.0),
    "evidence_time_constant": (0.05, 100.0),
}
N_LATENT = 2
LATENT_NAMES = ("reward_evidence", "last_choice_signed")


def init_latents(p, session):
    # Trial 0 is chance: zero evidence and no last choice (signed 0).
    return jnp.zeros(2)


def predict(latents, p):
    return jnp.array([0.0, latents[0] + p["choice_history_weight"] * latents[1]])


def update(latents, choice, reward, p):
    signed_choice = 2.0 * choice - 1.0
    decay = jnp.exp(-1.0 / p["evidence_time_constant"])
    evidence = decay * latents[0] + p["reward_evidence_weight"] * reward * signed_choice
    return jnp.array([evidence, signed_choice])

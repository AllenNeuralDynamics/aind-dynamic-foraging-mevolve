"""Lebedeva-PR: a perseveration trace and a reward-seeking trace, each a leaky average of signed
choices (times signed feedback for reward seeking), summed into the right-side logit
(ForagerLebedevaPR)."""
import jax.numpy as jnp

PARAMS = {
    "perseveration_learning_rate": (0.0, 1.0),
    "reward_learning_rate": (0.0, 1.0),
    "perseveration_weight": (-20.0, 20.0),
    "reward_weight": (-20.0, 20.0),
    "choice_bias": (-20.0, 20.0),
}
N_LATENT = 2
LATENT_NAMES = ("perseveration", "reward_seeking")


def init_latents(p, session):
    return jnp.zeros(2)


def predict(latents, p):
    return jnp.array([0.0, latents[0] + latents[1] + p["choice_bias"]])


def update(latents, choice, reward, p):
    signed_choice = 2.0 * choice - 1.0
    signed_feedback = 2.0 * reward - 1.0
    a_p, a_r = p["perseveration_learning_rate"], p["reward_learning_rate"]
    perseveration = (1.0 - a_p) * latents[0] + a_p * p["perseveration_weight"] * signed_choice
    reward_seeking = (1.0 - a_r) * latents[1] + a_r * p["reward_weight"] * signed_choice * signed_feedback
    return jnp.array([perseveration, reward_seeking])

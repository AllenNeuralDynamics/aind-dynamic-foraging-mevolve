"""Hattori2019: Q-learning with separate rates after reward and no reward, and forgetting of the
unchosen side (ForagerQLearning, two learning rates, one forget rate, softmax)."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate_rew": (0.0, 1.0),
    "learn_rate_unrew": (0.0, 1.0),
    "forget_rate_unchosen": (0.0, 1.0),
    "biasL": (-5.0, 5.0),
    "softmax_inverse_temperature": (0.0, 100.0),
}
N_LATENT = 2
LATENT_NAMES = ("q_left", "q_right")


def init_latents(p, session):
    return jnp.zeros(2)


def predict(latents, p):
    return p["softmax_inverse_temperature"] * latents + jnp.array([p["biasL"], 0.0])


def update(latents, choice, reward, p):
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward > 0, p["learn_rate_rew"], p["learn_rate_unrew"])
    learned = latents + rate * (reward - latents)
    forgotten = (1.0 - p["forget_rate_unchosen"]) * latents
    return jnp.where(chosen > 0, learned, forgotten)

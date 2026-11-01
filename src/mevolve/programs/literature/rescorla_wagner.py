"""Rescorla-Wagner: Q-learning of the chosen side with one learning rate and no forgetting, with
epsilon-greedy choice (ForagerQLearning, one learning rate, no forget rate, epsilon-greedy)."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate": (0.0, 1.0),
    "biasL": (-5.0, 5.0),
    "epsilon": (0.0, 1.0),
}
N_LATENT = 2
LATENT_NAMES = ("q_left", "q_right")


def init_latents(p, session):
    return jnp.zeros(2)


def predict(latents, p):
    # The greedy side gets 1 - epsilon / 2, the other epsilon / 2; a tie gives 0.5 each.
    adjusted = latents + jnp.array([p["biasL"], 0.0])
    greedy = jax.nn.one_hot(jnp.argmax(adjusted), 2)
    eps = p["epsilon"]
    probs = jnp.where(greedy > 0, 1.0 - eps / 2, eps / 2)
    probs = jnp.where(adjusted[0] == adjusted[1], jnp.array([0.5, 0.5]), probs)
    return jnp.log(jnp.maximum(probs, 1e-300))  # epsilon = 0 gives probability 0, kept finite


def update(latents, choice, reward, p):
    chosen = jax.nn.one_hot(choice, 2)
    return jnp.where(chosen > 0, latents + p["learn_rate"] * (reward - latents), latents)

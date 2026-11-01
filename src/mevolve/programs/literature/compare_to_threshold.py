"""CompareToThreshold: track the value of the current side and stay while it beats a threshold;
P(stay) is a sigmoid of value - threshold (ForagerCompareThreshold, one learning rate, reset to
threshold on a switch, no choice kernel)."""
import jax.numpy as jnp

PARAMS = {
    "learn_rate": (0.0, 1.0),
    "threshold": (-1.0, 1.0),
    "softmax_inverse_temperature": (1e-11, 100.0),
    "biasL": (-5.0, 5.0),
}
N_LATENT = 3
LATENT_NAMES = ("value", "previous_choice", "has_previous_choice")


def init_latents(p, session):
    return jnp.array([p["threshold"], 0.0, 0.0])  # the value starts at the threshold


def predict(latents, p):
    value, previous, has_previous = latents
    # biasL is added only when the last choice was left, and not on the first trial.
    logit = (p["softmax_inverse_temperature"] * (value - p["threshold"])
             + jnp.where((has_previous > 0) & (previous == 0), p["biasL"], 0.0))
    p_stay = jnp.clip(1.0 / (1.0 + jnp.exp(-logit)), 1e-12, 1.0 - 1e-12)
    p_left = jnp.where(previous == 0, p_stay, 1.0 - p_stay)
    p_right = jnp.where(previous == 0, 1.0 - p_stay, p_stay)
    probs = jnp.where(has_previous > 0, jnp.stack([p_left, p_right]), jnp.array([0.5, 0.5]))
    return jnp.log(probs)


def update(latents, choice, reward, p):
    value, previous, has_previous = latents
    switched = (has_previous == 0) | (choice != previous)  # the first choice counts as a switch
    start = jnp.where(switched, p["threshold"], value)
    value = start + p["learn_rate"] * (reward - start)
    return jnp.stack([value, jnp.asarray(choice, dtype=jnp.float64), 1.0])

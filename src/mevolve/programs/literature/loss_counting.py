"""LossCounting: stay on a side until the run of losses there passes a noisy threshold, then switch
(ForagerLossCounting, win_stay_lose_switch False, no choice kernel)."""
import jax.numpy as jnp
from jax.scipy.stats import norm

PARAMS = {
    "loss_count_threshold_mean": (0.0, 10.0),
    "loss_count_threshold_std": (0.0, 10.0),
    "biasL": (-1.0, 1.0),
}
N_LATENT = 3
LATENT_NAMES = ("loss_count", "previous_choice", "has_previous_choice")


def init_latents(p, session):
    return jnp.zeros(3)


def predict(latents, p):
    loss_count, previous, has_previous = latents
    # P(switch) = P(threshold <= loss count), threshold ~ Normal(mean, std); the offsets are the
    # reference's (">=" for an integer threshold, and std = 0).
    p_switch = norm.cdf(loss_count, p["loss_count_threshold_mean"] - 1e-10,
                        p["loss_count_threshold_std"] + 1e-16)
    p_left = jnp.where(previous == 0, 1.0 - p_switch, p_switch)
    p_left = jnp.clip(p_left + p["biasL"], 0.0, 1.0)
    # The first trial is 0.5 each, with no bias.
    p_left = jnp.where(has_previous > 0, p_left, 0.5)
    probs = jnp.stack([p_left, 1.0 - p_left])
    return jnp.log(jnp.maximum(probs, 1e-300))  # the clip can give probability 0, kept finite


def update(latents, choice, reward, p):
    loss_count, previous, has_previous = latents
    just_switched = (has_previous == 0) | (choice != previous)
    loss_count = jnp.where(reward != 0, 0.0, jnp.where(just_switched, 1.0, loss_count + 1.0))
    return jnp.stack([loss_count, jnp.asarray(choice, dtype=jnp.float64), 1.0])

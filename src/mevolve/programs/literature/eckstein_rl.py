"""Eckstein-RL: asymmetric learning rates with counterfactual update of the unchosen side,
perseveration bonus on the last choice, and a lapse floor on the choice probability
(ForagerEcksteinRL)."""
import jax
import jax.numpy as jnp

PARAMS = {
    "positive_learning_rate": (0.0, 1.0),
    "negative_learning_rate": (0.0, 1.0),
    "softmax_inverse_temperature": (0.0, 15.0),
    "perseveration_bonus": (-1.0, 1.0),
}
N_LATENT = 3
LATENT_NAMES = ("q_left", "q_right", "last_choice_signed")


def init_latents(p, session):
    # Values start at 0.5; no last choice on trial 0 (signed 0, so no perseveration bonus).
    return jnp.array([0.5, 0.5, 0.0])


def predict(latents, p):
    right_logit = p["softmax_inverse_temperature"] * (
        latents[1] - latents[0] + p["perseveration_bonus"] * latents[2])
    p_right = 0.0001 + 0.9998 * jax.nn.sigmoid(right_logit)
    return jnp.log(jnp.array([1.0 - p_right, p_right]))


def update(latents, choice, reward, p):
    q = latents[:2]
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward == 1.0, p["positive_learning_rate"], p["negative_learning_rate"])
    target = jnp.where(chosen > 0, reward, 1.0 - reward)  # counterfactual outcome for the unchosen side
    new_q = q + rate * (target - q)
    return jnp.array([new_q[0], new_q[1], 2.0 * choice - 1.0])

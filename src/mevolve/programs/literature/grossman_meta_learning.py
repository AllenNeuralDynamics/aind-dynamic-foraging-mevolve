"""Grossman-MetaLearning: Q-learning whose negative learning rate tracks unexpected uncertainty,
with forgetting of the unchosen side and an expected-uncertainty discount on the update
(ForagerGrossmanMetaLearning)."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate_rew": (0.0, 1.0),
    "learn_rate_unrew": (0.0, 1.0),
    "forgetting_factor": (0.0, 1.0),
    "expected_uncertainty_step_size": (0.0, 1.0),
    "negative_learning_rate_step_size": (0.0, 1.0),
    "choice_bias": (-1.0, 1.0),
    "softmax_inverse_temperature": (0.0, 10.0),
}
N_LATENT = 4
LATENT_NAMES = ("q_left", "q_right", "expected_uncertainty", "negative_learning_rate")


def init_latents(p, session):
    return jnp.array([0.0, 0.0, 0.0, p["learn_rate_unrew"]])


def predict(latents, p):
    right_logit = p["softmax_inverse_temperature"] * (latents[1] - latents[0] + p["choice_bias"])
    # The reference gives chance when the meta-learning state has diverged.
    right_logit = jnp.where(jnp.isfinite(right_logit), right_logit, 0.0)
    return jnp.array([0.0, right_logit])


def update(latents, choice, reward, p):
    q = latents[:2]
    expected, negative_rate = latents[2], latents[3]
    chosen = jax.nn.one_hot(choice, 2)
    q_chosen = jnp.sum(chosen * q)
    prediction_error = reward - q_chosen
    unexpected = jnp.abs(prediction_error) - expected

    s = p["negative_learning_rate_step_size"]
    adapted = s * (unexpected + p["learn_rate_unrew"]) + (1.0 - s) * negative_rate
    adapted = jnp.where(adapted > 0.0, adapted, 0.0)  # Python max(0.0, x): nan and -0.0 give 0.0
    negative = prediction_error < 0.0
    new_negative_rate = jnp.where(negative, adapted, negative_rate)
    learning_rate = jnp.where(negative, adapted, p["learn_rate_rew"])

    learned = q_chosen + learning_rate * (prediction_error * (1.0 - expected))
    forgotten = p["forgetting_factor"] * q
    new_q = jnp.where(chosen > 0, learned, forgotten)
    new_expected = expected + p["expected_uncertainty_step_size"] * unexpected
    return jnp.array([new_q[0], new_q[1], new_expected, new_negative_rate])

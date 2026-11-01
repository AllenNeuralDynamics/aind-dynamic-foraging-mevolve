"""Alsio-DualRateSticky-RL: dual-rate Rescorla-Wagner (as FeedbackDependent-RL) with a one-step
spatial stickiness bonus (ForagerAlsioRL). Values start at 0.5;
p(right) = sigmoid(beta * (q_right - q_left) + bonus * (+1 if last choice right, -1 if left)).
Trial 0 has no previous choice and gets no bonus."""
import jax
import jax.numpy as jnp

PARAMS = {
    "positive_learning_rate": (0.0, 1.0),
    "negative_learning_rate": (0.0, 1.0),
    "softmax_inverse_temperature": (0.0, 100.0),
    "perseveration_bonus": (-20.0, 20.0),
}
N_LATENT = 3
LATENT_NAMES = ("q_left", "q_right", "last_choice_sign")


def init_latents(p, session):
    return jnp.array([0.5, 0.5, 0.0])  # last_choice_sign 0: no previous choice yet


def predict(latents, p):
    right_logit = (p["softmax_inverse_temperature"] * (latents[1] - latents[0])
                   + p["perseveration_bonus"] * latents[2])
    return jnp.stack([jnp.zeros_like(right_logit), right_logit])


def update(latents, choice, reward, p):
    q = latents[:2]
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward == 1.0, p["positive_learning_rate"], p["negative_learning_rate"])
    q = jnp.where(chosen > 0, q + rate * (reward - q), q)
    sign = 2.0 * choice.astype(jnp.float64) - 1.0
    return jnp.concatenate([q, sign[None]])

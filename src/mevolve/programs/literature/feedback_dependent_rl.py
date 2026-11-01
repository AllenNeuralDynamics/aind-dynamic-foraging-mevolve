"""FeedbackDependent-RL: Costa's Rescorla-Wagner model with separate learning rates after reward
and no reward; only the chosen value learns, the unchosen value is kept (ForagerFeedbackDependentRL).
Values start at 0.5; p(right) = sigmoid(beta * (q_right - q_left))."""
import jax
import jax.numpy as jnp

PARAMS = {
    "positive_learning_rate": (0.0, 1.0),
    "negative_learning_rate": (0.0, 1.0),
    "softmax_inverse_temperature": (0.0, 100.0),
}
N_LATENT = 2
LATENT_NAMES = ("q_left", "q_right")


def init_latents(p, session):
    return jnp.full(2, 0.5)


def predict(latents, p):
    right_logit = p["softmax_inverse_temperature"] * (latents[1] - latents[0])
    return jnp.stack([jnp.zeros_like(right_logit), right_logit])


def update(latents, choice, reward, p):
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward == 1.0, p["positive_learning_rate"], p["negative_learning_rate"])
    learned = latents + rate * (reward - latents)
    return jnp.where(chosen > 0, learned, latents)

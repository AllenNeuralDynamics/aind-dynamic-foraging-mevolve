"""FeedbackDependent-RL+Bias+CK1: Costa's feedback-dependent Rescorla-Wagner model, a fixed
right-choice bias and Bari's one-step choice kernel (ForagerFeedbackDependentRLBiasCK1).
Values start at 0.5, the kernel at 0; the kernel step size is clamped to 1, so the kernel is the
last choice. p(right) = sigmoid(beta * (dq + w * dk) + choice_bias)."""
import jax
import jax.numpy as jnp

PARAMS = {
    "positive_learning_rate": (0.0, 1.0),
    "negative_learning_rate": (0.0, 1.0),
    "softmax_inverse_temperature": (0.0, 100.0),
    "choice_bias": (-5.0, 5.0),
    "choice_kernel_relative_weight": (0.0, 1.0),
}
CHOICE_KERNEL_STEP_SIZE = 1.0  # clamped (frozen) in the reference fit
N_LATENT = 4
LATENT_NAMES = ("q_left", "q_right", "kernel_left", "kernel_right")


def init_latents(p, session):
    return jnp.array([0.5, 0.5, 0.0, 0.0])


def predict(latents, p):
    value_difference = latents[1] - latents[0]
    kernel_difference = latents[3] - latents[2]
    right_logit = p["softmax_inverse_temperature"] * (
        value_difference + p["choice_kernel_relative_weight"] * kernel_difference
    ) + p["choice_bias"]
    return jnp.stack([jnp.zeros_like(right_logit), right_logit])


def update(latents, choice, reward, p):
    q, kernel = latents[:2], latents[2:]
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward == 1.0, p["positive_learning_rate"], p["negative_learning_rate"])
    q = jnp.where(chosen > 0, q + rate * (reward - q), q)
    kernel = kernel + CHOICE_KERNEL_STEP_SIZE * (chosen - kernel)
    return jnp.concatenate([q, kernel])

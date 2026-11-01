"""Bari2019: Q-learning with one learning rate, forgetting of the unchosen side, and a one-step
choice kernel (the last choice) added inside the softmax (ForagerQLearning, one learning rate, one
forget rate, choice kernel "one_step", softmax)."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate": (0.0, 1.0),
    "forget_rate_unchosen": (0.0, 1.0),
    "choice_kernel_relative_weight": (0.0, 1.0),
    "biasL": (-5.0, 5.0),
    "softmax_inverse_temperature": (0.0, 100.0),
}
CHOICE_KERNEL_STEP_SIZE = 1.0  # clamped by the reference fit: the kernel is the last choice
N_LATENT = 4
LATENT_NAMES = ("q_left", "q_right", "kernel_left", "kernel_right")


def init_latents(p, session):
    return jnp.zeros(4)


def predict(latents, p):
    q, kernel = latents[:2], latents[2:]
    beta = p["softmax_inverse_temperature"]
    return (beta * q + jnp.array([p["biasL"], 0.0])
            + beta * p["choice_kernel_relative_weight"] * kernel)


def update(latents, choice, reward, p):
    q, kernel = latents[:2], latents[2:]
    chosen = jax.nn.one_hot(choice, 2)
    learned = q + p["learn_rate"] * (reward - q)
    forgotten = (1.0 - p["forget_rate_unchosen"]) * q
    q = jnp.where(chosen > 0, learned, forgotten)
    kernel = kernel + CHOICE_KERNEL_STEP_SIZE * (chosen - kernel)
    return jnp.concatenate([q, kernel])

"""Chen-RLCK: Q-learning on the chosen side plus a choice kernel, each with its own inverse
temperature (ForagerRLCK). Values and kernel start at 0; the unchosen value is kept;
logits = beta * q + beta_ck * kernel."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate": (0.0, 1.0),
    "choice_kernel_step_size": (0.0, 1.0),
    "softmax_inverse_temperature": (0.0, 100.0),
    "choice_kernel_inverse_temperature": (0.0, 100.0),
}
N_LATENT = 4
LATENT_NAMES = ("q_left", "q_right", "kernel_left", "kernel_right")


def init_latents(p, session):
    return jnp.zeros(4)


def predict(latents, p):
    return (p["softmax_inverse_temperature"] * latents[:2]
            + p["choice_kernel_inverse_temperature"] * latents[2:])


def update(latents, choice, reward, p):
    q, kernel = latents[:2], latents[2:]
    chosen = jax.nn.one_hot(choice, 2)
    q = jnp.where(chosen > 0, q + p["learn_rate"] * (reward - q), q)
    kernel = kernel + p["choice_kernel_step_size"] * (chosen - kernel)
    return jnp.concatenate([q, kernel])

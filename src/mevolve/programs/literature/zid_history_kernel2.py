"""Zid-HistoryKernel2-ForagingRL: compare-to-threshold foraging RL with a history kernel on staying
(ForagerZidHistoryKernel). The value of the current side is compared with a threshold; the value
restarts at the threshold after a switch, and the stay kernel grows while the animal stays."""
import jax.numpy as jnp

PARAMS = {
    "learn_rate": (0.0, 1.0),
    "threshold": (0.0, 2.0),
    "choice_kernel_step_size": (0.0, 1.0),
    "softmax_inverse_temperature": (0.0, 100.0),
    "choice_kernel_inverse_temperature": (0.0, 100.0),
}
N_LATENT = 4
LATENT_NAMES = ("value", "stay_kernel", "last_choice", "has_last_choice")


def init_latents(p, session):
    # The value starts at 1; before the first choice there is no side to stay on.
    return jnp.array([1.0, 0.0, 0.0, 0.0])


def predict(latents, p):
    value, kernel, last, has_last = latents[0], latents[1], latents[2], latents[3]
    stay_logit = (p["softmax_inverse_temperature"] * (value - p["threshold"])
                  + p["choice_kernel_inverse_temperature"] * kernel)
    stay_logit = jnp.where(has_last > 0, stay_logit, 0.0)
    # Stay logit on the last chosen side, 0 on the other: softmax gives p(stay) = expit(stay_logit).
    return jnp.where(jnp.arange(2) == last, stay_logit, 0.0)


def update(latents, choice, reward, p):
    value, kernel, last, has_last = latents[0], latents[1], latents[2], latents[3]
    c = choice.astype(jnp.float64)
    switched = (has_last > 0) & (c != last)
    previous_value = jnp.where(switched, p["threshold"], value)
    new_value = previous_value + p["learn_rate"] * (reward - previous_value)
    grown = kernel + p["choice_kernel_step_size"] * (1.0 - kernel)
    new_kernel = jnp.where((has_last > 0) & ~switched, grown, 0.0)
    return jnp.array([new_value, new_kernel, c, 1.0])

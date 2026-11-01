"""Lopez-DoubleTrace: Lopez-Yepez et al. reward value plus fast and slow choice traces
(ForagerLopezDoubleTrace). Both values move towards (chosen * reward), so the unchosen value decays;
both traces move towards the one-hot choice. Values start at 0.5, traces at 0;
logits = beta * (q + w_fast * fast_trace + w_slow * slow_trace)."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate": (0.0, 1.0),
    "softmax_inverse_temperature": (0.0, 50.0),
    "fast_choice_trace_weight": (-25.0, 25.0),
    "slow_choice_trace_weight": (-25.0, 25.0),
    "fast_choice_trace_step_size": (0.0, 1.0),
    "slow_choice_trace_step_size": (0.0, 1.0),
}
N_LATENT = 6
LATENT_NAMES = ("q_left", "q_right", "fast_trace_left", "fast_trace_right",
                "slow_trace_left", "slow_trace_right")


def init_latents(p, session):
    return jnp.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])


def predict(latents, p):
    q, fast, slow = latents[:2], latents[2:4], latents[4:]
    values = q + p["fast_choice_trace_weight"] * fast + p["slow_choice_trace_weight"] * slow
    return p["softmax_inverse_temperature"] * values


def update(latents, choice, reward, p):
    q, fast, slow = latents[:2], latents[2:4], latents[4:]
    chosen = jax.nn.one_hot(choice, 2)
    q = q + p["learn_rate"] * (chosen * reward - q)
    fast = fast + p["fast_choice_trace_step_size"] * (chosen - fast)
    slow = slow + p["slow_choice_trace_step_size"] * (chosen - slow)
    return jnp.concatenate([q, fast, slow])

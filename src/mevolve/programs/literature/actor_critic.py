"""ActorCritic: a softmax actor over per-side preferences and a critic that tracks one reward
baseline; both learn from the same reward-prediction error (ForagerActorCritic, no choice
kernel)."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate_actor": (0.0, 1.0),
    "learn_rate_critic": (0.0, 1.0),
    "biasL": (-5.0, 5.0),
    "softmax_inverse_temperature": (0.0, 100.0),
}
N_LATENT = 3
LATENT_NAMES = ("preference_left", "preference_right", "value")


def _rounded(x):
    # Round a product before it is added. Under jit, XLA fuses a multiply and an add into one
    # FMA (one rounding); numpy rounds twice. The policy feeds back into the preferences, so this
    # model amplifies that 1-ulp difference to over 1e-6 within a 1000-trial session. The select
    # (NaN stays NaN) stops the fusion; optimization_barrier and reduce_precision do not.
    return jnp.where(jnp.isnan(x), jnp.nan, x)


def init_latents(p, session):
    return jnp.zeros(3)


def predict(latents, p):
    return _rounded(p["softmax_inverse_temperature"] * latents[:2]) + jnp.array([p["biasL"], 0.0])


def update(latents, choice, reward, p):
    preference, value = latents[:2], latents[2]
    # The policy that made this choice, computed as the reference does: exp(x) / sum(exp(x)),
    # with no max shift. Above 700 the reference goes greedy; a plain softmax is used there.
    logits = predict(latents, p)
    expo = jnp.exp(jnp.minimum(logits, 700.0))
    policy = jnp.where(jnp.max(logits) > 700.0, jax.nn.softmax(logits), expo / jnp.sum(expo))
    td_error = reward - value
    preference = preference + _rounded(
        p["learn_rate_actor"] * td_error * (jax.nn.one_hot(choice, 2) - policy))
    value = value + _rounded(p["learn_rate_critic"] * td_error)
    return jnp.concatenate([preference, value[None]])

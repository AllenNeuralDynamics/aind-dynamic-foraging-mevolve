"""Eckstein-BI: Bayesian inference over which side is currently the correct one, with a
subjective switch probability, a perseveration bonus on the last choice, and a lapse floor on the
choice probability (ForagerEcksteinBI)."""
import jax.numpy as jnp
import jax

PARAMS = {
    "subjective_switch_probability": (0.0, 1.0),
    "subjective_reward_probability": (0.0, 1.0),
    "softmax_inverse_temperature": (0.0, 15.0),
    "perseveration_bonus": (-1.0, 1.0),
}
N_LATENT = 2
LATENT_NAMES = ("belief_right_correct", "last_choice_signed")

INCORRECT_CHOICE_REWARD_PROBABILITY = 1e-5


def init_latents(p, session):
    # Belief 0.5 and no last choice give exactly chance on trial 0.
    return jnp.array([0.5, 0.0])


def predict(latents, p):
    biased_belief = latents[0] + latents[1] * p["perseveration_bonus"]
    right_logit = p["softmax_inverse_temperature"] * (2.0 * biased_belief - 1.0)
    p_right = 0.0001 + 0.9998 * jax.nn.sigmoid(right_logit)
    return jnp.log(jnp.array([1.0 - p_right, p_right]))


def update(latents, choice, reward, p):
    prior_right = latents[0]
    p_reward = p["subjective_reward_probability"]
    p_noisy = INCORRECT_CHOICE_REWARD_PROBABILITY
    chose_right = choice == 1
    if_right = jnp.where(chose_right, p_reward, p_noisy)
    if_left = jnp.where(chose_right, p_noisy, p_reward)
    rewarded = reward == 1.0
    likelihood_right = jnp.where(rewarded, if_right, 1.0 - if_right)
    likelihood_left = jnp.where(rewarded, if_left, 1.0 - if_left)
    denominator = likelihood_right * prior_right + likelihood_left * (1.0 - prior_right)
    safe = jnp.where(denominator > 0.0, denominator, 1.0)
    posterior_right = jnp.where(denominator > 0.0, likelihood_right * prior_right / safe, 0.5)
    s = p["subjective_switch_probability"]
    belief = (1.0 - s) * posterior_right + s * (1.0 - posterior_right)
    return jnp.array([belief, 2.0 * choice - 1.0])

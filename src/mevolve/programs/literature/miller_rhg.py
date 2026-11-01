"""Miller-RHG: reward-seeking, habit and gambler's-fallacy traces, each a leaky average with a
logit-parameterised retention, weighted into the choice (ForagerMillerRHG)."""
import jax
import jax.numpy as jnp

PARAMS = {
    "reward_weight": (-20.0, 20.0),
    "habit_weight": (-20.0, 20.0),
    "gambler_fallacy_weight": (-20.0, 20.0),
    "reward_retention_logit": (-10.0, 10.0),
    "habit_retention_logit": (-10.0, 10.0),
    "gambler_fallacy_retention_logit": (-10.0, 10.0),
    "choice_bias": (-20.0, 20.0),
}
N_LATENT = 3
LATENT_NAMES = ("reward_seeking", "habit", "gambler_fallacy")


def init_latents(p, session):
    return jnp.zeros(3)


def predict(latents, p):
    choice_term = (p["reward_weight"] * latents[0] + p["habit_weight"] * latents[1]
                   + p["gambler_fallacy_weight"] * latents[2] + p["choice_bias"])
    # Logits [-term, +term], as in the published implementation.
    return jnp.array([-choice_term, choice_term])


def update(latents, choice, reward, p):
    signed_choice = 2.0 * choice - 1.0
    signed_feedback = 2.0 * reward - 1.0
    r_rew = jax.nn.sigmoid(p["reward_retention_logit"])
    r_hab = jax.nn.sigmoid(p["habit_retention_logit"])
    r_gam = jax.nn.sigmoid(p["gambler_fallacy_retention_logit"])
    reward_seeking = r_rew * latents[0] + (1.0 - r_rew) * signed_feedback * signed_choice
    habit = r_hab * latents[1] + (1.0 - r_hab) * signed_choice
    gambler = r_gam * latents[2] + (1.0 - r_gam) * (signed_choice - signed_feedback * signed_choice)
    return jnp.array([reward_seeking, habit, gambler])

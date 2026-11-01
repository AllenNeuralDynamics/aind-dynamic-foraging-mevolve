"""The seed library's index: each literature model, its program file and its reference forager.

Built from the dispatcher's ``baseline_rl_lit_*`` model configs (agent class and ``agent_kwargs``)
at 6450af3, plus ``baseline_rl_lit_loss_counting`` (dispatcher PR 214), the configs study 12 fit.
The keys are study 12's names; which of them are seeds follows study 12 r1's ranking
(``cross_study_literature.literature_ranking()``): Win-Stay-Lose-Shift is fitted there but not
ranked, so it has no program here.
"""
from __future__ import annotations

LITERATURE = {
    # study 12 name: (program file in programs/literature, forager class, agent_kwargs)
    "Hattori2019": ("hattori2019", "ForagerQLearning",
                    {"number_of_learning_rate": 2, "number_of_forget_rate": 1, "choice_kernel": "none",
                     "action_selection": "softmax"}),
    "Bari2019": ("bari2019", "ForagerQLearning",
                 {"number_of_learning_rate": 1, "number_of_forget_rate": 1, "choice_kernel": "one_step",
                  "action_selection": "softmax"}),
    "Rescorla-Wagner": ("rescorla_wagner", "ForagerQLearning",
                        {"number_of_learning_rate": 1, "number_of_forget_rate": 0, "choice_kernel": "none",
                         "action_selection": "epsilon-greedy"}),
    "LossCounting": ("loss_counting", "ForagerLossCounting",
                     {"win_stay_lose_switch": False, "choice_kernel": "none"}),
    "CompareToThreshold": ("compare_to_threshold", "ForagerCompareThreshold", {"choice_kernel": "none"}),
    "ActorCritic": ("actor_critic", "ForagerActorCritic", {"choice_kernel": "none"}),
    "Chen-RLCK": ("chen_rlck", "ForagerRLCK", {}),
    "Zid-HistoryKernel2-ForagingRL": ("zid_history_kernel2", "ForagerZidHistoryKernel", {}),
    "Grossman-MetaLearning": ("grossman_meta_learning", "ForagerGrossmanMetaLearning", {}),
    "Lebedeva-PR": ("lebedeva_pr", "ForagerLebedevaPR", {}),
    "Beron-RFLR": ("beron_rflr", "ForagerBeronRFLR", {}),
    "Miller-RHG": ("miller_rhg", "ForagerMillerRHG", {}),
    "Eckstein-RL": ("eckstein_rl", "ForagerEcksteinRL", {}),
    "Eckstein-BI": ("eckstein_bi", "ForagerEcksteinBI", {}),
    "FeedbackDependent-RL": ("feedback_dependent_rl", "ForagerFeedbackDependentRL", {}),
    "FeedbackDependent-RL+Bias": ("feedback_dependent_rl_bias", "ForagerFeedbackDependentRLBias", {}),
    "FeedbackDependent-RL+Bias+CK1": ("feedback_dependent_rl_bias_ck1", "ForagerFeedbackDependentRLBiasCK1", {}),
    "Alsio-DualRateSticky-RL": ("alsio_dual_rate_sticky_rl", "ForagerAlsioRL", {}),
    "Lopez-DoubleTrace": ("lopez_double_trace", "ForagerLopezDoubleTrace", {}),
}


def oracle(name: str, params: dict | None = None):
    """The reference forager of one literature model, from ``aind-dynamic-foraging-models``."""
    from aind_dynamic_foraging_models import generative_model

    _, cls, kwargs = LITERATURE[name]
    return getattr(generative_model, cls)(**kwargs, params=params or {})


def fit_spec(name: str) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """The parameters the reference fit frees, with their bounds, and the ones it clamps.

    Follows ``DynamicForagingAgentMLEBase.fit`` with no overrides, as study 12 ran it: parameters
    frozen by construction, or with equal bounds, are clamped; the rest are fitted.
    """
    agent = oracle(name)
    bounds = agent.ParamFitBoundModel().model_dump()
    clamped = dict(agent.params_list_frozen)
    free = {}
    for key, (lo, hi) in bounds.items():
        if key in clamped:
            continue
        if lo == hi:
            clamped[key] = lo
        else:
            free[key] = (float(lo), float(hi))
    return free, clamped
